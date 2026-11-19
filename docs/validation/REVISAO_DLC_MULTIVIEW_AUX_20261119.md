# Revisão — DeepLabCut (DLC) para as duas webcams auxiliares (aux0/aux1)

**Data:** 19/11/2026
**Tipo:** revisão + plano (NÃO implementado; sem hardware; sem commit/push; sem `git add -A`)
**Escopo corrigido pelo usuário:** o alvo é **DeepLabCut** (estimador de pose 2D), não o
Anipose. O Anipose/aniposelib entra apenas como backend de calibração/triangulação
do DLC no modo >2 câmeras.
**Ambiente desta revisão:** Windows 11, Python 3.11.9 do projeto; nenhuma
instalação no `.venv` do projeto — os testes de instalação/conversão rodaram em
**venvs temporários isolados** em `%TEMP%`. Nada foi alterado em `outputs/eeg`,
`configs/calibracao`, câmeras ou código do pipeline.

---

## 0. Sumário executivo (veredito)

| Pergunta | Veredito |
|---|---|
| Substituir **agora** o MediaPipe pelo DLC no pipeline das auxiliares? | **NO-GO** |
| Fazer um **spike isolado** de DLC 2D (anotar/treinar/avaliar) como comparação científica? | **GO condicional** (escopo fechado, critérios pré-registrados) |
| Adotar **DLC nativo 3D** (2 câmeras) no lugar da triangulação própria? | **NO-GO agora** |
| Adotar **Anipose/aniposelib** como backend 3D (incluindo Kinect como 3ª vista)? | **NO-GO agora** (rever só se o Kinect entrar como 3ª câmera) |
| Ganho barato e imediato fora do DLC? | **GO**: suavização temporal por track + gravação com FPS casado |

Resumo dos motivos:

1. **DLC não é um detector de mão pré-treinado.** O Model Zoo oficial tem
   `superanimal_quadruped`, `superanimal_topviewmouse` e `superanimal_humanbody`
   (17 keypoints de corpo COCO) — **não há modelo de mão/21 landmarks**. Para o
   nosso caso o DLC exigiria anotação manual do zero e treino de rede.
2. **Custo/benefício desfavorável agora:** a máquina de bancada **não tem GPU
   NVIDIA** (só Intel Arc A580 via `torch` XPU), e o DLC suporta `cpu`, `cuda` e
   `mps` — **não suporta XPU oficialmente**. Treino em CPU é "consideravelmente
   mais lento" (doc do DLC cita ~10× para ResNets). O MediaPipe atual já entrega
   reprojeção ~1,4 px e cobertura mediana 75,9% no modo 1 mão.
3. **O DLC não substitui a geometria**: a triangulação continua sendo nossa
   (`kinect_imu_groundtruth.triangulate_pair_points`) ou o 3D nativo do DLC
   (só 2 câmeras) / Anipose (>2 câmeras). Eu testei numericamente que a conversão
   da nossa calibração para o formato do Anipose é **exata** e a triangulação do
   Anipose é **idêntica ao nosso DLT** — ou seja, adotar Anipose não melhora a
   precisão geométrica por si só.
4. **Sincronismo é o gargalo maior**: DLC 3D e Anipose casam vídeos por **índice
   de frame** (não usam timestamp). Nossa gravação entrega FPS diferentes por
   câmera (~10,6 vs ~8,0) e contagens de frames diferentes (196 vs 160 em 20 s).
   Para usar DLC 3D/Anipose seria preciso um novo passo de reamostragem/
   re-encodagem para uma grade de frames comum — custo e risco não triviais.
5. **Licenças:** código DLC é LGPL-3.0 (ok para uso acadêmico/thèse; alterações
   distribuídas precisam ser LGPL); **pesos SuperAnimal são "research use only
   (non-commercial)"**. MediaPipe é Apache-2.0. Anipose/aniposelib são BSD-2.

O que **recomendo fazer já** (sem DLC): (a) suavização temporal 3D por track
(Savitzky-Golay/robusto) — o teste sintético do `triangulate_optim` do Anipose
mostrou que só isso já reduz jitter de 3,48→2,04 mm/passo; (b) gravar as duas
auxiliares com **FPS casado** (reduzir resolução ou separar controladores USB)
para qualquer trabalho multiview futuro; (c) manter o MediaPipe como default.

---

## 1. Revisão do pipeline atual (o que já existe e onde plugar)

### 1.1 Arquivos revisados

| Arquivo | Papel |
|---|---|
| `src/mestradopy/tools/handmvnet_video_record.py` | Gravador "burro" das 2 auxiliares em disco (H.264→mp4v→MJPEG), `meta.json`, CSV de timestamps por frame (wall/monotonic/brilho/flash/manual_sync), cheque de espaço, lock de exposição, sync por flash/clap/tecla S. |
| `src/mestradopy/vision/handmvnet_video_io.py` | Biblioteca compartilhada: formato de sessão, leitura dos CSVs de timestamp, detecção de flash, estimativa de offset de sincronismo, `pair_by_time`, espaço/codecs. |
| `src/mestradopy/tools/handmvnet_video_process.py` | Processador offline: carrega calibração, sincroniza timelines, roda MediaPipe HandLandmarker offline por vista, triangula par (ou 3 vistas), gate de reprojeção, track IDs por continuidade 3D, dedup por distância 3D, suavização de lateralidade por track, escrita de `landmarks_3d.csv/.npz`, `coverage_report.json/.txt`, `meta_processado.json`, estágio opcional HandMvNet. |
| `src/mestradopy/core/kinect_imu_groundtruth.py` | Matemática canônica: `CameraAlignment`, `AuxCameraView` (convenção `X_aux = R X_kin + T`), `relative_transform`, `triangulate_pair_points`, `triangulate_three_views`, `erro_reprojecao_px`, `build_projection_matrices`, mirror, handedness flip. |
| `src/mestradopy/core/stereo_calibration.py` | Calibração Kinect↔aux com tabuleiro 7×5 (quadrado 45 mm), `cv2.calibrateCamera` + `stereoCalibrate`, critérios de RMS/span/poses, grava `configs/calibracao/stereo_calibration*.npz`. |
| `src/mestradopy/vision/handmvnet_dataset.py` | Dataset HandMvNet: `labels.npz` com `joints_2d (N,V,21,2)`, `joints_3d (N,21,3)`, `bboxes`, `intrinsics`, `extrinsics`, `valid`, `hand`, `timestamps`; crop 256×256/heatmap 32×32. |
| `src/mestradopy/vision/handmvnet_adapter.py` | Contrato `HandMvNetView`/adaptador para o modelo oficial (3 vistas), fallback de triangulação. |
| `src/mestradopy/tools/visualiza_landmarks3d.py` | Visualização dos 21 landmarks 3D processados (PNG de trajetórias + GIF do esqueleto). |
| `src/mestradopy/tools/compara_tracking_mao.py` + `tools/metricas_tracking_mao.py` | Comparação online Kinect-depth × multi-view e biblioteca de métricas (cobertura, reprojeção, jitter, lateralidade, continuidade). |
| `src/mestradopy/tools/handmvnet_capture.py` / `handmvnet_train.py` | Captura 3 vistas (Kinect RGB + aux0 + aux1) com pseudo-3D da triangulação, e trainer HandMvNet (sem checkpoint oficial). |
| `src/mestradopy/tools/handmvnet_export_2views.py` | Exporta uma sessão das 2 auxiliares (`aux00/aux01.mp4` + `processed/landmarks_3d.npz` + `meta.json`) para o dataset HandMvNet de **2 vistas** (`frames/*.jpg` + `labels.npz`), com `joints_2d` vindos de `pixels_aux0/aux1` do npz. É agnóstico ao detector 2D — se o DLC substituir o MediaPipe no processador, este exporter continua funcionando. |
| `docs/validation/VALIDACAO_AQUISICAO_20261119.md` | Baseline validado da aquisição/processamento offline. |
| `docs/guias/HANDMVNET_VIDEO_OFFLINE.md`, `HANDMVNET_INTEGRACAO.md`, `HANDMVNET_TREINO.md` | Guias do pipeline e do HandMvNet. |

### 1.2 Fatos/números do baseline (usados como critério de comparação)

- Calibração vigente (`configs/calibracao/`): **aux0 RMS 1,6988 px**, **aux1 RMS
  1,3147 px**, 15 poses; Kinect intrinsics congeladas (RMS 0,631 px no doc).
  Obs.: a tabela 1 e a tabela 3 do `VALIDACAO_AQUISICAO_20261119.md` citam
  1,1580 e 1,315 px para aux1; o npz atual tem **1,3147** — corrigir a tabela
  do doc na próxima edição (não afeta este parecer).
- Sessão validada `teste_sync3`: 1280×720, H.264, **aux0 196 frames ~10,6 FPS**,
  **aux1 160 frames ~8,0 FPS**, sync manual (S): offset +16,0 ms, residual 0 ms.
- Modo par **1 mão** (`--num-hands 1`): 124 pares → **112 saídas válidas**;
  `hand`: 85 right / 27 unknown; **2 trocas de rótulo**; cobertura por junção
  min 49,1% / mediana 75,9%; **reproj aux0 1,37 px / aux1 1,40 px**;
  triangulação 112/112 ok.
- Modo 2 mãos: 187 saídas, 113 right / 72 left / 2 unknown, **143 trocas de
  rótulo** — modo não confiável para sessão de 1 mão (já mitigado por
  pareamento geométrico + dedup + track IDs nos commits recentes).
- HandMvNet: **sem checkpoint oficial**; treino atual usa pseudo-labels da
  triangulação (circularidade reconhecida nos docs).

### 1.3 Onde o DLC entraria (interfaces exatas)

1. **Troca do detector 2D:** `MediaPipeHandDetector.detect(frame_bgr, view_name,
   timestamp_ms, aux_index) -> List[HandDetection]` em
   `handmvnet_video_process.py` (linhas ~208-291). Um `DLC2DDetector` que leia
   HDF5/CSV do DLC e devolvesse `HandDetection(side, side_raw, score,
   landmarks_px (21,2))` entraria sem mexer na geometria, no sync por timestamp,
   no pareamento, no gate de reprojeção, nos track IDs nem no formato de saída.
2. **Dataset HandMvNet:** o `labels.npz` consome `joints_2d`/`joints_3d`; um
   adaptador DLC→`joints_2d` (21 keypoints) reordena as bodyparts escolhidas.
   O exportador de 2 vistas (`tools/handmvnet_export_2views.py`, commit
   `5ba5ac4`) lê `pixels_aux0/aux1` de `processed/landmarks_3d.npz` e já é
   detector-agnóstico; o trainer (`handmvnet_train.py`) passou a aceitar
   datasets de 2 vistas (commit `14711af`). Ou seja, o caminho de dataset
   HandMvNet **não precisa mudar** se o DLC entrar só como detector 2D.
3. **3D:** `triangulate_pair_points`/`triangulate_three_views` continuam sendo a
   geometria; o DLC nativo 3D/Anipose seriam um **backend alternativo** só se
   houver ganho comprovado (ver §6).

Ponto forte do desenho atual: o detector é desacoplado; a integração 2D é
pequena. Ponto fraco para DLC-3D: o pipeline **pair_by_time** por timestamps
(necessário porque os FPS são diferentes) não é usado pelo DLC/Anipose.

---

## 2. DeepLabCut — o que é, versão, requisitos e licença

- **O que é:** DeepLabCut é um **estimador de pose 2D** (detecção de keypoints
  definidos pelo usuário), com treino supervisionado e GUI de anotação; a
  triangulação 3D é um módulo separado (nativo, 2 câmeras; Anipose para >2).
- **Versão/atividade:** repo `DeepLabCut/DeepLabCut` **5.828 stars**, ativo
  (push 2026-11-19), branch main **3.0.3**; PyPI mais recente **3.0.2**
  (2026-09-23). Licença **LGPL-3.0-or-later** (`pyproject.toml`), Windows
  suportado, Python **3.10–3.12** (`requires-python >=3.10`).
- **Engine:** DLC 3.x usa **PyTorch** como engine padrão (TensorFlow em
  depreciação). Dependências base (`deeplabcut==3.0.2`): `torch>=2`,
  `torchvision`, `timm`, `albumentations<=1.4.3`, `numba>=0.60`,
  `opencv-python-headless>=4.10`, `pandas[hdf5]>=2.2,<3`, `tables>3.8`,
  `scikit-image`, `scikit-learn`, `statsmodels`, `pycocotools`, `huggingface-hub`,
  `dlclibrary`, `ruamel-yaml`, `filterpy`, etc. GUI de anotação =
  `napari-deeplabcut` + PySide6 (extra `[gui]`).
- **Dry-run de instalação (venv temporário limpo, Windows/Py3.11):** o pip
  resolveria ~96 pacotes, incluindo `torch 2.15.0`, `torchvision 0.29.2`,
  `opencv-python-headless 5.1.0.95`, `pandas 2.3.3`, `numba 0.68.1`, `tables`,
  `scikit-image` — sem conflitos em venv limpo, mas são **vários GB** e não roda
  no `.venv` do projeto (ver §7). Nenhuma instalação real foi feita no projeto.
- **GPU/suporte oficial:** docs citam `cpu`, NVIDIA CUDA (`cuda:0`) e Apple
  `mps`; **não há suporte a Intel XPU**. Doc: "A GPU is not strictly necessary,
  but on a CPU the (training and evaluation) code is considerably slower (10x)
  for ResNets".
- **Model Zoo / modelos pré-treinados:** `superanimal_quadruped`,
  `superanimal_topviewmouse`, `superanimal_humanbody` (17 keypoints COCO
  body7, RTMPose) — **não existe modelo oficial de mão**. Pesos SuperAnimal:
  **"research use only (non-commercial)"** (README do DLC).
- **Transfer learning:** para PyTorch, `deeplabcut.modelzoo.build_weight_init(
  cfg, super_animal=..., model_name=..., detector_name=..., with_decoder=False)`
  + `create_training_dataset(config, weight_init=...)` permite inicializar o
  backbone com pesos SuperAnimal e **novo decoder para qualquer número de
  keypoints** (o `superanimal_transfer_learning` clássico só vale para o engine
  TensorFlow). É a única via plausível de acelerar um treino de mão, mas o
  ganho para mão (objeto pequeno) é **não comprovado** — exige validação.
- **Anotação/treino:** doc recomenda 100–200 frames bem diversos (multi-animal)
  ou 10–20 frames de 10 vídeos (single-animal); anotação por GUI napari; treino
  com data augmentation e múltiplos shuffles; para mão em vista lateral
  esperamos precisar de **300–500 frames anotados** (2 vistas, 21 pontos) numa
  primeira iteração — várias horas de anotação + treino.
- **Saída 2D:** HDF5/CSV com MultiIndex `scorer / bodyparts / coords(x,y,likelihood)`
  — fácil de ler por pandas e de mapear para `HandDetection`.

---

## 3. Anipose/aniposelib — apenas como backend 3D do DLC (>2 câmeras)

- `anipose` (PyPI **1.1.24**, 2025-01-29; repo 474 stars, BSD-2, ativo) é o
  wrapper que o próprio DLC recomenda para **>2 câmeras** e bundle adjustment;
  instala `aniposelib` (PyPI **0.8.0**, 2026-05-07, engine clássico com
  **JAX+numba**). Existe um fork novo `aniposelib-pytorch` (1.1.0, 2026-10-27)
  com features de sync/BA, mas o **Anipose publicado usa o aniposelib clássico**.
- **Formato de calibração:** TOML com seções `[cam_i]` contendo `name`, `size`,
  `matrix`, `distortions`, `rotation` (rvec), `translation` (tvec),
  `[metadata]`; `CameraGroup.dump/load` faz o round-trip. Nosso `npz` mapeia
  direto: `matrix=auxiliary_matrix`, `distortions=auxiliary_dist`,
  `rotation=Rodrigues(R)`, `translation=T` (convenção `X_aux = R X_kin + T`,
  igual ao `cv2.projectPoints`); para o par aux0+aux1, `R_ab,T_ab` saem de
  `relative_transform(view_a, view_b)`.
- **Triangulação:** `CameraGroup.triangulate` usa SVD com JAX (mesmo DLT);
  `triangulate_optim` adiciona suavização temporal, loss robusto (soft_l1) e
  restrições de comprimento de ossos. `filter_pose.py`/`filter_3d.py` têm
  medfilt/offset/score/spline.
- **Entrada 2D:** só DLC HDF5 (`pose-2d/*.h5`) ou SLEAP (`.slp`), com
  `model_type=deeplabcut|sleap`; precisa de árvore de pastas Anipose
  (`videos-raw`, `pose-2d`, `calibration`), `cam_regex` nos nomes e
  **vídeos por frame** (sem timestamp); `convert_videos.py` só checa
  `|nframes1-nframes2| < 5`. O pipeline espera câmeras sincronizadas por frame.

### 3.1 Teste de conversão/validação feito nesta revisão (venv temporário)

Executei um teste sintético com a **calibração real** (`stereo_calibration.npz` +
`stereo_calibration_aux1.npz`), 120 frames × 21 juntas projetadas nas duas
auxiliares com ruído gaussiano de 1,4 px (≈ reprojeção validada atual):

| Teste | Resultado |
|---|---|
| `CameraGroup.dump` → `CameraGroup.load` (TOML 934 B) | round-trip **exato** (diff 0,0 em K/rvec/tvec) |
| Conversão npz→CameraGroup e triangulação sem ruído | erro 3D mediano **0,000 mm** (máx 2×10⁻⁵ mm → conversão exata) |
| Aniposelib `triangulate` × nosso DLT equivalente | **numericamente idêntico** (erro mediano 1,864 mm, p90 3,185 mm; diff 7×10⁻¹³) |
| Aniposelib `triangulate_optim` (temporal + soft_l1, sem constraints) | erro mediano **0,542 mm** (p90 1,711) e jitter/passo **2,037 mm** vs 3,478 do DLT (verdade 2,144) — em 120×21 pontos, ~4,1 s |
| Custo | 1ª chamada `triangulate` 1,3–2,2 s (compile JAX) vs DLT 0,07 s para 2.520 pontos |

**Leitura:** (a) dá para consumir nossa calibração no Anipose **sem recalibrar**,
com conversão trivial e exata; (b) a triangulação do Anipose **não é melhor** que
a nossa — é a mesma matemática; (c) o ganho real do `triangulate_optim` é
**suavização temporal/robustez**, que pode ser obtida com bem menos dependências
(filtro Savitzky-Golay/Huber por track no nosso próprio pipeline); (d) a
instalação clássica arrasta JAX/numba (~centenas de MB) — no venv temporário
funcionou (jaxlib 0.10.2 cp311 Windows), mas o `triangulate_optim` emitiu erro
de compilação numba no encerramento (resultados saíram corretos, mas o ambiente
precisa de versões pinadas/reteste antes de produção).

---

## 4. DLC × MediaPipe para o NOSSO caso (mão, 2 vistas laterais, 21 landmarks)

| Critério | MediaPipe Hands (atual) | DLC (custom hand model) |
|---|---|---|
| Pré-treino p/ 21 landmarks de mão | **sim** (modelo oficial, 21 landmarks) | **não** (Model Zoo sem mão; anotar/treinar) |
| Anotação necessária | 0 | ~300–500 frames × 21 pts × 2 vistas (estimativa) |
| Dados escassos (1 sujeito, poucas sessões) | ok (zero-shot) | risco de overfit; precisa split por sessão |
| Controle dos keypoints | fixo (21 do MediaPipe, ordem fixa) | livre (ordem/nomes nossos, ex. incluir ponta de dedo) |
| Robustez à vista lateral/oclusão | boa, mas falha em ângulos extremos (motivo dos 143 swaps no modo 2 mãos) | pode melhorar se treinada nesses ângulos; não resolve vista ausente |
| Likelihood/confiança | score de handedness + presença | likelihood por keypoint (mais informativa) |
| Tempo de inferência | leve (CPU/GPU delegado) | CNN (HRNet) — offline ok; CPU mais lento |
| Treino | não precisa | GPU recomendada; **sem NVIDIA/XPU oficial** nesta bancada → CPU/Colab |
| Licença | Apache-2.0 | código LGPL-3.0; pesos SuperAnimal research-only |
| Esforço de integração | já integrado | aderência baixa (detector plugável), mas exige treino + ambiente separado |
| Ganho esperado | baseline 1,37/1,40 px reproj, cobertura mediana 75,9% | **incerto**; meta ≥10% melhor em pixel/cobertura em sessões held-out |

Conclusão desta tabela: DLC vale a pena **só** se (1) o MediaPipe falhar de forma
sistemática nas vistas laterais difíceis e (2) houver orçamento de anotação +
treino (GPU/Colab) e tempo para validar em sessões separadas. Não é um
"drop-in replacement" barato.

---

## 5. DLC 3D nativo (2 câmeras) × nossa triangulação

- O DLC **só faz 3D nativo com 2 câmeras** ("We directly support 2-camera based
  3D"; >2 → Anipose/AcinoSet etc.). Para nosso par aux0+aux1 isso bastaria.
- Calibração DLC: `deeplabcut.create_new_project_3d(...)`, pasta
  `calibration_images` com pares nomeados `camera-1-XX.jpg`/`camera-2-XX.jpg`,
  `deeplabcut.calibrate_cameras(config3d, cbrow, cbcol, calibrate=True)`, que
  grava `camera-1_intrinsic_params.pickle` e `stereo_params.pickle` (dict com
  `cameraMatrix1/2`, `distCoeffs1/2`, `R`, `T`, `E`, `F`, `R1`, `R2`, `P1`,
  `P2`, `Q`, `roi1/2`, `image_shape`) — o DLC oficial usa tabuleiro de xadrez
  (ideal 8×6 quadrados; o nosso é 7×5 cantos internos, 45 mm — calibração já
  validada com RMS < 2 px, mas o CLI do DLC esperaria re-capturar imagens).
  **É viável converter nosso npz → pickles do DLC** (mesma convenção R/T do
  `stereoCalibrate`; P1/P2/Q via `cv2.stereoRectify` com `CALIB_FIX_INTRINSIC`),
  evitando recalibrar.
- Triangulação DLC 3D: `deeplabcut.triangulate(config3d, pasta_videos)` →
  `cv2.triangulatePoints(P1[:3],P2[:3])` depois de undistort, saída HDF5/CSV.
  **Sem timestamp**: pareia por nome de arquivo/índice de frame e **trunca o
  vídeo mais longo** para o tamanho do menor (`triangulation.py` linhas
  ~303-313). Como nossas câmeras gravam FPS diferentes e offsets de até dezenas
  de ms, o pareamento por índice fica **errado ao longo do tempo** (ex.: 10,6
  vs 8,0 FPS → ~0,5-0,6 s de deriva em 20 s). Para usar DLC-3D seria necessário
  reamostrar/re-encodar os dois vídeos para uma grade de frames comum usando
  nossos CSVs de timestamps — um passo novo, offline, com perda de frames.
- A geometria DLC-3D é equivalente ao nosso DLT (mostramos isso com o Anipose);
  o **diferencial** seria a qualidade 2D do DLC (se o modelo treinado for melhor
  que o MediaPipe) e filtros median/ARIMA do DLC — não a triangulação em si.
- O Kinect: o DLC 3D nativo não aceita 3ª câmera; para incluir o RGB do Kinect
  seria necessário Anipose (e converter a calibração/estrutura de pastas), ou
  manter nossa `triangulate_three_views` (que já existe e foi testada:
  erro 0,80 mm em teste sintético). **Não recomendo adotar Anipose agora.**

---

## 6. Instalação, ambiente e licenças (riscos concretos)

### 6.1 Não instalar no `.venv` do projeto — conflitos reais

O `.venv` do projeto tem: `pandas 3.0.6`, `opencv-python 5.0.0`, `torch
2.14.0+xpu`, **sem torchvision**, **sem numba**. O DLC 3.0.2 exige:
`pandas>=2.2,<3` (rebaixaria o projeto), `opencv-python-headless>=4.10` (outra
distribuição de cv2, risco de conflito), `torch>=2` + `torchvision` (pip
provavelmente baixaria torch CPU e sobrescreveria o wheel XPU), `numba>=0.60`
(novo), `tables`, `albumentations`, etc. **Instalar no venv do projeto é
altamente arriscado** — usar ambiente isolado (conda/venv separado ou
Colab/GPU). O dry-run em venv limpo mostra que a instalação é viável, mas o
pacote é grande (vários GB com torch/torchvision/GUI).

### 6.2 Hardware disponível

- **Sem GPU NVIDIA** (`nvidia-smi` ausente); `torch.cuda.is_available()=False`.
- `torch.xpu.is_available()=True` (Intel Arc A580), mas **DLC não suporta XPU**
  oficialmente (device docs: `cpu`, `cuda:0`, `mps`). Treino de mão em CPU =
  horas/dias; alternativa = Google Colab (o próprio doc do DLC sugere) ou
  workstation com GPU NVIDIA.
- Labeling GUI (napari/PySide6) exige `[gui]` extra; pode ser feito em outra
  máquina (ex.: notebook) e os dados copiados para a bancada.

### 6.3 Licenças

| Item | Licença | Observação |
|---|---|---|
| DeepLabCut (código) | LGPL-3.0-or-later | uso/import ok; se distribuirmos versão modificada, cumprir LGPL |
| Pesos SuperAnimal (Model Zoo) | **research use only / non-commercial** | aceitável para a tese (acadêmico), mas não redistribuir/comercializar |
| MediaPipe | Apache-2.0 | sem restrição prática |
| Anipose / aniposelib | BSD-2-Clause | permissiva |
| Modelo DLC treinado por nós | nossos pesos | sem restrição de terceiros (arquitetura LGPL) |

---

## 7. Prós/contras consolidados

### 7.1 DLC 2D (substituindo MediaPipe)

**Prós:** keypoints customizáveis; likelihood por ponto; pode aprender
aparência/iluminação/ângulos da nossa bancada; estado-da-arte em tracking de
animais; ativo (3.0.x, 2026); GUI e ecossistema maduros; possível ancoragem em
SuperAnimal-Human para transfer learning (novo decoder p/ 21 keypoints).

**Contras:** sem modelo de mão pré-treinado; exige anotação e treino (esforço e
tempo); sem GPU NVIDIA/XPU suportada (treino lento); ambiente isolado pesado;
não resolve oclusão de uma das vistas; adaptação/validação em sessões held-out
consome tempo; LGPL/weights research-only; ganho sobre MediaPipe **incerto** no
nosso caso (MediaPipe já entrega ~1,4 px com ressalvas de lateralidade).

### 7.2 DLC 3D nativo (2 cams) / Anipose

**Prós:** 3D "de fábrica" com filtros e bundle adjustment (Anipose);
`triangulate_optim` reduz jitter/suaviza (comprovado em sintético);
Anipose escala para >2 câmeras; nossa calibração converte sem recalibrar.

**Contras:** pareamento por frame (sem timestamp) quebra com nossos FPS
diferentes → precisa re-amostrar vídeos; DLC-3D nativo não inclui Kinect;
geometria equivalente ao nosso DLT (sem ganho de precisão intrínseco);
Anipose arrasta JAX/numba, estrutura de pastas rígida e DLC HDF5 no formato
dele; bundle adjustment exige calibração multi-câmera conjunta (teria de
re-processar tabuleiro com as 2 auxiliares simultaneamente ou confiar na
composição via Kinect); manutenção/dependências extras.

### 7.3 Nosso pipeline atual

**Prós:** validado (teste sintético 73/73; reproj 1,37-1,40 px; 112/112
triangulações), sync por timestamp robusto a FPS diferente, calibração
consistente/validada, track IDs/dedup, saída integrada ao HandMvNet/dataset e
EEG, leve e offline, sem dependências novas.

**Contras:** MediaPipe é caixa-preta fixa (21 pontos, sem likelihood fina);
laterais extremas/oclusão geram trocas de lado; sem bundle adjustment; sem
keypoints customizados.

---

## 8. Plano incremental (se o Lead aprovar o spike)

### Fase 0 — infraestrutura de avaliação (sem DLC; 1-2 dias)

Criar:

1. `src/mestradopy/vision/dlc_io.py`
   - `ler_h5_dlc(caminho) -> (points (F,21,3), metadata)` (MultiIndex
     scorer/bodyparts/coords; aceitar `_filtered.h5` e CSV);
   - `mapear_bodyparts(config, ordem_mp) -> ordem`;
   - `frame_para_tempo(sessao, cam_index) -> array` (reaproveita
     `handmvnet_video_io`/CSV de timestamps);
   - `DLCPredictionSource` com `get(frame, cam) -> HandDetection | None`.
2. `src/mestradopy/tools/dlc_prelabel.py`
   - Gera, para frames selecionados de uma sessão, um CSV
     `CollectedData_*.csv` no formato de labels do DLC (MultiIndex
     scorer/bodyparts/x,y) pré-preenchido com os 21 landmarks do MediaPipe
     (conversão de ordem/resolução/mirror), para acelerar a anotação; NÃO
     importa DLC (só pandas/numpy).
   - `--revisar` lista frames ambíguos (baixa confiança, fora do gate) para
     anotação prioritária.
3. `src/mestradopy/tools/dlc_eval_2d.py`
   - Modos `--sintetico` (sem DLC: gera keypoints sintéticos/ruído e valida a
     métrica e o encanamento), `--mediapipe` (roda o detector atual nas
     frames anotadas) e `--dlc-h5` (lê predições DLC);
   - métricas por frame/junta: erro em px vs anotação manual, PCK@5px, cobertura
     (nº de juntas com score ≥ pcutoff), jitter, taxa de falha; saída CSV/JSON/MD
     + figuras; comparação pareada MediaPipe × DLC.
4. `tests/manual/test_dlc_avaliacao.py` (sintético): CSV/H5 fake, conversão de
   ordem, PCK/cobertura/jitter, `dlc_prelabel` com frames e timestamps; checar
   que o CSV gerado é lido de volta e que o pipeline não depende de DLC instalado.

Critério de aceite Fase 0: `tests/manual/test_dlc_avaliacao.py` OK; com dados
sintéticos, `dlc_eval_2d.py --sintetico` gera relatório com todas as métricas e
figuras; nenhum arquivo do pipeline atual alterado.

### Fase 1 — spike DLC 2D isolado (1-2 semanas)

1. Ambiente separado (não no projeto): `py -3.11 -m venv %TEMP%\dlc_env`,
   instalar `deeplabcut[gui]` (CPU); ou conda env do DLC; ou Colab para treinar.
2. Projeto DLC de mão: `create_new_project` com bodyparts = 21 nomes
   (wrist, thumb_cmc, ..., pinky_tip); `add_new_videos` com frames extraídos das
   sessões (aux0 e aux1 no **mesmo** projeto — recomendação DLC: 1 rede treinada
   em todas as vistas); `extract_frames(numframes2pick=20-30)`; pré-anotar com
   `dlc_prelabel.py`, revisar na GUI; mínimo ~150-300 frames por vista (alvo
   ~400), split por **sessão** (nunca por frame).
3. Avaliar (rede treinada × MediaPipe × anotação manual, nas frames anotadas e
   em sessões held-out):
   - **mensagens de aceite pré-registradas:** erro mediano de pixel ≤ MediaPipe
     −10% relativo; PCK@5px ≥ MediaPipe +10% relativo; cobertura por junção
     ≥ MediaPipe +10 p.p.; jitter não pior; inferência offline aceitável
     (<2× tempo do MediaPipe por clipe de 20 s em CPU);
   - se qualquer critério falhar → arquivar o experimento (relatório) e manter
     MediaPipe; **não** integrar só por curiosidade.
4. Registrar custo real (horas de anotação/treino) no relatório para decidir
   se vale a Fase 2.

Critério de aceite Fase 1: relatório `docs/validation/DLC_2D_AVALIACAO_<data>.md`
com números nas sessões held-out e veredito objetivo; modelo/pesos guardados
fora do repo (gitignored).

### Fase 2 — integração 2D no pipeline (só se Fase 1 passar; 2-3 dias)

Alterar:

1. `handmvnet_video_process.py`: `--detector {mediapipe,dlc}`,
   `--dlc-2d-dir`, `--dlc-bodyparts`; classe `DLC2DDetector` (via
   `vision/dlc_io.py`) com a mesma interface `detect(...)`; default continua
   MediaPipe; saídas com **mesmo schema** (`landmarks_3d.npz/csv`, report).
2. `tests/manual/test_dlc_avaliacao.py` → adicionar teste de integração
   sintético (DLC fake h5 + sessão fake) provando caminho completo sem DLC real.
3. Rodar `tests/manual/test_handmvnet_video_offline.py` (73/73 esperado) e
   regenerar a sessão `teste_sync3` com `--detector dlc` (quando o modelo
   existir) e comparar com o baseline: reproj ≤1,5 px (baseline 1,37-1,40),
   cobertura ≥75,9% ou +10 p.p., track continuity igual/melhor.
4. Atualizar `docs/guias/HANDMVNET_VIDEO_OFFLINE.md` (seção detector) e o
   `HANDMVNET_TREINO.md` (labels para HandMvNet).
5. Validar que `handmvnet_export_2views.py` → `handmvnet_train.py` continua
   funcionando com `--detector dlc` (o exporter é agnóstico; só revalidar
   shapes/ordem de bodyparts e o smoke `--smoke` de 2 vistas).

Critério de aceite Fase 2: suite sintética 100% verde; relatório de comparação
na sessão real; nenhuma regressão de sync/calibração; default MediaPipe.

### Fase 3 — DLC 3D / Anipose (só se necessária; 1-2 semanas)

Gatilhos: (a) precisamos do Kinect como 3ª vista no 3D do DLC; (b) o modelo 2D
DLC melhora tanto que vale usar o 3D nativo; (c) queremos bundle adjustment.

1. `src/mestradopy/tools/dlc_3d_bridge.py`:
   - `calibrar`: converte `stereo_calibration*.npz` → projeto 3D DLC
     (`camera_matrix/*_intrinsic_params.pickle`, `stereo_params.pickle`) com
     testes de sanidade (reprojetar pontos conhecidos com P1/P2);
   - `sincronizar`: usa `timestamps_aux*.csv` para reamostrar/trim dos 2 vídeos
     em grade comum (ex.: 10 FPS, offset do flash/manual), gerando `.mp4` com
     nomes que o DLC espera (`*camera-1*`, `*camera-2*`); registrar perda de
     frames/erro de sync no meta;
   - `importar`: converte o HDF5 3D do DLC (frame da câmera-1) para o frame do
     Kinect e para o nosso `landmarks_3d.npz`.
2. Comandos no ambiente isolado do DLC: `deeplabcut.create_new_project_3d`,
   `deeplabcut.triangulate(config3d, pasta_videos)`; depois `dlc_3d_bridge.py
   importar`.
3. Teste sintético no projeto: calibração real + pontos 3D conhecidos projetados
   nos 2 vídeos-fake; comparar erro 3D DLC × `triangulate_pair_points` × verdade
   (meta: <1 mm no sintético, igual ao teste atual 0,79/0,80 mm).
4. Anipose: só avaliar se entrar Kinect/3ª câmera; usar `aniposelib` (não o
   `anipose` completo) numa fase separada; revalidar o `triangulate_optim`
   (jitter) contra o nosso baseline.
5. Critério de aceite Fase 3: 3D do DLC reproduz o baseline em sessão real
   (reproj ≤1,5 px, jitter não pior) e o importador gera o mesmo schema; senão
   descartar e manter `triangulate_pair_points`/`triangulate_three_views`.

### Fase 4 — vitória rápida independente do DLC (GO imediato sugerido; 1-2 dias)

Criar `src/mestradopy/tools/suaviza_landmarks3d.py`:
- `--npz processed/landmarks_3d.npz --metodo savgol|mediana|huber --janela 7-9`;
- suaviza por track (não por índice de junta bruta), mantém `valid`, grava
  `landmarks_3d_suave.npz` + relatório de jitter;
- avaliar em sessões reais: jitter vs baseline, reproj (não pior), comparação com
  depth do Kinect quando disponível.
- Justificativa: o teste do `triangulate_optim` mostrou que suavização temporal
  é o ganho real disponível; dá para tê-lo sem DLC/Anipose/JAX.
- Critério: jitter reduzido ≥15% sem piorar cobertura/reproj; aprovar com Lead
  antes de virar default.

---

## 9. Riscos e limitações desta revisão

1. **Sem hardware nesta revisão**: os números de instalação/conversão são de
   venv temporário + dados sintéticos com a calibração real; não rodei DLC real
   (não instalado no projeto/não treinado). O ganho do DLC em dados reais é
   **não medido** — a Fase 1 é justamente para medir.
2. **Sem GPU NVIDIA**: qualquer plano de treino depende de CPU/Colab/outra
   máquina; tempo de treino real não foi medido.
3. **Anotação manual é o custo dominante** e não pode ser estimada com precisão
   sem uma rodada-piloto (a Fase 1 deve medir horas reais).
4. **Sincronização**: DLC-3D/Anipose exigem grade comum de frames; nosso
   pipeline atual é baseado em timestamp. A Fase 3 inclui reamostragem, mas há
   risco de perda de frames/artefatos e o ganho é incerto.
5. **DLC 3.0.3 (repo) ≠ 3.0.2 (PyPI)**: fixar versão (3.0.2 do PyPI, ou commit
   do repo) e validar em ambiente isolado antes de qualquer automação.
6. **Anipose clássico** arrasta JAX/numba (instalação testada; `triangulate_optim`
   acusou erro de compilação numba no encerramento do processo, resultados
   corretos) — se adotado, pinar versões e retestar.
7. **Licenças**: LGPL do DLC e pesos SuperAnimal research-only precisam constar
   no relatório da tese se o DLC for usado; não redistribuir pesos.
8. **HandMvNet**: sem checkpoint oficial; o dataset usa pseudo-labels — um
   DLC melhor pode melhorar labels futuros, mas não resolve a circularidade por
   si.
9. **Métricas de aceite** são relativas ao baseline atual; mudanças de câmera/
   iluminação/luz podem alterar o baseline (registrar calibração/sessão).
10. Esta revisão não alterou código, dados ou configurações; os venvs/clones
    usados ficaram em `%TEMP%` (fora do repo).

---

## 10. Anexo — evidências, comandos e links

### 10.1 Comandos de teste (ambiente temporário isolado)

```powershell
# dry-run de instalacao do DLC 3.0.2 em venv limpo (sem tocar o .venv do projeto)
$py = "$env:TEMP\anipose_review\venv_dlc\Scripts\python.exe"
& $py -m pip install --dry-run deeplabcut==3.0.2

# conversao/teste Anipose com a calibracao REAL (venv isolado com aniposelib 0.8.0)
$py = "$env:TEMP\anipose_review\venv_aniposelib\Scripts\python.exe"
& $py "$env:TEMP\anipose_review\teste_anipose_calib.py"
# -> TOML round-trip exato; triangulacao identica ao DLT (diff 7e-13);
#    triangulate_optim: mediana 1.864 -> 0.542 mm; jitter 3.478 -> 2.037 mm
```

### 10.2 Fontes consultadas (links)

- DeepLabCut (código): https://github.com/DeepLabCut/DeepLabCut (v3.0.3, LGPL-3.0)
- DLC docs instalação: https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/installation.md
- DLC Model Zoo (SuperAnimal): https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/ModelZoo.md
- DLC 3D (2 câmeras; >2 → Anipose): https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/Overviewof3D.md
- Anipose: https://github.com/lambdaloop/anipose (BSD-2)
- Aniposelib: https://github.com/lambdaloop/aniposelib (BSD-2)
- PyPI: https://pypi.org/project/deeplabcut/ (3.0.2, 2026-09-23), https://pypi.org/project/anipose/ (1.1.24)
- Papers: Mathis et al. 2018 (DLC); Nath et al. 2019 (DLC protocol); Lauer et al.
  2022 (maDLC 3D); Karashchuk et al. 2021 (Anipose); Ye et al. 2024
  (SuperAnimal).

### 10.3 Arquivos do projeto revisados (caminhos relativos à raiz)

- `src/mestradopy/tools/handmvnet_video_record.py`
- `src/mestradopy/tools/handmvnet_video_process.py`
- `src/mestradopy/tools/compara_tracking_mao.py`, `tools/metricas_tracking_mao.py`
- `src/mestradopy/tools/visualiza_landmarks3d.py`, `tools/handmvnet_capture.py`,
  `tools/handmvnet_train.py`, `tools/handmvnet_infer.py`
- `src/mestradopy/vision/handmvnet_video_io.py`,
  `vision/handmvnet_adapter.py`, `vision/handmvnet_dataset.py`
- `src/mestradopy/core/kinect_imu_groundtruth.py` (`AuxCameraView`,
  `relative_transform`, `triangulate_pair_points`, `triangulate_three_views`,
  `erro_reprojecao_px`, `build_projection_matrices`)
- `src/mestradopy/core/stereo_calibration.py` (tabuleiro 7×5, quadrado 45 mm,
  `stereo_calibration*.npz`)
- `configs/calibracao/{stereo_calibration,stereo_calibration_aux1,
  kinect_intrinsics,kinect_depth_para_cor,camera_alignment}.npz`
- `docs/validation/VALIDACAO_AQUISICAO_20261119.md`,
  `docs/guias/HANDMVNET_VIDEO_OFFLINE.md`, `HANDMVNET_INTEGRACAO.md`,
  `HANDMVNET_TREINO.md`
