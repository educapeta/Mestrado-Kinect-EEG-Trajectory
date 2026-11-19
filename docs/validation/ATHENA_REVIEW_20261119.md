# Revisão — ATHENA (Automatically Tracking Hands Expertly with No Annotations) para as 2 webcams auxiliares

**Data:** 19/11/2026
**Tipo:** revisão técnica profunda + **teste executável real** (ambiente isolado; sem hardware; sem commit/push; sem `git add -A`; nada alterado no `.venv` do projeto, em `configs/calibracao/` ou nos dados piloto)
**Alvo:** ATHENA ([GitHub](https://github.com/neural-control-and-computation-lab/athena), [docs](https://neural-control-and-computation-lab.github.io/athena/), PyPI `athena-tracking`, paper J Neurophysiol 2025/2026)
**Pergunta:** substituir ou complementar o pipeline atual MediaPipe + triangulação ([handmvnet_video_process.py](../../src/mestradopy/tools/handmvnet_video_process.py)) nas 2 auxiliares (aux0 SIGMA-W780M, aux1 Trust USB)?
**Versões testadas:** `athena-tracking==0.3` (o que o `pip install athena-tracking` instala no Python 3.11 do projeto) **e** `main` atual do GitHub (`__version__ = 1.0`, não publicado no PyPI)
**Sessão de teste:** `F:\handmvnet_videos\teste_sync3` (aux00 196 frames ~10,6 FPS, aux01 160 frames ~8,0 FPS, H.264 1280×720, sync manual S, offset +16,0 ms) + `configs/calibracao/stereo_calibration*.npz`
**Ambientes dos testes (fora do repo):** `%TEMP%\athena_smoke_venv` (main v1.0) e `%TEMP%\athena03_venv` (0.3), Python 3.11.9, mediapipe 0.10.21, av 18.1.0, numpy 1.26.4, opencv 4.11, scipy 1.17.1
**Revisão paralela relacionada:** [REVISAO_DLC_MULTIVIEW_AUX_20261119.md](REVISAO_DLC_MULTIVIEW_AUX_20261119.md) (DeepLabCut/Anipose) — este documento complementa, não duplica.

---

## 0. Sumário executivo (veredito)

| Pergunta | Veredito |
|---|---|
| **Substituir** o MediaPipe+triangulação atual pelo ATHENA nas 2 auxiliares? | **NO-GO agora** |
| **Complementar** o pipeline (usar o ATHENA como backend alternativo/validação e portar seus 2 passes de correção + suavização)? | **GO condicional** (spike de 2–3 dias) |
| Usar `athena-tracking==0.3` do PyPI em produção? | **NO-GO** (versão de 2024, instalação quebrada, bugs; serve só como prova de conceito) |
| Usar a versão `main v1.0` do GitHub? | **GO condicional** em venv Python **3.12** isolado, como backend de comparação; não como default |
| Recalibrar as câmeras para o ATHENA? | **NÃO é necessário** — a conversão da nossa calibração é exata (testada) |
| Kinect como 3ª câmera no ATHENA? | Possível em princípio (RGB, mesma world frame), mas **sem depth** e exige vídeo sincronizado frame-locked; não recomendado agora |

O ATHENA **rodou de ponta a ponta nos nossos dados reais** (com adaptador mínimo de vídeos/calibração) e a **geometria concordou com a nossa triangulação atual em 1,5 mm (mediana)** após corrigir a inversão de rótulos L/R. Mas ele **não melhora a precisão geométrica** (nosso DLT já entrega reprojeção 1,37–1,40 px; o ATHENA deu 2,3–3,9 px), exige **videos frame-locked** (reamostragem que o nosso sync por timestamp já resolve melhor) e adiciona **ambiente Python 3.12 separado**. O ganho real e portável é barato: (a) **reasseguramento/rejeição de mãos por punho 3D do Pose** (resolve lateralidade/duplicatas — na sessão real: 11 swaps, 10 rejeições); (b) **suavização temporal SavGol/estabilização 3D** (jitter 3,32 → 0,65 mm/frame²); (c) validação científica independente com paper em J Neurophysiol.

---

## 1. O que é e como funciona

### 1.1 Identidade

- **ATHENA** (Mulla, Costantino, Freud, Michaels; *J Neurophysiol* 2025, doi [10.1152/jn.00407.2025](https://doi.org/10.1152/jn.00407.2025); preprint [bioRxiv 2025.08.12.669753](https://www.biorxiv.org/content/10.1101/2025.08.12.669753)):
  toolbox Python **open-source (MIT)** para tracking 3D markerless de **mãos + corpo + face** a partir de vídeos multi-câmera sincronizados.
- **Nada de anotação manual**: usa MediaPipe (Hands/Pose/Face pré-treinados) para 2D e triangulação DLT para 3D, com refinamento (correção de troca de mãos entre câmeras, filtro de câmeras outlier, Savitzky-Golay, e — na versão atual — bundle adjustment opcional de calibração e backend HaMeR opcional).
- Validação publicada contra **OptiTrack** (5 câmeras, 100 Hz): R² > 0,90 na maioria das tarefas; RMSD < 1 cm (grip aperture), < 4 cm/s (velocidade do punho), 5–10° (flexão MCP); 8 câmeras FLIR 1280×1024 @ 60 Hz, calibração JARVIS com reprojeção < 0,5 px.

### 1.2 Pipeline (versão `main` v1.0, a atual)

```
main_folder/
├── videos/<recording>/cam0.avi, cam1.avi, ...     # videos SINCRONIZADOS (frame i = frame i)
├── calibration/cam0.yaml, cam1.yaml, ...          # JARVIS YAML (ou um .toml Anipose)
└── (saídas) images/, imagesrefined/, landmarks/, videos_processed/
```

**Fase 1 — `athena.labels2d`** (por câmera, em paralelo):
1. abre o vídeo com **PyAV** (`av`), lê `average_rate` do container para o FPS;
2. **remapeia a imagem NÃO distorcida** (`cv2.initUndistortRectifyMap(K, dist, None, K, size)` + `cv2.remap`) antes de inferir — logo, os 2D salvos estão em **pixels da imagem ideal (sem distorção)**;
3. roda **MediaPipe HandLandmarker** (`num_hands=2`, 3 confianças) e **PoseLandmarker** (33 landmarks) no modo VIDEO, com `timestamp_ms = framenum*1000/fps` (relógio sintético por índice, sem timestamp real);
4. salva por câmera: `2Dlandmarks_{body,right,left}.npy` shape `(nframes, N, 5)` = `[x_px, y_px, z, visibility, presence]`, sentinela `-1`; `2Dworldlandmarks_*.npy`; `handedness_score.npy`;
5. **passe cross-camera** (`_reassign_hands_from_3d_wrists`): triangula os **punhos do Pose** (landmarks 15=left, 16=right) entre todas as câmeras, reprojeta nos pixels de cada câmera e: (a) troca rótulos left/right da mão se o punho da mão estiver mais perto do punho 3D da outra mão; (b) **rejeita** mão com raiz (landmark 0) a > 60 px do punho 3D reprojetado (padrão `reject_threshold=60.0`);
6. opcionalmente: face mesh (478 landmarks) e/ou backend **HaMeR** (malha MANO 778 vértices, PyTorch opcional).

**Fase 2 — `athena.triangulaterefine`**:
1. carrega os 2D de todas as câmeras; **`np.stack` exige o MESMO número de frames em todas** (vídeos frame-locked);
2. `_switch_hands`: 2ª correção de lateralidade (Parte A por proximidade dos punhos do Pose em 2D; Parte B por triangulação dos punhos das mãos com filtro de reprojeção e comparação com as estimativas 3D reprojetadas);
3. undistort dos 2D com `dist=0` (já são da imagem ideal) → DLT em SVD (`_triangulate_batch`) com **filtro iterativo de câmera por erro de reprojeção** (limiar **30 px**, mínimo 2 câmeras);
4. **Savitzky-Golay** por landmark (`_smooth3d`): `window_length = max(3, int(2*fps/cutoff+1) | 1)`, `polyorder=3`, interpola NaNs e restaura runs > 5 frames; cutoff default de GUI/paper = **10 Hz**;
5. reprojeta o 3D suavizado de volta em 2D e salva:
   - `landmarks/<trial>/<trial>_3Dlandmarks.npy` — **`(nframes, 75, 3)`**, unidades da calibração (JARVIS em mm; nós usamos **metros**);
   - `<trial>_2Dlandmarksrefined.npy` — `(ncams, nframes, 75, 2)`;
   - por câmera: `2Dlandmarks_*.npy`;
   - opcional: imagens/vídeos com overlay 2D/3D.
6. Ordem fixa dos 75 landmarks: **0–32 corpo (Pose), 33–53 mão direita, 54–74 mão esquerda**; punhos = 33 (direito) e 54 (esquerdo). Com face mesh: 75 + 478 = 553.
7. **Não há timestamps, track IDs nem confiança no arquivo 3D** — a identidade é só o rótulo left/right (e o índice do frame).

### 1.3 Opcional (só na versão `main`, não no PyPI): refinamento de calibração e HaMeR

- **`athena.calibration_refine`**: bundle adjustment em janelas deslizantes sobre os landmarks de mão, otimizando **só as extrínsecas** (intrínsecas fixas), loss soft-L1, guia `ref_cam=0`, detecção de drift (rotação > 2°, translação > 20 "mm"). Salva em `calibration/refined/` sem sobrescrever o original.
  - **Limitação dura para nós:** o default é `min_cameras=3` (landmark precisa ser visto em ≥3 câmeras) → **com 2 auxiliares o refinamento é pulado**. Forçar `min_cameras=2` exigiria alterar o código e ficaria sub-restrito (sem câmera redundante).
- **`athena.hamer_hands`**: HaMeR + PyTorch opcional (malha densa); irrelevante para o nosso alvo de 21 keypoints.

### 1.4 Módulos e API

| Módulo | Papel |
|---|---|
| `athena.athena` | GUI Tkinter (CLI `athena`) |
| `athena.labels2d` | Fase 1; `read_calibration`, `run_mediapipe`, `process_camera`, `_reassign_hands_from_3d_wrists` |
| `athena.triangulaterefine` | Fase 2; `main(json)`, `_triangulate_batch`, `_triangulate_with_filtering`, `_switch_hands`, `_smooth3d` |
| `athena.calibration_refine` | refinamento opcional de extrínsecas (≥3 cams por default) |
| `athena.visualization` | esqueleto/cores/mesh |
| `athena.hamer_hands` | backend HaMeR opcional |
| `athena.montage` | vídeo montagem multi-câmera (`athena-montage`) |

- **CLI oficial:** `athena` (GUI Tk) e `athena-montage`; o fluxo por linha de comando **não existe** — para automação usam-se as funções `labels2d.main(json)` / `labels2d.run_mediapipe(...)` / `process_camera(...)` e `triangulaterefine.main(json)`. `labels2d.main` ainda abre uma janela Tk de progresso; o caminho headless é chamar `process_camera`/`run_mediapipe` + `_reassign_hands_from_3d_wrists` diretamente (foi o que fizemos).
- **Multiview/2 câmeras:** suporta **≥2 câmeras**; com apenas 2, o filtro de câmera outlier não tem o que rejeitar (mínimo 2), então a filtragem de reprojeção é inócua e a correção de lateralidade disponível é a do Pose.
- **Kinect:** **não há suporte a depth/point cloud**; é RGB puro. Em tese, o RGB do Kinect poderia entrar como mais uma câmera na mesma world frame (a nossa `world = frame de cor do Kinect` implicaria `E_kinect = [I|0]`) se houvesse vídeo + sincronismo frame-locked — mas sem depth, sem ganho sobre o que já fazemos, e não é o caso de uso do ATHENA.

---

## 2. Instalação / compatibilidade

### 2.1 O que o PyPI entrega vs o GitHub

| Item | `athena-tracking` no PyPI | GitHub `main` |
|---|---|---|
| Versão | **0.3** (2024-11-10), 0.4/0.9 (2025) | **1.0** (`__version__ = "1.0"`, não publicado) |
| Requires-Python | 0.2/0.3: `>=3.9`; 0.4/0.9: `==3.12.*` | `==3.12.*` |
| Deps | 0.3 metadata: **`pyav` (não existe!)**, `mediapipe`, `numpy`, `matplotlib`, `tqdm`, `scipy`, **`pymovie` (projeto de astronomia, não relacionado)** | `av`, `mediapipe==0.10.21`, `numpy`, `matplotlib`, `tqdm`, `scipy`, `toml`, `moviepy` |
| Código | API antiga (`readcalibration`, `createvideo`), sem reasseguramento cross-camera, sem HaMeR/face/refino | API atual + os passes e opcionais acima |
| Licença | MIT | MIT |

**Fatos medidos:**
1. No **Python 3.11.9 do projeto**, `pip install athena-tracking==0.3` **falha** com:
   `ERROR: Could not find a version that satisfies the requirement pyav (from athena-tracking) (from versions: none)`.
   O pip ainda **ignora silenciosamente** 0.4/0.9 (Requires-Python `==3.12.*`) — ou seja, no venv do projeto você não instala uma versão atual nem a antiga "limpa".
2. Workaround testado: `pip install --no-deps athena-tracking==0.3` + `pip install "numpy<2" opencv-python mediapipe==0.10.21 scipy matplotlib tqdm av` (2 min). Funciona.
3. **Bug no 0.3**: `hand_model_path = pkg_resources.path(models, "hand_landmarker.task")` devolve um **context manager**, passado direto a `BaseOptions(model_asset_path=...)` → `TypeError: expected str, bytes or os.PathLike object, not _GeneratorContextManager`. É preciso patch de 2 linhas (caminho real).
4. Em Python **3.12**, instalar `athena-tracking==0.9` do PyPI **falha resolução** (`ResolutionImpossible`): o metadata do 0.9 pede `pymovie` (astronomia, `Requires-Python <3.11`) e arrasta pinos antigos (`numpy==1.23.1`, `opencv-python==4.6.0.66`, `pyephem`, `ravf`, `winshell` sem distribuição). **Não use PyPI; instale do GitHub** (`pip install git+https://github.com/neural-control-and-computation-lab/athena.git` ou `main` local) em venv **3.12 limpo**.
5. `mediapipe==0.10.21` **tem wheel cp312 win_amd64** (download testado, 51 MB), então o pin do ATHENA roda no 3.12 do Windows.

### 2.2 Dependências/pegada

- Obrigatórias (main): `mediapipe==0.10.21`, `av` (PyAV/FFmpeg), `numpy`, `opencv-python`, `scipy`, `matplotlib`, `tqdm`, `toml`, `moviepy`.
- O wheel traz os modelos `.task` embutidos: `hand_landmarker.task` (~7,8 MB) e `pose_landmarker_full.task` (~9,4 MB); face/HaMeR opcionais baixam/instalam à parte.
- **Python 3.12 exato** — o ambiente é separado do `.venv` (3.11.9) do projeto; isso implica duas pilhas de MediaPipe em convivência (0.10.21 no ATHENA × 1.0.1 no projeto), o que é aceitável para um backend de comparação, mas é custo/risco de manutenção.
- Testado por nós em Python 3.11 **forçando** o source do main (com deps instaladas manualmente) — funcionou, mas **não** é o ambiente suportado pelo projeto (o pyproject bloqueia 3.11).

### 2.3 Manutenção, testes, issues

- Repo: 43 stars, 10 forks, **0 issues abertas**, último commit de código 12/05/2026 (o push mais recente é de 04/08/2026, só "Added sample data"), tags `v0.4` e `v0.9`; sem CI de testes (`.github/workflows` só tem `publish.yml`).
- Testes: `tests/test_athena.py` + `tests/test_calibration_refine.py` → **75 passed em 103,9 s** (na nossa máquina, venv isolado); `tests/test_face_reprojection.py` **falha na coleta** sem o `sampledata/` (é teste de integração dependente de dados: "Loading calibration... 0 cameras" → `ValueError: need at least one array to stack`). Não há teste end-to-end de MediaPipe/vídeo real.
- Sample data no repo: 8 câmeras AVI (~83–104 MB cada) em `sampledata/`, com `metadata.csv`/`triggerdata.csv` do JARVIS — **o ATHENA ignora esses arquivos** e assume frame-locked (evidência direta de que ele não usa timestamps).
- Docs: a página mkdocs está **desatualizada** em relação ao README (não menciona HaMeR/face/refino); o README é a referência mais atual.
- Licença **MIT** (código); modelos MediaPipe são Apache-2.0 — uso acadêmico/tese sem restrição; se portarmos código, incluir atribuição MIT.

---

## 3. Calibração: formato, conversão e validação

### 3.1 O que o ATHENA exige

- **Formato, não procedimento**: o ATHENA **não calibra câmeras** (não tem captura de tabuleiro). Ele consome:
  - **YAML JARVIS** por câmera (`intrinsicMatrix`, `distortionCoefficients`, `R`, `T`) — o código lê `R.transpose()` e `intrinsicMatrix.transpose()` (o JARVIS armazena transposto) e usa `T` como está;
  - **TOML Anipose** (um arquivo só, `cam_0..cam_N`; `rotation` tratado como **rvec** no código, `matrix`, `distortions`, `translation`);
  - **convenção:** `X_cam = R X_world + T` (world→camera) — os 3D finais ficam na **world frame da calibração** (na sample data: câmera `Camera_LU` com `R=I, T=0`).
- Câmeras calibradas **em um frame comum** (não é calibração par-a-par): no nosso caso isso é perfeito porque `stereo_calibration*.npz` dá `X_aux = R X_kin + T` para cada auxiliar, tudo no frame do Kinect → **world = frame de cor do Kinect** e os 3D saem exatamente no frame que o nosso pipeline usa (sem transformação rígida extra).
- **Não aceita depth do Kinect**; o Kinect só entraria como mais uma RGB calibrada.

### 3.2 Conversão da nossa calibração (testada: EXATA)

Para cada auxiliar (`stereo_calibration.npz` / `stereo_calibration_aux1.npz`), escrevendo YAML JARVIS:

```python
fs = cv2.FileStorage("cam0.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("intrinsicMatrix", K.T)            # read_calibration transpõe -> K
fs.write("distortionCoefficients", dist.reshape(1, -1))
fs.write("R", R.T)                          # read_calibration transpõe -> R (X_aux = R X_world + T)
fs.write("T", T.reshape(3, 1))              # ATENÇÃO ao metro vs milímetro (nós usamos METROS)
fs.release()
```

Resultado medido: `athena.labels2d.read_calibration(...)` devolve **R, T, K e dist idênticos ao npz** (diff máxima 0,0 em todas as entradas). Não é preciso recalibrar nem reestimar intrínsecas.

### 3.3 Validação da calibração com o ATHENA (teste sintético de fase 2)

Construímos 75 landmarks 3D no frame do Kinect (forma real da mão direita vinda de `processed_1mao/landmarks_3d.npz`, espelhada para a esquerda; corpo com só os punhos 15/16 válidos), projetamos nas 2 auxiliares com a **nossa** `K`/`R`/`T` (no espaço da imagem ideal, como o ATHENA vê), adicionamos ruído de 0,3 px, e rodamos a **fase 2 real do ATHENA main v1.0**:

| Cenário | Erro 3D mediano vs ground truth no frame do Kinect |
|---|---:|
| **Conversão correta** (YAML com `R.T`/`K.T`) | **0,317 mm** (p90 0,584; máx 1,962) |
| Controle negativo: `R`/`K` **sem** transposição | 2 426 mm (falha grosseira) |
| Corpo do Pose todo `-1` (só mãos) | 0,317 mm (mãos triangulam igual) |

→ **A conversão está correta e o 3D sai no frame do Kinect.** O controle negativo prova que a convenção `R.transpose()`/`K.transpose()` do YAML é obrigatória.

---

## 4. Encaixe no nosso caso (2 webcams, mão, 8–10 fps, sync manual)

### 4.1 O que funciona

| Nosso requisito | ATHENA | Observação |
|---|---|---|
| 2 câmeras | Sim (≥2) | filtro de câmera outlier fica inócuo com só 2 |
| mão 21 landmarks | Sim (blocos 33–53 / 54–74) | + corpo 33 e face opcional |
| ler nossos vídeos | Sim via PyAV, **mas** | H.264 MP4 decodifica; **fase 2 do 0.3 só lê `*.avi`**; e exige **mesmo nº de frames** em todas as câmeras |
| nossa calibração | Sim | YAML convertido (acima), sem recalibrar |
| 3D no frame do Kinect | Sim | world = frame do Kinect (se ordenarmos as câmeras aux00=0, aux01=1) |
| sync manual por flash/S | **Não** | ATHENA **ignora timestamps**; assume frame-locked (frame i de uma câmera = frame i da outra) |
| 8–10 fps reais com FPS de container 30 | **Problema** | ver 4.2 |
| track IDs / lateralidade | **Parcial** | lado L/R corrigido por Pose; **sem** track_id; no nosso setup os rótulos saem **invertidos** (ver 4.3) |
| exportar para o dataset HandMvNet | Precisa de contra-adaptação | 2D do ATHENA estão em pixels **não distorcidos**, e não há timestamps/pair/track_id |

### 4.2 Sincronismo é o gargalo (o mesmo do DLC/Anipose)

- O ATHENA **não lê os nossos CSVs de timestamp**. A fase 2 faz `np.stack` dos arrays por câmera, que **exige igual número de frames**; com nossos vídeos reais (196 × 160) o erro exato é:
  `ValueError: all input arrays must have the same shape`.
- Solução que testamos (adaptador): casar por timestamp com o nosso protocolo (eventos `manual_sync`: offset +16,0 ms) → **125 pares**; regravar 2 vídeos **frame-locked** com 1 frame por par (125 frames) na taxa escolhida (6 fps nos testes) e rodar o ATHENA nesses vídeos.
- Consequências: (a) o vídeo reamostrado duplica frames dos pares e **muda o eixo de tempo efetivo**; (b) o SavGol usa o **FPS do container** — e o nosso gravador grava container **30 fps** enquanto a captura real é ~9,7/7,9 fps; qualquer ferramenta que leia o FPS do container (ATHENA, Anipose, DLC-3D) **interpreta o tempo 3× errado**. No bridge é obrigatório regravar com o FPS efetivo correto (ou corrigir o metadata do gravador).

### 4.3 Lateralidade e IDs: o achado mais importante

- O ATHENA tem 2 passes de correção de troca de mão (Pose 2D + triangulação dos punhos 3D), e **eles funcionam**: no teste end-to-end real (v1.0) a triangulação dos punhos do Pose detectou 125/125 frames com os 2 punhos e aplicou **11 swaps e 10 rejeições** (>60 px).
- **Porém** o rótulo final herdou a convenção “selfie” do MediaPipe: no nosso setup físico (vídeos gravados sem espelho, após o `transform_auxiliary_frame`), o bloco “right” do ATHENA corresponde à **mão esquerda física** e o bloco “left” à **mão direita física**. Evidência quantitativa (sessão real, 125 pares, comparando com a nossa triangulação já calibrada contra o Kinect):
  - bloco “right” do ATHENA × nossa mão direita = **283,7 mm** (mediana) → não é a mesma mão;
  - bloco “left” do ATHENA × nossa mão direita = **1,5 mm** (mediana; p90 6,7 mm) → é a mesma mão.
- Isso é **exatamente** o problema que o nosso `configs/calibracao/aux_handedness_flip.json` já corrige (flip=true nas 2 câmeras, medido contra a geometria do Kinect). Ou seja: **para usar o ATHENA temos que aplicar um mapa de lado validado contra o Kinect** (swap global dos blocos), como já fazemos no nosso detector.
- **Track IDs:** o ATHENA não tem noção de identidade temporal (não há `track_id`, nem dedup, nem “uma mão por sessão”). Com `num_hands=2` fixo, nas nossas vistas ele vê as 2 mãos (a mão-alvo + a outra mão no teclado, etc.) e separa apenas por lado — o que é suficiente para trials unimanuais **se** o lado for confiável (ver remap acima), mas não substitui o nosso `track_id`/dedup/continuidade 3D.
- Cobertura (ATHENA v1.0, 125 pares, confiança 0,5): cam0 right 75/125 e left 106/125; cam1 right 118/125 e left 116/125; Pose detectado em 125/125 nas 2 câmeras. No 0.3 (sem os passes cross-camera): right 84/125, left 114/125 (cam0) e right 116/125, left 122/125 (cam1).

### 4.4 Pose: precisa baixar a confiança (default do ATHENA não serve)

Em frames reais das nossas câmeras:
- **conf 0,5** → Pose detectado **12/12** nas 2 câmeras;
- **conf 0,9** (default da GUI e do paper) → **aux0: 0/12** (!!!), aux1: 11/12.

Como o passe de correção de lateralidade depende do Pose, usar o default 0,9 **anula** o principal diferencial do ATHENA na nossa bancada. O bridge precisa fixar `pose_confidence≈0.5`.

### 4.5 Saídas → nosso dataset

- O que o ATHENA entrega: `(nframes,75,3)` 3D + `(ncams,nframes,75,2)` 2D refinado + 2D por câmera. **Não** entrega: timestamps, `pair`, `frame0/frame1`, `track_id`, `valid`, reprojeção por vista.
- Conversão necessária para o nosso `labels.npz` (`handmvnet_export_2views.py` espera `joints_xyz`, `valid`, `pixels_aux0/1`, `hand`, `track_id`, `frame0/frame1`, `time_sync_s`, `pair`):
  1. mapear frame k do ATHENA → par (i0,i1) do bridge → `frame0/frame1/pair/time_sync_s`;
  2. converter os 2D do ATHENA (imagem ideal) de volta para **pixels do vídeo original** (com distorção): `undist = cv2.undistortPoints(px_ideal, K, 0); px_orig = cv2.projectPoints(undist, rvec=0, tvec=0, K, dist)` (ou exportar frames não distorcidos);
  3. `track_id`: rodar o nosso `_atribui_tracks` sobre os 3D do ATHENA (ou id fixo por lado em sessão unimanual);
  4. `valid`: máscara de finitos + gate de reprojeção (o ATHENA não expõe a máscara de câmeras rejeitadas).
- Alternativa mais segura: usar o ATHENA como **backend de comparação/validação** e manter o nosso `npz` como fonte da verdade.

---

## 5. O que testamos de verdade (comandos, resultados, saídas)

> Tudo em `%TEMP%` (fora do repo). Nada de hardware. O `.venv` do projeto não foi tocado.

### 5.1 Instalação

```powershell
# 0.3 — tentativa exata (FALHA)
python -m venv $env:TEMP\athena03_venv
& $env:TEMP\athena03_venv\Scripts\python.exe -m pip install athena-tracking==0.3
# ERROR: Could not find a version that satisfies the requirement pyav ... (from versions: none)

# 0.3 — workaround que funcionou
& ...\python.exe -m pip install --no-deps athena-tracking==0.3
& ...\python.exe -m pip install "numpy<2" opencv-python mediapipe==0.10.21 scipy matplotlib tqdm av

# main v1.0 — fonte do GitHub (raw) + venv com deps equivalentes
# (pip do main exige Python 3.12; testamos em 3.11 com PYTHONPATH para o pacote baixado)
```

### 5.2 Adaptador usado (nosso formato → ATHENA)

1. `timestamps_aux00/01.csv` + eventos `manual_sync` → offset **+16,0 ms**; pareamento por tempo com tolerância 34 ms → **125 pares** (196×160 frames originais).
2. 2 vídeos frame-locked (frame k = par k): para o 0.3, **AVI MJPG** `cam0.avi` (17,9 MB) e `cam1.avi` (12,8 MB) @6 fps (fase 2 do 0.3 só lê `*.avi`); para o main v1.0, MP4 @6 fps (aceita mp4).
3. `calibration/cam0.yaml`/`cam1.yaml` JARVIS a partir dos npz (seção 3.2).
4. Estrutura: `mainroot/{videos/rec1/cam0.*, videos/rec1/cam1.*, calibration/, landmarks/rec1/}`.
5. Fase 1 headless (sem Tk): `readcalibration` → `process_camera` por câmera (v1.0: + `_reassign_hands_from_3d_wrists`) → fase 2 `triangulaterefine.main(json)`; no 0.3 passando `hand_centroid_lfc`/`all_landmarks_lfc`, no v1.0 passando `all_landmarks_lfc` — sempre com cutoff ajustado ao FPS para não estourar o SavGol.

### 5.3 Resultados (125 pares, 2 auxiliares, sessão `teste_sync3`)

**ATHENA 0.3 (PyPI):**

| Métrica | Resultado |
|---|---|
| Fase 1 (hand+pose, 2 cams, 720p, CPU) | **15,6 s** (250 frames) |
| Fase 2 | **0,8 s** |
| Saída 3D | `rec1_3Dlandmarks.npy` **(125, 75, 3)** + `rec1_2Dlandmarksrefined.npy` |
| Cobertura 2D (body/right/left) | cam0 125/84/114; cam1 125/116/122 (de 125) |
| Bloco 3D com dados | right 84/125 frames; left 125/125 |
| Reprojeção 3D-vs-2D do próprio ATHENA | right **2,93 px** (p90 210 px!); left **3,93 px** (p90 31,6) |
| Concordância com nossa triangulação (após remap de lado) | **1,7 mm** mediana (p90 34,1) |
| Jitter (2ª diferença) | right 0,49 / left 1,79 mm/frame² |
| Rótulos L/R | **invertidos** (bloco right ≈ mão esquerda física: 283,7 mm) |

**ATHENA main v1.0 (GitHub):**

| Métrica | Resultado |
|---|---|
| Fase 1 (2 cams, sequencial) | **14,0 s** (250 frames) |
| Fase 2 | **0,2 s** |
| Pose detectado (conf 0,5) | **125/125** nas 2 câmeras |
| Passe cross-camera | **125/125** frames com os 2 punhos 3D; **11 swaps, 10 rejeições** |
| Cobertura 2D (body/right/left) | cam0 125/75/106; cam1 125/118/116 |
| Bloco 3D com dados | right 70/125; left 112/125 |
| Reprojeção 3D-vs-2D do próprio ATHENA | right **2,32 px** (p90 5,1); left **2,76 px** (p90 13,7) |
| Concordância com nossa triangulação (após remap) | **1,5 mm** mediana (p90 6,7) |
| Jitter do bloco right | **0,65 mm/frame²** |
| Rótulos L/R | **invertidos** (bloco right ≈ mão esquerda: 283,7 mm) |

**Nosso pipeline atual (baseline `processed_1mao`, MediaPipe 1.0.1 + DLT):** 124 pares, 112 saídas válidas, **reproj 1,37/1,40 px**, cobertura por junção mediana 75,9%, triangulação 112/112, **7,0 s** de processamento; jitter da trajetória 3D **3,32 mm/frame²**.

### 5.4 Falhas/bugs encontrados (com erro exato)

1. `pip install athena-tracking==0.3` → `ERROR: Could not find a version that satisfies the requirement pyav ...`.
2. `pip install athena-tracking==0.9` em Python 3.12 → `ResolutionImpossible` (dependência `pymovie` incompatível; pinos antigos sem rodas).
3. 0.3: `TypeError: expected str, bytes or os.PathLike object, not _GeneratorContextManager` ao criar HandLandmarker/PoseLandmarker (path de modelo não resolvido).
4. 0.3: fase 2 lê só `videos/<trial>/*.avi` → MP4 é ignorado (falha silenciosa/erro no FPS).
5. 0.3: `smooth3d` com os **defaults da própria GUI (5 Hz / 20 Hz)** e vídeo a 6 fps → `ValueError: polyorder must be less than window_length`; só funciona com cutoffs ≤ ~fps/2 (usei 2/2).
6. v1.0: mesma classe de bug no `_smooth3d` com FPS=10 e cutoff=10 → `ValueError: polyorder must be less than window_length`; com cutoff 4 funciona. **Qualquer vídeo de 8–10 fps no setup atual + defaults de 10 Hz quebra o ATHENA.**
7. v1.0: Pose com conf 0,9 falha 0/12 na aux0 (ver 4.4).
8. Frame-locked: `np.stack` com 196×160 frames → `ValueError: all input arrays must have the same shape`.
9. `tests/test_face_reprojection.py` não roda sem `sampledata` (`ValueError: need at least one array to stack`).

### 5.5 Onde ficaram as saídas (fora do repo)

- `%TEMP%\athena03_venv\` — venv do 0.3; `%TEMP%\athena_smoke_venv\` — venv do main v1.0.
- `%TEMP%\athena03_run\mainroot\{videos\rec1\cam0.avi,cam1.avi, calibration\cam0.yaml,cam1.yaml, landmarks\rec1\...}` — saídas do 0.3.
- `%TEMP%\athena_e2e\mainroot\` — saídas do main v1.0; `%TEMP%\athena_scripts\{run03.py, e2e_real.py, e2e_metrics.py, synthetic_phase2.py, phase1_real.py, athena03_result.json, e2e_metrics.json}`.
- `%TEMP%\athena_pkg\athena\` — snapshot do pacote main v1.0 usado (com modelos).

---

## 6. Comparação honesta

### 6.1 ATHENA vs pipeline atual (MediaPipe + triangulação nossa) vs DeepLabCut + Anipose

| Critério | **ATHENA** (main v1.0) | **Nosso pipeline atual** | **DLC 2D + Anipose/DLC-3D** (ver doc paralelo) |
|---|---|---|---|
| Detector 2D | MediaPipe Pose+Hands pré-treinados | MediaPipe Hands (1.0.1) + correção de lateralidade nossa | DLC treinado por nós (sem modelo de mão no Model Zoo) |
| Anotação manual | **nenhuma** | nenhuma | **exige anotar/treinar** (semanas) |
| Geometria 3D | DLT + filtro câmera 30 px + correção cross-camera | DLT + gate de reprojeção + plausibilidade física | DLT idêntico (Anipose comprovadamente idêntico ao nosso — doc paralelo, erro mediano 1,86 mm; `triangulate_optim` 0,54 mm) |
| Sync | **frame-locked (ignora timestamp)** | **timestamp/eventos manuais** (FPS diferentes OK) | frame-locked (mesmo gargalo) |
| FPS variável / reamostragem | exige re-encode | resolve nativamente | exige re-encode |
| Lateralidade | pose cross-camera (funciona: 11 swaps/10 rejeições) **mas convenção invertida no nosso setup** | flip calibrado + pareamento geométrico + track IDs | sem semântica de lado (keypoints separados) |
| Track ID/dedup | **não tem** | **tem** (`track_id`, dedup 3D, suavização de lado) | DLC tem identidade por keypoint; 3D multi-animal é outro problema |
| Suavização/refino | **SavGol** (+ bundle adjustment opcional, mas ≥3 cams) | nenhuma (recomendação: adicionar) | Anipose `triangulate_optim` (jitter 3,48→2,04 mm no doc paralelo) |
| Saída | `.npy` 75 landmarks, sem tempo/ID/confiança | `.csv/.npz` rico (reproj, valid, lado, track, par, tempos) + relatório | HDF5/CSV + 3D CSV |
| Instalação | **Python 3.12 exato**; PyPI quebrado; GitHub main | roda no `.venv` 3.11 atual | ambiente pesado; sem GPU NVIDIA (XPU não suportado); LGPL + pesos research-only |
| Tempo (125 pares, nosso PC) | 14,2 s (main) / 16,4 s (0.3) | **7,0 s** | treino + inferência (maior) |
| Precisão reproj (nosso teste) | 2,3–2,9 px (main) / 2,9–3,9 px (0.3) | **1,37–1,40 px** | geometria igual à nossa; ganho só com temporal optimization |
| Validação científica publicada | **sim (J Neurophysiol, vs OptiTrack)** | parcial (nossa validação de aquisição) | ampla (DLC/Anipose) mas não para mão 21 pts |
| Licença | MIT (código) | nosso | LGPL + SuperAnimal research-only (pesos) |
| Esforço de integração | 2–3 dias (bridge) ou 1–2 dias (portar 2 passes) | 0 | semanas (anotação/treino/valid.) |

### 6.2 Onde o ATHENA ganha

1. **Zero anotação e específico para mão** (21 keypoints + pose), ao contrário do DLC.
2. **Correção cross-camera de lateralidade por geometria 3D do Pose** — ideia boa e comprovadamente funcional na nossa sessão (11 swaps/10 rejeições; concordância de 1,5 mm com nossa geometria). É o que mais nos interessa portar.
3. **Filtro iterativo de câmera outlier** (30 px) e **suavização SavGol** com resultado de jitter muito melhor (0,65 vs 3,32 mm/frame²) — barato de portar.
4. **Validação publicada contra OptiTrack** e paper revisado: excelente para citar na tese como método de referência/comparação.
5. **MIT**: portar código é legal (com atribuição); não há armadilha de licença.

### 6.3 Onde o ATHENA perde (para o nosso caso)

1. **Sincronismo frame-locked** (ignora timestamps) — o nosso ponto mais forte; usar ATHENA "puro" exigiria re-encode para grade comum (custo/risco), e a reamostragem muda o tempo efetivo.
2. **Requer ambiente Python 3.12** separado e pino `mediapipe==0.10.21` (o projeto usa 3.11/1.0.1) → duas pilhas; manutenção e reprodutibilidade piores.
3. **PyPI não confiável** (0.3 quebrado; 0.4/0.9 metadata com `pymovie`; main v1.0 só no GitHub).
4. **Sem track_id/timestamps/qualidade por frame** no 3D → não entra direto no nosso dataset; precisa de contra-adaptação.
5. **Lateralidade invertida** no nosso setup: os passes de Pose corrigem trocas entre câmeras, mas **não** a convenção global; sem remap validado contra o Kinect, todo o dataset sai com lado trocado. (O nosso `aux_handedness_flip` já resolve.)
6. **Não melhora a geometria**: reprojeção pior que a nossa (2,3–3,9 vs 1,4 px) e concordância de 1,5 mm apenas confirma o DLT; o ganho é estabilidade/correção, não acurácia intrínseca.
7. **Sem depth de Kinect** e sem fusão com IMU/esqueleto do Kinect — nosso ecossistema de ground truth é mais rico.
8. **Refino de calibração indisponível com 2 câmeras** (default ≥3).
9. **Só 2 câmeras = filtro de câmera outlier inócuo**; robustez menor do que o paper (8 câmeras).
10. **Bugs de robustez em baixo FPS** (SavGol) e **default de confiança inadequado** para as nossas câmeras (0,9 falha).
11. **Codebase acadêmica pequena**, sem CI; última mudança de código em 05/2026; docs desatualizadas; patches podem ser necessários.

---

## 7. Arquitetura de integração proposta

Premissa: **não substituir** o pipeline atual; manter o `.venv` 3.11 e o default atuais; o ATHENA entra como **backend alternativo comparável** e/ou suas ideias entram como **estágio opcional**.

### Opção A (recomendada para comparação científica) — bridge ATHENA v1.0 em venv 3.12

Novos arquivos (todos opt-in; nada muda no default):

```
src/mestradopy/vision/athena_bridge.py
    build_framelocked_bridge(session_dir, bridge_dir) -> dict
        # reusa plan_session/pair_by_time (timestamp sync)
        # grava videos/<rec>/{cam0,cam1}.mp4 (frame k = par k) @ fps efetivo correto
        # grava calibration/cam{i}.yaml (K.T, R.T, T em m) a partir de CameraAlignment
        # grava bridge_meta.json (lista de pares, offsets, sizes, fps, side_map)
    run_athena(runner_python, bridge_dir, *, pose_conf=0.5, hand_conf=0.5,
               lfc_hz, refine_calibration=False) -> dict
    import_athena(bridge_dir, session_dir, out_npz)
        # mapeia frame k -> (frame0,frame1,pair,time_sync_s)
        # aplica side_map validado (swap de blocos) e monta o npz no schema atual:
        # joints_xyz/valid/pixels_aux0/1/frame0/frame1/time_sync_s/hand/track_id/method
        # converte 2D px ideal -> px do vídeo original (projectPoints com dist)

scripts/athena/athena_runner.py      # roda DENTRO do venv 3.12 (sem Tk):
                                     # read_calibration -> process_camera (2 cams)
                                     # -> _reassign_hands_from_3d_wrists -> triangulaterefine.main

tests/manual/test_athena_bridge.py   # sintético (não exige ATHENA instalado):
                                     # YAML roundtrip, pair mapping, px ideal<->orig,
                                     # side_map, schema do npz; integração real marcada skip
```

Critérios de aceite (todos medidos no spike): conversão de calibração com erro 3D < 1 mm no teste sintético (**já: 0,317 mm**); Pose detectado ≥ 90% frames com conf 0,5 (**já: 125/125**); reproj do 3D contra 2D do próprio ATHENA ≤ 3,5 px (**já: 2,32/2,76**); concordância com a triangulação atual ≤ 10 mm mediana após side_map (**já: 1,5 mm**); cobertura da mão-alvo ≥ 85% dos pares (**já: 112/125 = 89,6%**); tempo ≤ 2× o atual (**já: 14,2 s vs 7,0 s**); side_map 100% correto na sessão de validação contra o Kinect; nenhum arquivo/flag default alterado; relatório final `docs/validation/ATHENA_BRIDGE_<data>.md`.

Riscos: venv 3.12 separado; mediapipe pinada; bridge quebra se o ATHENA mudar; reamostragem duplica frames; suavização SavGol pode atenuar movimentos rápidos de MI (validar cutoff contra métricas de tempo/velocidade).

### Opção B (recomendada se não quiser novo ambiente) — portar 2 componentes para o pipeline atual

1. `src/mestradopy/vision/lateralidade_pose.py` (MIT, atribuição no cabeçalho): triangula os punhos do Pose entre as câmeras disponíveis, reprojeta e aplica o mesmo critério de reassign/rejeição do ATHENA (`reject_threshold` configurável), usando a nossa `CameraAlignment`/`AuxCameraView`. Entra **depois** da detecção 2D e **antes** do pareamento/triangulação em `handmvnet_video_process.py`; precisa do PoseLandmarker (o projeto tem `pose_landmarker_lite.task` em `assets/models/`; ideal baixar o `full`).
2. Estágio opcional `--suavizar 3d --lfc-hz N` em `process_session` (SavGol por junta com `window=int(2*fps_eff/cutoff+1)`, polyorder adaptado ao window; NaNs interpolados/restaurados like ATHENA; fps_eff calculado dos timestamps, **não** do container).
3. Testes sintéticos (`tests/manual/test_lateralidade_pose.py` e `test_suavizacao_3d.py`): punhos 3D conhecidos + rótulos trocados → correção recupera; SavGol reduz jitter ≥50% sem atrasar passo além de X ms; sem regressão de cobertura/reproj (≤1,5 px) no teste offline 73/73.
4. Critérios de aceite: trocas de rótulo/dedup medidas na sessão real caem (baseline: 2 trocas em 1 mão; ATHENA real: 11 swaps/10 rejeições); jitter 3,32 → ≤1,5 mm/frame²; reproj ≤1,5 px; nenhum arquivo default alterado (flag opt-in).

### Opção C (mínima, se o tempo apertar) — ATHENA só como validação offline

Rodar o bridge uma vez por sessão, gerar `docs/validation/ATHENA_X_NOSSA_<data>.md` comparando 3D (concordância ~1,5 mm), cobertura, jitter e lateralidade; citar o paper como método independente. Nenhuma mudança no pipeline; custo ~1 dia.

### Sequência sugerida (2–3 dias)

1. **Dia 1:** Opção B item 1 (reassign/rejeição por Pose) + teste sintético; medir na `teste_sync3`/próxima sessão (trocas, cobertura, reproj).
2. **Dia 2:** Opção B item 2 (SavGol opt-in) + teste sintético; validar que não atrasa/atenua os movimentos de MI.
3. **Dia 3:** Opção A/C (bridge comparativo) + relatório com critérios de aceite acima; decidir com o Lead se o backend ATHENA vira opção de produção.

---

## 8. Riscos, limitações e licença

**Técnicos:** sincronismo frame-locked; FPS de container ≠ FPS real do nosso gravador; SavGol quebra em baixo FPS/cutoff alto; Pose com confiança 0,9 falha na aux0; inversão sistemática de lado (precisa mapa validado); ausência de track IDs/timestamps; filtro de câmera inócuo com 2 cams; refino de calibração exige ≥3 cams; 2D do ATHENA em pixels não distorcidos (conversão necessária).
**Operacionais:** Python 3.12 exato + pino mediapipe 0.10.21 (dois ambientes); PyPI quebrado; repo acadêmico pequeno (43 stars, 2 mantenedores, sem CI, última mudança de código 05/2026); docs desatualizadas; patches locais podem ser necessários; sample data de 800 MB no repo.
**Científicos:** validação do ATHENA é com 8 câmeras FLIR 60 Hz + JARVIS/OptiTrack; nosso cenário (2 webcams 8–10 fps, vistas laterais/close, mão pequena) é fora de distribuição; não há garantia de que o ganho publicado se mantenha; a suavização agressiva pode alterar métricas de velocidade/tempo do MI (validar antes de usar no label).
**Licença:** código **MIT** (uso, modificação e redistribuição permitidos com atribuição); modelos MediaPipe Apache-2.0; sem restrição comercial. Ao portar código, copiar o aviso MIT no cabeçalho e citar o paper (`Mulla & Michaels, J Neurophysiol 2025, doi:10.1152/jn.00407.2025`).

---

## 9. Recomendação final

1. **NÃO substituir** o pipeline MediaPipe+triangulação atual agora: ele é melhor no nosso gargalo (sincronismo por timestamp), na precisão de reprojeção (1,4 px vs 2,3–3,9), tem track IDs/dataset integrados e roda no `.venv` atual.
2. **SIM complementar** com os 2 componentes portáveis (Opção B): correção/rejeição de lateralidade por punho 3D do Pose + suavização SavGol opt-in — ganho claro e barato (11 swaps/10 rejeições reais; jitter 3,32 → 0,65–1,5 mm/frame²), sem novo ambiente.
3. **SIM manter o ATHENA main v1.0 como backend de comparação/validação** (Opção A/C) em venv 3.12 isolado, com o bridge de reamostragem; usar os números para um parágrafo/tabela da tese ("concordância de 1,5 mm com método independente validado contra OptiTrack").
4. **NÃO usar `athena-tracking==0.3` do PyPI** para nada além de prova de conceito (instalação quebrada, bug de modelo, AVI-only, defaults de suavização quebram no nosso FPS, sem correção cross-camera).
5. **Testar novamente apenas com sessão nova** de uma mão (idealmente FPS casado entre as auxiliares e gravação com FPS de container verdadeiro) para medir o ganho do passe de Pose e da suavização sem confundidores da sessão atual.

---

## 10. Referências e links

- Repo: <https://github.com/neural-control-and-computation-lab/athena> (MIT; v0.4/v0.9 tags; main v1.0)
- Docs: <https://neural-control-and-computation-lab.github.io/athena/>
- PyPI: <https://pypi.org/project/athena-tracking/> (0.2/0.3 de 2024; 0.4/0.9 de 2025)
- Paper: Mulla D, Costantino M, Freud E, Michaels JA. *ATHENA: Automatically Tracking Hands Expertly with No Annotations.* J Neurophysiol 2025. doi: [10.1152/jn.00407.2025](https://doi.org/10.1152/jn.00407.2025)
- Preprint: <https://www.biorxiv.org/content/10.1101/2025.08.12.669753>
- JARVIS (calibração recomendada pelo ATHENA): <https://jarvis-mocap.github.io/jarvis-docs/>
- Anipose: <https://github.com/lambdaloop/anipose> (BSD-2); DeepLabCut: <https://github.com/DeepLabCut/DeepLabCut> (LGPL-3.0)
- Revisão DLC/Anipose do projeto: [REVISAO_DLC_MULTIVIEW_AUX_20261119.md](REVISAO_DLC_MULTIVIEW_AUX_20261119.md)
- Nossos arquivos de referência: [HANDMVNET_VIDEO_OFFLINE.md](../guias/HANDMVNET_VIDEO_OFFLINE.md), [VALIDACAO_AQUISICAO_20261119.md](VALIDACAO_AQUISICAO_20261119.md), `src/mestradopy/core/stereo_calibration.py`, `src/mestradopy/core/kinect_imu_groundtruth.py`, `src/mestradopy/tools/handmvnet_video_process.py`, `src/mestradopy/tools/handmvnet_export_2views.py`.

**Citação (se portarmos código/ideias):**
```bibtex
@article{mulla2025athena,
  title     = {{ATHENA}: Automatically Tracking Hands Expertly with No Annotations},
  author    = {Mulla, Daanish and Michaels, Jonathan A.},
  journal   = {Journal of Neurophysiology},
  year      = {2025},
  doi       = {10.1152/jn.00407.2025}
}
```
