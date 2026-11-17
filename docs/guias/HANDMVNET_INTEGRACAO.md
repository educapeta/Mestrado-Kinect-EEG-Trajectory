# HandMvNet -- avaliacao e integracao ao pipeline multiview

**Data da avaliacao:** 17/11/2026
**Repositorio externo:** <https://github.com/pyxploiter/HandMvNet> (clonado em
`external/HandMvNet`, commit `aa08ed1`, 2026-08-27, ~4,9 MB)
**Paper:** Ali, Robertini, Stricker -- *HandMvNet: Real-Time 3D Hand Pose
Estimation using Multi-View Cross-Attention Fusion* (VISAPP 2025).

Este guia registra o que foi lido/validado do HandMvNet, o que foi integrado ao
nosso sistema (Kinect RGB + webcams auxiliares) e o que ficou **bloqueado**.
Nada aqui foi commitado; a revisao/commit e' do Lead.

---

## 1. Resumo executivo

| Item | Estado |
|---|---|
| Repositorio lido e arquitetura mapeada | **OK** (entrada/saida/pre-processamento/checkpoints documentados abaixo) |
| Adaptador integrado ao nosso pipeline multiview | **OK** -- `src/mestradopy/vision/handmvnet_adapter.py` |
| Fallback geometrico funcional (sem pesos, sem hardware) | **OK** -- triangulacao multi-view com as calibracoes reais do projeto |
| Caminho do modelo oficial (forward real) | **OK em encanamento** -- carrega o modelo oficial, roda forward e converte saida; validado com pesos ALEATORIOS |
| Pesos pre-treinados | **BLOQUEADO** -- nao publicados pelo repositorio oficial (ver secao 3) |
| Licenca | **BLOQUEADO/risco** -- repositorio SEM arquivo de licenca (ver secao 3) |
| Teste smoke sem hardware | **OK** -- `tests/manual/test_handmvnet_adapter.py`: **116/116 checks** em ~7 s |
| Hardware | CPU e **Intel Arc A580 (torch XPU)** testados; sem CUDA na maquina |

O adaptador **roda hoje** no pipeline pelo fallback de triangulacao: recebe
frames sincronizados (Kinect + auxiliares), usa `stereo_calibration*.npz` e
devolve 21 junções 3D no mesmo contrato da triangulacao atual. Quando um
checkpoint `.ckpt` existir, o modo `auto` passa a usar o modelo oficial e o
fallback vira a rede de seguranca (forward que falha/numero de vistas
incompativel/dominio estranho caem no geometrico com motivo explicito).

---

## 2. O que e' o HandMvNet (para quem for revisar o adaptador)

### Arquitetura (do codigo oficial)

- backbone 2D por vista: `HRNet` (w40/w64) ou `ResNet18/34/50_paper` (classes
  do proprio repo; `ResNet50_Paper` usa `torchvision` apenas para as URLs de
  pesos ImageNet);
- o backbone devolve features em varias escalas (`mlvl_img_feats`); `pose_net`
  produz **heatmaps 21x32x32** por vista; `soft_argmax_2d` extrai 21 coords 2D
  por vista;
- `SampleNet` amostra features nas coords dos joints em cada escala;
  concatenam-se as features (`feat_dim` = soma dos canais / 2) + `pos2d` (+
  `crop`/`sin` conforme `pos_enc`);
- `CrossAttentionFusion` (5 camadas, 8 cabecas) funde as features das V vistas:
  os 21 tokens da vista raiz funcionam como *query* e os demais como
  *key/value* -- por isso a ordem das vistas importa (a primeira e' a raiz);
- `JointsDecoderGCN` (ChebConv com as arestas MANO) ou `JointsDecoderNN`
  devolve `joints_cam` (21,3) em **metros, root-relative ao landmark 0
  (wrist)**; opcionalmente um decoder extra prediz a raiz absoluta quando
  `root_relative: false`.

### Entrada (o que o adaptador monta)

| Tensor | Shape | Conteudo |
|---|---|---|
| `x` | `(B, V, 3, 256, 256)` | crops RGB normalizados ImageNet (mean/std) |
| `bbox` | `(B, V, 4)` | `[x1,y1,x2,y2]` em pixels da imagem inteira (quadrada) |
| `cam_params["intrinsic"]` | `(B, V, 4)` | `[fx, fy, cx, cy]` da imagem inteira |
| `cam_params["extrinsic"]` | `(B, V, 4, 4)` | mundo->camera (so' usado em treino/loss; o forward de inferencia usa `intrinsic` apenas quando `pos_enc` tem `crop`) |

Pre-processamento oficial replicado no adaptador (`datasets/utils.py` +
`HO3DSamplePreprocessor`): bbox quadrada em volta dos 21 joints ->
`crop_and_pad_image` (zero-pad fora da imagem) -> resize `256x256` (antialias)
-> `/255` -> normalize ImageNet -> CHW. `joints_crop = (p - (x1,y1)) * 256 /
(w,h)`; o inverso recupera pixels da imagem inteira.

### Saida (o que o adaptador converte)

| Saida oficial | Shape | Uso no adaptador |
|---|---|---|
| `joints_crop_img` | `(B, V, 21, 2)` | 2D por vista no espaco do crop; convertido para pixels da imagem inteira |
| `joints_cam` | `(B, 21, 3)` | metros, root-relative; somado a uma raiz 3D para virar absoluto |
| `root_joint` | `(B, 1, 3)` | so' quando `root_relative: false`; usado como raiz se nao houver ancora externa |
| `heatmap` | `(B, V, 21, 32, 32)` | pico do softmax espacial -> `scores` (21,) por junção |

### Configuracoes publicas e numero de vistas

`configs/release/*.yaml` do repo oficial foram treinados com:

| Dataset | Vistas (`num_views`) | `pos_enc` |
|---|---|---|
| DexYCB | 8 | `pos2d`, `crop`, `sin` (`root_relative: true`) |
| HO3D | 5 | `pos2d`, `crop`, `sin` (`root_relative: true`) |
| MVHand | 4 | `pos2d`, `sin` (`wo_cam`: nao usa intrinseca) |

**Isto importa:** nenhum checkpoint publico seria de 3 vistas (o nosso rig e'
Kinect + 2 auxiliares). O adaptador recusa checkpoint com numero de vistas
diferente por padrao e explica o motivo; ha' uma opcao **experimental**
`adapt_num_views=True` (as camadas de atencao nao dependem do numero de
vistas, entao o `state_dict` carrega com `strict=False`), mas **a qualidade com
menos vistas e' desconhecida e nao foi validada** -- nao usar como resultado
cientifico sem medir.

---

## 3. Auditoria do repositorio externo

### O que foi clonado

- `external/HandMvNet` (ignorado pelo `.gitignore` do projeto: `external/` nao
  e' versionado; **nunca** versionar pesos).
- `src/models/handmvnet.py` + backbone/fusion/decoder; `configs/release/*.yaml`;
  `notebooks/infer_vis.ipynb`; `train.py`/`eval.py`.

### Pesos: **nao publicados** (bloqueio duro)

Evidencias coletadas em 17/11/2026:

- `README.md` nao linka pesos; nao ha `releases` nem `assets` no GitHub
  (API: `releases: []`, `has_downloads: false`);
- o notebook `notebooks/infer_vis.ipynb` aponta para caminhos **locais** que
  nao existem no repo (`../weights/dexycb/...`, `../weights/ho3d/...`,
  `../weights/mvhand/...`);
- issue **#2** "Can you share the model weights?" (aberta, 10/06/2025) sem
  resposta; issue **#4** "where is checkpoint?" (aberta, 06/11/2025) sem
  resposta;
- `eval.py` so' roda com `--checkpoint /caminho.ckpt` externo.

O adaptador procura, nesta ordem: argumento `checkpoint=`, env
`HANDMVNET_CHECKPOINT`, `external/HandMvNet/weights/**/*.ckpt`,
`external/HandMvNet/checkpoints/`, `assets/models/handmvnet*`, `weights/`.
O comando `status` mostra o que foi encontrado.

### Licenca: **sem licenca explicita** (risco institucional)

- nao existe `LICENSE`/`COPYING`/`NOTICE` no repositorio;
- a API do GitHub devolve `"license": null`;
- sem licenca, o padrao legal e' **todos os direitos reservados**: uso interno
  de pesquisa/avaliacao e' defensavel, mas **redistribuir codigo, pesos,
  derivados ou publicar resultados que incluam os pesos exige autorizacao dos
  autores** (tambem por causa dos datasets de treino: DexYCB, HO3D, MVHand e
  MANO tem licencas proprias de uso academico nao comercial).

**Acao recomendada ao Lead:** (1) nao commitar nada de `external/HandMvNet` no
repo do projeto; (2) se os pesos forem obtidos, guardar fora do git e registrar
no texto da tese a origem/licenca; (3) contatar os autores por e-mail/issue
pedindo pesos + termos de uso.

### Dependencias oficiais vs nosso ambiente

`README` oficial: **CUDA 11.7, Python 3.8, PyTorch 2.0.1, Lightning 2.0.6**
(ambiente de treino dos autores). Nosso ambiente: **Windows 10, Python 3.11.9,
torch 2.14.0+xpu (Intel Arc A580), sem CUDA**; instalados: numpy, opencv,
scipy, PyYAML, pandas, mediapipe. **Ausentes**: `lightning`, `einops`,
`torchvision`, `manopth`, `plotly`, `transforms3d`.

Como o forward de inferencia **nao usa** lightning/einops/torchvision/manopth/
plotly (so' o treino e a visualizacao usam), o adaptador instala **shims de
import** (mini-modulos registrados em `sys.modules`, sem tocar no repo externo
nem no venv):

| Modulo ausente | Shim cobre | Usado no forward? |
|---|---|---|
| `lightning` | base `LightningModule` (`save_hyperparameters`/`log` no-op) | nao (so' herança da classe) |
| `einops` | `rearrange` para os 2 padroes de `models/layers.py` | **sim** (verificado byte-a-byte contra `torch`) |
| `torchvision` | apenas enums `ResNet*_Weights` (URLs de pesos ImageNet) | nao (`backbone_pretrained=False` na inferencia) |
| `vis.visualizer` | `HandPoseVisualizer` | nao (visualizacao de treino) |
| `models.joints_to_vertices` | `JointsToVertices` (MANO) | nao (`get_vertices=False`) |

Para **treinar/avaliar com o codigo oficial** (nao e' o nosso caso agora),
instalar as dependencias reais (`pip install lightning einops torchvision
plotly` e `manopth` + modelos MANO, que exigem registro/licenca).

---

## 4. Arquivos entregues

| Arquivo | O que e' |
|---|---|
| `src/mestradopy/vision/handmvnet_adapter.py` | adaptador: vistas, calibracao, bbox/crop, triangulacao fallback, carregador do modelo oficial (com shims), contrato de saida |
| `src/mestradopy/vision/handmvnet_synthetic.py` | mao 3D sintetica + projecao com as calibracoes reais + imagens sinteticas (sem hardware, sem dado de participante) |
| `src/mestradopy/vision/__init__.py` | docstring do pacote |
| `src/mestradopy/tools/handmvnet_infer.py` | entry point: `status`, `synthetic`, `infer-npz` |
| `tests/manual/test_handmvnet_adapter.py` | suite manual (sem hardware, sem pytest): 116 checks |
| `docs/guias/HANDMVNET_INTEGRACAO.md` | este guia |

Nao foram alterados: `src/mestradopy/eeg/`, `outputs/eeg/`, `data/`,
`docs/thesis/`, `external/HandMvNet` (somente clone), nem o pipeline de EEG.

---

## 5. Convencoes geometricas (as mesmas do projeto)

- **Frame da saida = camera RGB do Kinect**, convencao OpenCV: X direita,
  Y baixo, Z frente, **metros**. E' o mesmo referencial de
  `fusion/aux_vision.py::AuxStereoFrame.points_3d` e da triangulacao; a fusao
  converte Y para cima quando precisa (`fusion/adapters.py::MultiViewSource`).
- **Calibracao auxiliar:** `X_aux = R @ X_kin + T` (campos `rotation` e
  `translation` dos `stereo_calibration*.npz`); `extrinsic` 4x4 = `[[R,T],[0,1]]`
  (mundo = frame do Kinect), como no `CameraAlignment` do projeto.
- **Auxiliares espelhadas:** o pipeline passa os frames por
  `transform_auxiliary_frame` (cv2.flip) ANTES da deteccao, e a calibracao foi
  feita nessa imagem espelhada. O adaptador espera imagem **e** landmarks nessa
  mesma convencao. Quem entregar frame cru pode usar
  `HandMvNetView.from_calibration_npz(..., mirror_to_calibration=True)` ou
  `mirror_view_data(image, landmarks)`.
- **Raiz:** os checkpoints publicos sao `root_relative` (predizem offsets a
  partir do wrist). Por isso o adaptador calcula a raiz 3D com a triangulacao
  das MESMAS vistas (ou usa `root_anchor=` de quem ja' tem ancora melhor) e
  soma. Sem raiz nao ha' posicao absoluta.

---

## 6. API do adaptador (integracao)

### Entrada: `HandMvNetView`

```python
from vision.handmvnet_adapter import HandMvNetAdapter, HandMvNetView

view = HandMvNetView.from_alignment(
    alignment, "aux0", frame_bgr, landmarks_px_21x2,
    aux_index=0,                 # None para a vista do Kinect
)
# ou, sem CameraAlignment (numpy puro; bom para frames gravados/testes):
view = HandMvNetView.from_calibration_npz(
    "kinect", kinect_bgra_or_bgr, kin_px, "configs/calibracao/stereo_calibration.npz",
    is_kinect=True)
```

A **primeira** vista da lista e' a raiz (frame da saida) -- use o Kinect.

### Saida: `HandMvNetOutput`

Campos principais: `backend` (`handmvnet` | `handmvnet_random_weights` |
`triangulation_fallback` | `unavailable`), `valid`, `reason`, `joints_3d`
(21,3 absolutos), `joints_3d_root_relative`, `root_3d`, `root_source`,
`joints_2d` (por vista), `reproj_errors` (por vista, px), `scores`, `latency_ms`,
`model_weights`, `warnings`. Helpers:

- `palm_points_3d()` / `palm_reproj_errors()` -- os 5 pontos do dorso
  `(0,5,9,13,17)`, exatamente o subconjunto usado pelo A/B de normal da palma;
- `as_aux_stereo_frame()` -- converte para o contrato
  `fusion/aux_vision.AuxStereoFrame`, encaixe direto no `MultiViewSource`;
- `to_dict()` -- JSON para log/CSV.

### Uso num loop ao vivo (esboco)

```python
adapter = HandMvNetAdapter(              # mode="auto"; fallback hoje
    checkpoint=None,                     # ou env HANDMVNET_CHECKPOINT
    device="auto",                       # cuda > xpu (Intel Arc) > cpu
)
...
views = [
    HandMvNetView.from_alignment(alignment, "kinect", kinect_bgr, kin_px,
                                 is_kinect=True),
    HandMvNetView.from_alignment(alignment, "aux0", aux0_bgr, aux0_px,
                                 aux_index=0),
    HandMvNetView.from_alignment(alignment, "aux1", aux1_bgr, aux1_px,
                                 aux_index=1),
]
out = adapter.infer(views, hand="right")
if out.valid:
    # backend="triangulation_fallback" hoje; "handmvnet" quando houver pesos
    normal = MultiViewBackend().compute(out.palm_points_3d(), out.hand, t,
                                        reproj_errors=out.palm_reproj_errors())
```

Pontos de atencao na integracao ao vivo:

1. os 3 frames tem de ser do **mesmo instante** (o `comparar_tracking_multiview`
   ja' tem o padrao de captura + threads de deteccao; reusar);
2. a vista raiz (Kinect) deve ser montada com a MESMA calibracao do
   `CameraAlignment` (use `from_alignment`);
3. se usar `adapt_num_views=True` (experimental), medir antes de confiar;
4. `as_aux_stereo_frame()` permite plugar no `MultiViewSource` sem mudar a
   fusao (o `MultiViewSource` ja' converte Y para cima).

---

## 7. Como rodar (comandos exatos)

Na raiz do repositorio, com o Python do projeto:

```powershell
# Diagnostico (repo, dependencias, checkpoint, licenca, calibracoes)
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_infer.py status
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_infer.py status --json

# Smoke sintetico SEM pesos (fallback de triangulacao com calibracoes reais)
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_infer.py synthetic --triangulation
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_infer.py synthetic --views 2

# Validar o ENCANAMENTO do modelo oficial sem pesos (pesos ALEATORIOS:
# valida imports/forward/formatos/latencia; NAO e' predicao)
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_infer.py synthetic --random-weights

# Quando houver checkpoint:
$env:HANDMVNET_CHECKPOINT = "C:\caminho\handmvnet.ckpt"
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_infer.py synthetic
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_infer.py synthetic --device xpu

# Frames gravados (.npz sincronizado; formato no --help do subcomando)
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_infer.py infer-npz frames.npz

# Suite de teste manual (116 checks; sem hardware)
.\.venv\Scripts\python.exe tests\manual\test_handmvnet_adapter.py
```

Observacao de ambiente: sem `PYTHONPATH`, os tools do projeto inserem
`src/mestradopy` e `src/mestradopy/core` no `sys.path`; a suite faz o mesmo.
**Nao** coloque `src/mestradopy/fusion` no `PYTHONPATH`: `fusion/logging.py`
sombreia o `logging` da stdlib (foi o que quebrou a primeira tentativa de
importar `kinect_imu_groundtruth`).

Formato do `.npz` para `infer-npz` (frames ja' espelhados na convencao da
calibracao):

```
kinect_img (H,W,3) uint8 BGR | kinect_px (21,2) float
aux0_img / aux0_px           | aux1_img / aux1_px   (opcionais)
```

---

## 8. Resultados medidos (sintetico, 17/11/2026)

Fallback de triangulacao -- mao 3D sintetica projetada nas 3 cameras com as
calibracoes reais (`stereo_calibration.npz`, `stereo_calibration_aux1.npz`) e
erro medido contra o ground truth:

| Cenario | Erro 3D medio | Erro mediano | Reprojeção mediana |
|---|---|---|---|
| 3 vistas, sem ruído | **0,036 mm** | 0,037 mm | 0,01 px |
| 3 vistas, ruído 0,5 px | 1,76 mm | 1,74 mm | 0,44 px |
| 3 vistas, ruído 1,0 px | 3,50 mm | 3,49 mm | 0,9 px |
| 3 vistas, ruído 2,0 px | 7,08 mm | 7,05 mm | 1,8 px |
| 2 vistas (Kinect+aux0), sem ruído | 0,078 mm | 0,02 mm | 0,01 px |

Comparacao com a triangulacao canonica do projeto
(`kinect_imu_groundtruth.triangulate_three_views`, mesmo frame e mesmos
pixels): concordam nas 21 junções e a **diferenca mediana e' 0,002 mm**; o
erro da canonica contra o GT foi 0,034 mm.

Modelo oficial (sem pesos pre-treinados; pesos aleatorios, so' encanamento):

- arquitetura montada: `HandMvNet(resnet/18, views=3, pos_enc=[pos2d, crop,
  sin])`, **17.504.704 parametros**;
- forward CPU: ~70 ms/frame (3 vistas, 256x256) depois do aquecimento;
- forward no **Intel Arc A580 (XPU)**: funciona; primeira chamada ~2,3 s
  (compilacao/oneDNN) -- a segunda em diante e' a medir no pipeline real;
- saidas conferidas: `joints_cam (1,21,3)`, `joints_crop_img (1,3,21,2)`,
  `heatmap (1,3,21,32,32)`; conversao para pixels da imagem e ancora da raiz
  conferida analiticamente (`joints_3d = joints_rel + raiz`).

O caminho de **checkpoint real** tambem foi exercitado no teste com um `.ckpt`
FAKE no formato Lightning (mesmo `state_dict` + `hyper_parameters` oficial),
salvo em diretorio temporario: reconstrucao da arquitetura pelas
`hyper_parameters`, `strict=True` sem chaves faltando, backend virando
`handmvnet` e `adapt_num_views` (experimental) carregando com `strict=False`.

---

## 9. Limitacoes e riscos (para o texto/metodologia)

1. **Pesos indisponiveis** -> nenhuma predicao real do HandMvNet foi medida;
   tudo que envolve o modelo e' validacao de encanamento (aleatorio) ou de
   carregamento (fake checkpoint). Nao escrever resultado de qualidade.
2. **Licenca ausente** -> nao redistribuir codigo/pesos; pedir autorizacao.
3. **`num_views` dos checkpoints publicos (5/8/4) != nosso rig (3)** ->
   `adapt_num_views` e' experimental; o caminho cientificamente seguro e'
   treinar/finetunar um modelo de 3 vistas com dados/labels do nosso protocolo.
4. **Domain gap** mesmo com pesos: treino em DexYCB/HO3D/MVHand (câmeras,
   iluminacao, distancia, mãos de outros sujeitos); nosso rig usa Kinect v2 +
   webcams calibradas e uma mesa com tarefa de EEG. Medir MPJPE/PCK no nosso
   ground truth antes de qualquer conclusao.
5. **Shims de import** so' para inferencia; nao usar esse caminho para treinar
   nem para o `eval.py` oficial (que, alem disso, esta' quebrado: importa
   `datasets.dexycb`/`datasets.mvhand`, arquivos ausentes no repo).
6. **Malha/MANO**: vertices (778) exigem `manopth` + modelos MANO (licenca a
   parte); o adaptador entrega **21 junções** (suficiente para o nosso
   tracking/e fusão).
7. **XPU**: torch 2.14+xpu roda; nao ha' CUDA. Para o pipeline ao vivo,
   medir FPS com o adaptador + detector no mesmo loop (o teste mede so' o
   forward isolado).
8. **LGPD/CEP**: o adaptador nao grava frames; a suite usa desenhos
   sinteticos. Se frames reais forem salvos para `infer-npz`, valem as regras
   do projeto (`data/`/`outputs/` nao versionados).

---

## 10. Proximos passos sugeridos ao Lead

1. Revisar e commitar os arquivos da secao 4 (o clone `external/HandMvNet` e'
   ignorado pelo git e nao deve ser commitado).
2. Abrir contato com os autores (issue/email) pedindo **pesos + termos de
   licenca**; se liberarem, baixar para fora do git
   (`external/HandMvNet/weights/` ou `HANDMVNET_CHECKPOINT`) e rodar
   `status` + `synthetic --checkpoint ... --device xpu`.
3. Coletar uma sessao curta com o rig (Kinect + aux0/aux1), exportar frames
   sincronizados no formato `.npz` do `infer-npz` com os landmarks do
   MediaPipe e comparar `handmvnet` x `triangulation_fallback` contra o ground
   truth do projeto (MPJPE/PCK, erro por junção, latencia) -- e' o experimento
   que falta para decidir se o modelo entra no pipeline.
4. Se `num_views` incompativel: avaliar `adapt_num_views=True` (experimental)
   OU finetunar o decoder/fusion com 3 vistas no nosso dataset (exige labels
   3D do ground truth do projeto).
5. Integracao ao vivo: criar uma fonte adaptadora (como `MultiViewSource`) que
   chame `HandMvNetAdapter.infer` no loop de aquisição; a API ja' devolve o
   contrato `AuxStereoFrame` (`as_aux_stereo_frame`) e o adaptador nao abre
   câmeras, entao o loop existente continua dono da aquisição/sincronismo.
