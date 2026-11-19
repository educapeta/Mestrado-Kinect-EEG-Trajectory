# MAPA DOS ESSENCIAIS — EEG g.Nautilus + tracking Kinect/câmeras + IMU

**Projeto:** `C:\mestradopy` (reorganizado em 17/11/2026; caminhos da raiz antiga
estão em `docs/guias/MIGRACAO_ESTRUTURA.md`)
**Escopo deste mapa:** o que é essencial para gravar o piloto ME/MI com ground truth
da mão, o que é experimental, o que é legado e onde mexer (e onde NÃO mexer).
**Regra de ouro:** a raiz antiga virou `src/mestradopy/`; código ativo lê calibração
de `configs/calibracao/` e escreve em `outputs/<area>/`.

---

## 1. Resumo executivo (o que importa)

| Camada | Arquivo essencial | Papel |
|---|---|---|
| Aquisição EEG + paradigma | `src/mestradopy/core/eeg_motor_paradigm.py` | programa 1 (offline) completo: g.Pype→CSV, Qt/QtWidgets, trial 22 s, ME 8 s + MI 8 s, 4 classes, 2 sessões |
| Aquisição do piloto atual (2 classes) | `src/mestradopy/tools/validacao_eeg_3classes.py` | **é o que o `scripts/run_piloto_me_mi.ps1` roda hoje**: ME 4 s + MI 4 s, L/R, 2 sessões |
| Ground truth / visão | `src/mestradopy/core/kinect_imu_groundtruth.py` | BIBLIOTECA: Kinect v2 + MediaPipe + câmeras auxiliares (stereo) + IMU + palma/braço |
| Calibração stereo | `src/mestradopy/core/stereo_calibration.py` | calibra aux0/aux1 contra o Kinect (tabuleiro 8×6 quadrados = 7×5 cantos internos, 45 mm) |
| IMU | `src/mestradopy/core/imu.py` | UDP ESP32/MPU6050, fusão IMU×câmera, Kalman da trajetória |
| Calibração de bias IMU | `src/mestradopy/core/imu_reset.py` | zera bias do giroscópio no início da sessão |
| Cancelamento global | `src/mestradopy/core/hotkeys.py` | ESPAÇO/C/ESC globais (Windows) + saída garantida |
| Compat pykinect2 | `src/mestradopy/core/kinect_compat.py` | faz o pykinect2 (PyPI) funcionar no Python 64 bits |
| Nomes/índices de webcam | `src/mestradopy/core/camera_names.py` | mapa estável índice→nome (hoje: 0=SIGMA-W780M, 1=Trust USB Camera) |
| Controles UVC | `src/mestradopy/core/camera_uvc.py` | resolução/MJPG/exposição/ganho/FPS travados |
| Esqueleto do corpo | `src/mestradopy/core/body_joints_recorder.py` | grava nós 3D do esqueleto (ombro/cotovelo/punho) |
| Estatística BCI | `src/mestradopy/core/estat_bci.py` | acaso/permutação/IC/FDR/ITR/clusters |
| EEGNet | `src/mestradopy/eeg/mi_eegnet/` | pacote EEGNet 5.870 parâmetros (pré-treino 2b, fine-tune, baselines) |
| CSP+LDA (numpy/scipy) | `src/mestradopy/tools/treino_baseline_3classes.py` | CSP+LDA sem sklearn; base do treino/decoder L/R |
| Análise ME/MI do piloto | `src/mestradopy/tools/csp_lda_duas_sessoes.py`, `analise_erd_novo_piloto.py`, `rsf_novo_piloto.py`, `eegnet2b_finetune_novo_piloto.py` | avalições do piloto novo |
| Tracking comparativo | `src/mestradopy/tools/compara_tracking_mao.py` + `metricas_tracking_mao.py` | grava frames/landmarks e calcula cobertura/jitter/reprojeção |
| Multiview 3 vistas | `src/mestradopy/tools/compara_tracking_multiview.py` + `kinect_imu_groundtruth.triangulate_three_views` | Kinect+aux0+aux1 com fallback de pares |
| Diagnóstico offline calibração | `src/mestradopy/tools/diagnostico_calibracao.py` | sensibilidade a Z, consistência do modelo do Kinect (sem hardware) |

**O que é o "programa 1" da dissertação:** `eeg_motor_paradigm.py` (4 classes,
garrafa/bola × esquerda/direita, 100 trials/sessão = 25/classe, 2 sessões, trial 22 s,
ME 8 s + animação 4 s + MI 8 s, 200 trials/participante). **O piloto novo** (2 classes
L/R, ME 4 s + MI 4 s, 2 sessões) é `validacao_eeg_3classes.py`, e é o que está
gerando `outputs/eeg/piloto_me_mi_s01|s02/`.

---

## 2. Árvore atual (só o que importa)

```text
C:\mestradopy
├─ src/mestradopy/
│  ├─ core/                     # aquisição + visão + IMU + calibração (código ativo)
│  │  ├─ eeg_motor_paradigm.py         # programa 1 (paradigma completo Qt + g.Pype)
│  │  ├─ kinect_imu_groundtruth.py     # biblioteca de ground truth (7.5k linhas)
│  │  ├─ stereo_calibration.py         # calibra aux0/aux1 vs Kinect
│  │  ├─ imu.py / imu_reset.py         # IMU UDP + bias
│  │  ├─ kinect_compat.py              # patch pykinect2
│  │  ├─ camera_names.py / camera_uvc.py
│  │  ├─ hotkeys.py / body_joints_recorder.py
│  │  ├─ freeze_kinect_direto.py       # congelar intrínsecas do Kinect (legado de bancada)
│  │  ├─ palm_orientacao.py / palm_normal_edge_on.py
│  │  ├─ sand_trajectory_model.py / sand_traj_treino.py / sand_traj_tempo_real.py   # SAND (experimental)
│  │  ├─ fusion_demo.py / estat_bci.py / mi_animation_method.py
│  │  └─ __init__.py
│  ├─ tools/                    # executáveis (análise, treino, diagnóstico, aquisição)
│  │  ├─ validacao_eeg_3classes.py     # AQUISIÇÃO do piloto atual (Gate C)
│  │  ├─ compara_tracking_mao.py       # gravação/análise Kinect × multiview
│  │  ├─ compara_tracking_multiview.py # demo 3 vistas (Kinect+aux0+aux1)
│  │  ├─ metricas_tracking_mao.py      # relatório offline dos CSVs de tracking
│  │  ├─ diagnostico_calibracao.py     # auditoria offline da calibração
│  │  ├─ kinect_groundtruth_tool.py    # UI completa do ground truth (5.1k linhas)
│  │  ├─ multiview_janela.py           # janela Kinect × auxiliar
│  │  ├─ test_triangulation_3view_synthetic.py  # teste sintético da triangulação
│  │  ├─ treino_baseline_3classes.py   # CSP+LDA (numpy/scipy)
│  │  ├─ csp_lda_duas_sessoes.py, analise_erd_novo_piloto.py, rsf_novo_piloto.py,
│  │  │  eegnet2b_finetune_novo_piloto.py, baselines_csp_lda_eegnet.py
│  │  ├─ handmvnet_*.py                # pipeline HandMvNet (novo, 17-18/11)
│  │  ├─ validacao_groundtruth_deslocamento.py  # Gate B (jig A4/nível)
│  │  ├─ roda_testes.py                # runner de testes (caminhos desatualizados pós-reorg)
│  │  └─ ... (bancada: probe_*, diagnostico_*, preview_kinect, etc.)
│  ├─ eeg/mi_eegnet/            # EEGNet configurável + baselines + testes (unittest)
│  ├─ fusion/                   # fusão/sincronização (V1: quaternion, vision backends, live)
│  ├─ vision/                   # handmvnet_adapter/dataset/synthetic/video_io
│  ├─ acquisition/ analysis/ calibration/   # vazios (espaço reservado; não há código)
│  └─ __init__.py
├─ scripts/                     # entradas .ps1/.bat (run_piloto_me_mi.ps1, run_eegnet_*.ps1,
│                               #   coletar_handmvnet.ps1, GRAVAR_ESP32.bat, BACKUP_ALINHAMENTO.bat)
├─ configs/
│  ├─ calibracao/               # stereo_calibration*.npz, camera_alignment.npz,
│  │                            # kinect_intrinsics.npz, hand_landmark_calibration*.npz,
│  │                            # aux_handedness_flip.json, imu_palm_alignment*.json, arm_model.json
│  ├─ handmvnet_3views.yaml
│  └─ sand_pose_piloto.yaml
├─ data/                        # raw/ (BCIC-IV, WAY-EEG-GAL), cache/, gravacoes/, calibracao/
├─ outputs/
│  ├─ eeg/                      # piloto_me_mi_s01|s02, csp_lda_novo_piloto, erd_novo_piloto, ...
│  ├─ visao/                    # smoke/tracking/handmvnet
│  ├─ fusao/                    # logs da fusão
│  └─ antigos/                  # _teste_* antigos (não usar como resultado)
├─ docs/                        # docs operacionais + guias/ + validation/ + thesis/ + literature/
├─ tests/manual/                # todos os test_*.py (antigos; muitos exigem hardware)
├─ external/                    # HandMvNet, RSF, _github-3d (código de terceiros)
├─ firmware/                    # AHRS e firmware ESP32
├─ assets/models/               # hand_landmarker.task, pose_landmarker_lite.task
├─ archive/root_legacy/         # legado da raiz antiga (não apagar sem revisar)
└─ .venv/                       # Python 3.11.9 (usar SEMPRE .venv\Scripts\python.exe)
```

> **Caminhos da raiz antiga** (para não se perder em docs antigos):
> `eeg_motor_paradigm.py` → `src/mestradopy/core/`, `tools/` → `src/mestradopy/tools/`,
> `Tese_Latex_Agente/` → `docs/thesis/Tese_Latex_Agente/` (cópia parcial) e a versão
> viva continua em `C:\Users\Usuario\OneDrive\Desktop\Mestrado_Projeto\Tese_Latex_Agente`.

---

## 3. Fluxo A — EEG: g.Nautilus → g.Pype → CSV/JSON

### 3.1 Programa 1 (`src/mestradopy/core/eeg_motor_paradigm.py`)

Um único processo, três threads principais:

```text
[thread principal — Qt]  Paradigma (protocolo 2.1-pose-anim, trial 22 s)
   PREPARAÇÃO → INSTRUÇÃO/PRÁTICA (--trials-teste) → BASELINE (olhos fechados/
   abertos/repouso) → calibração de origem → 4 blocos × 25 trials (pausa 3-5 min)
[thread g.Pype]          GNautilusSource 32 canais @ 500 Hz
   → bandpass 1-30 Hz → notch 50/60 Hz → CountsToMicrovolts (µV)
   → RecordingMuxer → RecordingCsvWriter → gravacao_MEMI_<P>_s<N>_<...>.csv
[thread tracking]        TrackingThread (Kinect+IMU, ver Fluxo B)
   → SharedState → RecordingMuxer (colunas KT_*/IMU_*/ARM_*/PALM_*/Marker)
[thread gravação]        MotionRecorder → gravacao_..._movimento.csv
```

- **Fonte:** `--source gnautilus` (real) ou `--source generator` (sintético, sem hardware).
  `build_source()`/`resolve_device()`/`_gds_*` concentram a inicialização do g.Pype/GDS.
- **Saídas por sessão** (em `outputs/eeg/<experimento>/` quando rodado pelo launcher):
  - `gravacao_..._<timestamp>.csv` — Time, `EEG_Ch01..ChNN` (µV), `KT_x/y/z_m`,
    `IMU_L_*`/`IMU_R_*` (11 colunas cada: rpy, acc, pos, valid, ZERO_lock),
    `IMU_hand`, `KTT_valid`, `ARM_*` (18 col.), `PALM_*` (11 col.), `Marker`.
  - `gravacao_..._eventos.json` — eventos por trial (marker, objeto, mão, bloco,
    tempos, fase ME/MI, metadados do participante). `_align_event_json`/`_build_session_meta`.
  - `gravacao_..._movimento.csv` — trajetória/resumo do movimento (MotionRecorder).
- **Marcadores** (BCI IV 2a estendida; ver docstring do módulo):
  768 início do trial, 769/770 cue esquerda/direita, 778/779 ME, 780/781 MI,
  811-814 condições (garrafa/bola × direita/esquerda), 794 animação, 789 prática,
  790/791 bloco, 796 cancelado, 500/501 origem, 32766/32765 sessão.
- **Cancelamento:** ESPAÇO (pula espera/pausa), C (cancela trial), ESC/Ctrl+C (aborta
  com `CancelamentoGarantido`/`os._exit`) e `--sem-hotkeys` restringe à janela Qt.
- **Impedância:** `--impedancia`, `ImpedanceMonitor`, `--sem-painel-impedancia`.
- **Protocolo/condições:** `planejamento_protocolo()`, `build_practice_list()`,
  `slides_instrucao()`, `_resolve_aux_cameras()`, `_parse_sessao()`.

### 3.2 Piloto atual 2 classes (`src/mestradopy/tools/validacao_eeg_3classes.py`)

`scripts/run_piloto_me_mi.ps1 -Sessao 01 -Sujeito S01` roda:

```text
validacao_eeg_3classes.py --source gnautilus --blocos 2 --trials-por-classe 15
  --classes esquerda,direita --me-sec 4 --mi-sec 4 --transicao-sec 1 --cue-sec 1
  --repouso-sec 2 --baseline-olhos-sec 60 --iti-min 1 --iti-max 2
  --sujeito S01 --sessao 01 --pasta-saida outputs/eeg/piloto_me_mi_s01
```

Trial: baseline → cue → ME 4 s → transição 1 s → MI 4 s → ITI. Saída: CSV
`eeg_mi3c_... .csv`, `eventos_...json`, `log_aquisicao_s<N>.txt` em
`outputs/eeg/piloto_me_mi_s0N/`. **É essa a aquisição a usar no treino L/R**
(não o paradigma 4 classes).

### 3.3 Análise/treino do EEG

| Script | Para quê |
|---|---|
| `tools/csp_lda_duas_sessoes.py` | CSP+LDA within/cross-session (2 sessões do piloto) |
| `tools/analise_erd_novo_piloto.py` | ERD/ERS mu/beta por trial e fase |
| `tools/rsf_novo_piloto.py` | RSF riemanniano vs CSP+LDA |
| `tools/eegnet2b_finetune_novo_piloto.py` | EEGNet 2b pré-treinado + fine-tune |
| `tools/baselines_csp_lda_eegnet.py`, `permutacao_baselines.py` | baselines 3.1/3.2 + permutação |
| `tools/treino_baseline_3classes.py` | CSP+LDA puro (numpy/scipy), sem sklearn |
| `src/mestradopy/eeg/mi_eegnet/*` | EEGNet configurável + LOSO 2b + fine-tune + testes |
| `scripts/run_eegnet_*.ps1` | runbooks EEGNet (CPU/4060/Arc) |
| `tools/gera_janelas_mi_piloto.py`, `casamento_eeg_groundtruth.py` | janelas 2 s de MI e casamento com o ground truth |
| Saídas | `outputs/eeg/{csp_lda_novo_piloto,erd_novo_piloto,rsf_novo_piloto,baselines_csp_eegnet,...}/` |

---

## 4. Fluxo B — tracking: Kinect + webcams + IMU → movimento

### 4.1 Cadeia (`src/mestradopy/core/kinect_imu_groundtruth.py`)

```text
Kinect v2 (SDK 2.0 via pykinect2 + kinect_compat)
  color 1920×1080 + depth 512×424
  → MediaPipe Hands (21 landmarks, VIDEO mode) → KinectHandTracker.detect_hands_all()
  → _landmark_camera_point(): pixel da COR → pixel de DEPTH (_color_to_depth_pixel)
    → depth robusto (patche 11×11; resgatar_profundidade com prior do esqueleto)
    → MapDepthPointToCameraSpace()  == frame da CÂMERA DE PROFUNDIDADE (atenção!)
  → 3D da mão (landmark 9 = palma) no frame do Kinect
Câmeras auxiliares (aux0=SIGMA-W780M idx 0, aux1=Trust USB Camera idx 1)
  → frame espelhado (AUXILIARY_MIRROR_HORIZONTAL) → MediaPipe → 21 px
  → CameraAlignment / AuxCameraView (configs/calibracao/stereo_calibration*.npz):
       triangulate_with_kinect(aux_px, kin_px) → 3D do multi-view (DLT)
       to_kinect_stereo(aux_px, z_depth)       → px no RGB do Kinect
  → lateralidade por câmera (aux_handedness_flip.json + ArbitroDeLateralidade)
IMU ESP32 ×2 (MPU6050) 100 Hz via UDP 4210/4211
  → imu.py ImuReceiver → ImuBank → PositionFusion (acc+câmera) / KalmanTrajectory
  → imu_palm_alignment_{left,right}.json (R,t IMU→palma)
Esqueleto do corpo (Kinect body) → arm_model.json + IK de elos rígidos (ARM_*)
Palma → palm_normal_from_plane / normal_robusta → PALM_*
               ↓ TUDO junto no SharedState
        eeg_motor_paradigm.TrackingThread → RecordingMuxer → CSV do trial
        (+ MotionRecorder: movimento csv)
```

- **`kinect_imu_groundtruth.py` é biblioteca** (sem `main` próprio). Classes/funções-âncora:
  `KinectHandTracker`, `CameraAlignment`, `AuxCameraView`, `triangulate_aux_pairs`,
  `triangulate_three_views`, `build_projection_matrices`, `relative_transform`,
  `palm_frame`/`palm_normal_from_plane`/`normal_robusta`, `ArbitroDeLateralidade`,
  `indices_auxiliares_automaticos`, `aux_stereo_file`, `aux_hand_file`,
  `AUXILIARY_MIRROR_HORIZONTAL`, `MAX_STEREO_RMS_PX`.
- **UI completa/antiga do ground truth:** `tools/kinect_groundtruth_tool.py`
  (era o `main()` do módulo; calibração de palma/teclas/overlay).
- **Fusão (V1, experimental):** `src/mestradopy/fusion/*` — `live.py` (FusionEngine),
  `complementary.py` (quaternion), `vision_backends.py` (KinectDepth × MultiView),
  `aux_vision.py` (AuxStereoSource), `logging.py`, `metrics.py`, `sync.py`,
  `eeg_export.py` (colunas `FUSED_*`), `contracts.py`, `calibration.py`.
  Entrada: `tools/fusion_live.py`; A/B: `tools/vision_ab.py`.

### 4.2 Comparação de tracking (o que mede a qualidade)

```text
compara_tracking_mao.py  → grava tracking_mao_frames.csv + tracking_mao_landmarks.csv
                           + tracking_mao_meta.json (sem vídeo; só landmarks/3D)
metricas_tracking_mao.py → relatório .md + PNGs (cobertura/jitter/reprojeção/suavização)
```

Colunas que importam: `ok_kinect_detect`, `ok_aux_detect`, `ambas`, `mv_triangulou`,
`mv_valid`, `mv_implausivel`, `mv_conf`, `reproj_mv_kin_med_px`,
`reproj_mv_aux_med_px`, **`kd_aux_err_med_px` = "depth 3D na AUXILIAR"**,
`d_kin_mv_dorso_mm`, `jitter_*`, `fps_inst`, `aux_lag_ms`, `pareamento`, `fase`.
Critérios pré-fixados em `metricas_tracking_mao.CRITERIOS` (cobertura ≥90%,
jitter ≤5 mm, saltos ≤1%, reprojeção ≤6 px).

---

## 5. Configuração e calibração (`configs/calibracao/`)

| Arquivo | O que é | Observações |
|---|---|---|
| `stereo_calibration.npz` | aux0 (SIGMA-W780M) × Kinect | K/dist do Kinect **congelado**, K/dist da aux, `rotation`/`translation` (X_aux=R·X_kin+T), `rms`, `per_pose_rms`, `checkerboard_size=(7,5)`, `square_size_m=0.045` |
| `stereo_calibration_aux1.npz` | aux1 (Trust USB Camera) × Kinect | mesmo schema, arquivo separado por câmera |
| `stereo_calibration_aux2.npz` | aux2 | **INCOMPATÍVEL** (tabuleiro 5×3, 57 mm, `refined_intrinsics=1` → reestimou o Kinect). Não usar. |
| `camera_alignment.npz` | homografia 2D aux↔Kinect (v7) + `refinement_affine` | usada como fallback 2D quando não há stereo 3D |
| `kinect_intrinsics.npz` | intrínsecas do RGB do Kinect (congeladas) | base para todas as calibrações; `frozen_at=2026-10-09` |
| `hand_landmark_calibration.npz`, `_aux1.npz` | homografia + offsets 3D por landmark (palma) | legado de calibração da palma |
| `aux_handedness_flip.json` | flip de lateralidade aprendido por câmera | hoje `{0:true, 1:true}` (as duas câmeras entregam frame espelhado) |
| `imu_palm_alignment_{left,right}.json` | R,t IMU→palma | calibrado por sessão (`.bak`/`.antes_item153` são backups) |
| `arm_model.json` | elos ombro→cotovelo→punho | usado pela IK do braço |
| `mesa_kinect.json` | plano da mesa + poses físicas de referência | para normal da palma |
| `kinect_intrinsics.npz` × `stereo_calibration*.kinect_matrix` | devem ser idênticos | é o "modelo congelado" checado por `diagnostico_calibracao.py` |

**Regra:** não sobrescrever `.npz` calibrados sem necessidade; se recalibrar, salvar
backup e comparar (o arquivo novo diz `camera_name`, `rms`, `per_pose_rms`).

---

## 6. Essencial × experimental × legado/descartável

### 6.1 Essencial (não mexer sem revisão)
- `core/eeg_motor_paradigm.py` — **reservado** (outro agente mexe na animação MI).
- `core/kinect_imu_groundtruth.py` — biblioteca usada por tudo; mudanças de convenção
  (frame, espelhamento, triangulação) quebram o pipeline inteiro.
- `core/stereo_calibration.py`, `kinect_compat.py`, `camera_names.py`, `camera_uvc.py`.
- `tools/validacao_eeg_3classes.py` — aquisição vigente do piloto novo.
- `configs/calibracao/*.npz` e `*.json` calibrados.
- `eeg/mi_eegnet/` (treino/validação do EEGNet).

### 6.2 Experimental / em desenvolvimento
- Multiview/3 vistas: `triangulate_three_views`, `tools/compara_tracking_multiview.py`,
  `tools/test_triangulation_3view_synthetic.py`, `docs/guias/HANDMVNET_*.md`.
- HandMvNet: `src/mestradopy/vision/handmvnet_*`, `tools/handmvnet_*.py`,
  `external/HandMvNet`, `scripts/coletar_handmvnet.ps1`.
- Fusão V1: `src/mestradopy/fusion/*`, `tools/fusion_live.py`, `tools/vision_ab.py`,
  `docs/ARQUITETURA_FUSAO.md`, `docs/INTEGRACAO_EEG_FUSED.md` (**não ativar FUSED_* no
  EEG sem decisão**).
- SAND: `sand_trajectory_model.py`, `sand_traj_treino.py`, `sand_traj_tempo_real.py`.
- `imu.py` PositionFusion/Kalman (usado, mas calibração fina ainda em ajuste).

### 6.3 Legado / descartável como código ativo (não apagar sem revisar)
- `archive/root_legacy/` — scripts `_tmp_*`, `apply_*`, `fix_*`, CSVs antigos,
  `eeg_motor_paradigm.py.bak*`, imagens de bancada.
- `outputs/antigos/_teste_*` — execuções antigas (não tratar como resultado).
- `tools/` de bancada pontuais: `checa_kinect.py`, `preview_kinect.py`,
  `pos_reinicio_kinect.py`, `probe_kinect_*.py`, `conserta_intrinsecas_kinect.py`,
  `diagnostico_tabuleiro.py`, `probe_tabuleiro.py`, `checklist_kinect_mudo.py`,
  `verifica_kinect_antes_depois.py`, `diagnostico_hotspot.py`, `tecla_global.py`,
  `emissores_imu_falsos.py`, `debug_test.py`… — servem para bancada, não para produção.
- `tools/roda_testes.py` — **caminhos desatualizados pós-reorg** (procura `test_*.py`
  em `src/mestradopy/` e `.venv` ali dentro); usar `tests/manual` diretamente.
- `fusion_demo.py` (demo sintética), `sand_*.py` (ver 6.2), `palm_normal_edge_on.py`/
  `palm_orientacao.py` (experimentos de normal; produção usa `palm_*` do
  `kinect_imu_groundtruth`).
- `.bak`, `.antes_item*`, `.bak_20261110` — backups, não editar.

---

## 7. Comandos de teste reproduzíveis

### 7.1 Ambiente (sempre)

```powershell
Set-Location C:\mestradopy
$env:PYTHONPATH = "$PWD\src\mestradopy;$PWD\src\mestradopy\core;$PWD\src\mestradopy\tools;$PWD\src\mestradopy\eeg;$PWD\src\mestradopy\vision;$PWD\src\mestradopy\acquisition"
```

### 7.2 Sem hardware (síntese / unidades)

```powershell
# programa 1 sintético (protocolo completo, sem g.Nautilus/Kinect)
.\.venv\Scripts\python.exe src\mestradopy\core\eeg_motor_paradigm.py --source generator --blocos 1 --trials-por-bloco 2 --sem-preparacao --sem-instrucoes --sem-kinect --sem-janela-tracking --sem-painel-impedancia --sem-questionario

# tracking sintético (gera CSV + relatório; NÃO é resultado de bancada)
.\.venv\Scripts\python.exe src\mestradopy\tools\compara_tracking_mao.py --sintetico --duracao 10 --saida-dir outputs\visao\_smoke_tracking
.\.venv\Scripts\python.exe src\mestradopy\tools\metricas_tracking_mao.py --csv outputs\visao\_smoke_tracking\tracking_mao_frames.csv

# geometria multiview (sintético)
.\.venv\Scripts\python.exe src\mestradopy\tools\test_triangulation_3view_synthetic.py
.\.venv\Scripts\python.exe src\mestradopy\tools\multiview_janela.py --help

# calibração (diagnóstico offline)
.\.venv\Scripts\python.exe src\mestradopy\tools\diagnostico_calibracao.py   # (lê a pasta do próprio script — ver riscos)

# suite EEGNet
.\.venv\Scripts\python.exe -m unittest discover -s src\mestradopy\eeg\mi_eegnet\tests -v
```

### 7.3 Com hardware (bancada)

```powershell
# checagem de câmeras/índices
.\.venv\Scripts\python.exe src\mestradopy\tools\diagnostico_cameras.py

# calibração stereo (salva configs/calibracao/stereo_calibration*.npz)
.\.venv\Scripts\python.exe src\mestradopy\core\stereo_calibration.py --aux-index 0
.\.venv\Scripts\python.exe src\mestradopy\core\stereo_calibration.py --aux-index 1

# piloto ME 4 s + MI 4 s (2 classes) — coleta de verdade
.\scripts\run_piloto_me_mi.ps1 -Sessao 01 -Sujeito S01

# tracking comparativo (Kinect × aux0, depois × aux1)
.\.venv\Scripts\python.exe src\mestradopy\tools\compara_tracking_mao.py --aux-index 0 --duracao 15
.\.venv\Scripts\python.exe src\mestradopy\tools\compara_tracking_mao.py --aux-index 1 --duracao 15

# demo 3 vistas (Kinect + aux0 + aux1)
.\.venv\Scripts\python.exe src\mestradopy\tools\compara_tracking_multiview.py

# ground truth completo (UI histórica)
.\.venv\Scripts\python.exe src\mestradopy\tools\kinect_groundtruth_tool.py --aux-index 0
```

> `tests/manual/test_*.py`: rodar individualmente (`python tests\manual\test_x.py`);
> muitos exigem Kinect/eeg/IMU. `tests/manual/__init__.py` existe, mas não há runner
> confiável pós-reorg.

---

## 8. Não tocar (ou só leitura)

1. `src/mestradopy/core/eeg_motor_paradigm.py` — **reservado** (animação MI em evolução).
2. `configs/calibracao/*.npz` — não sobrescrever sem backup/comparação; os `.npz`
   novos de `stereo_calibration*.npz` estão sob suspeita (ver §9.3).
3. `src/mestradopy/fusion/contracts.py`, `eeg_export.py` — schema de colunas `FUSED_*`
   (mexer quebra compatibilidade).
4. `outputs/eeg/`, `data/raw/` — dados/resultados; não editar/sobrescrever.
5. `docs/thesis/`, `Tese_Latex_Agente/` e `Comite Etica/` (fora do repo) — conteúdo da
   tese/CEP; escrita só por quem tem a tarefa.
6. `external/HandMvNet`, `external/RSF` — código de terceiros; integrar por adaptador.
7. `archive/` — legado; não apagar.
8. `.git` — nada de commit/push sem revisão do Lead.

---

## 9. Pontos de entrada e riscos para a tarefa de multiview

### 9.1 Arquivos que governam a geometria (ordem de leitura)

1. `src/mestradopy/core/kinect_imu_groundtruth.py`:
   - `AuxCameraView` (~L2865): convenção **X_aux = R·X_kin + T**, `to_kinect_stereo`,
     `triangulate_with_kinect` (P1=[I|0], P2=[R|T] sobre raios desdistorcidos).
   - `CameraAlignment` (~L3308): carrega `configs/calibracao/stereo_calibration*.npz`,
     `stereo_usable` = RMS ≤ 5 px (aceite "pronta" ≤ 2 px), `aux_view(idx)`.
   - `_triangulate_rays` (~L3030), `triangulate_pair_points`, `triangulate_aux_pairs`,
     `erro_reprojecao_px`, `relative_transform`.
   - `triangulate_three_views`/`build_projection_matrices` (implementados em 12/11;
     ainda **não ligados** ao pipeline principal de aquisição).
   - `transform_auxiliary_frame` + `AUXILIARY_MIRROR_HORIZONTAL` + `aux_handedness_flip`.
2. `src/mestradopy/tools/compara_tracking_mao.py`: `_kinect_3d_21`,
   `_projeta_cor_sdk`, `_processa_lado` (é onde `kd_aux` é medido), `_grava_frame_e_landmarks`.
3. `src/mestradopy/tools/metricas_tracking_mao.py`: seções 3.3 (`depth 3D na AUXILIAR`),
   3.4 (jitter), 3.5 (saltos) e `CRITERIOS`.
4. `src/mestradopy/tools/diagnostico_calibracao.py`: sensibilidade a Z
   (px/cm por 1 cm de depth), consistência do modelo do Kinect entre arquivos.
5. `src/mestradopy/tools/test_triangulation_3view_synthetic.py` e
   `tests/manual/test_aux_pair_stereo.py`: validação sintética da geometria.

### 9.2 Riscos conhecidos (medidos/observados)

- **Frame depth × color (crítico, descoberto em 18/11/2026):** `_kinect_3d_21`
  devolve o 3D de `MapDepthPointToCameraSpace` = frame da **câmera de profundidade**,
  mas a calibração stereo (`stereo_calibration*.npz`) foi feita no frame da **câmera
  de cor** (tabuleiro detectado no RGB do Kinect). Projetar esse 3D na auxiliar injeta
  a baseline depth→color (~6 cm; sistemática). Replay offline com um PnP ajustado nos
  próprios dados (resíduo 1,6-1,9 px): `depth 3D na AUXILIAR` cai de **84,6 → 15,2 px
  (aux0)** e **39,0 → 10,4 px (aux1)** na sessão de 12/11. **Qualquer métrica que
  compare `k_xyz` (depth) com multi-view 3D (`d_kin_mv_dorso_mm`) ou com a auxiliar
  está contaminada por esse frame.** A triangulação em si (que usa só pixels) não é
  afetada; o `triangulate_three_views`/`CameraAlignment` trabalham no frame da cor.
- **Calibração nova (17/11) suspeita:** comparada com a de 12/11, as extrínsecas
  mudaram 20,4° (aux0) / 10,8° (aux1), mas a baseline aux0↔aux1 ficou em ~1,11 m.
  Aplicada aos dados de 12/11, a nova calibração piora o `depth 3D na AUXILIAR`
  (aux0 149 px / aux1 39 px vs 15/10 px com a antiga). Se o usuário **não** moveu as
  câmeras ao recalibrar, a calibração nova está errada; se moveu, o dataset de 12/11
  não serve para a geometria nova e é preciso nova captura. **Não usar para
  labels/treino até validação com tabuleiro/mão.**
- **Espelhamento:** `AUXILIARY_MIRROR_HORIZONTAL=True` espelha o frame ANTES do
  MediaPipe (pixels e calibração vivem na imagem espelhada); `aux_handedness_flip.json`
  corrige o rótulo esquerda/direita por câmera (hoje flip=true nas duas). Mexer nisso
  troca mão e quebra o pareamento.
- **Ordem canônica de cantos:** `stereo_calibration.simetrias_do_tabuleiro`,
  `ordem_canonica_do_tabuleiro`, `solve_with_repair` tentam corrigir ordem/rotação/
  espelhamento do tabuleiro; conferir `per_pose_rms` e `kinect_deviation_px` (desvio
  do modelo estimado vs congelado; >10 px = sessão de tabuleiro fraca).
- **aux2 incompatível** (`checkerboard 5×3`, 57 mm, `refined_intrinsics=1`): ignorar.
- **Escala de resolução:** `AuxCameraView.effective_matrix`/`runtime_size` reescala K
  se a câmera rodar em resolução diferente da calibrada (calibrado 1280×720); aviso
  `auxiliary_scale_warned`.
- **RMS de aceite:** `MAX_STEREO_RMS_PX=2.0` (pronta) e `STEREO_OVERLAY_MAX_RMS_PX=5.0`
  (usável). RMS baixo **não garante** extrínseca certa (calibração planar tem
  ambiguidade) — validar com tabuleiro em ≥2 distâncias e com a mão.
- **Sincronismo Kinect↔aux:** frames lidos no mesmo laço, sem trigger; `aux_lag_ms`
  mediano ~15 ms (p95 ~31 ms) na sessão de 12/11 — não explica erro grosseiro, mas
  atrapalha jitter em movimento.
- **Cobertura dos 21 landmarks:** aux0/aux1 são 720p (vs Kinect 1080p); perdas
  concentram-se em punho e pontas dos dedos, e no gate de plausibilidade
  (`mv_implausivel` 9-12%) — não é falta de calibração.

### 9.3 Próximos passos técnicos (ordem)

1. Validar calibração (antiga × nova) com tabuleiro AO VIVO e comparar mão real; só
   então decidir qual `stereo_calibration*.npz` é a geometria de HOJE.
2. Corrigir o frame depth→color no caminho que usa `k_xyz` para comparar/projetar na
   auxiliar (adicionar conversão: `MapCameraPointToColorSpace` → espaço de câmera de
   cor, ou extrínseca depth→color salva em `configs/calibracao/`).
3. Ligar `triangulate_three_views` ao `compara_tracking_multiview.py`
   (já ligada) e a um tool de captura/validação (fallback de pares Kinect+aux0,
   Kinect+aux1, aux0+aux1) para cobertura ≥90% quando a mão é visível.
4. Reavaliar cobertura/jitter/reprojeção com `metricas_tracking_mao.py` e escrever
   o relatório em `Tese_Latex_Agente/Tracking/` (Uma Só Verdade: Kinect depth como
   label primário até a validação externa fechar).

---

## 10. Documentos canônicos (para não ler a raiz antiga)

| Tema | Documento |
|---|---|
| Reorganização/árvore | `docs/guias/MIGRACAO_ESTRUTURA.md`, `README.md` |
| Histórico completo | `docs/guias/HISTORICO_ITERACOES.txt` (676 kB; canônico; raiz tem atalho) |
| Pipeline de dados/janelas | `docs/PIPELINE_DADOS_JANELAS.md` |
| Schema do dataset de pose | `docs/DATASET_POSE_SCHEMA.md` |
| Calibração geométrica/palma | `docs/CALIBRACAO_GEOMETRICA.md` |
| Bifurcação visão (KinectDepth × MultiView) | `docs/BIFURCACAO_VISAO.md` |
| Fusão/sincronização | `docs/ARQUITETURA_FUSAO.md`, `docs/INTEGRACAO_EEG_FUSED.md` |
| HandMvNet | `docs/guias/HANDMVNET_INTEGRACAO.md`, `docs/guias/HANDMVNET_TREINO.md` |
| Escopo/CEP/cronograma | `docs/ESCOPO_TESE_POSE_MI.md`, `docs/EMENDA_CEP_POSE_MI.md`, `docs/CRONOGRAMA_6_MESES.md` |
| Validação | `docs/validation/`, `docs/thesis/Tese_Latex_Agente/Validacao/` |
| Piloto novo ME/MI | `outputs/eeg/piloto_me_mi_s01|s02/` + `docs/guias/` |
| Tracking/multiview (vivo) | `C:\Users\Usuario\OneDrive\Desktop\Mestrado_Projeto\Tese_Latex_Agente\Tracking\` |

---

*Mapa produzido em 18/11/2026 a partir da árvore atual (`src/mestradopy`, 149 módulos),
da migração documentada, dos launchers em `scripts/`, da leitura de docstrings/código
essencial e da análise dos CSVs de tracking de 12/11. Sem edições de código nesta
tarefa. Estimativas de mult-view medidas estão em
`Tese_Latex_Agente/Tracking/MULTIVIEW_AUX0_AUX1.md`.*
