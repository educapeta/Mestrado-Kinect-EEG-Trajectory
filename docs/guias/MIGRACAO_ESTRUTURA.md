# Mapa da reorganização (17/11/2026)

A raiz foi limpa em duas etapas: checkpoint do estado anterior e reorganização
em `src/`, `data/`, `outputs/`, `docs/`, `scripts/`, `tests/`, `configs/`,
`assets/`, `external/`, `firmware/` e `archive/`.

## Antes → depois

| Antes (raiz) | Depois |
|---|---|
| `eeg_motor_paradigm.py`, `body_joints_recorder.py`, `imu.py`, `imu_reset.py`, `hotkeys.py`, `kinect_compat.py`, `kinect_imu_groundtruth.py`, `camera_uvc.py`, `camera_names.py`, `stereo_calibration.py`, `freeze_kinect_direto.py`, `estat_bci.py`, `fusion_demo.py`, `mi_animation_method.py`, `palm_*`, `sand_*` | `src/mestradopy/core/` |
| `mi_eegnet/` | `src/mestradopy/eeg/mi_eegnet/` |
| `tools/` | `src/mestradopy/tools/` |
| `fusion/` | `src/mestradopy/fusion/` |
| `run_piloto_me_mi.ps1`, `run_eegnet_*.ps1`, `*.bat` | `scripts/` |
| `test_*.py` | `tests/manual/` |
| `Datasets/` | `data/raw/Datasets/` |
| `gravacoes/` | `data/gravacoes/` |
| `_piloto_me_mi_s01`, `_piloto_me_mi_s02`, `_eegnet*`, `_erd*`, `_csp*`, `_alpha*`, `_bci2a*` | `outputs/eeg/` (sem o `_` inicial nos resultados novos) |
| `_teste_*`, `_tmp_*`, `_cleanup`, `_toy_out`, `_recuperado*` | `outputs/antigos/` |
| `literatura/` | `docs/literature/` |
| `Validacao/` | `docs/validation/Validacao/` |
| `Tese_Latex_Agente/` | `docs/thesis/Tese_Latex_Agente/` |
| `stereo_calibration*.npz`, `camera_alignment.npz`, `kinect_intrinsics.npz`, `hand_landmark_calibration*.npz`, `imu_palm_alignment*.json`, `arm_model.json` | `configs/calibracao/` |
| `hand_landmarker.task`, `pose_landmarker_lite.task` | `assets/models/` |
| `_cue_*.png` | `assets/cues/` |
| `_github-3d/`, `_external_repo/` | `external/` |
| `AHRS/` | `firmware/AHRS/` |
| scripts/logs/temporários antigos | `archive/root_legacy/` |

## Caminhos de execução

O código ativo não usa mais `Path(__file__).with_name(...)` para calibração:
`stereo_calibration.py` e `kinect_imu_groundtruth.py` apontam para
`configs/calibracao/`. As saídas novas vão para `outputs/<area>/`.

Para executar comandos diretos, defina:

```powershell
$env:PYTHONPATH = "$PWD\src\mestradopy;$PWD\src\mestradopy\core;$PWD\src\mestradopy\tools;$PWD\src\mestradopy\eeg;$PWD\src\mestradopy\vision;$PWD\src\mestradopy\acquisition"
```

Ou use os launchers em `scripts/`, que já fazem isso.

## Git

- `9230311`: checkpoint do estado anterior.
- `c5c64d6`: reorganização da árvore.
- Dados (`data/`), saídas (`outputs/`), arquivos antigos (`archive/`) e
  externos (`external/`) ficam fora do Git.
