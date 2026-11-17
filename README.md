# Mestradopy — EEG, visão e fusão

Pipeline de aquisição e análise do mestrado: EEG g.Nautilus, Kinect v2,
webcams auxiliares, IMU e modelos neurais.

A raiz mantém este README, o `.gitignore` e um **atalho de compatibilidade**
para o log histórico ativo. O arquivo canônico é
`docs/guias/HISTORICO_ITERACOES.txt`; o atalho na raiz existe porque agentes e
scripts antigos ainda escrevem nesse caminho.

## Estrutura

```text
src/mestradopy/
  core/           módulos ativos de aquisição, visão, IMU e calibração
  tools/          ferramentas executáveis de análise, treino e diagnóstico
  eeg/mi_eegnet/  pacote EEGNet configurável (5.870 parâmetros)
  fusion/         biblioteca de fusão/sincronização
  vision/         espaço para novos módulos de visão
  acquisition/    espaço para novos protocolos
  calibration/    espaço para novos módulos de calibração

scripts/          entradas PowerShell/BAT (run_piloto_me_mi.ps1, run_eegnet_*.ps1)

configs/
  calibracao/     stereo_calibration*.npz, camera_alignment.npz,
                  kinect_intrinsics.npz, hand_landmark_calibration*.npz,
                  imu_palm_alignment*.json, arm_model.json

data/
  raw/            Datasets (BCIC-IV 2a/2b, WAY-EEG-GAL)
  cache/          caches de features
  calibracao/     dados de calibração (se necessário no futuro)
  gravacoes/      gravações de aquisição

outputs/
  eeg/            resultados de EEG e modelos
  visao/          resultados de visão
  fusao/
  antigos/

assets/
  models/         hand_landmarker.task, pose_landmarker_lite.task
  cues/           imagens de cue

docs/
  literature/     PDFs e texto de literatura
  validation/     relatórios de validação (CSP, ERD, EEGNet etc.)
  thesis/         Tese_Latex_Agente
  guias/          guias operacionais e README original

tests/
  legacy/         scripts de teste manuais antigos

archive/
  root_legacy/    scripts/logs/temporários antigos
  backups/

external/         _github-3d, _external_repo
firmware/         AHRS e firmware
```

## Ambiente

```powershell
Set-Location C:\mestradopy
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para comandos diretos, ative os caminhos:

```powershell
$env:PYTHONPATH = "$PWD\src\mestradopy;$PWD\src\mestradopy\core;$PWD\src\mestradopy\tools;$PWD\src\mestradopy\eeg;$PWD\src\mestradopy\vision;$PWD\src\mestradopy\acquisition"
```

## Comandos principais

### Piloto ME/MI

```powershell
.\scripts\run_piloto_me_mi.ps1 -Sessao 01 -Sujeito S01
```

Saída: `outputs/eeg/piloto_me_mi_s01/`.

### Testes do pacote EEGNet

```powershell
$env:PYTHONPATH = "$PWD\src\mestradopy\eeg;$PWD\src\mestradopy\core"
.\.venv\Scripts\python.exe -m unittest discover -s src\mestradopy\eeg\mi_eegnet\tests -v
```

### Pré-treino 2b

```powershell
$env:PYTHONPATH = "$PWD\src\mestradopy\eeg;$PWD\src\mestradopy\core"
.\.venv\Scripts\python.exe -m mi_eegnet.pretrain --device xpu
```

Config: `src/mestradopy/eeg/mi_eegnet/configs/default.yaml`.

### Calibração stereo

```powershell
$env:PYTHONPATH = "$PWD\src\mestradopy\core"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\python.exe src\mestradopy\core\stereo_calibration.py --aux-index 1
```

Calibrações ficam em `configs/calibracao/`.

## Regras

- Não deixar resultado de execução na raiz.
- Saídas sempre em `outputs/<area>/<experimento>/`.
- Dados brutos e caches sempre em `data/`; não versionar.
- Código novo em `src/mestradopy/` ou `scripts/`.
- Configuração de câmera/EEG em `configs/`.
- Relatórios em `docs/validation/`; literatura em `docs/literature/`.
- Experimentos antigos em `archive/`, não apagar sem revisar.

O README antigo, com os detalhes de bancada e calibração, está em
`docs/guias/README_ORIGINAL.md`.
