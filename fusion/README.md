# fusion/ -- V1 da fusão de orientação

Pacote isolado para o protótipo de fusão **Kinect v2 + MediaPipe + IMU**.
A V1 é Windows, sem estéreo e sem `JointOrientation` do Kinect.

## Estado

`q_body_cam` = quaternion **corpo do sensor -> câmera** (w,x,y,z).
A normal da palma é `n_cam = q_body_cam ⊗ f_body` (rotação de vetor).
Nunca use Euler para a fusão.

## Componentes

- `contracts.py`: `KinectObservation`, `MediaPipeObservation`, `ImuSample`,
  `Calibration`, `FusedState`, `SourceStatus`.
- `quaternion.py`: Hamilton, rotação de vetor, ZYX, matriz, slerp, exp SO(3).
- `sync.py`: buffers por fonte + `SyncBuffer.snapshot()` com idade/validade.
- `calibration.py`: carrega/salva o JSON `rotation`/`body` do projeto
  (`load_calibration`, `save_calibration`) e aplica T/f.
- `sensor_manager.py`: decide `fused`/`imu_only`/`vision_only`/`none`.
- `complementary.py`: filtro complementar em quaternion:
  predição por gyro, correção de gravidade e correção de yaw pela visão
  (correção de yaw pela ESQUERDA, no frame da câmera).
- `visualization.py`: overlay OpenCV leve (sem abrir janela por conta própria).

## Uso mínimo

```python
from fusion import (
    Calibration, ImuSample, QuaternionComplementaryFilter,
    ComplementaryConfig, FusedState, draw_overlay,
)

cal = Calibration.from_dict(json.load(open("imu_palm_alignment_right.json")),
                            hand="right")
filtro = QuaternionComplementaryFilter(
    config=ComplementaryConfig(gravity_cam=cal.T_cam_world @ [0, 0, -1]),
    calibration=cal,
)
q = filtro.step(imu, normal_cam, dt, vision_confidence=0.9)
```

## Testes

```powershell
.\venv\Scripts\python.exe test_fusion_quaternion.py
.\venv\Scripts\python.exe test_fusion_contracts.py
.\venv\Scripts\python.exe test_fusion_sync.py
.\venv\Scripts\python.exe test_fusion_calibration.py
.\venv\Scripts\python.exe test_fusion_sensor_manager.py
.\venv\Scripts\python.exe test_fusion_complementary.py
.\venv\Scripts\python.exe test_fusion_visualization.py
.\venv\Scripts\python.exe fusion_demo.py
```

## Bancada (Fase 3.5)

```powershell
# Sintético (sem hardware)
.\.venv\Scripts\python.exe tools\fusion_live.py --synthetic --sem-janela --duracao 1 --sem-calibracao

# Kinect real (sem janela; Ctrl+C limpo)
.\.venv\Scripts\python.exe tools\fusion_live.py --sem-janela --duracao 10

# Apenas Kinect / apenas IMU
.\.venv\Scripts\python.exe tools\fusion_live.py --sem-imu --sem-janela --duracao 10
.\.venv\Scripts\python.exe tools\fusion_live.py --sem-kinect --sem-janela --duracao 10
```

Smoke validado em 05/11/2026: Kinect + MediaPipe via `KinectMediaPipeSource`,
duas mãos detectadas, ~35 FPS com calibrações reais carregadas, modo
`vision_only` quando não há amostras IMU. A janela OpenCV é opcional
(`--sem-janela`); a lógica fica em `fusion/live.py`.

## O que falta para bancada

- Adaptadores reais: `KinectHandTracker`, MediaPipe, `ImuReceiver` -> contratos.
- Escolher `gravity_cam` a partir da calibração T (câmera pode estar inclinada).
- Loop de visualização/janela em `tools/` (o pacote só desenha).
- Validação sintética (já coberta) -> bancada Kinect+IMU -> jig A4/nível.
- Integração EEG futura via colunas `FUSED_*` versionadas, sem alterar
  `PALM_*`/`IMU_*`.

