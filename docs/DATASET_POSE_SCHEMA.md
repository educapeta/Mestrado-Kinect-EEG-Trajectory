# Dataset de pose da mão para EEG — schema v1

> Objetivo: coletar um dataset piloto **EEG → pose 6D da mão** (posição +
> orientação), sem fusão T/f online. O alvo pode ser a pose do pulso/TCP da
> garra; a mão robótica inicial é tratada como corpo rígido (dedos
> independentes ficam para uma versão futura).

## 0. Decisões que o schema fixa

| Item | Decisão | Por quê |
| --- | --- | --- |
| Frame dos labels | `base` = tronco/cabeça do sujeito; origem no esterno ou ponto médio dos ombros | EEG não carrega posição absoluta; frame corporal é repetível e não depende do robô |
| Pose | `T_base_hand` = posição `p` (m) + quaternion `q` (xyzw) da mão/TCP | 6 GDL; quaternion evita gimbal lock |
| Pose de repouso | gravada em `rest/` no início e no fim da sessão | serve de baseline/home e para calcular pose relativa |
| Rotação de treino | **6D** derivada de `R`, não Euler | representação contínua e boa para regressão |
| Câmeras | pelo menos 2 calibradas (intrínsecos + extrínsecos) | triangulação sem depth do Kinect |
| MediaPipe | salvar **landmarks brutos** + confiança; não usar `z` relativo como métrica | permite reprocessar/recalibrar depois |
| IMU | salvar **bruto** (acc/gyro/quaternion), sem calibrar com câmera | entrada auxiliar; T/f fica para o futuro |
| Taxas | EEG nativo (500 Hz); IMU nativo; labels a 30 Hz | manter o dado cru; a janela de treino reamostra |
| Split | por sessão/sujeito, nunca por frame aleatório | evita vazamento temporal |

Convenção de eixos do frame `base` (definir e manter em todas as sessões):

- `x`: direita do sujeito;
- `y`: para cima;
- `z`: para frente (para longe do sujeito);
- rotação dextrógira.

Se o robô usar outro frame, grave a matriz `T_robot_base` em `meta.yaml`.
O modelo aprende no frame `base`; a conversão para o robô é externa.

## 1. Árvore de diretórios

```text
dados_pose/
  manifest.csv                  # índice de todas as sessões
  subjects/
    S01/
      sessions/
        2026-11-10_01/
          meta.yaml
          trials.csv
          sync.csv
          events.csv
          eeg/
            eeg.npz
          vision/
            cameras.yaml
            landmarks.csv
            landmarks_3d.csv
          pose/
            pose_labels.csv
          imu/
            imu_raw.csv
          rest/
            rest_pose.csv
          quality/
            quality_report.json
```

Nomes: `S01`, `S02`; sessão `YYYY-MM-DD_NN` (duas dígitas para ordem).
Nunca sobrescrever uma sessão: criar `_02`, `_03`.

## 2. `manifest.csv` — índice do dataset

Cabeçalho exato:

```csv
session_id,subject_id,date,path,pose_frame,eeg_fs,camera_ids,has_imu,n_frames,duration_s,split,notes
S01_2026-11-10_01,S01,2026-11-10,subjects/S01/sessions/2026-11-10_01,base,500,"0;1",1,3120,104.0,train,piloto inicial
```

- `split`: `train`, `val`, `test` ou `pilot` (não usar teste para treinar).
- `path`: relativo à raiz `dados_pose/`.
- `has_imu`: `1`/`0`.
- Uma linha por sessão; o loader valida que o caminho existe.

## 3. `meta.yaml` — metadados da sessão

```yaml
schema_version: 1
session_id: S01_2026-11-10_01
subject_id: S01
date: 2026-11-10
operator: ""
task: "pose_pilot"
hand: both                  # right, left, both
pose_frame: base
base_origin: "sternum"      # sternum, shoulder_mid, head
base_axes: "x=right,y=up,z=forward,right_handed"
robot_frame: null           # ex.: T_robot_base (matriz 4x4) se já existir

rest_pose:
  file: rest/rest_pose.csv
  description: "braços apoiados na mesa, palmas para baixo, dedos relaxados"
  start_and_end: true

sensors:
  eeg:
    system: "g.Nautilus"
    n_channels: 32
    fs_hz: 500
    montage: "g.tec 32"
    unit: "uV"
    reference: "earlobe"    # conferir no sistema
  cameras:
    ids: ["0", "1"]
    model: ["SIGMA-W780M", "Trust USB Camera"]
    calibration: "stereo_calibration_aux0.npz"
    resolution: [1280, 720]
  imu:
    sides: ["right", "left"]
    fs_hz: 100
    unit_acc: "g"
    unit_gyro: "dps"
    firmware: "esp32_duas_imus_serial"

markers:
  source: "LED"             # LED, audio, serial, teclado
  note: "evento de início/fim de trial gravado em events.csv"

files:
  eeg: eeg/eeg.npz
  landmarks: vision/landmarks.csv
  landmarks_3d: vision/landmarks_3d.csv
  pose_labels: pose/pose_labels.csv
  imu: imu/imu_raw.csv
  trials: trials.csv
  sync: sync.csv
  quality: quality/quality_report.json

notes: ""
```

## 4. `eeg/eeg.npz`

Arrays dentro do `.npz`:

| Chave | Shape | Tipo | Unidade |
| --- | --- | --- | --- |
| `eeg` | `(n_samples, n_channels)` | float32 | µV |
| `t_us` | `(n_samples,)` | int64 | µs no relógio do PC |
| `fs` | escalar | float32 | Hz |
| `ch_names` | `(n_channels,)` | unicode | nomes dos canais |
| `t0_unix_us` | escalar | int64 | µs Unix do primeiro sample |
| `montage` | escalar | unicode | nome da montagem |

Não filtrar/notch dentro do `.npz`: guardar **cru**. O pré-processamento de
treino faz banda, CAR e z-score de forma reproduzível.

## 5. `events.csv` e `trials.csv`

`events.csv`:

```csv
t_us,event,trial_id,phase,note
1730000000123456,trial_start,1,reach,LED acendeu
1730000001987654,hand_movement_onset,1,reach,onset por cinemática
1730000004321000,trial_end,1,return,LED apagou
```

- `event`: `trial_start`, `hand_movement_onset`, `grasp`, `lift`, `release`,
  `return`, `rest_start`, `rest_end`.
- `t_us`: relógio do PC; o `sync.csv` relaciona com cada stream.

`trials.csv`:

```csv
trial_id,instruction,t_start_us,t_end_us,hand,rest_before,notes
1,"reach + grasp",1730000000123456,1730000004321000,right,1,""
2,"rotate wrist",1730000005000000,1730000009000000,right,1,""
```

## 6. `vision/landmarks.csv` — MediaPipe bruto

Uma linha por `(frame, câmera, landmark)`:

```csv
t_us,frame_id,camera_id,hand,landmark_id,x_px,y_px,z_rel,visibility
1730000000200000,101,cam0,right,0,640.2,360.1,-0.021,0.99
...
```

- `landmark_id`: 0–20 (índice MediaPipe).
- `hand`: `right` ou `left`.
- `x_px,y_px`: pixel na imagem original da câmera.
- `z_rel`: saída relativa do MediaPipe (não é metro; não usar como label).
- `visibility`: confiança/presença (0–1), quando disponível.

`vision/cameras.yaml`: intrínsecos e extrínsecos por câmera; usar os arquivos
`.npz` de calibração já existentes, mas gravar aqui o nome/versão usada.

## 7. `vision/landmarks_3d.csv` — triangulação multi-view

```csv
t_us,frame_id,hand,landmark_id,x_m,y_m,z_m,reproj_px,n_views,confidence
1730000000200000,101,right,0,0.031,-0.042,0.512,0.84,2,0.97
```

- `x_m,y_m,z_m`: triangulação no frame da **câmera de referência** (cam0).
- `reproj_px`: erro médio de reprojeção nas câmeras.
- `n_views`: 2+.
- Se usar Kinect color como câmera extra, manter o mesmo schema e registrar
  `camera_id=kinect` em `cameras.yaml`.

## 8. `pose/pose_labels.csv` — alvo supervisionado

```csv
t_us,frame_id,hand,pos_x_m,pos_y_m,pos_z_m,quat_w,quat_x,quat_y,quat_z,normal_x,normal_y,normal_z,finger_x,finger_y,finger_z,confidence,source
1730000000200000,101,right,0.031,-0.042,0.512,0.707,-0.001,0.707,0.001,0.021,0.998,0.051,0.999,0.010,-0.030,0.97,triangulated
```

Definições:

- `pos_*`: posição do **ponto de referência da mão** (pulso ou TCP da garra),
  em metros, no frame `base`.
- `quat_*`: quaternion unitário `(w,x,y,z)` de `R_base_hand`.
- `normal_*`: normal da palma (ou eixo de aproximação da garra) em `base`.
- `finger_*`: direção dos dedos / eixo de referência em `base`.
- `source`: `triangulated`, `kinect_depth`, `manual`, `rest`.
- `confidence`: 0–1; usar `0` ou vazio quando a pose não for confiável.

Regra de ouro: **não preencher quaternion inventado**. Se a orientação não foi
medida, deixe vazio e marque `confidence=0`.

## 9. `rest/rest_pose.csv`

Mesmo cabeçalho de `pose_labels.csv`, com uma linha por frame de repouso.
Gravar no início e no fim da sessão. O loader usa a mediana do repouso como
`T_base_rest` e pode gerar o alvo relativo:

```text
T_rel = T_base_rest^-1 · T_base_hand
```

Você pode treinar com pose absoluta (`T_base_hand`) ou relativa; escolha uma e
grave no config. Para o robô, a relativa ao repouso costuma ser mais estável.

## 10. `imu/imu_raw.csv`

```csv
t_us,side,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,qw,qx,qy,qz,boot_id,valid
1730000000200000,right,0.012,-0.003,0.998,0.21,-0.10,0.05,0.999,0.01,0.02,0.03,boot_7,1
```

- `side`: `right`/`left`.
- `boot_id`: identificar o boot do ESP32; indispensável para saber quando o
  yaw absoluto mudou.
- `valid`: `1` só quando a amostra passou pelos filtros de faixa/NaN.
- **Não** gravar T/f, normal calibrada ou pose derivada do IMU neste arquivo.

## 11. `sync.csv`

```csv
t_pc_us,stream,t_stream_us,offset_us,method,note
1730000000200000,eeg,1730000000100000,100000,hardware_marker,"trigger no início"
1730000000200000,cam0,1730000000190000,10000,pc_clock,"frame no mesmo PC"
```

- `offset_us`: `t_stream - t_pc` estimado.
- `method`: `hardware_marker`, `pc_clock`, `led`, `audio`, `xcorr`.
- O mínimo para o piloto: mesmo PC para EEG e câmeras + marcador de evento.
  Se EEG vier de outro PC, gravar um sinal de sincronização (LED/trigger).

## 12. `quality/quality_report.json`

Gerar depois de cada sessão:

```json
{
  "schema_version": 1,
  "eeg": {"n_channels": 32, "fs": 500, "nan_frac": 0.0, "line_noise_ok": true},
  "vision": {"n_frames": 3100, "mean_landmark_conf": 0.94, "pct_frames_2views": 0.91},
  "pose": {"n_labels": 3000, "pct_with_quat": 0.89, "median_reproj_px": 1.2},
  "imu": {"hz_right": 100.1, "hz_left": 99.8, "pct_valid": 0.99},
  "sync": {"max_offset_ms": 3.2},
  "warnings": []
}
```

Critérios mínimos para o piloto:

- `pct_frames_2views >= 0.80`;
- `pct_with_quat >= 0.80` (ou registrar por que não);
- `median_reproj_px <= 5`;
- `nan_frac` de EEG `< 1%`;
- `max_offset_ms <= 10`.

## 13. O que **não** entra no dataset

- T/f do IMU com o Kinect;
- quaternion/pose “calibrada” do IMU;
- Euler já convertido;
- frames filtrados sem registrar o cru;
- arquivos `.mat`/`.npz` gigantes sem manifest e SHA1.

## 14. Versionamento e privacidade

- `manifest.csv`, `meta.yaml`, `quality_report.json` e códigos: versionar.
- EEG/vídeo/landmarks: dados pessoais; **não** versionar no Git público.
- Guardar um `sha256` de cada arquivo grande em `manifest.csv` ou num
  `SHA256SUMS` da sessão.
