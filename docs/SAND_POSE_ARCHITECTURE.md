# SAND-Pose — especificação de treino e arquitetura

> Extensão do SAND atual (`sand_traj_treino.py` / `sand_trajectory_model.py`)
> para prever **posição + orientação** da mão, sem quebrar o modelo de
> trajetória que já funciona. O alvo é a pose do pulso/TCP da garra no frame
> `base` definido em `docs/DATASET_POSE_SCHEMA.md`.

## 1. Baseline atual (não mudar)

Hoje o SAND tem:

```text
X: (B, C, T)                 # EEG: canais × tempo
SANDTrajectory(X) -> (B, 30, 3)   # xyz em metros (normalizado)
loss = MSE
```

Componentes existentes em `sand_trajectory_model.py`:

- `SpectralFeatureExtractor`: `(B,C,T) -> (B,T',d_model)`;
- `FFTTransformerBlock`: atenção espectral/temporal;
- `TrajectoryProjector`: `(B,T',d_model) -> (B,30,3)`;
- `TrajectoryNormalizer`: normaliza `xyz` para o treino.

**Regra:** criar `SANDPose` em um arquivo novo (`sand_pose_model.py`) e um
treino novo (`sand_pose_treino.py`). Não alterar `SANDTrajectory` para não
quebrar `sand_traj_tempo_real.py`.

## 2. Novo alvo

Para cada janela, prever uma sequência de pose:

```text
pos:  (B, S, 3)    # metros
rot:  (B, S, 6)    # 6D de R_base_hand (ou R_rel_rest)
mask: (B, S)       # 1 = pose válida naquele passo
```

`S = 30` no piloto (mesma `output_seq_len` do SAND atual).

### 2.1. Posição

- `pos` em metros no frame `base`;
- `target_mode: relative_rest` → `p_rel = p_hand - p_rest` (recomendado no piloto);
- `target_mode: absolute` → `p_hand` direto;
- usar `TrajectoryNormalizer` existente para `pos` (média/desvio do treino,
  nunca do teste);
- guardar `normalizer` no checkpoint.

### 2.2. Rotação: 6D, não Euler

Para uma matriz `R ∈ SO(3)`, a representação 6D é:

```text
r6 = [R[:,0], R[:,1]]        # 6 números
```

Para voltar para rotação:

```text
a1 = normalize(r6[0:3])
a2 = normalize(r6[3:6] - dot(a1, r6[3:6]) * a1)
b1 = a1
b2 = cross(a1, a2)
R  = [b1, b2, cross(b1,b2)]   # colunas
```

No loader:

```text
R_base_hand  -> r6 (target)
R_model      -> 6D -> Gram-Schmidt -> R_pred
```

No checkpoint, guardar também `y_quat` `(B,S,4)` como **metadado de avaliação**;
não usar quaternion como alvo de treino no piloto (evita o problema `q` vs `-q`).

### 2.3. Perda

```text
L_pos = Huber(pred_pos, true_pos)
L_rot = arccos(clamp((trace(R_pred^T R_true) - 1) / 2, -1, 1))
L_smooth = mean(||pred[t] - pred[t-1]||^2)
L = λ_pos * L_pos + λ_rot * L_rot + λ_smooth * L_smooth
```

- `L_rot` em radianos; reportar em graus na validação.
- `mask` zera passos sem pose válida.
- Não usar MSE em Euler em nenhum ponto.

Equivalente em quaternion (só para avaliação/inspeção):

```text
theta = 2 * arccos(clamp(abs(dot(q_pred, q_true)), 0, 1))
```

## 3. Arquitetura `SANDPose`

```text
EEG (B,C,T)
  └─ SpectralFeatureExtractor ─┐
                              ├─ FFTTransformerBlocks ─ tokens (B,T',d)
IMU (B,10,T_imu) [opcional]   │
  └─ Conv1d + Transformer ────┘   (concatenação ou cross-attention)

tokens (B,T',d)
  ├─ PoseProjector_pos -> (B,S,3)
  └─ PoseProjector_rot -> (B,S,6)
```

### 3.1. Cabeças

Duas cabeças separadas, cada uma com a mesma estrutura do
`TrajectoryProjector` atual (Conv1d + BatchNorm + LeakyReLU + AdaptiveAvgPool1d),
mudando apenas `n_outputs`:

```python
class PoseProjector(nn.Module):
    # igual ao TrajectoryProjector, com n_outputs=3 ou 6
```

Por quê separadas: posição e rotação têm escalas e geometrias diferentes; uma
cabeça única de 9 saídas tende a favorecer a posição.

### 3.2. IMU opcional

No piloto, treinar **sem IMU primeiro** (para comparar com o SAND). Depois:

```text
x_imu: (B, T_imu, 10)  # ax,ay,az,gx,gy,gz,qw,qx,qy,qz (ou 9 sem quat)
Conv1d(10 -> d_model) + positional encoding -> Transformer
```

A fusão pode ser concatenação de tokens ou cross-attention EEG↔IMU. Não é
necessário T/f: a rede aprende a relação com o alvo de pose.

### 3.3. Saída no tempo real

- `R_pred` 6D → Gram-Schmidt → quaternion → comando do robô;
- posição relativa ao repouso → somar `p_rest` no frame `base`;
- transformar `base` → base do robô com `T_robot_base` (arquivo de calibração
  externo, não aprendido).

## 4. Tensores de treino

Arquivo `windows.npz` (gerado pelo loader a partir de `dados_pose/`):

| Chave | Shape | Tipo | Observação |
| --- | --- | --- | --- |
| `x_eeg` | `(N, C, T)` | float32 | EEG filtrado/normalizado |
| `x_imu` | `(N, 10, T)` | float32 | opcional; `T` reamostrado |
| `y_pos` | `(N, S, 3)` | float32 | metros, normalizado |
| `y_rot6d` | `(N, S, 6)` | float32 | 6D da rotação |
| `y_quat` | `(N, S, 4)` | float32 | só avaliação |
| `y_mask` | `(N, S)` | bool | pose válida |
| `t_target_us` | `(N, S)` | int64 | tempo de cada passo |

Arquivo `windows_index.csv`:

```csv
window_id,session_id,subject_id,trial_id,t_start_us,t_end_us,target_start_us,target_end_us,split
0,S01_2026-11-10_01,S01,1,1730000000123456,1730000002123456,1730000001623456,1730000002623456,train
```

`windows_meta.json`:

```json
{
  "schema_version": 1,
  "eeg_fs": 500,
  "target_fs": 30,
  "window_sec": 2.0,
  "window_start_sec": 0.0,
  "window_stride_sec": 0.1,
  "target_seq_len": 30,
  "target_mode": "relative_rest",
  "band_hz": [1.0, 30.0],
  "car": true,
  "zscore": "train",
  "ch_names": ["..."],
  "normalizer": {"pos_mean": [0, 0, 0], "pos_std": [1, 1, 1]},
  "sessions_train": ["..."],
  "sessions_val": ["..."]
}
```

## 5. Config de treino — cabeçalho exato

`configs/sand_pose_piloto.yaml`:

```yaml
data:
  root: dados_pose
  window_sec: 2.0
  window_start_sec: 0.0
  window_stride_sec: 0.1
  eeg_fs: 500
  target_fs: 30
  target_seq_len: 30
  target_mode: relative_rest     # relative_rest | absolute
  band_hz: [1.0, 30.0]
  car: true
  zscore: train
  channels: auto                 # interseção por nome da montagem

model:
  arch: SANDPose
  n_channels: 32
  n_timesteps: 1000
  d_model: 32
  n_layers: 4
  dropout: 0.1
  pos_dim: 3
  rot_dim: 6
  use_imu: false

train:
  batch_size: 16
  epochs: 100
  lr: 0.001
  weight_decay: 0.0001
  patience: 12
  seed: 42
  device: auto

loss:
  pos: huber
  lambda_pos: 1.0
  lambda_rot: 1.0
  lambda_smooth: 0.05

metrics:
  - pos_rmse_mm
  - pos_p95_mm
  - geodesic_deg_median
  - geodesic_deg_p95
  - pcc_x
  - pcc_y
  - pcc_z

checkpoint:
  out: sand_pose_piloto.pt
```

## 6. Métricas

- **posição:** RMSE e p95 em mm no frame `base`;
- **rotação:** erro geodésico mediano e p95 em graus;
- **normal da palma:** erro angular em graus (se você também prevê normal);
- **por eixo:** PCC(x,y,z) para comparar com o SAND atual;
- **por sujeito/sessão:** nunca reportar só a média global.

Critério inicial do piloto (não é promessa): posição p95 < 30 mm e rotação
p95 < 15° em sessão de validação. Ajuste depois de ver os dados reais.

## 7. Plano de treino do piloto

1. **Baseline A:** `SANDTrajectory` original, só posição (controle).
2. **Baseline B:** `SANDPose` com `λ_rot=0` (mesma posição; confirma que a
   cabeça extra não degrada).
3. **Pose C:** `SANDPose` posição + rotação, sem IMU.
4. **Pose D:** `SANDPose` + IMU bruto.
5. Ablação de janela: 1 s, 2 s; `S` = 30, 60.
6. Split por sessão; depois LOSO.

Não comece pelo IMU: primeiro prove que a pose é decodificável com EEG e
labels limpos.

## 8. Ordem de implementação

| # | Arquivo | O que entra |
| --- | --- | --- |
| 1 | `pose_dataset.py` | ler `dados_pose/`, casar timestamps, montar `windows.npz` + `windows_index.csv` |
| 2 | `sand_pose_model.py` | `PoseProjector`, `SANDPose`, conversão 6D→R, loss geodésica |
| 3 | `sand_pose_treino.py` | loop de treino, normalizer de posição, métricas, checkpoint |
| 4 | `test_sand_pose.py` | shape `(B,30,3)`/`(B,30,6)`, 6D→R, loss zero para predição perfeita |
| 5 | `sand_pose_tempo_real.py` | só depois de validar offline; converte 6D→quaternion |

Enquanto os dados do piloto não existem, os itens 2–4 podem ser testados com
dados sintéticos.
