# Arquitetura da V1 — Protótipo de Fusão para Orientação da Palma

**Versão:** 1.0  
**Data:** 05/11/2026  
**Alvo:** Windows, Python 3.11, protótipo de pesquisa separado (paradigma EEG futuro, mas SEM alterar o EEG na V1).

---

## 1. Objetivo e Escopo da V1

### 1.1 O que a V1 entrega
Um protótipo funcional de **fusão de orientação da palma** combinando três fontes:
- **Kinect v2**: posição 3D via depth + RGB para o MediaPipe. O esqueleto/body tracking do SDK **não entra na V1** (não rastreia corpo na montagem atual; ver `docs/DIAGNOSTICO_KINECT_BODY.md`).
- **MediaPipe Hands**: 21 landmarks 2.5D (profundidade relativa não métrica) → normal da palma via PCA/plano médio do dorso (5 pontos) + direção dos dedos.
- **IMU MPU6050 (duas luvas)**: quaternion do filtro Mahony 6-eixos no firmware (roll/pitch observáveis, yaw **deriva** sem magnetômetro) + aceleração/giroscópio brutos; envia UDP `timestamp_us, roll, pitch, yaw, ax, ay, az, gx, gy, gz` nas portas 4210 (direita) e 4211 (esquerda).

**Saída da V1:** `q_fundida` (quaternion corpo→câmera) + `n_fundida` (normal da palma no frame da câmera) + `dir_fundida` (direção dos dedos) + métricas de confiança/latência, tudo em uma janela OpenCV leve.

### 1.2 O que fica FORA da V1 (decisão de projeto)
| Item | Motivo | Futuro |
|------|--------|--------|
| Estéreo (Kinect + webcam auxiliar) | Baseline instável; triangulação só para posição, não orientação | V2 |
| `JointOrientation` do Kinect | Body tracker **não rastreia** na montagem atual (quadril/pernas fora do FOV) — ver `docs/DIAGNOSTICO_KINECT_BODY.md` | Reavaliar se reposicionar Kinect |
| EKF | Começar com filtro complementar em quaternion; EKF como opção futura | V2+ |
| Euler para fusão | Singularidades em pitch ≈ ±90°; quaternion é obrigatório | Nunca |
| Matplotlib | Visualização só OpenCV (leve, tempo-real) | Nunca |
| Substituir `PALM_*`/`IMU_*` no EEG | Colunas `FUSED_*` serão **opcionais/versionadas**; legado não quebra sem validação | Integração EEG futura |
| Jig físico (A4 + papelão + nível) | Validação sintética + dados gravados primeiro; não bloqueia arquitetura | Pós-V1 |

---

## 2. Diagrama de Blocos Textual

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  AQUISIÇÃO  │     │  AQUISIÇÃO  │     │  AQUISIÇÃO  │
│   KINECT    │     │ MEDIAPIPE   │     │     IMU     │
│  (pos/depth)│     │(landmarks/  │     │ (2x MPU6050 │
│  + RGB)     │     │ normal/PCA) │     │  Mahony 6e) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌───────────────────────────────────────────────────────┐
│                    SINCRONIZAÇÃO                       │
│  - Relógio monotônico (PC)                            │
│  - Buffer circular por fonte (Kinect ~30 Hz, IMU ~100 Hz, MP por frame)       │
│  - Mapeamento timestamp ESP32 (µs) → monotônico (s)   │
└─────────────────────┬─────────────────────────────────┘
                      ▼
┌───────────────────────────────────────────────────────┐
│                    CALIBRAÇÃO                          │
│  - Extrínseca T/f via alinhamento da palma (V1)       │
│  - f_body (palma no corpo do sensor) por luva         │
│  - Yaw de boot (AncoradorYawBoot)                      │
│  - Resíduo/LOO/validação cruzada                       │
└─────────────────────┬─────────────────────────────────┘
                      ▼
┌───────────────────────────────────────────────────────┐
│                      FUSÃO (V1)                        │
│  - Filtro complementar em QUATERNION                  │
│  - Predição: giroscópio (se diponível) senão q_IMU    │
│  - Correção gravidade: roll/pitch do acelerômetro     │
│  - Correção yaw: normal da palma da visão (3D, não 2D)│
│  - Ganhos adaptativos + histerese + fallback          │
└─────────────────────┬─────────────────────────────────┘
                      ▼
┌───────────────────────────────────────────────────────┐
│                  VISUALIZAÇÃO (OpenCV)                │
│  - Janela única, camadas/cores                        │
│  - Landmarks, normal Kinect, previsão IMU, estado     │
│    fundido, fonte ativa, confiança, latência, FPS     │
└─────────────────────┬─────────────────────────────────┘
                      ▼
┌───────────────────────────────────────────────────────┐
│              CAMINHO FUTURO → EEG                     │
│  - Pacote `fusion/` isolado (não engorda              │
│    `eeg_motor_paradigm.py`)                           │
│  - Colunas `FUSED_*` opcionais/versionadas (schema v1)│
│  - Feature flag `--fusion-enabled`                    │
│  - Regressão: PALM_*/IMU_* inalteradas                │
└───────────────────────────────────────────────────────┘
```

---

## 3. Mapa: Módulo Proposto → Código Existente → O que Falta

| Módulo Proposto | Código Existente (arquivo/classe real) | O que Falta / Novo na V1 |
|-----------------|----------------------------------------|--------------------------|
| **M_AQUISICAO_KINECT** | `kinect_imu_groundtruth.py::KinectHandTracker` (frames color/depth, MediaPipe no RGB do Kinect) | Wrapper `KinectSource` com interface padronizada `get_observation() → KinectObservation`; expor `sync_color_depth_ms`; **sem body/JointOrientation na V1** |
| **M_AQUISICAO_MEDIAPIPE** | Mesmo `KinectHandTracker` (detect_hand, detect_auxiliary_hands) + `palm_normal_edge_on.normal_mao_landmarks` | `MediaPipeSource` independente (pode rodar sem Kinect); normal robusta edge-on já existe em `palm_normal_edge_on.py` |
| **M_AQUISICAO_IMU** | `imu.py::ImuReceiver` (UDP 4210/4211), `ImuSample`, `ImuBank` (dois lados, bias/zeragem independentes) | `ImuSource` com `get_sample() → ImuSample`; expor quaternion do firmware (já disponível no `esp32_teste_ahrs`) se firmware evoluir |
| **M_SINCRONIZACAO** | Parcial: `ImuSample.received_at` (monotônico), `KinectHandTracker.sync_color_depth_ms`, `PositionFusion.last_timestamp_us` | `SyncBuffer` genérico: alinha por timestamp monotônico; interpola IMU→frame Kinect; mede latência por fonte |
| **M_CALIBRACAO_EXTRINSECA** | `imu.py::solve_palm_alignment_pesos` (`n_cam = T R f`) + `AncoradorYawBoot`; `CameraAlignment`/checkerboard 7×5/45mm são do estéreo de câmera auxiliar (V2) | `CalibrationManager`: carrega/valida `T_cam_world` e `f_body` por lado (JSON/shape), aplica yaw de boot, expõe resíduo/LOO; **não usa checkerboard na V1** |
| **M_FUSAO** | `imu.py::PositionFusion` (zeragem posição + bias), `solve_palm_alignment` (Kabsch alternado T/f), `AncoradorYawBoot` | **NOVO**: `ComplementaryQuaternionFusion` — predição (gyro) + correção gravidade (acc) + correção yaw (visão 3D); ganhos, histerese, fallback; **não usa Euler** |
| **M_SENSOR_MANAGER** | `imu.py::ImuBank` (dois lados), `TrackingThread` (lado ativo por cue) | `FusionSensorManager`: orquestra fontes, publica `FusedState`, gerencia confiança/degradação, feature flag EEG |
| **M_VISUALIZACAO** | `tools/kinect_groundtruth_tool.py` (janelas OpenCV, grade, escalas, badges, teclas C/B/O/D/T/K/H/L/R/ESC) | `FusionVisualizer`: janela única, camadas (landmarks, normal Kinect, seta IMU, seta fundida), painel status (fonte, confiança, latência, FPS), taxa de desenho ≤30 Hz |
| **M_VALIDACAO** | `tools/kinect_groundtruth_tool.py::DiagnosticoPalma` (fases PREPARO/MESA/REZANDO/KINECT, CSV + resumo + veredito), `CalibracaoGeometrica` (13 poses, LOO, validação mira) | `ValidationSuite`: unitários sintéticos (quaternion, slerp, filtro, calibração), integração com dados gravados, bancada Kinect+IMU, jig A4 posterior |
| **M_EXPORT_EEG** | `eeg_motor_paradigm.py::MOTION_COLUMNS`, `PALM_COLUMNS`, `IMU_COLUMNS`, `build_motion_row`, `SharedState.motion` | **NOVO**: `fusion.eeg_export.FusedColumns` (schema v1: `FUSED_q_w,x,y,z`, `FUSED_n_x,y,z`, `FUSED_dir_x,y,z`, `FUSED_confidence`, `FUSED_latency_ms`, `FUSED_source`), feature flag, teste de regressão automatizado |

**Observações:**
- `solve_palm_alignment` / `set_palm_alignment` / `apply_palm_alignment` / `AncoradorYawBoot` já existem em `imu.py` (fonte única, item #9 da revisão).
- `palm_frame`, `palm_angles`, `palm_normal_from_plane`, `palm_normal_from_points` já existem em `kinect_imu_groundtruth.py`.
- Normal robusta edge-on com 21 landmarks + estabilizador de sinal já existem em `palm_normal_edge_on.py` e `palm_orientacao.py::EstabilizadorSinal`.
- Firmware `esp32_duas_imus` já envia **roll/pitch/yaw + accel + gyro** (10 campos) a ~100 Hz (ODR 100 Hz, DLPF 42 Hz, FS ±1000 dps). O firmware `esp32_teste_ahrs` já publica **quaternion** (w,x,y,z) por serial — pode ser migrado para UDP no futuro.

---

## 4. Frames e Estado (Explícito)

### 4.1 Convenção de Referenciais
| Frame | Descrição | Eixos (mão direita) |
|-------|-----------|---------------------|
| **Câmera (Kinect)** | Origem no sensor IR; x→direita, y→baixo (imagem), z→dentro da cena (lente em -z) | Padrão Kinect v2 |
| **Mundo (IMU)** | Frame do filtro Mahony: gravidade alinhada a -Z (ou +Z conforme convenção); yaw **arbitrário** a cada boot (sem magnetômetro) | Mahony 6-eixos |
| **Corpo do Sensor (luva)** | Fixado na luva; `f_body` = direção da palma **no corpo do sensor** (unitário, calibrado por luva) | Definido na calibração |
| **Palma (anatômico)** | Normal aponta **para fora da palma** (não do dorso); direção dos dedos = punho → MCP médio ortogonalizado à normal | Anatômico, por lado |

### 4.2 Relações Matemáticas
- **Alinhamento IMU→Câmera (por luva):** `T ∈ SO(3)` tal que `n_cam = T · R_IMU(t) · f_body`
  - `R_IMU(t)` = atitude do IMU (corpo→mundo) no instante `t` (quaternion do Mahony).
  - `f_body` = vetor unitário da palma no corpo do sensor (constante por luva).
  - `T` = rotação **mundo-do-IMU → câmera** (absorve o yaw do boot + montagem da luva).
- **Normal da palma prevista pelo IMU:** `n_IMU(t) = T · R_IMU(t) · f_body`
- **Normal medida pela visão (Kinect/MP):** `n_vis(t)` = normal do plano médio do dorso (PCA/SVD em 5 pontos) com sinal anatômico por lado (`palm_normal_from_plane`).
- **Direção dos dedos:** `d_cam = ortho(wrist → middle_mcp, n_cam)`.

### 4.3 Por que NÃO `q_final = q_roll_pitch(q_IMU) ⊗ q_yaw(yaw_visao)`
Essa fórmula **pronta** assume:
1. Roll/pitch do IMU são "perfeitos" (não são: ruído de aceleração linear, bias).
2. Yaw da visão vem de um ângulo 2D (ex.: `atan2(nx, nz)` da normal) — **perde a informação 3D** e sofre com edge-on (normal ≈ horizontal → yaw mal condicionado).
3. A composição sequencial em Euler introduz **singularidades** e **ordem de rotação** arbitrária.

### 4.4 Formulação Correta em Quaternion (decisão de projeto)
O estado da fusão é um **quaternion** `q ∈ S³` representando a orientação **corpo-do-sensor → câmera** (ou mundo→câmera se preferir).  
A cada passo:
1. **Predição** (se giroscópio disponível no firmware):  
   `q_pred = q ⊗ exp(½ · ω_body · dt)`  onde `ω_body = [gx, gy, gz]` em rad/s (gyro corrigido de bias).
   - *Se firmware só manda RPY*: usar `q_IMU = quat_from_rpy(roll, pitch, yaw)` como predição (menos ideal, mas funcional).
2. **Correção gravidade (roll/pitch):**  
   Medida `a_world = R(q_pred)ᵀ · g` (vetor gravidade no corpo).  
   Erro `e_grav = a_world × g_ref` (g_ref = [0,0,-1] ou [0,0,1] conforme convenção).  
   `q_corr_grav = q_pred ⊗ exp(½ · Kp_grav · e_grav · dt)`.
3. **Correção yaw (visão 3D):**  
   Medida `n_vis` (normal da palma no frame da câmera, unitária, sinal anatômico).  
   Previsão `n_pred = quat_rotate(q_corr_grav, f_body)` (rotação de vetor: `q ⊗ f`, **não** `q ⊗ f ⊗ q⁻¹`).  
   Erro `e_yaw = n_pred × n_vis` (produto vetorial → eixo de rotação mínimo).  
   `q_fused = exp(½ · Kp_yaw · e_yaw · dt) ⊗ q_corr_grav` (correção pela **esquerda**, pois o erro está no frame da câmera).
4. **Normalização:** `q_fused ← q_fused / ‖q_fused‖`.

**Ganhos sugeridos (ajustáveis):**
- `Kp_grav = 2.0` (fecha tilt em ~1 s, igual ao Mahony no firmware).
- `Kp_yaw = 0.5..1.0` (visão mais lenta/ruidosa; histerese abaixo).
- **Limites:** `‖e_grav‖ < 0.5` (ignora aceleração linear forte), `‖e_yaw‖ < 0.3` (ignora outliers de visão).

---

## 5. Contratos de Interface (Tabela + Dataclasses Conceituais)

### 5.1 KinectObservation
| Campo | Tipo | Unidade | Validade | Taxa Esperada | Latência Típica |
|-------|------|---------|----------|---------------|-----------------|
| `timestamp_mono` | float | s (monotônico) | sempre | 30 Hz | — |
| `wrist_3d` | float[3] | m (camera space) | `KT_src ≠ 0` | 30 Hz | 10–20 ms |
| `palm_normal` | float[3] | unitário | `PALM_valid=1` | 30 Hz | 15–25 ms |
| `palm_dir` | float[3] | unitário | `PALM_valid=1` | 30 Hz | 15–25 ms |
| `palm_angles` | dict(elev,azim,pitch,yaw) | graus | `PALM_valid=1` | 30 Hz | — |
| `landmarks_3d` | dict[str,float[3]] | m | por landmark | 30 Hz | — |
| `source` | str | enum | sempre | — | — |
| `reproj_error_px` | float | px | se triangulado | — | — |
| `sync_color_depth_ms` | float | ms | se disponível | — | — |

```python
@dataclass
class KinectObservation:
    timestamp_mono: float
    wrist_3d: Optional[np.ndarray]        # (3,)
    palm_normal: Optional[np.ndarray]     # (3,) unitário, sinal anatômico
    palm_dir: Optional[np.ndarray]        # (3,) unitário
    palm_angles: Optional[Dict[str,float]]# elev, azim, pitch, yaw
    landmarks_3d: Dict[str, np.ndarray]   # até 21 chaves
    source: str                           # "kinect_depth", "mediapipe", ... (sem body/skeleton na V1)
    reproj_error_px: Optional[float]
    sync_color_depth_ms: Optional[float]
```

### 5.2 MediaPipeObservation
| Campo | Tipo | Unidade | Validade | Taxa Esperada |
|-------|------|---------|----------|---------------|
| `timestamp_mono` | float | s | sempre | por frame (≈30 Hz) |
| `landmarks_2d5` | dict[str, float[3]] | px, px, rel. | se detectado | — |
| `handedness` | str | "left"/"right" | se detectado | — |
| `palm_normal_pca` | float[3] | unitário | se `qualidade ≥ 0.05` | — |
| `palm_normal_quality` | float | [0,1] | — | — |
| `palm_normal_source` | str | "todos_pontos"/"dorso_5_pontos"/"falha" | — | — |

```python
@dataclass
class MediaPipeObservation:
    timestamp_mono: float
    landmarks_2d5: Dict[str, np.ndarray]   # x,y,z_rel (z_rel não métrico)
    handedness: Optional[str]
    palm_normal_pca: Optional[np.ndarray]
    palm_normal_quality: float
    palm_normal_source: str
```

### 5.3 ImuSample (já existe em `imu.py`)
| Campo | Tipo | Unidade | Validade | Taxa Esperada |
|-------|------|---------|----------|---------------|
| `timestamp_us` | int | µs (relógio ESP32) | sempre | ~100 Hz |
| `received_at` | float | s (monotônico PC) | sempre | — |
| `roll_deg, pitch_deg, yaw_deg` | float | graus | sempre | ~100 Hz |
| `accel_g` | float[3] | g | sempre | ~100 Hz |
| `gyro_dps` | float[3] | °/s | sempre | ~100 Hz |
| `quaternion` | float[4] | w,x,y,z (corpo→mundo) | **se firmware evoluir** | — |

```python
@dataclass
class ImuSample:
    timestamp_us: int
    received_at: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    accel_g: np.ndarray      # (3,)
    gyro_dps: np.ndarray     # (3,)
    # Futuro (quando firmware mandar quaternion):
    quaternion: Optional[np.ndarray] = None  # (4,) w,x,y,z corpo->mundo
```

### 5.4 Calibration (por lado)
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `T_cam_world` | float[3,3] | Extrínseca do modelo `n_cam = T · R_imu · f_body` (estimada por `solve_palm_alignment_pesos`; inclui yaw de boot) |
| `f_body` | float[3] | Direção da palma no corpo do sensor (unitário) |
| `T_imu_cam` | float[3,3] | `imu_to_camera` = alinhamento resolvido (`T` do Kabsch) |
| `yaw_boot_delta_deg` | float | Delta de yaw aplicado no boot atual (AncoradorYawBoot) |
| `residual_deg` | float | Resíduo médio ponderado do alinhamento |
| `source` | str | "bootstrap" / "calibrado" / "geometrico" |
| `validacao_mira_deg` | float | Mediana erro nas poses de mira (fora do ajuste) |
| `loo_mediana_deg` | float | Leave-one-out mediana |

```python
@dataclass
class Calibration:
    lado: str                    # "right" / "left"
    T_cam_world: np.ndarray      # (3,3)
    f_body: np.ndarray           # (3,)
    T_imu_cam: np.ndarray        # (3,3)
    yaw_boot_delta_deg: float
    residual_deg: float
    source: str
    validacao_mira_deg: Optional[float]
    loo_mediana_deg: Optional[float]
    timestamp_calib: str         # ISO
```

### 5.5 FusedState (saída da fusão V1)
| Campo | Tipo | Unidade | Descrição |
|-------|------|---------|-----------|
| `timestamp_mono` | float | s | Instante da fusão (monotônico) |
| `q_fused` | float[4] | w,x,y,z | Quaternion corpo→câmera (fundido) |
| `n_fused` | float[3] | unitário | Normal da palma no frame da câmera |
| `d_fused` | float[3] | unitário | Direção dos dedos no frame da câmera |
| `angles_fused` | dict | graus | elev, azim, pitch, yaw (de `n_fused`, `d_fused`) |
| `source` | str | enum | "imu_only", "vision_only", "fused", "fallback" |
| `confidence` | float | [0,1] | Confiança combinada (ver §7) |
| `latency_ms` | float | ms | Latência estimada da fusão |
| `imu_age_ms` | float | ms | Idade da amostra IMU usada |
| `vision_age_ms` | float | ms | Idade da observação visão usada |
| `fallback_reason` | str | — | Se source ≠ "fused" |

```python
@dataclass
class FusedState:
    timestamp_mono: float
    q_fused: np.ndarray            # (4,) w,x,y,z
    n_fused: np.ndarray            # (3,) unitário
    d_fused: np.ndarray            # (3,) unitário
    angles_fused: Dict[str,float]  # elev, azim, pitch, yaw
    source: str                    # "imu_only" | "vision_only" | "fused" | "fallback"
    confidence: float              # [0,1]
    latency_ms: float
    imu_age_ms: float
    vision_age_ms: float
    fallback_reason: str = ""
```

### 5.6 SourceStatus / Confidence (por fonte)
| Fonte | Métricas de Confiança | Degradação |
|-------|----------------------|------------|
| **Visão (Kinect/MP)** | `visibility` (landmarks detectados/21), `reproj_error_px`, `palm_normal_quality`, `edge_on_score = ‖n_xy‖`, `sync_color_depth_ms` | `edge_on` → peso yaw ↓; `reproj > 40px` → descarta triangulação; `visibility < 0.6` → vision_only ↓ |
| **IMU** | `sample_age_ms`, `accel_norm ≈ 1g`, `gyro_norm < saturação`, `bias_estimated`, `zero_lock` | `age > 200ms` → imu_only ↓; `‖accel‖ ∉ [0.5, 1.5]g` → ignora correção gravidade; `zero_lock=1` → bias confiável |

---

## 6. Algoritmo de Fusão V1 (Escolhido e Justificado)

### 6.1 Filtro Complementar em Quaternion (Mahony-style)
**Por que:**  
- Mahony 2008 (PDF p. 2, 4, 13) confirma: filtro complementar não-linear em SO(3) é **globalmente assintoticamente estável** para roll/pitch; yaw deriva sem magnetômetro → visão **deve** prover referência de yaw.  
- Tian (PDF p. 1, 3, 7) usa UKF com estado `[p, v, q]` — arquitetura de referência, mas EKF/UKF adicionam complexidade desnecessária na V1.  
- Acharya (PDF p. 1, 3, 6): IMU bom para tempo/relativo; visão boa para absoluto; yaw do acelerômetro **não observável**.  
- BIBE (PDF p. 1, 4): Kalman sequencial com múltiplos updates; RMSE fundido 6.18° < IMU 6.30° < visão 7.70° — fusão **melhora** ambas.

**Escolha:** Filtro complementar em quaternion (duas correções: gravidade + yaw-visão), **sem Euler**, **sem EKF** na V1.

### 6.2 Passos do Algoritmo (por lado, por frame)

```python
def fusion_step(q_prev: np.ndarray,           # (4,) w,x,y,z
                imu: Optional[ImuSample],
                vis: Optional[KinectObservation],
                cal: Calibration,
                dt: float) -> FusedState:
    # 1. PREDICAO
    if imu is not None and imu.gyro_dps is not None and np.linalg.norm(imu.gyro_dps) > 0:
        # Giroscópio disponível (firmware manda gyro_dps)
        omega = np.deg2rad(imu.gyro_dps)       # rad/s no corpo
        q_pred = quat_mult(q_prev, quat_exp(0.5 * omega * dt))
    elif imu is not None:
        # Fallback: usa q_IMU do RPY (menos ideal)
        q_imu = quat_from_rpy(imu.roll_deg, imu.pitch_deg, imu.yaw_deg)
        q_pred = slerp(q_prev, q_imu, alpha=0.9)  # puxa forte pro IMU
    else:
        q_pred = q_prev                         # sem IMU: mantém

    # 2. CORRECAO GRAVIDADE (roll/pitch)
    q_grav = q_pred
    if imu is not None and imu.accel_g is not None:
        a_body = imu.accel_g
        a_norm = np.linalg.norm(a_body)
        if 0.5 < a_norm < 1.5:                 # ~1g: sem aceleração linear forte
            a_body = a_body / a_norm
            g_ref = np.array([0.0, 0.0, -1.0])  # gravidade no mundo (Mahony)
            # Erro: a_body × g_ref no frame do corpo
            e_grav = np.cross(a_body, quat_rotate_inv(q_pred, g_ref))
            if np.linalg.norm(e_grav) < 0.5:
                q_grav = quat_mult(q_pred, quat_exp(0.5 * Kp_grav * e_grav * dt))

    # 3. CORRECAO YAW (visão 3D)
    q_fused = q_grav
    if vis is not None and vis.palm_normal is not None:
        n_vis = vis.palm_normal                 # unitário, sinal anatômico
        n_pred = quat_rotate(q_fused, cal.f_body)
        e_yaw = np.cross(n_pred, n_vis)         # eixo mínimo
        # Peso adaptativo: reduz se edge-on (normal ~horizontal) ou reproj alto
        w_yaw = Kp_yaw * vis_confidence(vis)
        if np.linalg.norm(e_yaw) < 0.3 and w_yaw > 0:
            # e_yaw esta' no frame da CAMERA -> correcao pela ESQUERDA
            q_fused = quat_mult(quat_exp(0.5 * w_yaw * e_yaw * dt), q_fused)

    # 4. NORMALIZA
    q_fused = q_fused / np.linalg.norm(q_fused)

    # 5. SAIDA
    n_fused = quat_rotate(q_fused, cal.f_body)
    d_fused = vis.palm_dir if vis is not None and vis.palm_dir is not None \
              else ortho(quat_rotate(q_fused, cal.f_fingers_ref), n_fused)
    return FusedState(...)
```

### 6.3 Calibração `T_cam_world` e `f_body` por Lado
- **Extrínseca da V1:** `T_cam_world` e `f_body` vêm de `solve_palm_alignment_pesos` (modelo `n_cam = T R f`) usando a normal do Kinect e as rotações do IMU; NÃO do checkerboard. O checkerboard 7×5/45 mm e o `CameraAlignment` pertencem ao estéreo de câmera auxiliar (V2), fora da V1.  
- **`f_body` (palma no corpo):** Resolvido por `solve_palm_alignment_pesos` (Kabsch alternado com pesos) usando:
  - 9 poses de **posição** (palma/dorso/mira para lente) — verdade = `-normalize(posição_mão)`, peso 1.5.
  - 4 poses **geométricas inclinadas** (50°/60°) — verdade exata no frame da câmera, peso 1.0.
  - 1 pose de **mira** (peso 1.5) ancorada pela posição.
- **Yaw de boot:** `AncoradorYawBoot` resolve `Δyaw` nas primeiras amostras com palma "fora da vertical" (componente horizontal de `R·f_body` > 0.35). Aplica `T_novo = T_antigo · Rz(-Δyaw)` **em memória**; JSON salvo continua valendo para qualquer boot.
- **Gates de validação:** resíduo ≤15°, LOO mediana ≤15°, validação mira ≤20°, espalhamento mín/máx ≥15°/30° (ver `CalibracaoGeometrica.resultado`).

### 6.4 Fallback quando Visão Some ou Fica Edge-on
| Condição | Ação |
|----------|------|
| `vis is None` ou `vis.palm_normal is None` | `source = "imu_only"`; `q_fused = q_grav`; `confidence = imu_confidence(imu)` |
| `‖n_vis_xy‖ < 0.2` (edge-on: normal ≈ vertical) | `w_yaw ← 0`; **não corrige yaw**; `source = "imu_only_yaw"`; avisa histerese |
| `reproj_error_px > 40` ou `visibility < 0.4` | Desconsidera visão; `source = "imu_only"` |
| `imu_age_ms > 200` | `source = "vision_only"`; `q_fused = q_from_vis(n_vis, cal.f_body)` (align mínimo) |
| Ambas degradadas | `source = "fallback"`; mantém `q_prev`; `confidence → 0`; `fallback_reason` preenchido |

**Histerese:** `source` só muda após **3 frames consecutivos** na nova condição (evita flapping).

### 6.5 Se a Luva Só Mandar Roll/Pitch/Yaw (Sem Giroscópio/Quaternion)
- **Predição:** `q_pred = slerp(q_prev, q_IMU, α=0.9)` (puxa forte pro IMU a cada frame).  
- **Correção gravidade:** Igual (usa `accel_g` do IMU).  
- **Correção yaw:** Igual (usa `n_vis`).  
- **Limitação:** Sem giroscópio, a predição não integra velocidade angular → **latência efetiva maior** e ruído de discretização do RPY. Funciona, mas **não é ideal**.  
- **Mitigação:** Firmware `esp32_teste_ahrs` já publica quaternion (w,x,y,z) por serial; migração para UDP é **recomendada** antes da V1 final.

---

## 7. Sensor Manager / Confiança (V1 Sem Estéreo)

### 7.1 Métricas de Confiança por Fonte
```python
def vision_confidence(vis: KinectObservation) -> float:
    if vis is None or vis.palm_normal is None: return 0.0
    w = 1.0
    w *= np.clip(vis.palm_normal_quality / 0.5, 0, 1)      # qualidade PCA/plano
    w *= np.clip(1.0 - vis.reproj_error_px / 40.0, 0, 1)   # reproj ≤40px
    w *= np.clip(vis.visibility / 0.6, 0, 1)               # landmarks detectados
    w *= np.clip(1.0 - abs(vis.sync_color_depth_ms or 0) / 30.0, 0, 1)  # sync ≤30ms
    # Penaliza edge-on: normal ~vertical -> yaw mal observável
    n_xy = np.linalg.norm(vis.palm_normal[:2])
    w *= np.clip(n_xy / 0.3, 0, 1)                         # horiz > 0.3
    return float(np.clip(w, 0.0, 1.0))

def imu_confidence(imu: ImuSample) -> float:
    if imu is None: return 0.0
    w = 1.0
    age = (time.monotonic() - imu.received_at) * 1000
    w *= np.clip(1.0 - age / 200.0, 0, 1)                  # idade ≤200ms
    a_norm = np.linalg.norm(imu.accel_g)
    w *= 1.0 if 0.8 < a_norm < 1.2 else 0.3                # ~1g
    g_norm = np.linalg.norm(imu.gyro_dps)
    w *= 1.0 if g_norm < 800 else 0.5                      # < saturação FS
    w *= 1.0 if imu.bias_estimated else 0.7
    w *= 1.2 if imu.zero_lock else 1.0
    return float(np.clip(w, 0.0, 1.0))
```

### 7.2 Estados de Degradação e Saída
| Estado | Condição | `source` | `confidence` | Ação na Saída |
|--------|----------|----------|--------------|---------------|
| **NOMINAL** | `vis_conf > 0.6` e `imu_conf > 0.6` | `fused` | `0.7–1.0` | Fusão completa |
| **VISÃO_FRACA** | `vis_conf ∈ [0.3, 0.6]` | `fused` | `0.4–0.7` | Peso yaw ↓ linear |
| **IMU_FRACO** | `imu_conf ∈ [0.3, 0.6]` | `fused` | `0.4–0.7` | Peso gravidade ↓ |
| **SEM_VISAO** | `vis_conf < 0.3` ou `vis is None` | `imu_only` | `imu_conf` | Sem correção yaw |
| **SEM_IMU** | `imu_conf < 0.3` ou `imu is None` | `vision_only` | `vis_conf` | Alinhamento mínimo `q_from_vis` |
| **FALLBACK** | Ambas < 0.3 | `fallback` | `< 0.3` | Mantém `q_prev`, avisa |

**Histerese de estado:** Transição requer **3 frames** consecutivos no novo estado.

---

## 8. Visualização Leve (OpenCV)

**Estilo:** `tools/kinect_groundtruth_tool.py` (janela única, grade de janelas, `WINDOW_SCALE=0.5`, badges, teclas).

### 8.1 Janela Única — Camadas e Cores
| Camada | Cor | Desenho |
|--------|-----|---------|
| Landmarks MediaPipe (21) | Verde (0,255,0) | Pontos + conexões padrão |
| Normal Kinect (plano médio) | Azul (255,0,0) | Seta 0.26 m a partir do punho |
| Previsão IMU (seta palma) | Amarelo (0,255,255) | Seta `T·R_IMU·f_body` |
| Estado Fundido | Vermelho (0,0,255) | Seta `n_fused` + `d_fused` (ciano) |
| Plano-alvo (calibração X) | Magenta (255,0,255) | Quadrado 0.30 m + normal alvo |
| (sem esqueleto/body na V1) | — | JointOrientation fora: body tracking não rastreia na montagem atual |

### 8.2 Painel de Status (Overlay Texto, Canto Superior Esquerdo)
```
FPS: 29.8 | LAT: 18 ms | SRC: fused (0.87)
IMU:  age=12ms  acc=1.00g  zero=1  bias=1
VIS:  age=15ms  q=0.82  rep=3.2px  edge=0.41
CAL:  resid=4.2°  yaw_boot=+3.1°  src=calibrado
PALM: elev=+12° azim=-5° pitch=+88° yaw=+3°
```
- **Taxa de desenho:** ≤30 Hz (mesmo loop da aquisição).  
- **Sem matplotlib.** Apenas `cv2.putText`, `cv2.arrowedLine`, `cv2.polylines`, `cv2.circle`.

### 8.3 Teclas Principais (herdadas do tool)
| Tecla | Ação |
|-------|------|
| `C` | Origem (zero posição) |
| `B` | Bias do acelerômetro (IMU parado) |
| `O` | Alinhamento palma IMU↔Câmera (14 s, por luva) |
| `J` | Plano do dorso (24 s, por luva) — verdade mais estável |
| `X` | Calibração geométrica (13 poses, por luva) |
| `D` | Diagnóstico palma (movimento programado 4 fases) |
| `T` | Diagnóstico tabuleiro (erro px) |
| `K` | Trim fixo overlay |
| `H` | Calibração esqueleto mão (homografia) |
| `L` | Log de desvio (CSV) |
| `R` | Reset fusão/zeragem |
| `ESC` | Sair |

---

## 9. Integração Futura com EEG

### 9.1 Onde Plugar
- **Pacote isolado:** `fusion/` (não engorda `eeg_motor_paradigm.py`).  
- **Entry point:** `fusion.FusionSensorManager` → publica `FusedState` a cada frame.  
- **EEG chama:** `fusion_manager.get_fused_state()` no loop de motion (30 Hz).  
- **Feature flag:** `--fusion-enabled` (default `False` na V1).  
- **CSV de motion:** Colunas `FUSED_*` **adicionadas** (não substituem `PALM_*`/`IMU_*`).

### 9.2 Colunas `FUSED_*` (Schema v1)
| Coluna | Tipo | Unidade | Descrição |
|--------|------|---------|-----------|
| `FUSED_q_w` | float | — | Quaternion w (corpo→câmera) |
| `FUSED_q_x` | float | — | Quaternion x |
| `FUSED_q_y` | float | — | Quaternion y |
| `FUSED_q_z` | float | — | Quaternion z |
| `FUSED_n_x` | float | — | Normal fundida x (câmera) |
| `FUSED_n_y` | float | — | Normal fundida y (câmera) |
| `FUSED_n_z` | float | — | Normal fundida z (câmera) |
| `FUSED_dir_x` | float | — | Direção dedos fundida x (câmera) |
| `FUSED_dir_y` | float | — | Direção dedos fundida y (câmera) |
| `FUSED_dir_z` | float | — | Direção dedos fundida z (câmera) |
| `FUSED_elev_deg` | float | graus | Elevação da normal fundida |
| `FUSED_azim_deg` | float | graus | Azimute da normal fundida |
| `FUSED_pitch_deg` | float | graus | Pitch da direção fundida |
| `FUSED_yaw_deg` | float | graus | Yaw da direção fundida |
| `FUSED_confidence` | float | [0,1] | Confiança combinada |
| `FUSED_latency_ms` | float | ms | Latência da fusão |
| `FUSED_source` | int | enum | 0=fallback, 1=imu_only, 2=vision_only, 3=fused |
| `FUSED_schema_ver` | int | — | Versão do schema (1) |

**No CSV:** `FUSED_schema_ver=1` gravado no cabeçalho do JSON de sessão.  
**Regressão:** Testes automatizados comparam `PALM_*` vs `FUSED_*` em dados gravados (correlação, erro angular) — `PALM_*`/`IMU_*` **inalteradas**.

---

## 10. Riscos e Mitigações

| Risco | Prob. | Impacto | Mitigação V1 |
|-------|-------|---------|--------------|
| **Yaw drift do IMU** (sem mag) | Alta | Yaw errado → normal fundida gira errado | Visão corrige yaw a cada frame; `AncoradorYawBoot` no boot; histerese |
| **Edge-on (palma ≈ vertical)** | Média | Normal ~horizontal → yaw mal condicionado | `w_yaw → 0` automaticamente; fallback `imu_only_yaw`; `EstabilizadorSinal` evita flip de sinal |
| **Latência/Desync Kinect↔IMU** | Média | Fusão com dados velhos → erro transitório | Buffer circular + interpolação IMU→frame Kinect; mede `imu_age_ms`, `vision_age_ms`; descarta se >200ms |
| **Calibração velha (T/f errados)** | Média | Erro sistemático em toda sessão | Gates de validação (resíduo, LOO, mira, espalhamento); `CalibracaoGeometrica` força recalibração |
| **Handedness trocado (luva errada)** | Baixa | `T`/`f` da luva oposta → erro grande | `--imu-portas "4210,4211"` (ordem fixa direita/esquerda); diagnóstico `D` flagra luva trocada |
| **Oclusão / mão fora do FOV** | Alta | Visão some → fusão degrada para IMU | `vision_confidence → 0`; fallback `imu_only` com bias já corrigido (zero_lock) |
| **Ruído depth Kinect (dedos/borda)** | Média | Normal PCA instável | Plano médio 5 pontos (PCA/SVD) + gates RMS/distância + `palm_normal_edge_on` robusto |
| **Firmware sem giroscópio** | Baixa | Predição só por RPY → latência/ruído | Funciona (slerp forte), mas **recomenda migrar para quaternion UDP** antes da V1 final |

---

## 11. Plano de Testes da V1

### 11.1 Unitários Sintéticos (`tests/fusion/test_*.py`)
| Teste | Cobertura |
|-------|-----------|
| `test_quaternion_ops.py` | `quat_mult`, `quat_exp`, `quat_rotate`, `slerp`, normalização |
| `test_complementary_filter.py` | Predição (gyro/RPY), correção gravidade, correção yaw, ganhos, limites |
| `test_calibration_kabsch.py` | `solve_palm_alignment_pesos` com ruído sintético, pesos, convergência |
| `test_yaw_boot_anchor.py` | `reancorar_yaw` com `Δyaw` conhecido, sensibilidade horizontal |
| `test_confidence.py` | `vision_confidence`, `imu_confidence`, transições de estado com histerese |

### 11.2 Integração com Dados Gravados/Simulados
- **Dataset sintético:** Trajetória conhecida `q(t)`, `n(t)`, `d(t)` + ruído IMU (bias, random walk) + ruído visão (outliers, edge-on, dropout). Métricas: erro angular RMS, convergência yaw, taxa de fallback.
- **Dados reais gravados:** `kinect_imu_groundtruth.py` + `imu.py` já gravam CSVs com `palm_err_deg`, `KT_*`, `IMU_*`, `PALM_*`. Reprocessar offline com `fusion.replay()`.

### 11.3 Bancada Kinect + IMU (Sem Jig)
1. Montar luvas, calibrar origem (10 s), alinhamento palma (`O` 14 s/lado).  
2. Rodar diagnóstico `D` (21 s): 4 fases → CSV + veredito.  
3. Gravar 5 min de movimento livre: medir `palm_err_deg` (mediana < 10° alvo V1).  
4. Testar edge-on intencional (palma para cima/baixo) → confirmar `w_yaw→0` e histerese.  
5. Testar oclusão (mão sai do FOV 2s) → confirmar fallback `imu_only` e recuperação suave.

### 11.4 Jig A4 + Papelão + Nível (Pós-V1, Não Bloqueia)
- Ângulos conhecidos (0°, 15°, 30°, 45°, 60°, 75°, 90°) no plano horizontal/vertical.  
- Medir erro absoluto vs. jig (não copiar 2.33 mm / 0.057° do IJMPE — aqueles são **RealSense + Witmotion**, não Kinect v2 + MPU6050).

### 11.5 Métricas Realistas (V1)
| Métrica | Alvo V1 (funcional) |
|---------|---------------------|
| Erro angular mediana (fundido vs. Kinect plano médio) | < 10° |
| Taxa de fallback (`source ≠ fused`) | < 15% dos frames |
| Latência fusão (mediana) | < 30 ms |
| Yaw boot anchor (delta aplicado) | Convergência < 10 s, erro < 5° |
| Diagnóstico `D` (mediana 3 poses) | < 15° (limite `PALM_DIAG_MAX_ERR_DEG`) |

---

## 12. Critérios de Aceite da V1 (Funcionais)

1. **Fusão roda em tempo real** (≤30 Hz loop principal, janela OpenCV responsiva) sem travar.  
2. **Três fontes ativas** (Kinect depth, MediaPipe landmarks, IMU UDP 4210/4211) → `FusedState` publicado a cada frame.  
3. **Calibração por luva** (`O` + `J` + `X`) resolve `T`, `f_body` com resíduo ≤15° e gates de validação passando.  
4. **Yaw de boot ancorado** automaticamente (< 10 s, palma fora da vertical) → seta vermelha aponta para lente na pose "palma para câmera".  
5. **Diagnóstico `D` passa** nas 3 poses programadas (mediana ≤15°).  
6. **Fallback gracioso**: visão some → `imu_only` (bias corrigido mantém posição por segundos); visão volta → `fused` sem salto.  
7. **Visualização mostra** landmarks, normal Kinect, previsão IMU, estado fundido, painel status (fonte, confiança, latência, FPS).  
8. **EEG intocado**: `eeg_motor_paradigm.py` roda igual sem `--fusion-enabled`; com flag, colunas `FUSED_*` v1 adicionadas, `PALM_*`/`IMU_*` inalteradas.  
9. **Testes unitários passam** (quaternion, filtro, calibração, confiança).  
10. **Reprocessamento offline** de dados gravados produz `FUSED_*` consistentes com fusão online.

---

## 13. Perguntas Abertas para o Usuário

| # | Pergunta | Impacto |
|---|----------|---------|
| 1 | **Migrar firmware `esp32_duas_imus` para enviar quaternion (w,x,y,z) via UDP?** Hoje envia RPY+accel+gyro. O firmware `esp32_teste_ahrs` já tem quaternion por serial. | Predição mais fiel, remove slerp forte, reduz latência efetiva. Esforço: ~2 h firmware + teste. |
| 2 | **Usar `palm_normal_edge_on.normal_mao_landmarks` (21 pontos PCA) como fonte principal de visão** em vez do plano médio 5 pontos do Kinect? | Mais robusto a edge-on/oclusão; mas depende de triangulação MediaPipe→3D (precisa stereo ou depth Kinect por landmark). |
| 3 | **Definir convenção final de eixos do `FUSED_*`: câmera (Kinect) ou mundo (gravidade alinhada)?** Hoje `PALM_*` usa câmera (y↑=baixo imagem). EEG futuro pode preferir mundo. | Afeta nomes/interpretacao das colunas; decidir antes de gravar CSV v1. |
| 4 | **Incluir `FUSED_pos_x,y,z` (posição fundida IMU+câmera) no schema v1?** `PositionFusion` já faz zeragem de posição. | Útil para EEG (trajetória fundida), mas acopla posição à orientação. |
| 5 | **Validar com jig A4 na V1 ou deixar para V2?** O jig dá erro absoluto; sem ele só temos erro relativo (Kinect vs. IMU). | Jig não bloqueia arquitetura; decide cronograma. |
| 6 | **Publicar `FUSED_*` no CSV de motion (30 Hz) ou em arquivo separado?** Hoje motion é arquivo separado a 30 Hz. | Arquivo separado mantém EEG leve; CSV único simplifica análise. |

---

## 13.1 Revisão do Lead (Fase 2)

Esta versão foi revisada pelo Lead depois da geração automática:

- **Kinect sem esqueleto/body na V1**: `JointOrientation` e esqueleto do SDK foram
  removidos do escopo; a montagem atual (Kinect alto, quadril fora do FOV) não
  rastreia corpo (ver `docs/DIAGNOSTICO_KINECT_BODY.md`).
- **Calibração da V1**: `T_cam_world`/`f_body` vêm de
  `solve_palm_alignment_pesos` + `AncoradorYawBoot`; o checkerboard 7×5/45 mm e
  `CameraAlignment` são do estéreo de câmera auxiliar e ficam para a V2.
- **Quaternion**: corrigida a rotação de vetor (`n = q ⊗ f`, não `q ⊗ f ⊗ q⁻¹`)
  e a correção de yaw pela esquerda (erro no frame da câmera).
- **Ganhos/limiares** (`Kp_grav`, `Kp_yaw`, limites de erro) são decisões de
  projeto a ajustar nos testes, não valores de literatura.
- **Firmware**: a luva envia roll/pitch/yaw + accel + gyro (UDP); o `esp32_teste_ahrs`
  envia quaternion por serial. O contrato de `ImuSample` já cobre os campos.
- A arquitetura continua **V1 sem estéreo**; `FUSED_*` só depois de validada.

## 14. Referências (Apenas Afirmações Seguras da Auditoria)

| Ref | Afirmação Segura | PDF + Página |
|-----|------------------|--------------|
| Mahony 2008 | Filtro complementar não-linear em SO(3); variante *passive* desacopla giro da atitude; sem magnetômetro yaw deriva | Mahony pp. 2, 4, 13 |
| Markley 2007 | Média de quaternions = autovetor principal de M; equivalência MLE e pesos | Markley pp. 2–4 |
| Tian 2015 | UKF estado `[p, v, q]`; transformação Kinect→global Sec. 5.1 | Tian pp. 1, 3, 7 |
| Acharya 2022 | MPU6050 + Kinect para membro superior; roll/pitch observáveis, yaw não observável por acelerômetro | Acharya pp. 1, 3, 6 |
| MediaPipe Hands | 21 landmarks 2.5D; profundidade relativa aprendida só de sintético | MediaPipe pp. 1–2 |
| IJMPE 2024 | Referência O = intersecção 1-17 e 0-9; yaw de ângulo 2D; eixos IMU X=dedos, Z=perp. palma; números são RealSense+Witmotion | IJMPE pp. 2–3 |
| Kusunose 2025 | Features normalizadas + seleção 22 landmarks; SHAP identifica triângulo palmar | Kusunose pp. 1, 5, 8 |
| Xefteris 2025 | Fusão tardia LSTM+RF; latência total 81,41 ms (~12,28 fps), não tempo-real estrito | Xefteris p. 12 |
| BIBE 2022 | Kalman sequencial multi-update; RMSE IMU 6,30°, visão 7,70°, fundido 6,18° | BIBE pp. 1, 4 |
| Virtual Kinect 2021 | Mapeia corpo 25↔33; **não** mapeia mãos 21 landmarks | Virtual Kinect pp. 2, 5–6 |

**Nota:** Páginas/equações dos relatórios A1/A2 **não são confiáveis** (auditoria `FASE1_5_AUDITORIA.md` §3–§6). Citações acima usam **páginas PDF reais (1-based)** confirmadas na extração.

---

*Fim do documento. Próximos passos: implementar `fusion/` package, `ComplementaryQuaternionFusion`, `FusionSensorManager`, `FusionVisualizer`, e testes unitários.*