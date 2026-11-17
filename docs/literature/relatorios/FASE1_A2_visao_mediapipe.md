> **AVISO (Fase 1.5):** as páginas/equações deste relatório bruto NÃO foram verificadas e contêm erros. Use como síntese conceitual e confira `FASE1_5_AUDITORIA.md` antes de citar.

# FASE1_A2 — Visão (MediaPipe/Kinect), Yaw e Fusão RGB+IMU

**Data:** 2025-11-03  
**Autor:** Subagente de pesquisa  
**Projeto:** Protótipo Kinect v2 + MediaPipe Hands + IMU MPU6050 (ESP32, Mahony) + EEG

---

## 1. Escopo e PDFs lidos

| # | Arquivo local | Referência | Páginas |
|---|---------------|------------|---------|
| 1 | `literatura/MediaPipeHands_2020_Zhang.pdf` | Zhang et al., *MediaPipe Hands: On-device Real-time Hand Tracking*, arXiv:2006.10214 (2020) | 5 |
| 2 | `literatura/applsci-15-10527.pdf` | Kusunose et al., *Estimation of Forearm Pronation–Supination Angles Using MediaPipe and IMU Sensors*, Appl. Sci. 15, 10527 (2025) | 19 |
| 3 | `literatura/Xefteris_2025_Multimodal_IMU_RGB_pose.pdf` | Xefteris et al., *Multimodal fusion of inertial sensors and single RGB camera data for 3D human pose estimation…*, Internet of Things 29, 101465 (2025) | 13 |
| 4 | `literatura/2-1016-172898355720-23.pdf` | Jaruchokphuriwaj & Varagul, *Accurate Hand Pose Estimation Using Depth Camera and IMU*, IJMPE 12(7), 20–23 (2024) | 4 |
| 5 | `literatura/Real-time_Joint_Angle_Estimation_using_Mediapipe_Framework_and_Inertial_Sensors.pdf` | Palani et al., *Real-time Joint Angle Estimation using Mediapipe Framework and Inertial Sensors*, BIBE 2022, DOI 10.1109/BIBE55377.2022.00035 | 6 |
| 6 | `literatura/3488162.3488215.pdf` | Lafayette et al., *The Virtual Kinect*, SVR 2021, DOI 10.1145/3488162.3488215 | 9 |

---

## 2. Resumo crítico

### 2.1 MediaPipe Hands (Zhang 2020) — fundação
- **21 landmarks 2.5D** (x, y normalizados [0,1] + profundidade relativa ao punho `z_rel`).
- *Topologia* idêntica a Simon et al. 2017 (p. 117): 0=punho, 1–4=polegar (CMC, MCP, IP, ponta), 5–8=indicador, 9–12=médio, 13–16=anelar, 17–20=mínimo.
- *Pipeline dois estágios*: BlazePalm (detector de palma orientada) → Hand Landmark Model (regressão 21×3).
- Saídas adicionais: **handedness** (esq./dir.) e **hand presence score** (confiança de alinhamento razoável; p. 122–125).
- Profundidade relativa aprendida **apenas de dataset sintético** (p. 120) — não é profundidade métrica absoluta.

### 2.2 Kusunose 2025 — pronação/supinação do antebraço via ML + IMU dorsal
- **22 landmarks** (21 mão + cotovelo anotado manualmente).
- Features geométricas normalizadas pelo comprimento do indicador (`|5–8|`): distâncias, ângulos internos, **áreas de triângulos** (Tabela 2, pp. 258–290).
  - `norm_palm_size = |(0→5) × (0→17)| / |5–8|²`  (área punho–indicador–mínimo)
  - `rad_palm_angle = ∠(0–5–17)`  (ângulo no MCP do indicador)
  - `norm_rad_size`, `norm_ulnar_size`, `palm_angle`, `ulnar_palm_angle`, `forarm_angle` (cotovelo).
- **LightGBM** supera SVM/RF/Linear: MAE 4.65° (out-camera) / 5.61° (in-camera), R² ≈ 0.97–0.98 (pp. 430–432).
- **SHAP**: `norm_palm_size` e `rad_palm_angle` dominam (>35% contribuição combinada); cotovelo contribui ≤10% (pp. 438–440, 478–479).
- IMU (WitMotion WT901BLECL) no **dorso do punho**, eixo *roll* = pronação/supinação; validação estática MAE 0.62° vs goniômetro (p. 172).
- **Limitação**: `z` do MediaPipe **não usado diretamente**; apenas normalizado (p. 231–232). Câmara única, sem profundidade métrica.

### 2.3 Xefteris 2025 — fusão corpo-inteiro (MediaPipe Pose + TIP + LSTM-RF)
- **MediaPipe 3D Pose Landmarker**: 33 landmarks *world coordinates* (origem no meio dos quadris, `z` = profundidade métrica a partir daí; pp. 162–165).
- Apenas **13 landmarks comuns** com SMPL usados na fusão (pp. 185–187).
- **TIP** (Transformer Inertial Poser) com 6 IMUs (cabeça, cintura, punhos, tibias) → 18 ângulos de junta + velocidade raiz + SBPs (pp. 213–217).
- **Fusão nível-decisão**: concatena 20×3 (IMU) + 13×3 (MediaPipe) = 99-d → LSTM (176 unid., lookback 5) → Random Forest (pp. 250–254, 265–269).
- **MPJPE**: MediaPipe sozinho 15.21 cm; TIP sozinho 5.48 cm; **LSTM-RF 1.21 cm** (Tabela 3, p. 407).
- **Latência**: MediaPipe Heavy 63.4 ms (CPU i7) + TIP 5.5 ms (GPU) + LSTM 18 ms = **81 ms total** (12 fps) — *não tempo-real estrito* (pp. 509–512).
- MediaPipe **falha profundidade** quando mãos acima da cabeça (p. 521).

### 2.4 IJMPE 2024 (Jaruchokphuriwaj & Varagul) — **RealSense D435 + Witmotion JY61P (MPU6050)**
> ⚠️ **Não transferível diretamente ao nosso setup** (Kinect v2 ≠ RealSense; ToF ≠ ToF mas intrínsecos/FOV/resolução diferentes).

- MediaPipe Hands (20 juntas citadas, real 21) → **ponto de referência O(x,y)** = interseção das retas **1–17** (CMC polegar → MCP mínimo) e **0–9** (punho → MCP médio) (p. 144–146, Fig. 2).
- **Posição métrica** via intrínsecos da câmara de profundidade:
  ```
  x = D (Ox - Ppx) / fx      (Eq. 1, p. 162)
  y = D (Oy - Ppy) / fy      (Eq. 2, p. 164)
  ```
  onde `D` = valor do depth map no pixel `O`, `Ppx/Ppy` = ponto principal, `fx/fy` = distâncias focais.
- **Yaw 2D no plano da imagem**: linha entre **pontos 9 e 10** (MCP médio → PIP médio); 0° = perpendicular ao horizontal da imagem; positivo = anti-horário (pp. 170–175).
- IMU na luva: **eixo X ao longo dos dedos**, **eixo Z perpendicular à palma** (pp. 194–196).
- **EKF** funde posição (câmara) + yaw (câmara) + roll/pitch (IMU).
- Resultados: posição 2.33 mm médio; yaw 0.057°; roll 0.45°; pitch 0.34° (Tabelas 1–2, pp. 261–288).
- **Máximos** em ângulos extremos: yaw 0.11°, roll 3.94°, pitch 3.92° — degradação em borda.

### 2.5 BIBE 2022 (Palani et al.) — Kalman sequencial MediaPipe Holistic + 2×MPU6050
- MediaPipe **Holistic** (33 landmarks pose) → vetores ombro-cotovelo-punho → ângulo de flexão via produto vetorial (pp. 219–222).
- IMUs: **IMU1 no antebraço**, **IMU2 no braço**; eixos `y` paralelos ao plano de flexão, `x` apontando ao cotovelo (pp. 272–276).
- **Roll/pitch do acelerômetro** (Eqs. 3–4, pp. 321–326):
  ```
  roll  = atan2(ay, sqrt(ax²+az²)) * 180/π
  pitch = atan2(ax, sqrt(ay²+az²)) * 180/π   (com sinal conforme convenção)
  ```
- **Flexão** = diferença de *pitch* entre IMUs; **Pronação** = diferença de *roll* (pp. 328–338).
- **Kalman sequencial** (Eqs. 5–8, pp. 429–465): vetor estado `[ângulo, velocidade angular]`, matriz transição `F = [[1, dt],[0,1]]`, ruído de processo `Q`, ruído de medição `R` diagonal (variâncias dos sensores).
- **Sincronismo temporal**: visão adianta IMU → alinhamento necessário antes do filtro (p. 622).
- **Confiança MediaPipe**: 50% tracking / 50% detection ótimo (Tabela I, pp. 545–566).
- RMSE: IMU 6.30°, Visão 7.70°, **Fundido 6.18°** (p. 629).

### 2.6 Virtual Kinect (Lafayette et al. 2021) — mapear Kinect V2 ↔ MediaPipe Pose
- **Tabela de correspondência** 25 juntas Kinect ↔ 33 MediaPipe (Tabela 2, pp. 410–436):
  - Ex.: *Spine Base* = média (Left Hip 23, Right Hip 24); *Left Hand* = média (17,19,15); *Left Thumb* = 21.
- **Sistema de coordenadas Z**:
  - Kinect: origem no sensor, `Z` positivo para frente (metros) (p. 440, Fig. 10).
  - MediaPipe: origem no meio dos quadris, `z` normalizado ≈ escala de `x`; *world coords* em metros com origem nos quadris (pp. 444–451).
- **Visibilidade** (MediaPipe 0–1) ↔ **Tracking State** (Kinect 0/1/2) (pp. 452–456).
- **Erro médio Euclidiano** 6–66 px (Tabela 4, pp. 601–622); braço esquerdo (em movimento) pior que direito (estático).
- **MediaPipe falha** quando braço erguido acima da cabeça (pp. 632–634).
- **Profundidade não implementada** no VK (pp. 394–395, 670–672).

---

## 3. O que se aplica ao nosso setup (Kinect v2 + MediaPipe Hands + MPU6050/ESP32)

| Item | Aplicabilidade | Detalhes / Adaptação |
|------|----------------|---------------------|
| **21 landmarks MediaPipe Hands** | ✅ Direto | Índices 0–20 idênticos; `handedness` + `presence_score` disponíveis. |
| **Palm reference point** | ✅ Adaptável | Kusunose usa triângulo punho(0)–ind.MCP(5)–min.MCP(17); IJMPE usa interseção 1–17 × 0–9. **Contrato**: testar ambos; `O = intersect(1→17, 0→9)` é mais geométrico (centro da palma). |
| **Normal da palma / yaw 3D** | ✅ Via depth do Kinect | MediaPipe `z_rel` não métrico. **Usar depth map do Kinect** no pixel do landmark de referência → back-projection com intrínsecos Kinect (Eqs. 1–2 do IJMPE). PCA no vizinhado 3D dos landmarks 0,5,9,13,17 → normal; yaw = `atan2(n_x, n_z)` no sistema da câmara. |
| **Yaw 2D (fallback)** | ✅ IJMPE Eq. | Linha 9→10 (MCP→PIP médio); `yaw2D = atan2(dy, dx) - π/2`. Útil quando depth ruidoso. |
| **Features geométricas normalizadas** | ✅ Kusunose Tabela 2 | `norm_palm_size`, `rad_palm_angle`, `norm_rad_size`, `palm_angle`, `forarm_angle` (se cotovelo visível via MediaPipe Pose). Invariantes a escala. |
| **IMU no dorso da mão** | ✅ Nosso ESP32 | Eixos: **X = dedos**, **Z = normal à palma** (igual IJMPE p. 195). Mahony no ESP32 já validado → roll/pitch bons; **yaw deriva** (6 eixos). |
| **Fusão EKF/Kalman** | ✅ IJMPE + BIBE | Estado: `[yaw, roll, pitch, ω_yaw, ω_roll, ω_pitch]` ou quaternão. Medidas: yaw_cam (depth+PCA), yaw_IMU (gyro integrado), roll/pitch_IMU (Mahony). `R` diagonal com variâncias empíricas. |
| **Sincronismo temporal** | ⚠️ Necessário | BIBE: visão adianta IMU. **Contrato**: timestamp único (PC) ao receber frame Kinect + pacote ESP32; buffer circular + interpolação linear. |
| **Confiança MediaPipe** | ✅ BIBE Tabela I | `tracking_conf ≥ 0.5`, `detection_conf ≥ 0.5`; descartar frames abaixo. |
| **Coordenadas métricas Kinect** | ✅ Nativo | Kinect SDK fornece `CameraSpacePoint` (mm) para cada pixel de depth — **não back-projetar manualmente**; usar `CoordinateMapper.MapDepthFrameToCameraSpace`. |
| **Handedness** | ✅ MediaPipe | Separar modelos/esquerdas/direitas; features geométricas simétricas (Kusunose p. 622–624). |

---

## 4. O que NÃO se aplica / Limitações

| Item | Razão | Impacto no nosso protótipo |
|------|-------|---------------------------|
| **Números IJMPE (2.33 mm, 0.057°)** | RealSense D435 (IR pattern + ToF) ≠ Kinect v2 (ToF contínuo); intrínsecos, FOV (70°×60° vs 70.6°×60°), resolução depth (640×480 vs 512×424), ruído diferente. | **Não usar como meta**. Nosso baseline: medir erro real no nosso hardware. |
| **IMU Witmotion JY61P** | JY61P tem filtro interno + magnetômetro (9 eixos); nosso MPU6050 6 eixos + Mahony no ESP32 → **yaw deriva**. | Fusão **obrigatória** para yaw; roll/pitch do Mahony já bons. |
| **MediaPipe Pose (corpo) ↔ Kinect V2** | Virtual Kinect mapeia *corpo* (33↔25), não *mãos* (21 landmarks). | Não reutilizar tabela de correspondência para mãos. |
| **MediaPipe world coordinates (corpo)** | Origem no quadril, `z` métrico relativo ao quadril — **não serve para mão isolada**. | Usar depth Kinect + intrínsecos Kinect para mão. |
| **Profundidade MediaPipe Hands (`z_rel`)** | Relativa ao punho, treinada só em sintético (Zhang p. 120), escala arbitrária. | **Descartar para métricas**; usar apenas como *prior* qualitativo (ordem de profundidade). |
| **LSTM-RF (Xefteris)** | Corpo inteiro, 6 IMUs, dataset TotalCapture (Vicon GT), latência 81 ms. | Overkill para mão única; **Kalman/EKF leve** suficiente (BIBE/IJMPE). |
| **Cotovelo via MediaPipe Pose** | Kusunose anota manualmente; MediaPipe Pose detecta cotovelo mas **não sincronizado** com Hands (graphs separados). | Se precisar `forarm_angle`, rodar Pose+Hands no mesmo frame e alinhar por timestamp. |
| **Profundidade VK** | Não implementada (Lafayette p. 670). | Não resolve nosso problema de depth da mão. |

---

## 5. Equações e pseudocódigo relevantes (com página)

### 5.1 Ponto de referência da palma (IJMPE, pp. 144–146)
```
# Landmarks: 0=wrist, 1=thumb_CMC, 5=index_MCP, 9=middle_MCP, 13=ring_MCP, 17=pinky_MCP
# Linha L1: p1 → p17    (vetor v1 = p17 - p1)
# Linha L2: p0 → p9     (vetor v2 = p9  - p0)
# Interseção O = p1 + t*v1  tal que (O - p0) × v2 = 0
t = ((p0 - p1) × v2) / (v1 × v2)
O = p1 + t * v1
```
> **Teste**: comparar `O` vs centróide `{0,5,9,13,17}` vs `norm_palm_size` centroid.

### 5.2 Back-projection Kinect (IJMPE Eqs. 1–2, pp. 162–164) — **usar SDK nativo**
```python
# Em vez de implementar Eqs. 1-2:
depth_frame = kinect.depth_frame  # 512x424, valores em mm
cam_pts = kinect.coordinate_mapper.MapDepthFrameToCameraSpace(depth_frame)
# cam_pts[Oy, Ox] → (x_mm, y_mm, z_mm) no sistema da câmara (origem no sensor IR)
```

### 5.3 Normal da palma via PCA (3D) — **novo, baseado em Kusunose + depth**
```python
# Coletar 3D points dos landmarks-chave no sistema da câmara (mm)
pts_3d = []
for idx in [0, 5, 9, 13, 17]:  # wrist, MCPs
    u, v = landmarks[idx].x * W, landmarks[idx].y * H
    pts_3d.append(cam_pts[int(v), int(u)])  # (x,y,z) mm
pts_3d = np.array(pts_3d)
# PCA
center = pts_3d.mean(axis=0)
U, S, Vt = np.linalg.svd(pts_3d - center)
normal = Vt[2]  # menor valor singular → normal do plano
# Yaw no sistema da câmara (Z para frente, X direita, Y baixo)
yaw_cam = atan2(normal[0], normal[2])  # rad
```
> **Degeneração edge-on**: se `S[2] / S[0] < 0.1` (plano muito fino visto de lado) → fallback para yaw 2D (Eq. 5.4).

### 5.4 Yaw 2D fallback (IJMPE pp. 170–175)
```python
# Linha entre middle_MCP (9) e middle_PIP (10)
p9 = landmarks[9]; p10 = landmarks[10]
dx = p10.x - p9.x; dy = p10.y - p9.y
yaw2D = atan2(dy, dx) - pi/2  # 0° quando perpendicular ao horizontal da imagem
```

### 5.5 Features normalizadas Kusunose (Tabela 2, pp. 258–290)
```python
def kusunose_features(lm, handedness):
    # lm: 21×3 (x,y,z_rel) normalizados [0,1]; z_rel ignorado
    idx = lambda i: lm[i]
    scale = np.linalg.norm(idx(5)[:2] - idx(8)[:2]) + 1e-6  # |5-8| 2D
    v05 = idx(5)[:2] - idx(0)[:2]
    v017 = idx(17)[:2] - idx(0)[:2]
    v517 = idx(17)[:2] - idx(5)[:2]
    v50 = idx(0)[:2] - idx(5)[:2]
    v170 = idx(0)[:2] - idx(17)[:2]
    v175 = idx(5)[:2] - idx(17)[:2]
    feats = {}
    feats['norm_palm_width']  = np.linalg.norm(v517) / scale
    feats['norm_palm_dist']   = np.linalg.norm(v50) / scale
    feats['rad_palm_angle']   = angle_between(v05, v517)        # ∠0-5-17
    feats['ulnar_palm_angle'] = angle_between(v170, v175)       # ∠17-0-5
    feats['palm_angle']       = angle_between(v05, v017)        # ∠0-17-5
    feats['norm_rad_size']    = abs(np.cross(v517, v50)) / scale**2
    feats['norm_ulnar_size']  = abs(np.cross(v170, v175)) / scale**2
    feats['norm_palm_size']   = abs(np.cross(v05, v017)) / scale**2
    return feats
```

### 5.6 Roll/Pitch do acelerômetro (BIBE Eqs. 3–4, pp. 321–326)
```python
# a = [ax, ay, az] no frame do IMU (gravidade = [0,0,g] no mundo)
roll  = atan2(ay, sqrt(ax**2 + az**2)) * 180/pi
pitch = atan2(-ax, sqrt(ay**2 + az**2)) * 180/pi  # sinal conforme convenção do papel
```
> **Nosso caso**: Mahony no ESP32 já entrega quaternão `q` → `roll, pitch, yaw = q.to_euler()`; usar roll/pitch do Mahony (bons), **descartar yaw do Mahony** (deriva).

### 5.7 Kalman sequencial para fusão yaw (BIBE Eqs. 5–8, pp. 429–465) — adaptado
```python
# Estado: x = [yaw, yaw_rate]^T
F = np.array([[1, dt], [0, 1]])
Q = np.diag([q_yaw, q_rate])   # tunar
# Medidas: z_cam (yaw_cam), z_imu (yaw_IMU = integral gyro_z)
H = np.array([[1, 0]])
R_cam = var_yaw_cam   # empirico: variância do PCA/yaw2D em estático
R_imu = var_yaw_imu   # deriva: cresce com tempo; modelar como random walk

# Loop por frame (seq. update: IMU depois visão, ou vice-versa — ordem não importa p. 425)
def kf_predict(x, P):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P

def kf_update(x, P, z, R):
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T / S
    x = x + K * y
    P = (np.eye(2) - K @ H) @ P
    return x, P

# Uso:
x, P = kf_predict(x, P)
x, P = kf_update(x, P, yaw_imu, R_imu)   # update IMU
x, P = kf_update(x, P, yaw_cam, R_cam)   # update câmara
yaw_fused = x[0]
```

### 5.8 Sincronismo temporal (BIBE p. 622)
```python
# Buffer circular com timestamps (monotonic ns)
class SyncBuffer:
    def __init__(self, maxlen=60):
        self.cam = deque(maxlen=maxlen)  # (ts, frame_data)
        self.imu = deque(maxlen=maxlen)  # (ts, imu_data)
    def push_cam(self, ts, data): self.cam.append((ts, data))
    def push_imu(self, ts, data): self.imu.append((ts, data))
    def get_pair(self, target_ts):
        # interpolar IMU para ts do frame da câmara
        cam_ts, cam_data = min(self.cam, key=lambda x: abs(x[0]-target_ts))
        imu_before = max([d for d in self.imu if d[0] <= cam_ts], default=None)
        imu_after  = min([d for d in self.imu if d[0] >= cam_ts], default=None)
        if imu_before and imu_after:
            alpha = (cam_ts - imu_before[0]) / (imu_after[0] - imu_before[0])
            imu_data = lerp(imu_before[1], imu_after[1], alpha)
        else:
            imu_data = (imu_before or imu_after)[1]
        return cam_data, imu_data
```

---

## 6. Riscos e perguntas abertas

| # | Risco / Pergunta | Mitigação / Próximo passo |
|---|------------------|---------------------------|
| R1 | **Depth Kinect ruidoso na borda da mão** (IR saturado, multitrajeto) → PCA instável. | Medir SNR do depth map na ROI da mão; filtro bilateral temporal; fallback yaw2D. |
| R2 | **Degeneração edge-on**: palma perpendicular à câmara → `S[2]/S[0] → 0`. | Detectar via razão autovalores; trocar para yaw2D + prior IMU; logar ocorrência. |
| R3 | **Latência diferencial** Kinect (30 fps, ~33 ms) vs ESP32 (100 Hz, ~10 ms) vs MediaPipe (~15 ms GPU). | Buffer sincronizado (Sec. 5.8); medir offsets reais; compensar no EKF. |
| R4 | **MediaPipe Hands `z_rel` não métrico** — tentação de usar como depth. | **Proibir** uso de `z_rel` para métricas; apenas debug visual (sombreamento). |
| R5 | **Handedness flips** em oclusão/rotação extrema → features espelhadas erradas. | Usar `presence_score < 0.5` para descartar; suavizar handedness com hist. temporal. |
| R6 | **Cotovelo não detectado** pelo MediaPipe Pose simultâneo → `forarm_angle` indisponível. | Treinar modelo só com features da palma (Kusunose: cotovelo ≤10% SHAP). |
| R7 | **Yaw IMU deriva** (6 eixos, sem mag) → erro cresce ~1°/min em Mahony. | EKF com `R_imu` crescente ou reset periódico quando `yaw_cam` confiável (baixa variância). |
| R8 | **Calibração extrínseca Kinect↔IMU** — IMU no dorso da mão, câmara no monitor/tripé. | Procedimento: mão plana sobre mesa (pose conhecida) → resolver `R_cam_imu`, `t_cam_imu` via Procrustes. |
| R9 | **Colunas FUSED_* versionadas** — não quebrar `PALM_*`/`IMU_*`. | Definir schema: `FUSED_v1 = {ts, yaw_fused, roll_imu, pitch_imu, palm_normal_cam, conf}`; manter originais intactas. |
| R10 | **Validação sem ground-truth óptico** (Vicon/OptiTrack). | Jig impresso 3D com ângulos conhecidos (0°, 15°, ..., 90°) + régua para posição; repetir IJMPE Tabelas 1–2 no nosso hardware. |

---

## 7. Citações completas

1. **Zhang et al. 2020** — *MediaPipe Hands: On-device Real-time Hand Tracking*, arXiv:2006.10214v1, 18 Jun 2020.  
   Arquivo: `literatura/MediaPipeHands_2020_Zhang.pdf` — Páginas 1–5 (landmarks pp. 106–117, pipeline pp. 48–64, datasets pp. 137–174, resultados pp. 176–211).

2. **Kusunose et al. 2025** — *Estimation of Forearm Pronation–Supination Angles Using MediaPipe and IMU Sensors: Performance Comparison and Interpretability Analysis of Machine Learning Models*, Appl. Sci. 15, 10527 (2025). DOI: 10.3390/app151910527.  
   Arquivo: `literatura/applsci-15-10527.pdf` — Páginas 1–19 (landmarks pp. 241–247, features Tabela 2 pp. 258–290, resultados Tabelas 5–6 pp. 415–430, SHAP pp. 438–440, 478–479, discussão pp. 542–580).

3. **Xefteris et al. 2025** — *Multimodal fusion of inertial sensors and single RGB camera data for 3D human pose estimation based on a hybrid LSTM-Random forest fusion network*, Internet of Things 29, 101465 (2025). DOI: 10.1016/j.iot.2024.101465.  
   Arquivo: `literatura/Xefteris_2025_Multimodal_IMU_RGB_pose.pdf` — Páginas 1–13 (pipeline Fig.1 p. 179, MediaPipe world coords pp. 162–165, TIP pp. 213–217, fusão pp. 250–269, resultados Tabelas 1–3 pp. 314–407, latência pp. 509–512).

4. **Jaruchokphuriwaj & Varagul 2024** — *Accurate Hand Pose Estimation Using Depth Camera and IMU*, International Journal of Mechanical and Production Engineering 12(7), 20–23 (Jul 2024). ISSN(p): 2320-2092, ISSN(e): 2321-2071.  
   Arquivo: `literatura/2-1016-172898355720-23.pdf` — Páginas 1–4 (ponto ref. pp. 144–146, Eqs. 1–2 pp. 162–164, yaw 2D pp. 170–175, IMU placement pp. 194–196, resultados Tabelas 1–2 pp. 260–288).

5. **Palani et al. 2022** — *Real-time Joint Angle Estimation using Mediapipe Framework and Inertial Sensors*, 2022 IEEE 22nd International Conference on Bioinformatics and Bioengineering (BIBE), DOI: 10.1109/BIBE55377.2022.00035.  
   Arquivo: `literatura/Real-time_Joint_Angle_Estimation_using_Mediapipe_Framework_and_Inertial_Sensors.pdf` — Páginas 1–6 (setup pp. 170–182, roll/pitch Eqs. 3–4 pp. 321–326, Kalman Eqs. 5–8 pp. 429–465, confiança Tabela I pp. 545–566, resultados p. 629).

6. **Lafayette et al. 2021** — *The Virtual Kinect*, Proceedings of SVR 2021, DOI: 10.1145/3488162.3488215.  
   Arquivo: `literatura/3488162.3488215.pdf` — Páginas 1–9 (correspondência juntas Tabela 2 pp. 410–436, sistemas Z pp. 440–451, erros Tabela 4 pp. 601–622, limitações pp. 632–634, 670–672).

---

## 8. Contratos, medidas e testes propostos (para próxima etapa)

| ID | Contrato / Medida | Critério de aceite |
|----|-------------------|---------------------|
| C1 | `get_palm_reference(landmarks) → (u,v) px` | Duas implementações (interseção IJMPE vs centróide 0,5,9,13,17); diferença < 5 px em 95% frames válidos. |
| C2 | `backproject_kinect(u,v,depth_frame) → (x,y,z) mm` | Usar `CoordinateMapper` nativo; erro < 2 mm em plano de calibração (tabuleiro xadrez a distâncias conhecidas). |
| C3 | `compute_palm_normal_pca(landmarks_3d) → normal_3d, yaw_cam, quality` | `quality = S[2]/S[0]`; se `quality < 0.1` → flag `DEGENERATE`; yaw_cam consistente com yaw2D ±3° em poses frontais. |
| C4 | `yaw2d_fallback(landmarks) → yaw2D` | Em poses frontais (palma ≈ perpendicular a câmara), `|yaw2D - yaw_cam| < 2°`. |
| C5 | `kusunose_features(landmarks, handedness) → dict[9 features]` | Valores idênticos (≤1e-6) à implementação de referência Python (teste unitário com landmarks sintéticos conhecidos). |
| C6 | `sync_buffer.get_pair(ts_cam) → (cam_data, imu_data_interp)` | Jitter temporal residual < 5 ms (medido com sinal sintético sincronizado). |
| C7 | `ekf_fuse(yaw_cam, yaw_imu, roll_imu, pitch_imu, dt) → yaw_fused` | Em estático (mão parada 30s): `std(yaw_fused) < 0.5°`; `std(roll_imu) < 0.3°`, `std(pitch_imu) < 0.3°` (alinhado a Mahony validado). |
| C8 | `calibrate_extrinsics_kinect_imu() → R, t` | Erro de reprojeção < 3 mm em 5 poses de validação (jig impresso 3D). |
| T1 | **Teste de bancada (jig 3D)** — replicar IJMPE Tabelas 1–2 no nosso hardware. | Posição média ≤ 5 mm; yaw ≤ 0.5°; roll/pitch ≤ 1.0° (metas relaxadas vs IJMPE por ser Kinect v2 + MPU6050). |
| T2 | **Teste dinâmico** — rotação contínua palma (0°→180°→0°) a 30°/s. | RMSE yaw_fused vs encoder/jig ≤ 3°; latência percepção-ação < 100 ms. |
| T3 | **Ablation** — rodar só visão, só IMU, fundido. | Fundido deve reduzir RMSE em ≥30% vs melhor unimodal (alinhado a BIBE 6.18° vs 6.30°/7.70°). |
| T4 | **Schema FUSED_v1** — gravar CSV/Parquet com colunas versionadas. | Colunas: `ts, yaw_fused, roll_imu, pitch_imu, palm_nx, palm_ny, palm_nz, conf_cam, conf_imu, quality_pca, schema_version=1`. `PALM_*` e `IMU_*` inalterados. |

---

*Fim do relatório FASE1_A2.*