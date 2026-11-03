# FASE 1 — Relatório A1: Atitude, Fusão e Calibração

**Data:** 2026-11-03  
**Projeto:** Protótipo Kinect v2 + MediaPipe RGB + IMU MPU6050 (ESP32, Mahony 6-eixos)  
**Objetivo:** Melhorar orientação da palma → integrar ao paradigma EEG (colunas `FUSED_*` versionadas, sem quebrar `PALM_*`/`IMU_*`)

---

## 1. Escopo e PDFs lidos

| PDF | Autor/Ano | Título | Páginas | Arquivo local |
|-----|-----------|--------|---------|---------------|
| **Mahony 2008** | Mahony, Hamel, Pflimlin | *Complementary Filter Design on SO(3)* (IEEE TAC) | 17 | `literatura/2007_Mahony.etal_TAC-06-396_v3.pdf` |
| **Markley 2002** | Markley, Cheng, Crassidis, Oshman | *Quaternion Averaging* (J. Guidance, Control, Dynamics) | 10 | `literatura/Markley_2002_Quaternion_Averaging.pdf` |
| **Tian 2015** | Tian, Meng, Tao, Liu, Feng | *Upper limb motion tracking with the integration of IMU and Kinect* (Neurocomputing) | 12 | `literatura/1-s2.0-S0925231215001472-main.pdf` |
| **Acharya 2022** | Acharya, Bhat, Kanthi | *A Comparative Analysis of Two Approaches for Estimation of Upper Limb Orientation Using Inertial and Kinect Sensors* (AECE) | 8 | `literatura/aece_2022_4_10.pdf` |

> **Nota importante:** O artigo IJMPE de referência (RealSense D435 + Witmotion JY61P) **não** usa Kinect v2 nem MPU6050; seus números **não** são meta do nosso protótipo.

---

## 2. Resumo crítico

### 2.1 Mahony 2008 — Filtro complementar em SO(3) com 6 eixos (acel + giro)
- **Arquitetura:** Duas variantes — *direct* (usa atitude reconstruída $R_y$ para mapear $\omega$ ao frame inercial) e *passive* (usa atitude filtrada $R$; Eq. 9–10, pp. 5–6). A passiva evita injetar ruído de $R_y$ no termo preditivo e mantém análise de Lyapunov idêntica (Eq. 8, p. 5).
- **Correção:** $\omega_{\text{corr}} = k_P \, \text{vex}(P_a(R^T R_y))$ (Eq. 7, p. 5), onde $P_a$ é a parte anti-simétrica. Em quaternions (Apêndice B, pp. 15–16): $q = \frac{1}{2} q \otimes p(\omega - b + 2 k_P s v)$, $b = 2 k_I s v$ (Eq. 47, p. 16).
- **Deriva de yaw (6 eixos):** Sem magnetômetro, não há referência absoluta de rumo. O vetor gravidade corrige apenas *roll/pitch* (2 DOF). O *yaw* deriva com o viés do giroscópio $b_z$. O filtro estima $b$ (Eq. 12b/13b), mas observabilidade de $b_z$ exige excitação persistente ou referência externa.
- **Por que sair de Euler:** Singularidades (gimbal lock), representação não-única, derivada não-linear. Em SO(3)/quaternions a cinemática é $\dot{R} = R \Omega$ / $\dot{q} = \frac{1}{2} q \otimes \omega$ — integrador linear no grupo de Lie, sem singularidades.

### 2.2 Markley 2002 — Média ótima de quaternions (pesos escalares/matriciais)
- **Princípio:** Mediar *atitudes*, não quaternions diretamente ($q$ e $-q$ representam a mesma rotação). Média = autovetor de $M = \sum_i w_i q_i q_i^T$ associado ao maior autovalor (Eq. 12, p. 3).
- **Pesos matriciais (covariância):** $M = \sum_i E(q_i) R_i E(q_i)^T$ (Eq. 26, p. 7), onde $E(q)$ mapeia erro de atitude para $\mathbb{R}^3$. Equivalente a estimação de máxima verossimilhança (Eq. 36, p. 8).
- **Condição de unicidade:** Todos $q_i$ dentro de $\pi/2$ de um referencial comum (pp. 5–6).
- **Covariância do erro:** $\Sigma \approx (\sum_i F_i)^{-1}$ (Eq. 38, p. 8) — útil para fundir múltiplas estimativas com confianças conhecidas.
- **Algoritmo eficiente:** QUEST (p. 3) evita decomposição completa de autovalores.

### 2.3 Tian 2015 — UKF para fusão IMU (Inertiacube 3) + Kinect v1
- **Três estágios:** (1) IMU apenas (dupla integração + UKF), (2) IMU + restrição geométrica (cotovelo ~2 DOF), (3) IMU + Kinect (UKF).
- **Vetor de estado:** $x = [p, v, q]^T$ (posição, velocidade, quaternion) — Eq. 1, p. 3.
- **Modelo de processo:** Cinemática de quaternion (Eq. 3–5, pp. 3–4), aceleração no frame global $a = q \otimes a_B \otimes q^{-1} - g$ (Eq. 6), integração discreta $p_t, v_t$ (Eq. 7).
- **Modelo de medição IMU:** Acelerômetro e magnetômetro no frame corpo (Eq. 10–11, p. 4).
- **Restrição geométrica:** Ângulo de adução do cotovelo $\approx 0$ → produto escalar $Y_u^T V \approx 0$ (Eq. 13–14, p. 5). Melhora mas não elimina deriva.
- **Fusão IMU+Kinect:** Transformação $R_{WK}$ (Kinect → mundo) necessária (Sec. 5.1, p. 7). Medição Kinect: posição da mão $p_K$ transformada para frame global. UKF funde posições (Kinect, sem deriva mas ruidosa/ocluída) + IMU (suave mas com deriva).
- **Calibração de viés:** Estático 45 min, modelo exponencial $a_{\text{err}} = C_1 e^{-t/T} + C_2$ (Eq. 12, p. 5).
- **Resultado:** UKF IMU+Kinect elimina deriva e reduz flutuação vs. Kinect isolado (Fig. 9–10, pp. 9–10).

### 2.4 Acharya 2022 — Comparação MPU6050 (Kalman) vs. Kinect v2
- **MPU6050 (6 eixos):** Roll/pitch do acelerômetro (Eq. 4–8, pp. 3–4: $\tan\phi = A_{ny}/A_{nx}$, $\tan\theta = A_{nx}/\sqrt{A_{ny}^2+A_{nz}^2}$). **Yaw não observável** pelo acelerômetro (p. 3: "rotation around z-axis is of no significance"). Filtro de Kalman funde accel + giro.
- **Kinect v2:** Esqueleto 25 juntas, mãos = juntas 12/13. Sensível a oclusão, iluminação, luz solar; requer calibração por usuário (levantar braço).
- **Estatística:** Wilcoxon signed-rank (p=0.0198 mãos, p=0.00003 métodos), Spearman $\rho \approx 0.94–0.96$ (IMU vs. cronômetro). IMU tem menor erro padrão (0.63–0.64 s) vs. Kinect (0.73–0.74 s).
- **Conclusão:** IMU mais preciso para tempo/orientação relativa; Kinect dá posição absoluta mas ruidosa. **Nenhuma fusão real** — apenas comparação paralela.

---

## 3. O que se aplica ao nosso setup

| Item | Aplicabilidade | Referência |
|------|----------------|------------|
| **Mahony passivo 6-eixos no ESP32** | ✅ Já validado isoladamente. Usar forma quaternion Eq. 47 (p. 16 Mahony) para evitar conversão matriz↔quaternion. | Mahony pp. 5–6, 15–16 |
| **Deriva de yaw intrínseca (6 eixos)** | ✅ Confirmada: sem mag, yaw deriva com $b_z$. Necessita referência externa (Kinect/MediaPipe) para corrigir. | Mahony pp. 4–5, Sec. II-B |
| **Média de quaternions (Markley)** | ✅ Para fundir: (a) Mahony IMU, (b) orientação da palma MediaPipe (via landmarks 3D), (c) eventuais múltiplas estimativas IMU. Pesos = inverso da covariância estimada. | Markley pp. 3–7 |
| **UKF IMU+Kinect (Tian)** | ✅ Arquitetura reutilizável: estado $[p, v, q]$, modelo de processo idêntico. Medição Kinect → posição da mão (MediaPipe 3D ou Kinect skeleton). Transformação $T_{\text{IMU}\to\text{cam}}$ e $R_{WK}$ necessárias. | Tian Sec. 3, 5 |
| **Restrição geométrica (cotovelo/palma)** | ✅ Adaptável: distância fixa palma-pulso, plano da palma ortogonal ao antebraço, etc. Entram como medições pseudo-lineares no UKF. | Tian Eq. 13–14 |
| **Calibração de viés IMU (estático)** | ✅ Procedimento de 45 min + modelo exponencial (Tian Eq. 12) ou média simples (Acharya). Fazer no ESP32 *offline* e gravar offsets. | Tian p. 5, Acharya Sec. III-B |
| **Comparação IMU vs. Kinect/MediaPipe** | ✅ Acharya valida: IMU = suave/preciso relativo; visão = absoluto/ruidoso/ocluído. Fundamenta fusão ponderada. | Acharya Sec. IV–V |

---

## 4. O que NÃO se aplica / Limitações

| Item | Limitação | Ação |
|------|-----------|------|
| **Mahony: magnetômetro para yaw** | Nosso MPU6050 **não tem mag**. Equação 11 (p. 4 Mahony) assume $v_m$; no nosso caso $v_m$ inexistente → yaw livre. | Usar visão (MediaPipe/Kinect) como referência de yaw. |
| **Tian: Kinect v1 + Inertiacube 3** | Hardware diferente (Kinect v2 RGB+depth, MPU6050, MediaPipe). Ruído/latência distintos. | Recalibrar matrizes de ruído $Q, R$ do UKF experimentalmente. |
| **Tian: 2 IMUs (antebraço + braço)** | Temos **1 IMU na palma** (ou pulso). Modelo de estado simplificado. | Reduzir estado para $[p_{\text{palma}}, v_{\text{palma}}, q_{\text{palma}}]$. |
| **Acharya: MPU6050 em objeto (Tin), não no corpo** | Nosso IMU está na palma/pulso → aceleração linear do movimento contamina estimativa de gravidade. | Usar detecção de movimento (limiar em $\|\omega\|$) para suspender correção accel durante aceleração linear alta. |
| **Markley: unicidade $\pi/2$** | Se MediaPipe der orientação invertida ($q \to -q$) ou com erro > 90°, média falha. | Normalizar sinal: forçar $q_{\text{MediaPipe}} \cdot q_{\text{IMU}} > 0$ antes de fundir. |
| **IJMPE (RealSense + JY61P)** | **Não aplicar números** (precisão, latência, deriva) ao nosso hardware. | Usar apenas como referência conceitual de arquitetura UKF. |

---

## 5. Equações e pseudocódigo relevantes (com página)

### 5.1 Mahony passivo (quaternion) — núcleo do filtro no ESP32
```python
# Mahony Eq. 47 (p. 16) — forma passiva com correção de viés
# q: quaternion [w, x, y, z] (escalar primeiro)
# omega: giroscópio rad/s no frame corpo (3,)
# b: viés estimado do giroscópio (3,)
# kP, kI: ganhos proporcionais e integrais
# dt: passo de tempo

def mahony_passive_step(q, omega, b, kP, kI, dt, accel_norm=None):
    # 1. Correção proporcional a partir do acelerômetro (se disponível)
    if accel_norm is not None and np.linalg.norm(accel_norm) > 0.1:
        # vetor gravidade estimado no frame corpo: v_g = q.rotate([0,0,1])
        v_g = quat_rotate(q, np.array([0., 0., 1.]))
        # erro = accel_norm x v_g  (produto vetorial)
        e = np.cross(accel_norm, v_g)
        omega_corr = omega + kP * e
    else:
        omega_corr = omega
    
    # 2. Subtrai viés
    omega_corr = omega_corr - b
    
    # 3. Integra quaternion: q_dot = 0.5 * q ⊗ [0, omega_corr]
    q_dot = 0.5 * quat_mult(q, np.concatenate([[0.], omega_corr]))
    q = q + dt * q_dot
    q = q / np.linalg.norm(q)
    
    # 4. Atualiza viés (integral do erro)
    if accel_norm is not None and np.linalg.norm(accel_norm) > 0.1:
        b = b + dt * kI * e
    
    return q, b
```
> **Nota:** Sem magnetômetro, `accel_norm` corrige apenas roll/pitch. Yaw deriva com `b[2]`.

### 5.2 Média de Markley (pesos escalares) — fusão IMU + MediaPipe
```python
# Markley Eq. 12 (p. 3) — média ponderada de quaternions
# qs: lista de quaternions [w,x,y,z] (N,4)
# ws: pesos escalares (N,) — ex.: 1/trace(cov) ou confiança heurística

def markley_average(qs, ws):
    # Garante consistência de sinal: qs[i] ⋅ qs[0] > 0
    qs = np.array(qs)
    ws = np.array(ws)
    for i in range(1, len(qs)):
        if np.dot(qs[i], qs[0]) < 0:
            qs[i] = -qs[i]
    
    # M = sum(w_i * q_i * q_i^T)
    M = np.zeros((4,4))
    for q, w in zip(qs, ws):
        M += w * np.outer(q, q)
    
    # Autovetor do maior autovalor
    eigvals, eigvecs = np.linalg.eigh(M)
    q_avg = eigvecs[:, -1]
    return q_avg / np.linalg.norm(q_avg)
```

### 5.3 UKF — Modelo de processo (Tian Eq. 3–7, pp. 3–4)
```python
# Estado: x = [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z] (10,)
# Controle: u = [omega_x, omega_y, omega_z, a_x, a_y, a_z] (6,) no frame corpo

def process_model(x, u, dt, g=np.array([0.,0.,-9.81])):
    p = x[0:3]
    v = x[3:6]
    q = x[6:10]
    omega = u[0:3]
    a_body = u[3:6]
    
    # 1. Quaternion kinematics (Eq. 5 discretizado)
    # q_{k+1} = q_k ⊗ exp(0.5 * omega * dt)
    half_omega_dt = 0.5 * omega * dt
    theta = np.linalg.norm(half_omega_dt)
    if theta > 1e-8:
        dq = np.concatenate([[np.cos(theta)], 
                             np.sin(theta)/theta * half_omega_dt])
    else:
        dq = np.array([1., 0., 0., 0.])
    q_new = quat_mult(q, dq)
    q_new = q_new / np.linalg.norm(q_new)
    
    # 2. Aceleração no frame global (Eq. 6)
    a_world = quat_rotate(q_new, a_body) - g
    
    # 3. Integração velocidade/posição (Eq. 7)
    v_new = v + dt * a_world
    p_new = p + dt * v + 0.5 * dt**2 * a_world
    
    return np.concatenate([p_new, v_new, q_new])
```

### 5.4 UKF — Medição Kinect/MediaPipe (posição da palma)
```python
# Tian Sec. 5.1 (p. 7): transformação Kinect → mundo
# p_kinect: posição da mão no frame Kinect (3,)
# R_WK: rotação Kinect → mundo (3,3) — calibração offline
# t_WK: translação Kinect → mundo (3,) — calibração offline

def kinect_measurement(p_kinect, R_WK, t_WK):
    return R_WK @ p_kinect + t_WK  # posição no frame mundo

# Medição MediaPipe (landmarks 3D no frame câmera RGB)
# Precisa T_cam_RGB → mundo (extrínsecos Kinect) + intrínsecos
```

### 5.5 Restrição geométrica (palma) — adaptado de Tian Eq. 13–14
```python
# Palma: distância fixa pulso-palma (~0.08-0.1 m), normal da palma ⊥ antebraço
# Se temos IMU no pulso (q_wrist) e IMU na palma (q_palm):
def palm_constraint(q_wrist, q_palm, p_wrist, p_palm, L_nominal=0.09):
    # 1. Distância pulso-palma
    d = np.linalg.norm(p_palm - p_wrist)
    err_dist = d - L_nominal
    
    # 2. Normal da palma (eixo z do frame palma) ⊥ eixo do antebraço (eixo y do pulso?)
    n_palm = quat_rotate(q_palm, np.array([0.,0.,1.]))
    y_wrist = quat_rotate(q_wrist, np.array([0.,1.,0.]))
    err_ortho = np.dot(n_palm, y_wrist)  # deve ser ~0
    
    return np.array([err_dist, err_ortho])
```

### 5.6 Calibração $T_{\text{IMU}\to\text{cam}}$ (mundo IMU → câmera) e $f$ (montante palma no corpo)
```python
# Problema: AX = XB  (hand-eye calibration)
# A = Δpose_cam (entre frames Kinect/MediaPipe)
# B = Δpose_IMU (entre amostras IMU sincronizadas)
# X = T_IMU_to_cam (4x4 homogênea)
# Resolver via SVD (Tsai 1989) ou otimização não-linear (mínimos quadrados)

def calibrate_T_IMU_cam(poses_cam, poses_imu):
    # poses_cam: lista de (R_cam, t_cam) no frame mundo (Kinect)
    # poses_imu: lista de (R_imu, t_imu) no frame mundo (IMU integrado)
    # Ambos sincronizados temporalmente
    # Retorna X = [R_X, t_X; 0, 1] tal que R_cam ≈ R_X @ R_imu @ R_X^T
    pass  # Implementar com base em movimentos de excitação (rotação + translação)

# Montante f (palma no frame corpo/IMU): medir offline com régua + alinhamento visual
# Ou estimar via otimização: minimizar erro de restrição geométrica (Sec. 5.5)
```

---

## 6. Riscos e perguntas abertas

| # | Risco / Pergunta | Impacto | Mitigação proposta |
|---|------------------|---------|---------------------|
| **R1** | **Yaw observável apenas com visão** — MediaPipe/Kinect dão posição 3D, não orientação absoluta da palma. Como extrair yaw da palma da visão? | Médio/Alto | (a) Usar landmarks MediaPipe (punho + base dedos) para definir frame palma; (b) Fundir via Markley com peso baixo no yaw; (c) Testar observabilidade no UKF. |
| **R2** | **Latência e assincronia** — IMU (ESP32, ~200 Hz) vs. Kinect/MediaPipe (~30 Hz, latência USB + processamento). | Alto | Timestamp sincronizado (NTP/PTP ou hardware trigger). Buffer + interpolação no UKF (multi-rate). |
| **R3** | **Aceleração linear contamina gravidade** — Movimento rápido da mão faz accel ≠ gravidade. Mahony corrige errado. | Alto | Detector de movimento: se $\|\omega\| > \tau_\omega$ ou $\|a\| \not\approx g$, suspender correção accel (só giro). |
| **R4** | **Calibração $T_{\text{IMU}\to\text{cam}}$ frágil** — Pequenos erros de extrínsecos causam viés sistemático na fusão. | Alto | Procedimento estruturado: (1) Movimentos de excitação (rotação pura + translação pura); (2) Mínimos quadrados não-lineares com inliers (RANSAC); (3) Validar com trajetória conhecida. |
| **R5** | **MediaPipe no RGB do Kinect** — Profundidade do Kinect não alinhada perfeitamente com RGB; landmarks 3D = back-projection via depth. Erro de profundidade → erro de posição 3D. | Médio | Usar depth do Kinect alinhado (registration) ou calibrar offset RGB-depth. Comparar Kinect skeleton vs. MediaPipe+profundidade. |
| **R6** | **UKF dimensionalidade** — Estado 10D + ruído de processo/medição. Convergência? Observabilidade de $b_z$ (viés yaw)? | Médio | Iniciar UKF com $P_0$ grande. Testar observabilidade via rank da matriz de observabilidade linearizada. |
| **R7** | **Colunas `FUSED_*` versionadas** — Como versionar sem quebrar `PALM_*`/`IMU_*`? | Baixo (arquitetura) | Schema: `FUSED_v1_{q_w,q_x,q_y,q_z, p_x,p_y,p_z, cov_diag, timestamp, source_mask}`. `source_mask` bits: 0=IMU, 1=MediaPipe, 2=Kinect, 3=geo_constraint. |
| **R8** | **Viés do giroscópio varia com temperatura** — Calibração estática de 45 min (Tian) não captura deriva térmica em sessão longa. | Médio | Calibração rápida no início de cada sessão (30 s parado) + modelo de temperatura se sensor tiver termômetro interno. |

---

## 7. Citações completas

1. **Mahony et al. 2008**  
   Mahony, R., Hamel, T., & Pflimlin, J.-M. (2008). *Complementary Filter Design on the Special Orthogonal Group SO(3)*. IEEE Transactions on Automatic Control, 53(5), 1203–1218. DOI: 10.1109/TAC.2008.923738  
   Arquivo: `literatura/2007_Mahony.etal_TAC-06-396_v3.pdf` (17 pp.)  
   Páginas-chave: 3–8 (teoria), 9–10 (estabilidade passiva), 15–16 (Apêndice B: quaternions).

2. **Markley et al. 2002**  
   Markley, F. L., Cheng, Y., Crassidis, J. L., & Oshman, Y. (2002). *Quaternion Averaging*. Journal of Guidance, Control, and Dynamics, 25(2), 273–278. DOI: 10.2514/2.4904  
   Arquivo: `literatura/Markley_2002_Quaternion_Averaging.pdf` (10 pp.)  
   Páginas-chave: 2–4 (média escalar), 6–7 (pesos matriciais/MLE), 8 (covariância erro).

3. **Tian et al. 2015**  
   Tian, Y., Meng, X., Tao, D., Liu, D., & Feng, C. (2015). *Upper limb motion tracking with the integration of IMU and Kinect*. Neurocomputing, 159, 207–218. DOI: 10.1016/j.neucom.2015.01.071  
   Arquivo: `literatura/1-s2.0-S0925231215001472-main.pdf` (12 pp.)  
   Páginas-chave: 3–4 (UKF IMU), 5 (restrição geométrica), 7–9 (UKF IMU+Kinect, transformações), 10 (resultados).

4. **Acharya et al. 2022**  
   Acharya, A., Bhat, S., & Kanthi, M. (2022). *A Comparative Analysis of Two Approaches for Estimation of Upper Limb Orientation Using Inertial and Kinect Sensors*. Advances in Electrical and Computer Engineering, 22(4), 85–92. DOI: 10.4316/AECE.2022.04010  
   Arquivo: `literatura/aece_2022_4_10.pdf` (8 pp.)  
   Páginas-chave: 2–3 (MPU6050 roll/pitch Eq. 4–8), 4–5 (Kinect setup), 5–6 (estatística Wilcoxon/Spearman), 6 (discussão limitações).

---

## 8. Próximos passos (contratos de calibração)

| Etapa | Descrição | Entradas | Saídas | Critério de aceitação |
|-------|-----------|----------|--------|------------------------|
| **C1** | Calibração viés giroscópio MPU6050 (estático) | IMU parado 60 s, temperatura ambiente | `gyro_bias[3]`, `gyro_noise_std[3]` | Desvio padrão da média < 0.01 rad/s; reprodutível em 3 tentativas |
| **C2** | Calibração extrínseca $T_{\text{IMU}\to\text{Kinect}}$ | Movimentos de excitação (rotação 360° + translação 0.5 m) sincronizados IMU + Kinect | `R_IMU_cam (3,3)`, `t_IMU_cam (3,)` | Erro de reprojeção médio < 5 mm em validação cruzada |
| **C3** | Montante $f$: pose da palma no frame IMU | Medição física (régua) + alinhamento visual + otimização via restrição geométrica | `R_palm_IMU (3,3)`, `t_palm_IMU (3,)` | Restrição geométrica (Sec. 5.5) residual < 2 mm / 2° |
| **C4** | Matrizes de ruído UKF ($Q, R$) | Dados de validação (trajetória conhecida via Qualisys ou marcador AprilTag) | `Q_proc (10,10)`, `R_meas_kinect (3,3)`, `R_meas_mp (3,3)` | NIS (Normalized Innovation Squared) dentro de intervalo $\chi^2$ 95% |
| **C5** | Validação fusão IMU+MediaPipe (offline) | Dataset sincronizado IMU + MediaPipe 3D + ground truth | Métricas: RMSE posição, RMSE orientação (geodésica), deriva yaw/h | RMSE pos < 20 mm, RMSE ori < 5°, deriva yaw < 2°/min |

---

*Fim do relatório A1. Arquivo salvo em `C:\mestradopy\literatura\relatorios\FASE1_A1_fusao_calibracao.md`.*