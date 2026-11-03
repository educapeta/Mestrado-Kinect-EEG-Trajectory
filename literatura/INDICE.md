# Literatura do projeto -- Fase 0

Pasta local para os PDFs e metadados do plano de fusão
**Kinect v2 + MediaPipe + IMU (MPU6050/Mahony)**. Os PDFs são para estudo local
e **não são versionados** (direitos autorais).

## PDFs baixados e verificados

| Arquivo | Referência | DOI/URL | Uso no projeto |
|---|---|---|---|
| `Markley_2002_Quaternion_Averaging.pdf` | Markley, F. L. (2002). Quaternion Averaging. NASA. | https://ntrs.nasa.gov/citations/20070017872 | média de rotações/quaternions |
| `MediaPipeHands_2020_Zhang.pdf` | Zhang et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. | https://arxiv.org/abs/2006.10214 | 21 landmarks, índices e pipeline |
| `2007_Mahony.etal_TAC-06-396_v3.pdf` | Mahony, Hamel & Pflimlin (2008). Nonlinear Complementary Filters on the Special Orthogonal Group. IEEE TAC 53(5). | 10.1109/TAC.2008.923738 | filtro de atitude; por que yaw de 6 eixos deriva |
| `1-s2.0-S0925231215001472-main.pdf` | Tian et al. (2015). Upper limb motion tracking with the integration of IMU and Kinect. Neurocomputing. | 10.1016/j.neucom.2015.01.071 | arquitetura UKF Kinect+IMU |
| `aece_2022_4_10.pdf` | Acharya, Bhat & Kanthi (2022). A Comparative Analysis of Two Approaches for Estimation of Upper Limb Orientation Using Inertial and Kinect Sensors. AECE 22(4). | 10.4316/AECE.2022.04010 | comparação MPU6050 x Kinect |
| `applsci-15-10527.pdf` | Kusunose et al. (2025). Estimation of Forearm Pronation-Supination Angles Using MediaPipe and IMU Sensors. Appl. Sci. 15, 10527. | 10.3390/app151910527 | MediaPipe + IMU para pronação/supinação |
| `Xefteris_2025_Multimodal_IMU_RGB_pose.pdf` | Xefteris et al. Multimodal fusion of inertial sensors and single RGB camera data for 3D human pose estimation. Internet of Things (Elsevier). | (conferir DOI no PDF) | fusão IMU+RGB, arquitetura de fusão |

## PDFs adicionados depois (Fase 0)

| Arquivo | Referência | DOI/URL | Observação |
|---|---|---|---|
| `2-1016-172898355720-23.pdf` | Jaruchokphuriwaj & Varagul (2024). Accurate Hand Pose Estimation Using Depth Camera and IMU. IJMPE 12(7), 20-23. | http://ijmpe.iraj.in/paperdetails/21035 | PDF completo. Setup: RealSense D435 + Witmotion JY61P (não Kinect v2). |
| `Real-time_Joint_Angle_Estimation_using_Mediapipe_Framework_and_Inertial_Sensors.pdf` | Palani & Panigrahi (2022). Real-time Joint Angle Estimation using Mediapipe Framework and Inertial Sensors. IEEE BIBE. | https://www.computer.org/csdl/proceedings-article/bibe/2022/848700a128/1J6hJGkP4nS | PDF completo. |
| `3488162.3488215.pdf` | Lafayette et al. The Virtual Kinect. ACM. | https://doi.org/10.1145/3488162.3488215 | PDF completo; mapeamento Kinect <-> MediaPipe. |

## Ainda faltando / atenção

| Item | Status |
|---|---|
| Tese Polytechnique Montréal "calibration framework for hand pose estimation" | **Não é** o `2022_MarkAsmar.pdf`: esse PDF é "A Computer Vision-Based Automatic Transcription of Guitar Music from RGBD Videos" (PolyPublie 10470), sem relação. A tese exata continua não confirmada. |
| Qualquer referência sem PDF | usar somente abstract/DOI; não pedir ao agente para "ler" o que não está na pasta. |

## Observações críticas

1. **O artigo principal não usa Kinect v2**: usa RealSense D435 e Witmotion JY61P. Os números 2,33 mm / 0,057° / 0,45° / 0,34° são do setup deles; não são meta automática do nosso protótipo.
2. Tian et al. (2015) usa **UKF** com Kinect + IMU; útil para arquitetura, mas o hardware/época é outro.
3. Kusunose et al. (2025) é **pronação/supinação do antebraço**, não orientação da palma; útil para o desenho MediaPipe+IMU e métricas.
4. Os agentes de leitura só devem ser acionados com o PDF na pasta; caso contrário, risco de alucinação.
5. Prioridade Fase 1: Mahony, Markley, Tian, Acharya, Kusunose, Xefteris, MediaPipe; o IJMPE entra quando o PDF for obtido.
