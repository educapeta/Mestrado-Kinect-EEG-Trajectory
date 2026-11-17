"""Teste da ZERAGEM do IMU pelas cameras (PositionFusion).

Usa a implementacao de kinect_imu_groundtruth.PositionFusion (a mesma usada
pelo Programa 1) com amostras sinteticas de IMU e camera. Verifica:

1. Repouso + camera estavel -> zero_lock, posicao ancorada na camera e
   velocidade zerada (a integracao nao "foge" mesmo com acel ruidosa).
2. Com a mao parada, o bias do acelerometro e ESTIMADO (EMA) e corrigido.
3. Camera em movimento -> sem lock, posicao puxada pelo ganho (nao ancorada).
4. Camera perdida -> com bias corrigido a deriva e menor que sem correcao.
"""
import sys
from types import SimpleNamespace

import numpy as np

import kinect_imu_groundtruth as gt

FS = 50.0
DT = 1.0 / FS
G = 9.80665


def imu_sample(timestamp_us, accel_g=(0.0, 0.0, 1.0), rpy=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        timestamp_us=int(timestamp_us), roll_deg=float(rpy[0]),
        pitch_deg=float(rpy[1]), yaw_deg=float(rpy[2]),
        accel_g=np.asarray(accel_g, np.float64),
        gyro_dps=np.zeros(3))


CAMERA = np.array([0.20, 1.00, 1.40])   # posicao fixa (mao parada)


def stable_run(fusion, n=gt.IMU_ZERO_WINDOW + 5, camera=CAMERA,
               accel=(0.0, 0.0, 1.0), start_us=100_000):
    """Alimenta n frames de IMU (a 50 Hz) + camera fixa; devolve resultado."""
    for index in range(n):
        sample = imu_sample(start_us + index * int(1e6 / FS), accel)
        fusion.update(sample, camera)
    return fusion


# ---------------------------------------------------------------------------
# 1) Repouso estavel com camera: ancoragem espacial + zero de velocidade
# ---------------------------------------------------------------------------
fusion = gt.PositionFusion()
stable_run(fusion, n=gt.IMU_ZERO_WINDOW + 2)
assert fusion.zero_lock, "camera estavel deveria travar (zero_lock)"
assert np.allclose(fusion.absolute_position, CAMERA, atol=1e-6), \
    fusion.absolute_position
assert np.linalg.norm(fusion.velocity) < 1e-9, fusion.velocity
# mesmo com acel ruidosa (mao parada mas sensor com ruido), a ancoragem
# mantem a posicao na camera (nada de integracao dupla descontrolada)
fusion = gt.PositionFusion()
noisy = (0.0, 0.0, 1.0 + 0.2)
stable_run(fusion, n=gt.IMU_ZERO_WINDOW * 2, accel=noisy)
assert fusion.zero_lock
assert np.allclose(fusion.absolute_position, CAMERA, atol=1e-6)

# ---------------------------------------------------------------------------
# 2) Bias do acelerometro estimado enquanto a mao esta parada
# ---------------------------------------------------------------------------
fusion = gt.PositionFusion()
stable_run(fusion, n=gt.IMU_ZERO_WINDOW, accel=(0.0, 0.0, 1.0))
# introduz um bias de +0.1 g no eixo z (sensores reais tem ~0.02-0.1 g)
for index in range(80):                     # 80 frames parados com bias
    sample = imu_sample(int(1e6 / FS) + index,
                        accel_g=(0.0, 0.0, 1.1))
    fusion.update(sample, CAMERA)
assert fusion.bias_estimated
assert 0.02 < fusion.accel_bias[2] < 0.1, fusion.accel_bias  # EMA convergindo
assert np.linalg.norm(fusion.accel_bias[:2]) < 0.05           # so no eixo z

# ---------------------------------------------------------------------------
# 3) Camera valida mas MAO EM MOVIMENTO -> sem lock, posicao segue o ganho
# ---------------------------------------------------------------------------
fusion = gt.PositionFusion()
for index in range(gt.IMU_ZERO_WINDOW):
    moving = CAMERA + np.array([0.005 * index, 0.0, 0.0])   # anda 5 mm/frame
    fusion.update(imu_sample(index * int(1e6 / FS)), moving)
assert not fusion.zero_lock, "mao em movimento nao deveria travar"
# sem lock, a posicao SEGUE a camera pelo ganho (nao e ancorada, mas nao foge)
assert fusion.position[0] > 0.04, fusion.position

# ---------------------------------------------------------------------------
# 4) Camera perdida: bias corrigido segura o drift (vs. sem correcao)
# ---------------------------------------------------------------------------
def drift_without_camera(accel_bias, n_frames=25):
    fusion = gt.PositionFusion()
    fusion.accel_bias = np.asarray(accel_bias, np.float64)
    first = imu_sample(0)
    fusion.update(first, None)                  # primeira amostra (dt=0)
    for index in range(1, n_frames):
        sample = imu_sample(index * int(1e6 / FS),
                            accel_g=(0.0, 0.0, 1.0 + 0.05))   # bias 0.05 g
        fusion.update(sample, None)
    return float(np.linalg.norm(fusion.position))

drift_corrected = drift_without_camera([0.0, 0.0, 0.05])
drift_bare = drift_without_camera([0.0, 0.0, 0.0])
assert drift_bare > 0.04, f"deriva sem bias deveria ser grande: {drift_bare}"
assert drift_corrected < drift_bare * 0.2, \
    f"bias corrigido deveria segurar o drift: {drift_corrected} vs {drift_bare}"

print(f"FUSION_ZERO_OK: 4/4 | derivas: {drift_corrected:.4f} m (com bias) vs "
      f"{drift_bare:.4f} m (sem bias)")
sys.exit(0)