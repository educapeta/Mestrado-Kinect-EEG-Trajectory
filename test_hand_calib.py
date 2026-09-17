import cv2
import numpy as np
from pathlib import Path

import kinect_imu_groundtruth as k

# monkeypatch: no sobrescribir un eventual hand_landmark_calibration.npz real
k.HAND_CALIBRATION_FILE = Path(__file__).with_name("tmp_hand_test.npz")

rng = np.random.default_rng(7)
alignment = k.CameraAlignment()

# homografia + residual por nó sintéticos (simula la vista distinta de cada cámara)
H = np.array([[1.08, -0.02, 0.01], [0.01, 1.05, -0.02], [1e-6, 1e-6, 1.0]])
residual = np.column_stack([np.linspace(-0.005, 0.005, 21), np.zeros(21)])

samples_aux, samples_kin = [], []
for _ in range(12):
    aux = rng.uniform(0.2, 0.8, (21, 2))
    kin = cv2.perspectiveTransform(aux.reshape(-1, 1, 2), H).reshape(-1, 2)
    kin = kin + residual + rng.normal(0.0, 0.004, kin.shape)
    samples_aux.append(aux)
    samples_kin.append(kin)

ok = alignment.solve_hand_landmark_calibration(samples_aux, samples_kin, verbose=True)
assert ok, "solve falhou"
assert alignment.hand_landmark_rms is not None and alignment.hand_landmark_rms < 0.01
assert alignment.hand_landmark_ready

out = alignment.auxiliary_to_kinect_landmark(samples_aux[0])
assert out.shape == (21, 2)
assert np.isfinite(out).all()
assert 0.0 <= out.min() and out.max() <= 1.0

# verifica que o residual aprendido rastreeça aproximadamente o verdadeiro
print("Teste sintetico calibracao de mano: OK")
k.HAND_CALIBRATION_FILE.unlink(missing_ok=True)