"""Teste do suporte a MULTIPLAS webcams auxiliares (Kinect + laptop + USB).

Sem hardware: cria arquivos de calibracao sinteticos para a camera auxiliar
de indice 1 (stereo_calibration_aux1.npz + hand_landmark_calibration_aux1.npz),
monkeypatcheia os resolvers de arquivo do modulo e verifica:

1. CameraAlignment registra as cameras 0 e 1 em aux_models.
2. aux_ready/aux_usable respondem por indice (0 usa o arquivo legado real).
3. set_active_aux troca os campos vivos (matriz/rotação/landmarks) e o
   recalculo de runtime por camera fica guardado por indice.
4. Indice nao registrado -> KeyError.
"""
import sys
from pathlib import Path

import numpy as np

import kinect_imu_groundtruth as gt

TMP_DIR = Path(__file__).with_name("_aux_test")
TMP_DIR.mkdir(exist_ok=True)
STF = TMP_DIR / "stereo_calibration_aux1.npz"
HCF = TMP_DIR / "hand_landmark_calibration_aux1.npz"

# --- arquivos sinteticos da camera 1 ---
matrix = np.array([[900.0, 0.0, 630.0], [0.0, 895.0, 360.0],
                   [0.0, 0.0, 1.0]])
np.savez(
    STF,
    kinect_matrix=np.eye(3) * 1000.0 + np.array([[0, 0, 959], [0, 0, 539],
                                                 [0, 0, 1]]) * 0.0,
    kinect_dist=np.zeros(5),
    auxiliary_matrix=matrix,
    auxiliary_dist=np.zeros(5),
    rotation=np.eye(3),
    translation=np.array([-0.22, 0.0, 0.01]),
    checkerboard_size=np.array([8, 6]),
    square_size_m=0.025,
    rms=0.4,
    auxiliary_size=np.array([1280, 720]),
    kinect_size=np.array([1920, 1080]),
)
np.savez(
    HCF,
    version=gt.HAND_CALIBRATION_VERSION,
    homography=np.eye(3),
    residuals=np.zeros((21, 2)),
    rms=0.01,
    scale_depth_params=np.array([1.5]),
    reference_scale=0.3,
    node_z_offset=np.zeros(21),
)

original_stereo = gt.aux_stereo_file
original_hand = gt.aux_hand_file


def fake_stereo(index):
    if int(index) == 1:
        return STF
    return original_stereo(int(index))


def fake_hand(index):
    if int(index) == 1:
        return HCF
    return original_hand(int(index))


gt.aux_stereo_file = fake_stereo
gt.aux_hand_file = fake_hand

try:
    alignment = gt.CameraAlignment()

    # 1) cameras registradas
    indices = alignment.aux_indices()
    assert 0 in indices and 1 in indices, indices

    # 2) prontidão por indice
    assert alignment.aux_ready(1), "camera 1 deveria estar pronta"
    assert alignment.aux_usable(1), "camera 1 deveria ter stereo utilizavel"

    # 3) troca de ativa
    alignment.set_active_aux(1)
    assert alignment.active_aux == 1
    assert np.allclose(alignment.auxiliary_matrix, matrix)
    assert alignment.calibrated_auxiliary_size == (1280, 720)
    assert alignment.hand_landmark_homography is not None
    # omita o stereo para camara 0 (legado) nao existe? verifica so a chave
    alignment.set_active_aux(0)
    assert alignment.active_aux == 0

    # runtime por camera: tamanho diferente da calibracao -> matriz escalada
    alignment.set_active_aux(1)
    small = np.zeros((400, 640, 3), np.uint8)      # 640x400 != 1280x720
    alignment.set_auxiliary_size(small)
    assert alignment.auxiliary_matrix_runtime is not None
    assert alignment.aux_models[1]["runtime_matrix"] is not None
    assert abs(alignment.auxiliary_matrix_runtime[0, 0]
               - matrix[0, 0] * (640 / 1280)) < 1e-6
    alignment.set_active_aux(1)                     # runtime restaurado
    assert alignment.auxiliary_matrix_runtime is not None

    # 4) indice invalido
    try:
        alignment.set_active_aux(99)
        raise AssertionError("indice 99 nao registrado deveria falhar")
    except KeyError:
        pass

    # 5) lista de indices PERSONALIZADA: ex. (1, 2) quando a webcam do laptop
    # fica no lugar do Kinect (sem baseline) e nao entra no par estereo
    custom = gt.CameraAlignment(aux_indices=(1, 2))
    assert custom.aux_indices() == [1, 2], custom.aux_indices()
    assert 0 not in custom.aux_indices()
    assert custom.aux_ready(1), "camera 1 deveria carregar do arquivo _aux1"
    custom.set_active_aux(2)
    assert custom.active_aux == 2

    print("CAMERAS_MULTI_OK: 5/5 (laptop + usb + kinect via calibracao por "
          "indice; lista personalizada suportada)")
finally:
    gt.aux_stereo_file = original_stereo
    gt.aux_hand_file = original_hand
    STF.unlink(missing_ok=True)
    HCF.unlink(missing_ok=True)
    TMP_DIR.rmdir()
sys.exit(0)