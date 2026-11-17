"""Teste sintetico da triangulacao stereo (espaco vetorial 3D).

Gera uma mao 3D sintetica (21 nos, metros), projeta nas duas cameras usando
uma calibracao stereo conhecida (R, T, K, dist), adiciona ruido de deteccao
tipico do MediaPipe (~0.4 px) e verifica que triangulate_hand_points recupera
os nos 3D com erro sub-centrimetrico e reprojecao ~1 px.

Tambem valida contra a calibracao stereo REAL (stereo_calibration.npz), se
existir: triangula pontos de tabuleiro sinteticos e checa erro de reprojecao.
"""
import sys

import cv2
import numpy as np

from kinect_imu_groundtruth import CameraAlignment


def make_synthetic_alignment():
    al = CameraAlignment()
    al.kinect_matrix = np.array(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
    )
    al.kinect_dist = np.zeros((1, 5))
    al.auxiliary_matrix = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
    )
    al.auxiliary_dist = np.zeros((1, 5))
    # Rotacao composta ~8 graus em X e ~5 em Y (ortogonal por construcao)
    ax, ay = np.deg2rad(8.0), np.deg2rad(5.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    al.stereo_rotation = Ry @ Rx
    al.stereo_translation = np.array([[0.35], [0.02], [0.01]])  # baseline 35 cm
    al.stereo_rms = 1.4  # < STEREO_OVERLAY_MAX_RMS_PX
    return al


def synthetic_hand(rng, palm=(0.0, 0.0, 1.1), spread=0.10):
    pts = np.asarray(palm, np.float64) + rng.normal(0.0, spread * 0.5, size=(21, 3))
    pts[0] = palm  # punho fixo
    return pts


def project_kinect(al, pts3d):
    p, _ = cv2.projectPoints(
        pts3d.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        al.kinect_matrix, al.kinect_dist,
    )
    return p.reshape(-1, 2)


def project_aux(al, pts3d_kin):
    X_aux = (np.asarray(al.stereo_rotation) @ pts3d_kin.T).T + np.asarray(
        al.stereo_translation
    ).reshape(3)
    p, _ = cv2.projectPoints(
        X_aux.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        al.auxiliary_matrix, al.auxiliary_dist,
    )
    return p.reshape(-1, 2)


def main():
    rng = np.random.default_rng(7)
    al = make_synthetic_alignment()

    # --- 1) Mao sintetica: ruido tipico de deteccao ------------------------
    pts3d = synthetic_hand(rng)
    kin_px = project_kinect(al, pts3d) + rng.normal(0, 0.4, size=(21, 2))
    aux_px = project_aux(al, pts3d) + rng.normal(0, 0.4, size=(21, 2))
    tri, reproj = al.triangulate_hand_points(aux_px, kin_px)
    assert tri is not None, "triangulacao falhou com dados validos"
    err3d = np.linalg.norm(tri - pts3d, axis=1)
    print(f"erro 3D mediano: {np.median(err3d) * 1000:.2f} mm | "
          f"max: {np.nanmax(err3d) * 1000:.2f} mm")
    print(f"reprojecao mediana: {np.median(reproj):.2f} px | "
          f"max: {np.nanmax(reproj):.2f} px")
    assert np.median(err3d) < 0.005, "erro 3D > 5 mm com ruido 0.4 px"
    assert np.median(reproj) < 1.5, "reprojecao > 1.5 px"
    # Z da palma deve ser recuperado com precisao sub-cm (o problema da
    # inclinacao desaparece: profundidade MEDIDA, nao estimada)
    assert abs(tri[9, 2] - pts3d[9, 2]) < 0.010, "Z da palma errado > 1 cm"

    # --- 2) Ruido maior (deteccao ruim em uma vista) ------------------------
    kin_px2 = project_kinect(al, pts3d) + rng.normal(0, 2.0, size=(21, 2))
    aux_px2 = project_aux(al, pts3d) + rng.normal(0, 2.0, size=(21, 2))
    tri2, reproj2 = al.triangulate_hand_points(aux_px2, kin_px2)
    assert tri2 is not None
    err3d2 = np.linalg.norm(tri2 - pts3d, axis=1)
    print(f"ruido 2.0 px -> erro 3D mediano: {np.median(err3d2) * 1000:.2f} mm | "
          f"reproj mediana: {np.median(reproj2):.2f} px")
    assert np.median(err3d2) < 0.030, "erro 3D > 3 cm com ruido 2 px"

    # --- 3) Entradas invalidas -> (None, None) ------------------------------
    assert al.triangulate_hand_points(aux_px[:5], kin_px) == (None, None)
    trash = project_kinect(al, np.array([[0.0, 0.0, 9.0]] * 21))  # fora do alcance
    tri3, _ = al.triangulate_hand_points(project_aux(al, np.array([[0.0, 0.0, 9.0]] * 21)), trash)
    assert tri3 is None, "deveria rejeitar Z fora do volume fisico"

    # --- 4) Calibracao stereo REAL (se existir) -----------------------------
    try:
        data = np.load("stereo_calibration.npz")
        al.stereo_rotation = data["rotation"]
        al.stereo_translation = data["translation"]
        al.kinect_matrix = data["kinect_matrix"]
        al.kinect_dist = data["kinect_dist"]
        al.auxiliary_matrix = data["auxiliary_matrix"]
        al.auxiliary_dist = data["auxiliary_dist"]
        al.stereo_rms = float(data["rms"])
        pts3d = synthetic_hand(rng, palm=(0.05, -0.1, 1.2))
        kin_px = project_kinect(al, pts3d) + rng.normal(0, 0.4, size=(21, 2))
        aux_px = project_aux(al, pts3d) + rng.normal(0, 0.4, size=(21, 2))
        tri4, reproj4 = al.triangulate_hand_points(aux_px, kin_px)
        assert tri4 is not None
        err3d4 = np.linalg.norm(tri4 - pts3d, axis=1)
        print(f"CALIBRACAO REAL -> erro 3D mediano: {np.median(err3d4) * 1000:.2f} mm | "
              f"reproj mediana: {np.median(reproj4):.2f} px")
        assert np.median(err3d4) < 0.010, "erro 3D > 1 cm com a calibracao real"
    except FileNotFoundError:
        print("stereo_calibration.npz ausente; teste real pulado")

    print("TESTE TRIANGULACAO 3D: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
