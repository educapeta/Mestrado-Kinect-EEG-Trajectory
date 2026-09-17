"""Validacao sintetica do pipeline de calibracao stereo.

Simula duas cameras (Kinect 1920x1080 e auxiliar 1280x720) com
intrinsecas, distorcao e pose relativa conhecidas, gera observacoes do
tabuleiro em poses variadas, simula a ordem canonica do detector,
injeta trocas de ordem (o bug que elevava o RMS stereo a 30-70 px),
adiciona ruido de pixel e verifica:

1. Reproducao do bug: stereoCalibrate com intrinsecas congeladas e
   correspondencias trocadas da RMS alto (como o usuario observava).
2. solve_with_repair corrige as poses trocadas e atinge RMS < 1.5 px
   com dados limpos e com trocas injetadas.
3. R recuperada com erro < 1 grau e T com erro < 10 mm.
4. A projecao 3D do runtime (equivalente a auxiliary_to_kinect_stereo)
   reprojeta a menos de 1 px.

Execute: .\\.venv\\Scripts\\python.exe test_stereo_solve.py
"""

import cv2
import numpy as np

import stereo_calibration as sc

KINECT_SIZE = (1920, 1080)
AUX_SIZE = (1280, 720)
K1_GT = np.array([[1090.0, 0, 960], [0, 1090, 540], [0, 0, 1]])
D1_GT = np.array([-0.12, 0.05, 0.001, -0.001, 0.0])
K2_GT = np.array([[820.0, 0, 640], [0, 820, 360], [0, 0, 1]])
D2_GT = np.array([-0.15, 0.06, 0.002, -0.002, 0.0])
T_GT = np.array([[0.35], [-0.04], [0.02]])


def rotation_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Camera auxiliar deslocada 0.35 m para a direita e convergida para a cena
# (yaw de -16 graus), como em um rig real apontando para o mesmo volume.
R_GT = rotation_y(np.radians(-16.0))


def rotation_angle_between(a, b):
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def generate_poses(count, rng):
    """Poses do tabuleiro visiveis nas duas cameras (board -> kinect)."""
    board = sc.board_object_points()
    poses = []
    attempts = 0
    while len(poses) < count and attempts < 800:
        attempts += 1
        center = np.array(
            [rng.uniform(-0.1, 0.2), rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.7)]
        )
        tilt = rotation_x(rng.uniform(-0.55, 0.55)) @ rotation_y(rng.uniform(-0.5, 0.5))
        spin = rotation_z(rng.uniform(0.0, 2.0 * np.pi))
        rot_board = tilt @ spin
        rvec_board, _ = cv2.Rodrigues(rot_board)
        tvec_board = center.reshape(3, 1)
        points_kinect = cv2.projectPoints(board, rvec_board, tvec_board, K1_GT, D1_GT)[0]
        rvec_aux, _ = cv2.Rodrigues(R_GT @ rot_board)
        tvec_aux = R_GT @ tvec_board + T_GT
        points_aux = cv2.projectPoints(board, rvec_aux, tvec_aux, K2_GT, D2_GT)[0]
        if not _visible(points_kinect, KINECT_SIZE) or not _visible(points_aux, AUX_SIZE):
            continue
        poses.append((rvec_board, tvec_board, rvec_aux, tvec_aux))
    if len(poses) < count:
        raise RuntimeError(f"gerou apenas {len(poses)} poses validas")
    return poses


def _visible(points, size):
    width, height = size
    xy = points.reshape(-1, 2)
    return bool(
        np.all(xy[:, 0] > 10)
        and np.all(xy[:, 0] < width - 10)
        and np.all(xy[:, 1] > 10)
        and np.all(xy[:, 1] < height - 10)
        and sc.board_span(xy) >= sc.MIN_BOARD_SPAN_FRACTION * np.hypot(width, height)
    )


def simulate_observations(poses, rng, flip_indices):
    """Detecta (ordem canonica), injeta trocas na auxiliar e adiciona ruido."""
    board = sc.board_object_points()
    object_points, kinect_points, auxiliary_points = [], [], []
    for index, (rvec_k, tvec_k, rvec_a, tvec_a) in enumerate(poses):
        kinect = cv2.projectPoints(board, rvec_k, tvec_k, K1_GT, D1_GT)[0].reshape(-1, 2)
        auxiliary = cv2.projectPoints(board, rvec_a, tvec_a, K2_GT, D2_GT)[0].reshape(-1, 2)
        kinect = sc.canonical_order(kinect)[1]
        auxiliary = sc.canonical_order(auxiliary)[1]
        if index in flip_indices:
            name = rng.choice(["rot180", "hflip", "vflip"])
            auxiliary = auxiliary[sc.symmetry_indices()[name]]
        kinect = kinect + rng.normal(0.0, 0.3, kinect.shape)
        auxiliary = auxiliary + rng.normal(0.0, 0.3, auxiliary.shape)
        object_points.append(board.copy())
        kinect_points.append(kinect.reshape(-1, 1, 2))
        auxiliary_points.append(auxiliary.reshape(-1, 1, 2))
    return object_points, kinect_points, auxiliary_points


def aux_to_kinect_stereo_px(aux_pixel, target_z, result):
    """Equivalente em numpy de CameraAlignment.auxiliary_to_kinect_stereo."""
    normalized = cv2.undistortPoints(
        np.asarray(aux_pixel, np.float64).reshape(1, 1, 2),
        result["auxiliary_matrix"],
        result["auxiliary_dist"],
    ).reshape(2)
    ray = np.array([normalized[0], normalized[1], 1.0])
    rotation = np.asarray(result["rotation"], np.float64)
    translation = np.asarray(result["translation"], np.float64).reshape(3)
    rotation_inverse = rotation.T
    direction = rotation_inverse @ ray
    offset = -rotation_inverse @ translation
    scale = (target_z - offset[2]) / direction[2]
    point_kinect = rotation_inverse @ (scale * ray - translation)
    pixel, _ = cv2.projectPoints(
        point_kinect.reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        result["kinect_matrix"],
        result["kinect_dist"],
    )
    return pixel.reshape(2), point_kinect


def run_scenario(name, object_points, kinect_points, auxiliary_points, poses, rng, flips):
    print(f"\n=== {name} ===")
    broken = sc.run_stereo(
        object_points, kinect_points, auxiliary_points,
        KINECT_SIZE, AUX_SIZE, refine_intrinsics=False,
    )
    print(f"Sem reparo (bug original): RMS stereo={broken['stereo_rms']:.2f} px")
    final, name_map, active, errors, _, _, _ = sc.solve_with_repair(
        object_points, kinect_points, auxiliary_points, KINECT_SIZE, AUX_SIZE
    )
    print(
        f"Com reparo: RMS stereo={final['stereo_rms']:.2f} px | "
        f"RMS Kinect={final['kinect_rms']:.2f} px | RMS auxiliar={final['auxiliary_rms']:.2f} px"
    )
    print(
        "Simetrias aplicadas: "
        + ", ".join(f"pose {i + 1}:{name_map[i]}" for i in sorted(name_map) if name_map[i] != "id")
    )
    rot_gt = R_GT
    angle = rotation_angle_between(rot_gt, final["rotation"])
    translation_error = float(np.linalg.norm(final["translation"].reshape(3) - T_GT.reshape(3)))
    print(f"Erro de rotacao: {angle:.3f} graus | erro de translacao: {translation_error * 1000:.2f} mm")

    # projecao 3D do runtime (autoconsistencia do modelo estimado, que e o
    # que a sobreposicao usa: pixel da auxiliar + Z do Kinect -> pixel Kinect)
    board = sc.board_object_points()
    rvec_k, tvec_k, _, _ = poses[2]
    sample = board.reshape(-1, 3)[[0, 10, 27, 47]]
    points_kinect = sample @ cv2.Rodrigues(rvec_k)[0].T + tvec_k.reshape(1, 3)
    rot_est = np.asarray(final["rotation"], np.float64)
    trans_est = np.asarray(final["translation"], np.float64).reshape(1, 3)
    points_aux = points_kinect @ rot_est.T + trans_est
    pixels_aux = cv2.projectPoints(
        points_aux.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        final["auxiliary_matrix"], final["auxiliary_dist"],
    )[0].reshape(-1, 2)
    pixels_kinect_true = cv2.projectPoints(
        points_kinect.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        final["kinect_matrix"], final["kinect_dist"],
    )[0].reshape(-1, 2)
    reproject = []
    for pixel_aux, point_kinect, pixel_true in zip(pixels_aux, points_kinect, pixels_kinect_true):
        pixel_est, point_est = aux_to_kinect_stereo_px(pixel_aux, point_kinect[2], final)
        reproject.append(np.linalg.norm(pixel_est - pixel_true))
        assert np.linalg.norm(point_est - point_kinect) < 3e-3, "ponto 3D reconstruido errado"
    print(f"Reprojecao 3D do runtime: max {max(reproject):.3f} px")

    assert final["stereo_rms"] < 1.5, f"RMS stereo alto: {final['stereo_rms']:.2f}"
    assert angle < 1.0, f"erro de rotacao grande: {angle:.2f} graus"
    assert translation_error < 0.010, f"erro de translacao grande: {translation_error * 1000:.1f} mm"
    assert max(reproject) < 1.0, "reprojecao 3D do runtime acima de 1 px"
    print("OK")
    return broken, final


def main():
    rng = np.random.default_rng(42)
    poses = generate_poses(16, rng)
    flip_indices = set(rng.choice(16, size=5, replace=False).tolist())
    print(f"Poses geradas: {len(poses)} | trocas de ordem injetadas nas poses: "
          + ", ".join(str(i + 1) for i in sorted(flip_indices)))

    obj, kin, aux = simulate_observations(poses, rng, set())
    run_scenario("dados limpos (sem trocas)", obj, kin, aux, poses, rng, set())

    obj, kin, aux = simulate_observations(poses, rng, flip_indices)
    broken, final = run_scenario("dados com trocas de ordem (bug reproduzido e reparado)", obj, kin, aux, poses, rng, flip_indices)
    assert broken["stereo_rms"] > 5.0, (
        "esperava RMS alto sem reparo (reproducao do bug); "
        f"obteve {broken['stereo_rms']:.2f} px"
    )
    print("\nTodos os cenarios passaram.")


if __name__ == "__main__":
    main()

