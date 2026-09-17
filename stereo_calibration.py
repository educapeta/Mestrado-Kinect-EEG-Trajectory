"""Calibracao stereo Kinect RGB + camera auxiliar.

Tabuleiro: 9 x 7 quadrados, cantos internos 8 x 6, quadrado de 25 mm.
As capturas sao automaticas (2 s com o tabuleiro estavel nas duas
cameras); ESC encerra sem salvar.

Por que esta versao existe (o RMS stereo ficava em 30-70 px com RMS
individual de ~0.5-1.0 px):

1. ORDEM DOS CANTOS. findChessboardCorners/SB devolve os 48 cantos em
   ordem canonica de IMAGEM (o canto mais proximo do canto superior
   esquerdo da imagem primeiro). Como cada camera ve o tabuleiro de um
   angulo diferente, o "primeiro canto" pode corresponder a cantos
   fisicos diferentes nas duas cameras, e a correspondencia muda de
   pose para pose (quando o tabuleiro gira no plano da imagem). Uma
   troca de 180 graus e uma rotacao rigida do tabuleiro: ela e
   absorvida pela pose individual de cada view e por isso o RMS
   individual continua excelente, mas a pose RELATIVA entre as cameras
   fica inconsistente e o RMS stereo explode. Aqui cada captura e
   reordenada de forma canonica e o solver testa as 4 simetrias da
   grade por pose: corrige as poses trocadas e descarta as poses ruins
   (diagnostico por residuo stereo por pose).
2. INTRINSECAS CONGELADAS. CALIB_FIX_INTRINSIC congela fx/fy/cx/cy/
   distorcao obtidos separadamente. Se as poses tiverem pouca variacao
   de distancia/inclinacao (tabuleiro sempre de frente e na mesma
   distancia), essas intrinsecas ficam imprecisas e o erro so aparece
   no stereo. O solver final tenta tambem CALIB_USE_INTRINSIC_GUESS
   (refinamento conjunto de intrinsecas + extrinsecas) e salva o
   resultado com menor RMS.
3. QUALIDADE DAS AMOSTRAS. Capturas com tabuleiro pequeno na imagem
   (foco/escala ruins) ou com erro planar do par alto (movimento,
   deteccao ruim) sao rejeitadas antes de entrar na solucao.
"""

from pathlib import Path

import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime

AUX_CAMERA_INDEX = 0
# Must match the resolution used by kinect_imu_groundtruth.py.
AUX_REQUEST_SIZE = (1280, 720)  # (width, height)
# The raw auxiliary stream arrives mirrored (virtual camera / phone app);
# flipping restores the physical, unmirrored coordinates. Keep the same
# value as AUXILIARY_MIRROR_HORIZONTAL in kinect_imu_groundtruth.py.
AUXILIARY_MIRROR_HORIZONTAL = True
AUXILIARY_DISPLAY_MIRROR = False
CHECKERBOARD_COLS = 8
CHECKERBOARD_ROWS = 6
CHECKERBOARD_SIZE = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)
SQUARE_SIZE_M = 0.025
SAMPLES_REQUIRED = 15
SOLVE_EVERY_EXTRA = 3          # depois das 15, resolve a cada +3 amostras
MAX_STEREO_RMS_PX = 2.0
STABLE_CAPTURE_SECONDS = 2.0
MIN_BOARD_SPAN_FRACTION = 0.08  # tabuleiro >= 8% da diagonal da imagem (duas cams)
MAX_PAIR_ERROR_PX = 3.0         # erro planar homografia aux -> kinect
MAX_POSE_RMS_PX = 4.0           # residuo stereo por pose (limite duro)
OUTLIER_MEDIAN_FACTOR = 2.5     # pior pose vs mediana
MIN_ACTIVE_POSES = 8
RANSAC_SEED_TRIALS = 40         # mini-solves para achar um subconjunto limpo
RANSAC_SEED_SIZE = 5
ASSIGNMENT_ROUNDS = 3
MAX_TOTAL_SAMPLES = 60
DETECTION_INTERVAL_S = 0.1     # detecta cantos a ~10 Hz e reusa entre frames
FALLBACK_EVERY = 5             # detector classico so a cada N deteccoes (custo)
# k1, k2, k3, p1, p2 livres (k3 congelado degradava a precisao nas bordas
# da imagem, onde a distorcao do Kinect e maior).
CALIB_FLAGS = 0
CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
STEREO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
OUTPUT_FILE = Path(__file__).with_name("stereo_calibration.npz")
# Regenerado junto com a calibracao stereo para manter os dois arquivos
# consistentes (a homografia e derivada dos pontos do tabuleiro da
# propria solucao; o refinamento afim recomeca da identidade).
CAMERA_ALIGNMENT_FILE = Path(__file__).with_name("camera_alignment.npz")
CAMERA_ALIGNMENT_VERSION = 7


def output_file(aux_index):
    """Arquivo stereo por camera auxiliar (legado = indice 0)."""
    if int(aux_index) == 0:
        return OUTPUT_FILE
    return Path(__file__).with_name(
        f"stereo_calibration_aux{int(aux_index)}.npz")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--aux-index", type=int, default=0,
                        help="Indice da webcam auxiliar a calibrar "
                             "(0 = laptop, 1 = usb). Salva em "
                             "stereo_calibration_aux{N}.npz (0 usa o arquivo "
                             "legado stereo_calibration.npz).")
    return parser.parse_args()


def find_corners(frame, allow_fallback=True):
    """Detecta os 48 cantos internos (float64, ordem de deteccao).

    O detector SB (rapido e subpixel) roda primeiro; o classico e usado
    apenas como fallback eventual (allow_fallback) porque e caro em
    imagens grandes e dobraria o custo quando o tabuleiro nao esta
    visivel.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            CHECKERBOARD_SIZE,
            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if found:
            return corners.reshape(-1, 2).astype(np.float64)
    if not allow_fallback:
        return None
    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return corners.reshape(-1, 2).astype(np.float64)


def board_object_points():
    board = np.zeros((CHECKERBOARD_COLS * CHECKERBOARD_ROWS, 1, 3), np.float64)
    grid = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    board[:, 0, :2] = grid
    return board * SQUARE_SIZE_M


def symmetry_indices():
    """Mapeia cada simetria da grade para uma permutacao de indices.

    Chaves: 'id' (identidade), 'rot180' (grade girada 180 graus),
    'hflip' (espelhada ao longo do eixo de 8 cantos) e 'vflip'
    (espelhada ao longo do eixo de 6 cantos). O layout dos indices e
    row-major com 8 cantos por linha.
    """
    cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
    rr, cc = np.mgrid[0:rows, 0:cols]
    return {
        "id": np.arange(rows * cols),
        "rot180": (rows * cols - 1) - np.arange(rows * cols),
        "hflip": (rr * cols + (cols - 1 - cc)).reshape(-1),
        "vflip": ((rows - 1 - rr) * cols + cc).reshape(-1),
    }


def canonical_order(corners):
    """Reordena a lista de cantos detectados em um caminho canonico.

    Regra: o primeiro canto e o canto externo mais proximo do canto
    superior esquerdo da imagem; a primeira linha segue o eixo mais
    horizontal a partir do inicio; as linhas avancam para baixo.
    Aplicada as duas cameras antes de calibrar para que os dois
    conjuntos de pontos usem a mesma convencao.
    """
    cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
    points = np.asarray(corners, np.float64).reshape(cols * rows, 2)
    best_name, best_points, best_penalty = "id", points, None
    for name, indices in symmetry_indices().items():
        candidate = points[indices]
        start = candidate[0]
        row_vec = candidate[cols - 1] - start
        col_vec = candidate[cols] - start
        others = np.array([candidate[cols - 1], candidate[cols], candidate[-1]])
        penalty = (
            max(0.0, float(others.sum(axis=1).min()) - float(start.sum()))
            + max(0.0, abs(row_vec[1]) - abs(row_vec[0]))
            + max(0.0, -col_vec[1])
        )
        if best_penalty is None or penalty < best_penalty - 1e-9:
            best_name, best_points, best_penalty = name, candidate, penalty
    return best_name, best_points


def board_span(corners):
    """Maior distancia entre cantos externos do tabuleiro (px)."""
    cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
    points = np.asarray(corners, np.float64).reshape(-1, 2)
    outer = np.array([points[0], points[cols - 1], points[(rows - 1) * cols], points[-1]])
    diffs = outer[:, None, :] - outer[None, :, :]
    return float(np.max(np.linalg.norm(diffs, axis=2)))


def board_tilt_ratio(corners):
    """Razao entre as diagonais do quadrilatero externo.

    ~1.0 significa tabuleiro de frente; valores maiores indicam
    inclinacao (bom para diversidade de poses).
    """
    cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
    points = np.asarray(corners, np.float64).reshape(-1, 2)
    d1 = np.linalg.norm(points[0] - points[-1])
    d2 = np.linalg.norm(points[cols - 1] - points[(rows - 1) * cols])
    return float(max(d1, d2) / max(1e-9, min(d1, d2)))


def pair_planar_error(kinect_corners, auxiliary_corners):
    """Erro medio da homografia planar aux -> kinect.

    Detecta capturas ruins (movimento entre frames, detecco ruim), mas
    NAO detecta troca de ordem dos cantos (a homografia absorve as
    simetrias da grade); por isso a correcao de simetria e feita depois
    pelo residuo stereo 3D em solve_with_repair.
    """
    aux = np.asarray(auxiliary_corners, np.float64).reshape(-1, 1, 2)
    kin = np.asarray(kinect_corners, np.float64).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(aux, kin, cv2.RANSAC, 3.0)
    if homography is None:
        return float("inf")
    projected = cv2.perspectiveTransform(aux, homography).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(kin.reshape(-1, 2) - projected, axis=1)))


def calibrate_single(object_points, image_points, size):
    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        size,
        None,
        None,
        flags=CALIB_FLAGS,
        criteria=CALIB_CRITERIA,
    )
    return rms, matrix, dist, rvecs, tvecs


def _unpack_stereo(output, kinect_matrix, kinect_dist, auxiliary_matrix, auxiliary_dist):
    """Lida com os diferentes tamanhos de tupla de cv2.stereoCalibrate.

    Os bindings Python do OpenCV 4.x/5.x retornam 9 valores:
    (rms, K1, D1, K2, D2, R, T, E, F). Bindings mais antigos retornam
    (rms, R, T, E, F). Com CALIB_USE_INTRINSIC_GUESS os K/D devolvidos
    sao os REFINADOS e devem substituir os de calibrateCamera.
    """
    if len(output) >= 9:
        rms = output[0]
        kinect_matrix, kinect_dist = output[1], output[2]
        auxiliary_matrix, auxiliary_dist = output[3], output[4]
        rotation, translation = output[5], output[6]
    else:
        rms, rotation, translation = output[0], output[1], output[2]
    return (
        rms,
        kinect_matrix,
        kinect_dist,
        auxiliary_matrix,
        auxiliary_dist,
        rotation,
        translation,
    )


def _as_float32_points(points_list):
    """cv2.calibrateCamera/stereoCalibrate exigem Point3f/Point2f."""
    return [np.asarray(points, np.float32) for points in points_list]


def run_stereo(
    object_points,
    kinect_points,
    auxiliary_points,
    kinect_size,
    auxiliary_size,
    refine_intrinsics,
):
    """Roda uma solucao completa (individual + stereo) e devolve um dict."""
    object_points = _as_float32_points(object_points)
    kinect_points = _as_float32_points(kinect_points)
    auxiliary_points = _as_float32_points(auxiliary_points)
    (
        kinect_rms,
        kinect_matrix,
        kinect_dist,
        kinect_rvecs,
        kinect_tvecs,
    ) = calibrate_single(object_points, kinect_points, kinect_size)
    auxiliary_rms, auxiliary_matrix, auxiliary_dist, _, _ = calibrate_single(
        object_points, auxiliary_points, auxiliary_size
    )
    if refine_intrinsics:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | CALIB_FLAGS
    else:
        flags = cv2.CALIB_FIX_INTRINSIC
    output = cv2.stereoCalibrate(
        object_points,
        kinect_points,
        auxiliary_points,
        kinect_matrix,
        kinect_dist,
        auxiliary_matrix,
        auxiliary_dist,
        kinect_size,
        flags=flags,
        criteria=STEREO_CRITERIA,
    )
    (
        stereo_rms,
        kinect_matrix,
        kinect_dist,
        auxiliary_matrix,
        auxiliary_dist,
        rotation,
        translation,
    ) = _unpack_stereo(
        output, kinect_matrix, kinect_dist, auxiliary_matrix, auxiliary_dist
    )
    return {
        "stereo_rms": float(stereo_rms),
        "kinect_rms": float(kinect_rms),
        "auxiliary_rms": float(auxiliary_rms),
        "kinect_matrix": kinect_matrix,
        "kinect_dist": kinect_dist,
        "auxiliary_matrix": auxiliary_matrix,
        "auxiliary_dist": auxiliary_dist,
        "rotation": np.asarray(rotation, np.float64),
        "translation": np.asarray(translation, np.float64),
        "kinect_size": tuple(kinect_size),
        "auxiliary_size": tuple(auxiliary_size),
        "refine_intrinsics": bool(refine_intrinsics),
        "kinect_rvecs": kinect_rvecs,
        "kinect_tvecs": kinect_tvecs,
    }


def _project_camera_points(points_camera, matrix, dist):
    projected, _ = cv2.projectPoints(
        np.asarray(points_camera, np.float64).reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        matrix,
        dist,
    )
    return projected.reshape(-1, 2)


def _single_pose_error(result, object_point, kinect_point, auxiliary_point, rvec, tvec):
    """RMS stereo ( Kinect + auxiliar, px) de uma unica pose."""
    rot_i, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    tvec_i = np.asarray(tvec, np.float64).reshape(1, 3)
    points_kinect = object_point.reshape(-1, 3) @ rot_i.T + tvec_i
    points_aux = points_kinect @ np.asarray(result["rotation"], np.float64).T + np.asarray(
        result["translation"], np.float64
    ).reshape(1, 3)
    kinect_error = np.linalg.norm(
        _project_camera_points(points_kinect, result["kinect_matrix"], result["kinect_dist"])
        - np.asarray(kinect_point, np.float64).reshape(-1, 2),
        axis=1,
    )
    auxiliary_error = np.linalg.norm(
        _project_camera_points(
            points_aux, result["auxiliary_matrix"], result["auxiliary_dist"]
        )
        - np.asarray(auxiliary_point, np.float64).reshape(-1, 2),
        axis=1,
    )
    combined = np.concatenate([kinect_error, auxiliary_error])
    return float(np.sqrt(np.mean(combined ** 2)))


def refit_view_poses(object_points, kinect_points, kinect_matrix, kinect_dist, kinect_size):
    """Poses por view do Kinect consistentes com intrinsecas fixas."""
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_FOCAL_LENGTH
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_TANGENT_DIST
    )
    _, _, _, rvecs, tvecs = cv2.calibrateCamera(
        _as_float32_points(object_points),
        _as_float32_points(kinect_points),
        kinect_size,
        np.asarray(kinect_matrix, np.float64).copy(),
        np.asarray(kinect_dist, np.float64).copy(),
        flags=flags,
        criteria=CALIB_CRITERIA,
    )
    return rvecs, tvecs


def pose_rms(result, object_points, kinect_points, auxiliary_points, rvecs=None, tvecs=None):
    """Residuo stereo por pose (px), combinando as duas cameras."""
    if rvecs is None or tvecs is None:
        rvecs, tvecs = refit_view_poses(
            object_points,
            kinect_points,
            result["kinect_matrix"],
            result["kinect_dist"],
            result["kinect_size"],
        )
    return np.array(
        [
            _single_pose_error(
                result,
                object_points[i],
                kinect_points[i],
                auxiliary_points[i],
                rvecs[i],
                tvecs[i],
            )
            for i in range(len(object_points))
        ],
        dtype=np.float64,
    )


def solve_best(object_points, kinect_points, auxiliary_points, kinect_size, auxiliary_size):
    """Resolve com intrinsecas congeladas e com refinadas; guarda a melhor."""
    candidates = []
    for refine in (False, True):
        try:
            candidates.append(
                run_stereo(
                    object_points,
                    kinect_points,
                    auxiliary_points,
                    kinect_size,
                    auxiliary_size,
                    refine_intrinsics=refine,
                )
            )
        except cv2.error as exc:
            print(f"  modo {'refinado' if refine else 'congelado'} falhou: {exc}")
    if not candidates:
        raise RuntimeError("stereoCalibrate falhou nos dois modos")
    return min(candidates, key=lambda item: item["stereo_rms"])


def solve_with_repair(
    object_points,
    kinect_points,
    auxiliary_points,
    kinect_size,
    auxiliary_size,
    verbose=True,
):
    """Resolve a calibracao stereo reparando trocas de ordem dos cantos.

    Estrategia:
    1. Semente RANSAC: resolve mini-subconjuntos aleatorios de poses
       (simetrias 'id'); um subconjunto sem poses trocadas da RMS ~1 px,
       enquanto qualquer subconjunto contaminado da RMS dezenas de px.
       Com menos de 50% das poses trocadas, algumas dezenas de tentativas
       acham um subconjunto limpo com probabilidade altissima.
    2. Atribuicao por pose: para cada pose, a pose do Kinect e obtida por
       solvePnP (independente do R/T global) e as 4 simetrias da grade sao
       testadas contra o modelo da semente; a simetria correta da erro
       ~1 px e as erradas dezenas de px. Re-resolve e repete.
    3. Poses que continuam ruins mesmo com a melhor simetria sao
       descartadas. Termina com solve final testando os dois modos de
       intrinsecas.
    """
    symmetries = symmetry_indices()
    total = len(object_points)
    name_map = {i: "id" for i in range(total)}
    active = list(range(total))

    def subset(indices):
        obj = [object_points[i] for i in indices]
        kin = [kinect_points[i] for i in indices]
        aux = []
        for i in indices:
            points = np.asarray(auxiliary_points[i], np.float64).reshape(-1, 2)
            aux.append(points[symmetries[name_map[i]]].reshape(-1, 1, 2))
        return obj, kin, aux

    def assign_symmetries(model, indices):
        assignments, errors = {}, {}
        for i in indices:
            found, rvec_i, tvec_i = cv2.solvePnP(
                np.asarray(object_points[i], np.float32),
                np.asarray(kinect_points[i], np.float32),
                model["kinect_matrix"],
                model["kinect_dist"],
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not found:
                assignments[i], errors[i] = "id", float("inf")
                continue
            best_name, best_error = "id", float("inf")
            points = np.asarray(auxiliary_points[i], np.float64).reshape(-1, 2)
            for name, indices_perm in symmetries.items():
                trial = points[indices_perm].reshape(-1, 1, 2)
                error = _single_pose_error(
                    model,
                    object_points[i],
                    kinect_points[i],
                    trial,
                    rvec_i,
                    tvec_i,
                )
                if error < best_error:
                    best_name, best_error = name, error
            assignments[i], errors[i] = best_name, best_error
        return assignments, errors

    # ---- etapa 1: semente RANSAC ----
    seed_size = min(RANSAC_SEED_SIZE, total)
    seed_result, seed_subset, seed_rms = None, None, float("inf")
    if total >= 6:
        rng = np.random.default_rng(12345)
        if verbose:
            print(f"  procurando subconjunto limpo ({RANSAC_SEED_TRIALS} tentativas)...")
        for trial_index in range(RANSAC_SEED_TRIALS):
            if verbose and trial_index and trial_index % 10 == 0:
                print(f"  ... {trial_index}/{RANSAC_SEED_TRIALS}")
            trial = sorted(rng.choice(total, size=seed_size, replace=False).tolist())
            try:
                trial_result = run_stereo(
                    *subset(trial), kinect_size, auxiliary_size, refine_intrinsics=False
                )
            except cv2.error:
                continue
            if trial_result["stereo_rms"] < seed_rms:
                seed_result, seed_subset, seed_rms = trial_result, trial, trial_result["stereo_rms"]
        if verbose and seed_result is not None:
            print(
                f"  semente RANSAC: poses {[i + 1 for i in seed_subset]} "
                f"(RMS {seed_rms:.2f} px)"
            )
    if seed_result is None:
        obj, kin, aux = subset(active)
        seed_result = run_stereo(obj, kin, aux, kinect_size, auxiliary_size, refine_intrinsics=False)

    # ---- etapa 2: atribuicao de simetrias por pose + re-solve ----
    model = seed_result
    for _ in range(ASSIGNMENT_ROUNDS):
        assignments, assignment_errors = assign_symmetries(model, active)
        changed = [
            f"pose {i + 1}:'{assignments[i]}'"
            for i in active
            if assignments[i] != name_map[i]
        ]
        if verbose and changed:
            print("  ordem dos cantos corrigida: " + ", ".join(changed))
        for i in active:
            name_map[i] = assignments[i]
        obj, kin, aux = subset(active)
        model = run_stereo(obj, kin, aux, kinect_size, auxiliary_size, refine_intrinsics=False)

    # ---- etapa 3: descarte de poses ruins ----
    obj, kin, aux = subset(active)
    errors = pose_rms(model, obj, kin, aux, model["kinect_rvecs"], model["kinect_tvecs"])
    limit = max(MAX_POSE_RMS_PX, OUTLIER_MEDIAN_FACTOR * float(np.median(errors)))
    order = np.argsort(errors)[::-1]
    for position in order:
        if errors[position] <= limit or len(active) <= MIN_ACTIVE_POSES:
            break
        dropped = active[position]
        if verbose:
            print(
                f"  pose {dropped + 1} descartada (erro {errors[position]:.2f} px "
                "mesmo apos reordenar os cantos)"
            )
        active.remove(dropped)
        del name_map[dropped]
        obj, kin, aux = subset(active)
        model = run_stereo(obj, kin, aux, kinect_size, auxiliary_size, refine_intrinsics=False)
        errors = pose_rms(model, obj, kin, aux, model["kinect_rvecs"], model["kinect_tvecs"])

    # ---- solve final ----
    obj, kin, aux = subset(active)
    final = solve_best(obj, kin, aux, kinect_size, auxiliary_size)
    errors = pose_rms(final, obj, kin, aux)
    return final, name_map, active, errors, obj, kin, aux


def print_solve_report(final, errors, active, spans, tilt_ratios):
    mode = "intrinsecas refinadas" if final["refine_intrinsics"] else "intrinsecas congeladas"
    print(
        f"RMS stereo={final['stereo_rms']:.4f} px ({mode}) | "
        f"RMS Kinect={final['kinect_rms']:.4f} px | RMS auxiliar={final['auxiliary_rms']:.4f} px"
    )
    for position, index in enumerate(active):
        print(f"  pose {index + 1}: {errors[position]:.2f} px")
    if spans and tilt_ratios:
        span_ratio = max(spans) / max(1e-9, min(spans))
        tilted = sum(1 for value in tilt_ratios if value > 1.2)
        print(
            f"Diversidade: span do tabuleiro max/min = {span_ratio:.2f} | "
            f"poses inclinadas (diag ratio > 1.2): {tilted}/{len(tilt_ratios)}"
        )
        if span_ratio < 1.6:
            print(
                "  AVISO: pouca variacao de distancia do tabuleiro "
                "(aproxime e afaste o tabuleiro entre capturas)."
            )
        if tilted < 5:
            print(
                "  AVISO: poucas poses inclinadas (inclua o tabuleiro 20-45 graus "
                "para os lados em varias capturas)."
            )
    if final["auxiliary_rms"] > 3.0:
        print(
            "  AVISO: RMS individual da auxiliar alto; verifique "
            "AUXILIARY_MIRROR_HORIZONTAL (a imagem pode estar espelhada)."
        )


def print_capture_guidance():
    print("COMO CAPTURAR (o RMS stereo depende muito disso):")
    print("- Segure o tabuleiro de frente para as cameras, sempre com a parte")
    print("  de cima para cima (nao gire o tabuleiro no plano da imagem).")
    print("- Varie a distancia (0.8 a 2.5 m) e a posicao na imagem; todas as")
    print("  regioes da imagem devem aparecer em pelo menos uma captura.")
    print("- Incline o tabuleiro 20-45 graus para os lados/cima/baixo em")
    print("  varias capturas; sem inclinacao o foco fica impreciso.")
    print("- Inclua capturas com o tabuleiro nas bordas e cantos da imagem")
    print("  das duas cameras (o modelo de distorcao precisa delas).")
    print("- Tabuleiro PARADO na captura (as cameras nao sao sincronizadas),")
    print("  imagem nitida e sem reflexo.")


def save_calibration(final, active, errors, obj, kin, aux):
    output = output_file(AUX_CAMERA_INDEX)
    np.savez(
        output,
        kinect_matrix=final["kinect_matrix"],
        kinect_dist=final["kinect_dist"],
        auxiliary_matrix=final["auxiliary_matrix"],
        auxiliary_dist=final["auxiliary_dist"],
        rotation=final["rotation"],
        translation=final["translation"],
        rms=final["stereo_rms"],
        kinect_rms=final["kinect_rms"],
        auxiliary_rms=final["auxiliary_rms"],
        square_size_m=SQUARE_SIZE_M,
        checkerboard_size=CHECKERBOARD_SIZE,
        kinect_size=np.asarray(final["kinect_size"]),
        auxiliary_size=np.asarray(final["auxiliary_size"]),
        per_pose_rms=errors,
        used_poses=np.asarray(active),
        refined_intrinsics=1 if final["refine_intrinsics"] else 0,
    )
    print(
        f"Melhor calibracao salva em {output}; "
        f"RMS stereo={final['stereo_rms']:.4f} px"
    )
    save_alignment(obj, kin, aux)


def save_alignment(obj, kin, aux):
    """Regenera o camera_alignment.npz a partir da calibracao stereo.

    Mantem camera_alignment.npz e stereo_calibration.npz sincronizados:
    a homografia e ajustada (RANSAC) aos pontos do tabuleiro usados na
    propria solucao stereo e o refinamento afim recomeca da identidade.
    """
    aux_pixels = np.concatenate(
        [np.asarray(points, np.float64).reshape(-1, 2) for points in aux]
    )
    kinect_pixels = np.concatenate(
        [np.asarray(points, np.float64).reshape(-1, 2) for points in kin]
    )
    homography, _ = cv2.findHomography(aux_pixels, kinect_pixels, cv2.RANSAC, 3.0)
    if homography is None:
        print("AVISO: nao foi possivel derivar a homografia do alinhamento")
        return
    np.savez(
        CAMERA_ALIGNMENT_FILE,
        homography=homography,
        square_size_m=SQUARE_SIZE_M,
        version=CAMERA_ALIGNMENT_VERSION,
        refinement_affine=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
    )
    print(f"Alinhamento (homografia) regenerado em {CAMERA_ALIGNMENT_FILE}")


def main():
    global AUX_CAMERA_INDEX
    args = parse_args()
    AUX_CAMERA_INDEX = int(args.aux_index)
    print(f"Calibrando a webcam auxiliar de indice {AUX_CAMERA_INDEX} -> "
          f"{output_file(AUX_CAMERA_INDEX)}")
    print_capture_guidance()
    kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    auxiliary = cv2.VideoCapture(AUX_CAMERA_INDEX)
    auxiliary.set(cv2.CAP_PROP_FRAME_WIDTH, AUX_REQUEST_SIZE[0])
    auxiliary.set(cv2.CAP_PROP_FRAME_HEIGHT, AUX_REQUEST_SIZE[1])
    aux_width = int(auxiliary.get(cv2.CAP_PROP_FRAME_WIDTH))
    aux_height = int(auxiliary.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kinect_width = kinect.color_frame_desc.Width
    kinect_height = kinect.color_frame_desc.Height
    kinect_size = (kinect_width, kinect_height)
    auxiliary_size = (aux_width, aux_height)
    print(
        f"Resolucao Kinect RGB: {kinect_width}x{kinect_height} | "
        f"auxiliar: {aux_width}x{aux_height}"
    )
    board = board_object_points()
    object_points = []
    kinect_points = []
    auxiliary_points = []
    kinect_spans = []
    tilt_ratios = []
    kinect_diagonal = float(np.hypot(kinect_width, kinect_height))
    auxiliary_diagonal = float(np.hypot(aux_width, aux_height))
    stable_since = None
    best_rms = float("inf")
    last_solved = 0
    last_detect = 0.0
    detection_count = 0
    cached_kinect_corners = None
    cached_auxiliary_corners = None
    try:
        while True:
            kinect_frame = None
            if kinect.has_new_color_frame():
                data = kinect.get_last_color_frame()
                kinect_frame = data.reshape((kinect_height, kinect_width, 4))
                kinect_frame = cv2.cvtColor(kinect_frame, cv2.COLOR_BGRA2BGR)
            auxiliary_ok, auxiliary_frame = auxiliary.read()
            if AUXILIARY_MIRROR_HORIZONTAL and auxiliary_ok:
                auxiliary_frame = cv2.flip(auxiliary_frame, 1)
            if kinect_frame is None or not auxiliary_ok:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            now = cv2.getTickCount() / cv2.getTickFrequency()
            # Detecta a ~10 Hz e reusa os cantos entre frames: SB+fallback
            # em 1080p+720p por frame deixaria a interface inviavel.
            if now - last_detect >= DETECTION_INTERVAL_S:
                allow_fallback = detection_count % FALLBACK_EVERY == 0
                kinect_corners = find_corners(kinect_frame, allow_fallback=allow_fallback)
                auxiliary_corners = find_corners(auxiliary_frame, allow_fallback=allow_fallback)
                cached_kinect_corners = kinect_corners
                cached_auxiliary_corners = auxiliary_corners
                last_detect = now
                detection_count += 1
            else:
                kinect_corners = cached_kinect_corners
                auxiliary_corners = cached_auxiliary_corners
            kinect_found = kinect_corners is not None
            auxiliary_found = auxiliary_corners is not None
            kinect_display = kinect_frame.copy()
            auxiliary_display = auxiliary_frame.copy()
            if kinect_found:
                cv2.drawChessboardCorners(
                    kinect_display,
                    CHECKERBOARD_SIZE,
                    kinect_corners.astype(np.float32).reshape(-1, 1, 2),
                    True,
                )
            if auxiliary_found:
                cv2.drawChessboardCorners(
                    auxiliary_display,
                    CHECKERBOARD_SIZE,
                    auxiliary_corners.astype(np.float32).reshape(-1, 1, 2),
                    True,
                )
            if AUXILIARY_DISPLAY_MIRROR:
                auxiliary_display = cv2.flip(auxiliary_display, 1)
            if kinect_found and auxiliary_found and stable_since is not None:
                stable_text = f"estavel {max(0.0, now - stable_since):.1f}/2.0 s"
            else:
                stable_text = "aguardando ambos"
            status = (
                f"Kinect: {'OK' if kinect_found else 'falhou'} | "
                f"Aux: {'OK' if auxiliary_found else 'falhou'} | "
                f"{len(object_points)}/{SAMPLES_REQUIRED} | {stable_text}"
            )
            for display in (kinect_display, auxiliary_display):
                cv2.putText(
                    display,
                    status,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if kinect_found and auxiliary_found else (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display,
                    "mantenha ambos por 2 s | ESC encerra",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            cv2.imshow("Stereo - Kinect RGB", kinect_display)
            cv2.imshow("Stereo - camera auxiliar", auxiliary_display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if not (kinect_found and auxiliary_found):
                stable_since = None
                continue
            if stable_since is None:
                stable_since = now
            if now - stable_since < STABLE_CAPTURE_SECONDS:
                continue
            stable_since = now
            kinect_span = board_span(kinect_corners)
            auxiliary_span = board_span(auxiliary_corners)
            if (
                kinect_span < MIN_BOARD_SPAN_FRACTION * kinect_diagonal
                or auxiliary_span < MIN_BOARD_SPAN_FRACTION * auxiliary_diagonal
            ):
                print("Tabuleiro pequeno na imagem; aproxime o tabuleiro das cameras.")
                continue
            pair_error = pair_planar_error(kinect_corners, auxiliary_corners)
            if pair_error > MAX_PAIR_ERROR_PX:
                print(
                    f"Par rejeitado (erro planar {pair_error:.2f} px > "
                    f"{MAX_PAIR_ERROR_PX:.1f}); mantenha o tabuleiro PARADO e repita."
                )
                continue
            _, kinect_canonical = canonical_order(kinect_corners)
            _, auxiliary_canonical = canonical_order(auxiliary_corners)
            object_points.append(board.copy())
            kinect_points.append(kinect_canonical.reshape(-1, 1, 2))
            auxiliary_points.append(auxiliary_canonical.reshape(-1, 1, 2))
            kinect_spans.append(kinect_span)
            tilt_ratios.append(board_tilt_ratio(kinect_corners))
            print(
                f"Amostra aceita: {len(object_points)}/{SAMPLES_REQUIRED} "
                f"(par {pair_error:.2f} px, span {kinect_span / kinect_diagonal:.0%} da imagem)"
            )


            total = len(object_points)
            if total < SAMPLES_REQUIRED:
                continue
            if total != SAMPLES_REQUIRED and total - last_solved < SOLVE_EVERY_EXTRA:
                continue
            last_solved = total
            print(f"Resolvendo com {total} poses...")
            final, name_map, active, errors, final_obj, final_kin, final_aux = solve_with_repair(
                object_points, kinect_points, auxiliary_points, kinect_size, auxiliary_size
            )
            print_solve_report(final, errors, active, kinect_spans, tilt_ratios)
            if final["stereo_rms"] < best_rms:
                best_rms = final["stereo_rms"]
                save_calibration(final, active, errors, final_obj, final_kin, final_aux)
            if final["stereo_rms"] <= MAX_STEREO_RMS_PX:
                print(f"Calibracao aceita; RMS stereo={final['stereo_rms']:.4f} px")
                break
            if total >= MAX_TOTAL_SAMPLES:
                print(
                    f"Limite de {MAX_TOTAL_SAMPLES} amostras atingido com RMS "
                    f"{final['stereo_rms']:.2f} px; a melhor foi salva. Revise a "
                    "variedade das poses (distancia, inclinacao, posicao) e recomece."
                )
                break
            print(
                f"Calibracao ainda acima do limite ({MAX_STEREO_RMS_PX:.1f} px); "
                "coletando mais amostras (varie distancia e inclinacao!)"
            )
    finally:
        auxiliary.release()
        kinect.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
