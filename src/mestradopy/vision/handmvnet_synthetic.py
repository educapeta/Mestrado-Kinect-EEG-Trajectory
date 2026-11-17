"""Frames sinteticos para validar o adaptador HandMvNet SEM hardware.

Motivo: o adaptador tem dois caminhos (modelo oficial e triangulacao de
fallback) e ambos precisam de frames + landmarks calibrados. Num laboratorio
sem Kinect/webcams ligados (ou em CI), a unica forma honesta de testar e' gerar
uma mao 3D conhecida, PROJETA-LA com os npz de calibracao reais do projeto e
entregar ao adaptador como se fossem deteccoes do MediaPipe. A triangulacao,
os bboxes, os crops e o empilhamento de vistas exercitam o mesmo codigo que a
bancada exercita.

Uso tipico (tool `handmvnet_infer.py --modo synthetic`):

    from vision.handmvnet_synthetic import build_synthetic_views
    views, gt_joints, info = build_synthetic_views(noise_px=0.5)
    out = adapter.infer(views, hand="right")

As imagens sao desenhos sinteticos (fundo cinza + esqueleto), sem dado de
participante: nada de privacidade/LGPD e nada de peso grande no repositorio.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:                                                    # cv2 e' dependencia do projeto
    import cv2
except ImportError:                                     # pragma: no cover
    cv2 = None

try:                                                    # import como pacote
    from .handmvnet_adapter import (
        DEFAULT_CALIBRATION_DIR,
        KINECT_VIEW_NAME,
        HandMvNetView,
        crop_and_pad_image,
        load_stereo_calibration,
        project_points_px,
        scale_intrinsic,
    )
except ImportError:                                     # import "flat" (path do projeto)
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    _CORE = _HERE.parent / "core"
    if str(_CORE) not in sys.path:
        sys.path.insert(0, str(_CORE))
    from handmvnet_adapter import (                     # type: ignore[no-redef]
        DEFAULT_CALIBRATION_DIR,
        KINECT_VIEW_NAME,
        HandMvNetView,
        crop_and_pad_image,
        load_stereo_calibration,
        project_points_px,
        scale_intrinsic,
    )

#: Arestas do esqueleto MediaPipe/MANO (mesma ordem de 21 junções do projeto).
HAND_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

#: Template da mao no "plano da palma" em metros (x ~ dedos, y ~ comprimento,
#: z ~ fora do plano); wrist (landmark 0) na origem. Ordem = MediaPipe.
PALM_PLANE_TEMPLATE: np.ndarray = np.array([
    [0.000, 0.000, 0.000],      # 0 wrist
    [0.030, -0.015, 0.006],     # 1 thumb_cmc
    [0.055, -0.030, 0.010],     # 2 thumb_mcp
    [0.072, -0.045, 0.012],     # 3 thumb_ip
    [0.085, -0.058, 0.012],     # 4 thumb_tip
    [0.020, -0.045, 0.004],     # 5 index_mcp
    [0.018, -0.080, 0.006],     # 6 index_pip
    [0.016, -0.105, 0.007],     # 7 index_dip
    [0.014, -0.125, 0.007],     # 8 index_tip
    [-0.002, -0.050, 0.003],    # 9 middle_mcp
    [-0.004, -0.090, 0.004],    # 10 middle_pip
    [-0.005, -0.118, 0.005],    # 11 middle_dip
    [-0.006, -0.140, 0.005],    # 12 middle_tip
    [-0.024, -0.046, 0.002],    # 13 ring_mcp
    [-0.027, -0.084, 0.003],    # 14 ring_pip
    [-0.029, -0.110, 0.004],    # 15 ring_dip
    [-0.031, -0.130, 0.004],    # 16 ring_tip
    [-0.045, -0.038, 0.001],    # 17 little_mcp
    [-0.048, -0.070, 0.002],    # 18 little_pip
    [-0.050, -0.092, 0.002],    # 19 little_dip
    [-0.052, -0.108, 0.002],    # 20 little_tip
], dtype=np.float64)


def rotation_matrix(rotation_deg: Sequence[float]) -> np.ndarray:
    """R = Rz @ Ry @ Rx (graus)."""
    ax, ay, az = np.deg2rad(np.asarray(rotation_deg, dtype=np.float64))
    rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def make_synthetic_hand(
    center: Sequence[float] = (0.30, 0.20, 0.80),
    rotation_deg: Sequence[float] = (10.0, -15.0, 8.0),
    *,
    scale: float = 1.0,
    noise_m: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """(21,3) junções sinteticas em metros no frame do Kinect (OpenCV).

    A mao e' o template do plano da palma rotacionado e transladado na regiao
    de bancada visivel pelas TRES cameras calibradas (~0,8 m do Kinect, um
    pouco a' direita). Com a calibracao atual do projeto as 21 junções caem
    dentro das 3 imagens (o teste verifica), o que permite medir o erro de
    reconstrucao contra o ground truth.
    """
    rng = np.random.default_rng(seed)
    R = rotation_matrix(rotation_deg)
    template = PALM_PLANE_TEMPLATE * float(scale)
    points = (R @ template.T).T + np.asarray(center, dtype=np.float64).reshape(1, 3)
    if noise_m > 0:
        points = points + rng.normal(0.0, float(noise_m), size=points.shape)
    return points


def hand_skeleton_connections() -> Tuple[Tuple[int, int], ...]:
    """Arestas do esqueleto (para desenhar/visualizar)."""
    return HAND_EDGES


def render_hand_image(size: Tuple[int, int], landmarks_px: np.ndarray,
                      *, background: int = 40) -> np.ndarray:
    """Desenha a mao sintetica numa imagem BGR (fundo cinza escuro)."""
    if cv2 is None:                                     # pragma: no cover
        raise RuntimeError("opencv necessario para renderizar imagem sintetica")
    w, h = int(size[0]), int(size[1])
    image = np.full((h, w, 3), int(background), dtype=np.uint8)
    pts = np.asarray(landmarks_px, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    for a, b in HAND_EDGES:
        if not (finite[a] and finite[b]):
            continue
        pa = (int(round(pts[a, 0])), int(round(pts[a, 1])))
        pb = (int(round(pts[b, 0])), int(round(pts[b, 1])))
        cv2.line(image, pa, pb, (60, 160, 60), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if not finite[i]:
            continue
        color = (0, 80, 255) if i == 0 else (220, 220, 220)
        cv2.circle(image, (int(round(p[0])), int(round(p[1]))), 4, color, -1,
                   cv2.LINE_AA)
    return image


def _load_kinect_view_calibration(calibration_dir: Path) -> Dict[str, Any]:
    """K/dist/tamanho do Kinect a partir das calibracoes do projeto.

    Prefere `kinect_intrinsics.npz` (arquivo dedicado). Se nao existir, cai no
    `kinect_matrix` do stereo da aux0 -- e' o mesmo par fisico usado na
    calibracao stereo.
    """
    dedicated = calibration_dir / "kinect_intrinsics.npz"
    if dedicated.is_file():
        data = np.load(dedicated)
        return {
            "matrix": np.asarray(data["kinect_matrix"], np.float64),
            "dist": np.asarray(data["kinect_dist"], np.float64).reshape(-1),
            "size": (tuple(int(v) for v in data["kinect_size"])
                     if "kinect_size" in data.files else (1920, 1080)),
            "source": str(dedicated),
        }
    stereo = load_stereo_calibration(calibration_dir / "stereo_calibration.npz")
    return {
        "matrix": np.asarray(stereo["kinect_matrix"], np.float64),
        "dist": np.asarray(stereo["kinect_dist"], np.float64),
        "size": tuple(stereo["kinect_size"] or (1920, 1080)),
        "source": stereo["path"],
    }


def available_aux_indices(calibration_dir: str | os.PathLike = DEFAULT_CALIBRATION_DIR
                          ) -> List[int]:
    """Indices de auxiliares com `stereo_calibration*.npz` no projeto."""
    folder = Path(calibration_dir)
    indices: List[int] = []
    if (folder / "stereo_calibration.npz").is_file():
        indices.append(0)
    for path in sorted(folder.glob("stereo_calibration_aux*.npz")):
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        if digits:
            indices.append(int(digits))
    return sorted(set(indices))


def build_synthetic_views(
    calibration_dir: str | os.PathLike = DEFAULT_CALIBRATION_DIR,
    *,
    aux_indices: Sequence[int] = (0, 1),
    num_aux: int = 2,
    render_scale: float = 1.0,
    noise_px: float = 0.0,
    center: Sequence[float] = (0.30, 0.20, 0.80),
    rotation_deg: Sequence[float] = (10.0, -15.0, 8.0),
    seed: int = 0,
    draw_images: bool = True,
) -> Tuple[List[HandMvNetView], np.ndarray, Dict[str, Any]]:
    """Gera vistas sinteticas calibradas + ground truth 3D (frame do Kinect).

    Args:
        calibration_dir: pasta `configs/calibracao`.
        aux_indices: indices de auxiliares a usar (default 0 e 1).
        num_aux: quantas auxiliares usar no maximo (2 = Kinect+aux0+aux1).
        render_scale: 1.0 = resolucao calibrada; 0.25 gera imagens menores
            (mais rapido) escalando intrinsics e projecoes de forma coerente.
        noise_px: ruido gaussiano (px) somado aos landmarks de cada vista, para
            simular erro do MediaPipe e medir sensibilidade.
        center/rotation_deg: pose da mao sintetica no frame do Kinect.
        seed: semente do ruido.
        draw_images: desenha o esqueleto nas imagens (False = fundo liso).

    Returns:
        (views, gt_joints_3d (21,3), info dict com calibracoes/projecoes/erros)
    """
    folder = Path(calibration_dir)
    available = available_aux_indices(folder)
    chosen = [int(i) for i in aux_indices if int(i) in available][:max(int(num_aux), 0)]
    if not chosen:
        raise FileNotFoundError(
            f"sem stereo_calibration*.npz em {folder}: nao da' para gerar vistas")
    rng = np.random.default_rng(seed + 1000)
    kinect = _load_kinect_view_calibration(folder)
    gt = make_synthetic_hand(center=center, rotation_deg=rotation_deg, seed=seed)

    def _render_size(size: Optional[Sequence[int]]) -> Tuple[int, int]:
        w, h = (int(size[0]), int(size[1])) if size else (1920, 1080)
        if abs(float(render_scale) - 1.0) < 1e-9:
            return w, h
        return (max(int(round(w * render_scale)), 32),
                max(int(round(h * render_scale)), 32))

    views: List[HandMvNetView] = []
    projections: Dict[str, np.ndarray] = {}
    calibs: Dict[str, Dict[str, Any]] = {}

    # ---- vista do Kinect ----
    kin_size = _render_size(kinect["size"])
    K_kin = scale_intrinsic(kinect["matrix"], kinect["size"], kin_size)
    kin_px = project_points_px(gt, K_kin, kinect["dist"])
    kin_px = _add_noise(kin_px, noise_px, rng)
    kin_img = (render_hand_image(kin_size, kin_px) if draw_images
               else np.zeros((kin_size[1], kin_size[0], 3), np.uint8))
    views.append(HandMvNetView(
        name=KINECT_VIEW_NAME, image=kin_img, landmarks_px=kin_px,
        intrinsic=K_kin, dist=kinect["dist"], is_kinect=True))
    projections[KINECT_VIEW_NAME] = kin_px
    calibs[KINECT_VIEW_NAME] = kinect

    # ---- auxiliares ----
    for aux_index in chosen:
        npz = (folder / "stereo_calibration.npz" if aux_index == 0 else
               folder / f"stereo_calibration_aux{aux_index}.npz")
        calib = load_stereo_calibration(npz)
        calibs[f"aux{aux_index}"] = calib
        aux_size = _render_size(calib["auxiliary_size"])
        K_aux = scale_intrinsic(calib["auxiliary_matrix"],
                                calib["auxiliary_size"], aux_size)
        aux_px = project_points_px(gt, K_aux, calib["auxiliary_dist"],
                                   rotation=calib["rotation"],
                                   translation=calib["translation"])
        aux_px = _add_noise(aux_px, noise_px, rng)
        aux_img = (render_hand_image(aux_size, aux_px) if draw_images
                   else np.zeros((aux_size[1], aux_size[0], 3), np.uint8))
        views.append(HandMvNetView(
            name=f"aux{aux_index}", image=aux_img, landmarks_px=aux_px,
            intrinsic=K_aux, dist=calib["auxiliary_dist"],
            rotation=calib["rotation"], translation=calib["translation"],
            aux_index=int(aux_index)))
        projections[f"aux{aux_index}"] = aux_px

    inside = {}
    for view in views:
        wt, ht = view.size
        pts = projections[view.name]
        finite = np.isfinite(pts).all(axis=1)
        inside[view.name] = bool(
            finite.all()
            and (pts[:, 0] >= 0).all() and (pts[:, 0] < wt).all()
            and (pts[:, 1] >= 0).all() and (pts[:, 1] < ht).all())
    info = {
        "calibration_dir": str(folder),
        "aux_indices": chosen,
        "render_scale": float(render_scale),
        "noise_px": float(noise_px),
        "gt_joints": gt,
        "projections": projections,
        "calibrations": calibs,
        "all_projections_inside_image": all(inside.values()),
        "projections_inside": inside,
        "view_sizes": {v.name: list(v.size) for v in views},
    }
    return views, gt, info


def _add_noise(points_px: np.ndarray, noise_px: float,
               rng: np.random.Generator) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2).copy()
    if noise_px and noise_px > 0:
        pts = pts + rng.normal(0.0, float(noise_px), size=pts.shape)
    return pts


def save_synthetic_frames(views: Iterable[HandMvNetView],
                          output_dir: str | os.PathLike,
                          info: Optional[Dict[str, Any]] = None) -> List[Path]:
    """Salva PNGs das vistas (debug visual; nao e' necessario para o smoke)."""
    if cv2 is None:                                     # pragma: no cover
        raise RuntimeError("opencv necessario para salvar frames")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for view in views:
        path = out / f"synthetic_{view.name}.png"
        cv2.imwrite(str(path), view.image)
        written.append(path)
    if info is not None:
        meta = out / "synthetic_meta.json"
        payload = {
            "view_sizes": info.get("view_sizes"),
            "aux_indices": info.get("aux_indices"),
            "noise_px": info.get("noise_px"),
            "all_projections_inside_image": info.get("all_projections_inside_image"),
            "gt_joints": np.asarray(info.get("gt_joints")).tolist(),
        }
        meta.write_text(__import__("json").dumps(payload, indent=2),
                        encoding="utf-8")
        written.append(meta)
    return written
