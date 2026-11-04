"""
Aquisicao multi-view (stereo) + comparacao A/B entre backends de normal.

Fornece:
- AuxStereoSource: fonte que triangula maos entre Kinect RGB + camera auxiliar
- compare_frame(): compara KinectDepthBackend vs MultiViewBackend nas mesmas amostras
"""

from __future__ import annotations
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .vision_backends import (
    VisionNormal,
    KinectDepthBackend,
    MultiViewBackend,
    compare_backends,
    robust_plane_normal,
)
import kinect_imu_groundtruth as kinect_gt


@dataclass
class AuxStereoFrame:
    """Resultado de um frame stereo para uma mao."""
    hand: str                       # "right" ou "left"
    points_3d: np.ndarray           # (N, 3) metros, frame da camera Kinect (multi-view)
    reproj_errors: np.ndarray       # (N,) erro de reprojecao em pixels (Kinect)
    timestamp: float                # time.monotonic()
    valid: bool                     # True se triangulacao confiavel
    #: Pontos 3D medidos pelo DEPTH do Kinect (5 pontos do dorso), para o
    #: backend KinectDepth. Sao independentes da triangulacao multi-view.
    kinect_points: Optional[np.ndarray] = None


class AuxStereoSource:
    """
    Fonte multi-view: triangula maos entre Kinect RGB e uma camera auxiliar.

    Fluxo por frame:
    1. tracker.frames() -> color_bgra, depth
    2. tracker.detect_hands_all(color_bgra, depth) -> lista de maos Kinect
    3. Captura frame da camera auxiliar (cv2.VideoCapture)
    4. tracker.detect_auxiliary_hands_sides(aux_frame, aux_index) -> maos auxiliar
    5. Para cada lado (right/left): casa mao Kinect com mao auxiliar pelo lado
    6. Extrai landmarks 2D (pixels normalizados) de ambas as cameras
    7. Chama alignment.triangulate_hand_points(aux_pixels, kinect_pixels)
    8. Devolve dict {hand: AuxStereoFrame} ou {} se sem mao/triangulacao falhou
    """

    def __init__(
        self,
        tracker: Optional[kinect_gt.KinectHandTracker] = None,
        alignment: Optional[kinect_gt.CameraAlignment] = None,
        aux_indices: Optional[tuple[int, ...]] = None,
        aux_index: int = 0,
    ):
        """
        Args:
            tracker: KinectHandTracker ja instanciado (ou None para criar lazy)
            alignment: CameraAlignment ja instanciado (ou None para criar lazy)
            aux_indices: Tupla de indices das cameras auxiliares configuradas
            aux_index: Indice da camera auxiliar ATIVA para esta fonte
        """
        self._tracker = tracker
        self._alignment = alignment
        self._aux_indices = tuple(int(i) for i in (aux_indices or kinect_gt.AUX_CAMERA_INDICES))
        self._aux_index = int(aux_index)
        self._cap = None
        self._started = False
        self._fake_mode = False  # True quando tracker/alignment injetados sao fakes de teste

    def _create_tracker(self) -> kinect_gt.KinectHandTracker:
        """Factory lazy para KinectHandTracker."""
        import kinect_compat
        kinect_compat.aplicar()
        return kinect_gt.KinectHandTracker(aux_indices=self._aux_indices)

    def _create_alignment(self) -> kinect_gt.CameraAlignment:
        """Factory lazy para CameraAlignment."""
        return kinect_gt.CameraAlignment(aux_indices=self._aux_indices)

    def start(self) -> None:
        """Inicia tracker, alignment e abre camera auxiliar (se nao fake)."""
        if self._started:
            return

        if self._tracker is None:
            self._tracker = self._create_tracker()
        if self._alignment is None:
            self._alignment = self._create_alignment()

        # Detecta modo fake: se tracker/alignment nao tem kinect real, nao abre cv2
        self._fake_mode = (
            getattr(self._tracker, "kinect", None) is None
            or getattr(self._alignment, "aux_models", None) is None
        )

        if not self._fake_mode and cv2 is not None:
            self._cap = cv2.VideoCapture(self._aux_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                warnings.warn(
                    f"AuxStereoSource: camera auxiliar indice {self._aux_index} "
                    f"nao abriu (pode nao existir). Triangulacao nao funcionara.",
                    RuntimeWarning,
                )
                self._cap = None
            else:
                # A calibracao stereo foi feita em uma resolucao especifica
                # (normalmente 1280x720). Abrir a camera em outra resolucao
                # faz o alignment ESCALAR a matriz e perde precisao. Forca a
                # resolucao salva no npz e tenta MJPEG.
                try:
                    view = self._alignment.aux_view(self._aux_index)
                    size = getattr(view, "size", None)
                    if size:
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))
                    self._cap.set(cv2.CAP_PROP_FPS, 30)
                    try:
                        self._cap.set(cv2.CAP_PROP_FOURCC,
                                     cv2.VideoWriter_fourcc(*"MJPG"))
                    except Exception:
                        pass
                except Exception:
                    pass
        elif not self._fake_mode:
            warnings.warn(
                "AuxStereoSource: cv2 nao disponivel; camera auxiliar nao abrira.",
                RuntimeWarning,
            )

        self._started = True

    def close(self) -> None:
        """Fecha camera auxiliar e libera tracker/alignment."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._tracker is not None:
            close = getattr(self._tracker, "close", None)
            if callable(close):
                close()
        self._started = False

    def poll(self) -> Dict[str, AuxStereoFrame]:
        """
        Poll para o proximo frame stereo.

        Returns:
            Dict mapeando hand ("right"/"left") -> AuxStereoFrame.
            Retorna {} se:
              - tracker/alignment nao iniciados
              - sem frame Kinect ou auxiliar
              - stereo calibration nao usable
              - triangulacao falhou (menos de 5 pontos plausiveis)
        """
        if not self._started or self._tracker is None or self._alignment is None:
            return {}

        # 1. Pega frames do Kinect
        frames_result = self._tracker.frames()
        if frames_result is None:
            return {}
        color_bgra, depth = frames_result
        if color_bgra is None:
            return {}

        # 2. Detecta maos no Kinect
        detect_result = self._tracker.detect_hands_all(color_bgra, depth)
        if detect_result is None:
            return {}
        if isinstance(detect_result, tuple) and len(detect_result) == 2:
            _, kinect_hands = detect_result
        else:
            kinect_hands = detect_result
        if not kinect_hands:
            return {}

        # 3. Pega frame da camera auxiliar
        aux_frame = None
        if not self._fake_mode and self._cap is not None:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                # MESMA convencao do groundtruth: a calibracao/pixels da
                # auxiliar foram feitos na imagem espelhada quando
                # AUXILIARY_MIRROR_HORIZONTAL=True.
                try:
                    aux_frame = kinect_gt.transform_auxiliary_frame(frame)
                except Exception:
                    aux_frame = frame
        elif self._fake_mode:
            # Modo fake: frame auxiliar dummy (testes injetam landmarks direto)
            aux_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if aux_frame is None:
            return {}

        # 4. Detecta maos na auxiliar (com lateralidade)
        try:
            aux_hands = self._tracker.detect_auxiliary_hands_sides(
                aux_frame, aux_index=self._aux_index
            )
        except Exception:
            return {}
        if not aux_hands:
            return {}

        # 5. Atualiza o tamanho da auxiliar (pixels da imagem lida) e verifica
        # se a calibracao stereo desta camera esta usable.
        try:
            self._alignment.set_active_aux(self._aux_index)
            self._alignment.set_auxiliary_size(aux_frame)
        except Exception:
            pass
        view = self._alignment.active_view() if hasattr(self._alignment, "active_view") else None
        if view is None:
            view = self._alignment.aux_view(self._aux_index)
        if view is None or not view.usable:
            warnings.warn(
                f"AuxStereoSource: stereo calibration da camera {self._aux_index} "
                f"nao usable (stereo_usable=False). Retornando {{}}.",
                RuntimeWarning,
            )
            return {}

        # 6. Casa maos Kinect <-> auxiliar pelo lado
        kinect_by_side = {h["side"]: h for h in kinect_hands if h.get("side")}
        aux_by_side = {h["side"]: h for h in aux_hands if h.get("side")}

        results: Dict[str, AuxStereoFrame] = {}
        now = time.monotonic()

        for hand in ("right", "left"):
            k_hand = kinect_by_side.get(hand)
            a_hand = aux_by_side.get(hand)
            if k_hand is None or a_hand is None:
                continue

            k_lm = k_hand.get("landmarks")
            a_lm = a_hand.get("landmarks")
            if k_lm is None or a_lm is None:
                continue

            # 7. Extrai pixels 2D (normalizados 0..1 -> pixels)
            # Kinect RGB size
            kw = self._tracker.color_width
            kh = self._tracker.color_height
            kinect_pixels = np.array(
                [(lm.x * kw, lm.y * kh) for lm in k_lm],
                dtype=np.float64,
            ).reshape(-1, 2)

            # Auxiliar size
            aw = view.size[0] if view.size else aux_frame.shape[1]
            ah = view.size[1] if view.size else aux_frame.shape[0]
            aux_pixels = np.array(
                [(lm.x * aw, lm.y * ah) for lm in a_lm],
                dtype=np.float64,
            ).reshape(-1, 2)

            # 8. Triangula
            points_3d, reproj = self._alignment.triangulate_hand_points(
                aux_pixels, kinect_pixels
            )

            if points_3d is None or reproj is None:
                continue

            # Filtra pontos validos (reproj finita + points_3d finito + Z > 0.3)
            valid_mask = (
                np.isfinite(points_3d).all(axis=1)
                & np.isfinite(reproj)
                & (points_3d[:, 2] > 0.3)
                & (points_3d[:, 2] < 4.0)
            )
            if valid_mask.sum() < 5:
                continue

            points_3d = points_3d[valid_mask]
            reproj = reproj[valid_mask]

            # Pontos 3D do depth do Kinect (mesma verdade do Y): 5 pontos do
            # dorso. Usados pelo KinectDepthBackend no A/B REAL.
            kinect_points = []
            for idx in (0, 5, 9, 13, 17):
                try:
                    ponto = self._tracker._landmark_camera_point(k_lm[idx], hand)
                except (AttributeError, IndexError, TypeError):
                    ponto = None
                if ponto is not None and np.isfinite(ponto).all():
                    kinect_points.append(np.asarray(ponto, np.float64).reshape(3))
            kinect_points = (np.asarray(kinect_points, np.float64)
                             if len(kinect_points) >= 4 else None)

            results[hand] = AuxStereoFrame(
                hand=hand,
                points_3d=points_3d,
                reproj_errors=reproj,
                timestamp=now,
                valid=True,
                kinect_points=kinect_points,
            )

        return results


def compare_frame(
    kinect_points: np.ndarray,
    multiview_points: np.ndarray,
    hand: str,
    timestamp: float,
    reproj_errors: Optional[np.ndarray] = None,
    normal_anterior: Optional[np.ndarray] = None,
) -> dict:
    """
    Compara os dois backends nas MESMAS amostras 3D.

    Args:
        kinect_points: (N, 3) pontos 3D do Kinect depth (para KinectDepthBackend)
        multiview_points: (M, 3) pontos 3D triangulados (para MultiViewBackend)
        hand: "right" ou "left"
        timestamp: time.monotonic()
        reproj_errors: (M,) erros de reprojecao para MultiView (opcional)
        normal_anterior: normal anterior para continuidade temporal (opcional)

    Returns:
        Dict com:
            - "kinect_depth": VisionNormal do KinectDepthBackend
            - "multiview": VisionNormal do MultiViewBackend
            - "angle_deg": angulo entre as duas normais (graus)
            - "agree": True se angle_deg < 15 graus
    """
    # KinectDepthBackend usa pontos do Kinect (sem pesos)
    kinect_backend = KinectDepthBackend()
    vn_kinect = kinect_backend.compute(
        points_3d=kinect_points,
        hand=hand,
        timestamp=timestamp,
        normal_anterior=normal_anterior,
    )

    # MultiViewBackend usa pontos triangulados + pesos de reprojecao
    mv_backend = MultiViewBackend()
    vn_mv = mv_backend.compute(
        points_3d=multiview_points,
        hand=hand,
        timestamp=timestamp,
        reproj_errors=reproj_errors,
        normal_anterior=normal_anterior,
    )

    # Angulo entre normais
    angle_deg = 180.0
    if vn_kinect.valid and vn_mv.valid:
        n1 = vn_kinect.normal_cam
        n2 = vn_mv.normal_cam
        dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(dot)))

    return {
        "kinect_depth": vn_kinect,
        "multiview": vn_mv,
        "angle_deg": angle_deg,
        "agree": angle_deg < 15.0,
    }


def _synthetic_kinect_points(
    hand: str,
    n_points: int = 21,
    noise_mm: float = 3.0,
    outlier_frac: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Gera pontos 3D sinteticos simulando hand landmarks do Kinect depth."""
    rng = np.random.default_rng(seed)
    # Palma centrada em ~0.8m, tamanho ~0.1m
    center = np.array([0.0, 0.0, 0.8])
    # Gera pontos no plano da palma (normal ~ -Z para right, +Z para left)
    if hand == "right":
        normal = np.array([0.0, 0.0, -1.0])
    else:
        normal = np.array([0.0, 0.0, 1.0])

    # Base no plano
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    points = []
    for _ in range(n_points):
        pu = rng.uniform(-0.05, 0.05)
        pv = rng.uniform(-0.05, 0.05)
        pt = center + pu * u + pv * v
        points.append(pt)
    points = np.array(points, dtype=np.float64)

    # Ruido gaussiano (mm -> m)
    if noise_mm > 0:
        noise = rng.normal(0, noise_mm / 1000.0, points.shape)
        points += noise

    # Outliers
    if outlier_frac > 0:
        n_out = int(n_points * outlier_frac)
        idx = rng.choice(n_points, n_out, replace=False)
        points[idx] += rng.normal(0, 0.05, (n_out, 3))  # 5cm outliers

    return points


def synthetic_compare_run(
    duration: float = 5.0,
    hand: str = "right",
    n_points: int = 21,
    noise_mm: float = 3.0,
    outlier_frac: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """
    Executa comparacao A/B sintetica (sem hardware).

    Gera pontos 3D com ruido/outliers, roda os dois backends,
    retorna lista de resultados por frame.
    """
    rng = np.random.default_rng(seed)
    results = []
    t0 = time.monotonic()
    normal_anterior = None

    while time.monotonic() - t0 < duration:
        # Gera pontos Kinect (com ruido)
        k_pts = _synthetic_kinect_points(
            hand, n_points, noise_mm, outlier_frac, seed=int(rng.integers(2**31))
        )
        # Gera pontos MultiView (triangulados) - menos ruidosos, com erros reproj
        mv_pts = _synthetic_kinect_points(
            hand, n_points, noise_mm * 0.5, outlier_frac * 0.5, seed=int(rng.integers(2**31))
        )
        # Erros de reprojecao sinteticos (pixels)
        reproj = rng.uniform(0.5, 3.0, n_points).astype(np.float64)

        cmp_result = compare_frame(
            kinect_points=k_pts,
            multiview_points=mv_pts,
            hand=hand,
            timestamp=time.monotonic(),
            reproj_errors=reproj,
            normal_anterior=normal_anterior,
        )

        # Atualiza anterior para continuidade
        if cmp_result["multiview"].valid:
            normal_anterior = cmp_result["multiview"].normal_cam.copy()

        results.append(cmp_result)
        time.sleep(0.033)  # ~30 fps

    return results