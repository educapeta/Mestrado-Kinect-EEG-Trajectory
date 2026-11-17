"""Adaptador HandMvNet <-> pipeline multiview do projeto (Kinect RGB + auxiliares).

HandMvNet (paper VISAPP 2025, codigo em https://github.com/pyxploiter/HandMvNet)
estima pose 3D de mao (21 junções MANO/MediaPipe, em metros) a partir de N vistas
SINCRONIZADAS, fundindo features com cross-attention. Este modulo e' a ponte
entre aquele modelo e o NOSSO sistema:

  - entram frames ja' sincronizados (Kinect RGB + webcams auxiliares), com os
    landmarks 2D que o MediaPipe do projeto ja' detecta;
  - a calibracao stereo existente (`configs/calibracao/stereo_calibration*.npz`)
    fornece intrinsics/distorcao de cada camera e a pose relativa ao Kinect;
  - saem 21 junções 3D em METROS no referencial da camera RGB do Kinect
    (mesma convencao OpenCV usada pela triangulacao do projeto), mais os
    landmarks 2D reprojetados, erros de reprojecao por vista e metadados.

Dois modos de operacao (o modo `auto` escolhe sozinho):

  1. `handmvnet` -- carrega o modelo OFICIAL do repositorio externo
     (`external/HandMvNet`) e roda `HandMvNet.forward`. Requer um checkpoint
     `.ckpt` do Lightning. O repositorio oficial NAO publica pesos (ver
     `WEIGHTS_STATUS` e `docs/guias/HANDMVNET_INTEGRACAO.md`), entao este modo
     fica indisponivel ate' o usuario obter um checkpoint.
  2. `triangulation_fallback` -- quando o modelo/pesos/dependencias nao estao
     disponiveis (ou quando o checkpoint tem `num_views` incompativel), o
     adaptador cai na triangulacao multi-view classica (DLT por junção,
     undistort com a calibracao, gate de reprojecao e queda para o melhor par
     quando uma vista falha). Devolve O MESMO contrato de saida, com
     `backend="triangulation_fallback"` e o motivo em `reason`.

O fallback nao e' um trecho morto: e' ele que permite o pipeline rodar HOJE sem
pesos e serve de linha de base geometrica para comparar com o modelo quando os
pesos existirem.

Referenciais (os mesmos do projeto):
  - frame Kinect = camera RGB, convencao OpenCV: X para a direita, Y para
    baixo, Z para frente, metros. A fusao/IMU converte Y para cima onde precisa
    (`fusion/adapters.py::MultiViewSource`), entao o adaptador entrega igual a
    `fusion/aux_vision.py::AuxStereoFrame.points_3d`.
  - calibracao por camera auxiliar: X_aux = R @ X_kin + T (convencao dos npz).
  - frames das auxiliares JA' espelhados pelo pipeline quando
    `AUXILIARY_MIRROR_HORIZONTAL=True` (`transform_auxiliary_frame`); landmarks
    e calibracao vivem nessa mesma imagem espelhada.

Dependencias:
  - obrigatorias: numpy, opencv (cv2) -- ja' no venv do projeto;
  - modo modelo: torch (o venv tem torch 2.14 + XPU/Intel Arc), o codigo
    oficial e o checkpoint. `lightning`, `einops`, `torchvision` e `manopth`
    nao sao usados no forward de inferencia; quando faltam, o carregador
    instala SHIMS apenas para importar o modelo oficial (documentado em
    `load_official_model`), sem alterar o repositorio externo nem o venv.

Nao faz parte do escopo: abrir cameras (isso e' do pipeline/tools existentes),
treinar, baixar pesos, gerar malha/MANO (o repositorio oficial so' faz vertices
com o pacote `manopth` e os modelos MANO, que exigem licenca a parte).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:                                     # pragma: no cover
    cv2 = None

# =============================================================================
# Constantes / caminhos do projeto
# =============================================================================
NUM_HAND_LANDMARKS = 21
#: Indices do dorso da palma (os MESMOS usados pelo contrato do projeto:
#: `fusion/vision_backends.py`, `fusion/aux_vision.py` e `fusion/adapters.py`).
PALM_LANDMARK_INDICES: Tuple[int, ...] = (0, 5, 9, 13, 17)
KINECT_VIEW_NAME = "kinect"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_HEATMAP_SIZE = 32
DEFAULT_CROP_MARGIN = 1.2
#: Gate de reprojecao usado na triangulacao fallback (px, em cada vista usada).
DEFAULT_MAX_REPROJ_PX = 10.0
#: Mesmo limite de `STEREO_OVERLAY_MAX_RMS_PX` em `kinect_imu_groundtruth.py`.
STEREO_USABLE_MAX_RMS_PX = 5.0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO_ROOT = _PROJECT_ROOT / "external" / "HandMvNet"
DEFAULT_CALIBRATION_DIR = _PROJECT_ROOT / "configs" / "calibracao"

OFFICIAL_REPO_URL = "https://github.com/pyxploiter/HandMvNet"

#: Evidencia sobre licenca (auditada em 17/11/2026, ver README de integracao).
LICENSE_STATUS = (
    "SEM LICENCA EXPLICITA: o repositorio oficial nao tem arquivo LICENSE e a "
    "API do GitHub devolve license=null em 17/11/2026. Sem licenca, o padrao e' "
    "'todos os direitos reservados': uso interno de pesquisa e' defensavel, mas "
    "redistribuir codigo/pesos/derivados exige autorizacao dos autores."
)
#: Evidencia sobre pesos (auditada em 17/11/2026).
WEIGHTS_STATUS = (
    "PESOS NAO PUBLICADOS: o repositorio nao versiona checkpoint (0 releases, 0 "
    "assets), o README nao linka pesos, o notebook `notebooks/infer_vis.ipynb` "
    "aponta para caminhos locais (`../weights/...`) que nao existem no repo e as "
    "issues #2 ('Can you share the model weights?') e #4 ('where is checkpoint?') "
    "estao abertas sem resposta. Sem `.ckpt` o modo modelo nao roda."
)

__all__ = [
    "NUM_HAND_LANDMARKS",
    "PALM_LANDMARK_INDICES",
    "KINECT_VIEW_NAME",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_REPO_ROOT",
    "DEFAULT_CALIBRATION_DIR",
    "LICENSE_STATUS",
    "WEIGHTS_STATUS",
    "HandMvNetView",
    "HandMvNetOutput",
    "TriangulationResult",
    "HandMvNetUnavailable",
    "HandMvNetAdapter",
    "load_stereo_calibration",
    "scale_intrinsic",
    "square_bbox_from_landmarks",
    "crop_and_pad_image",
    "joints_to_crop",
    "crop_to_joints",
    "undistort_landmarks",
    "triangulate_views",
    "resolve_torch_device",
    "describe_official_repo",
    "load_official_model",
    "load_random_model_for_validation",
]


class HandMvNetUnavailable(RuntimeError):
    """Modelo oficial (pesos/dependencias) indisponivel; o fallback cobre."""


class HandMvNetOutError(RuntimeError):
    """Forward do modelo devolveu um contrato inesperado."""


# =============================================================================
# Utilitarios numericos / calibracao
# =============================================================================
def _as_intrinsic(value: Any) -> np.ndarray:
    """Aceita K 3x3 ou [fx, fy, cx, cy] e devolve K 3x3 float64."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4,):
        arr = np.array([[arr[0], 0.0, arr[2]],
                        [0.0, arr[1], arr[3]],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"intrinseca deve ser 3x3 ou [fx,fy,cx,cy]; veio {arr.shape}")
    return arr


def _as_dist(value: Any) -> np.ndarray:
    """Coeficientes de distorcao como (N,) float64 (vazio -> zeros(5))."""
    if value is None:
        return np.zeros(5, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr if arr.size else np.zeros(5, dtype=np.float64)


def scale_intrinsic(matrix: Any, from_size: Optional[Sequence[int]],
                    to_size: Optional[Sequence[int]]) -> np.ndarray:
    """Reescala a intrinseca quando o frame em uso difere do calibrado.

    Args:
        matrix: K 3x3 (ou [fx,fy,cx,cy]).
        from_size: (w, h) da resolucao em que a matriz foi calibrada.
        to_size: (w, h) da resolucao do frame atual.

    Devolve K novo (nao modifica o original).
    """
    K = _as_intrinsic(matrix).copy()
    if from_size is None or to_size is None:
        return K
    fw, fh = float(from_size[0]), float(from_size[1])
    tw, th = float(to_size[0]), float(to_size[1])
    if fw <= 0 or fh <= 0 or tw <= 0 or th <= 0:
        return K
    sx, sy = tw / fw, th / fh
    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        return K
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def load_stereo_calibration(path: str | os.PathLike) -> Dict[str, Any]:
    """Le um `stereo_calibration*.npz` do projeto e devolve um dict normalizado.

    Chaves: `kinect_matrix`, `kinect_dist`, `kinect_size`, `auxiliary_matrix`,
    `auxiliary_dist`, `auxiliary_size`, `rotation` (3x3), `translation` (3,),
    `rms`, `camera_name`, `stereo_usable`, `path`.

    Convencao (a mesma do projeto): X_aux = rotation @ X_kinect + translation.
    Nao depende de `kinect_imu_groundtruth`/pykinect2: leitura numpy pura, para
    poder ser usado em smoke test sem hardware.
    """
    p = Path(path)
    data = np.load(p)
    required = {"kinect_matrix", "kinect_dist", "auxiliary_matrix",
                "auxiliary_dist", "rotation", "translation"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{p} nao e' calibracao stereo completa; faltam {missing}")
    out: Dict[str, Any] = {
        "path": str(p),
        "kinect_matrix": data["kinect_matrix"].astype(np.float64),
        "kinect_dist": _as_dist(data["kinect_dist"]),
        "auxiliary_matrix": data["auxiliary_matrix"].astype(np.float64),
        "auxiliary_dist": _as_dist(data["auxiliary_dist"]),
        "rotation": data["rotation"].astype(np.float64),
        "translation": data["translation"].astype(np.float64).reshape(3),
        "rms": float(data["rms"]) if "rms" in data.files else None,
        "camera_name": (str(data["camera_name"]).strip()
                        if "camera_name" in data.files else None),
        "kinect_size": (tuple(int(v) for v in data["kinect_size"])
                        if "kinect_size" in data.files else None),
        "auxiliary_size": (tuple(int(v) for v in data["auxiliary_size"])
                           if "auxiliary_size" in data.files else None),
    }
    rms = out["rms"]
    out["stereo_usable"] = bool(rms is not None
                                and rms <= STEREO_USABLE_MAX_RMS_PX)
    return out


def _load_kinect_only_calibration(path: str | os.PathLike) -> Dict[str, Any]:
    """Le so' a intrinseca do Kinect (`kinect_intrinsics.npz`) como fallback."""
    data = np.load(Path(path))
    if "kinect_matrix" not in data.files or "kinect_dist" not in data.files:
        raise ValueError(f"{path} nao tem kinect_matrix/kinect_dist")
    return {
        "path": str(path),
        "kinect_matrix": data["kinect_matrix"].astype(np.float64),
        "kinect_dist": _as_dist(data["kinect_dist"]),
        "kinect_size": (tuple(int(v) for v in data["kinect_size"])
                        if "kinect_size" in data.files else None),
        "auxiliary_matrix": None, "auxiliary_dist": None,
        "auxiliary_size": None, "rotation": None, "translation": None,
        "rms": None, "camera_name": None, "stereo_usable": False,
    }


def _undistort_points(pixels: np.ndarray, K: np.ndarray,
                      dist: np.ndarray) -> np.ndarray:
    """Pixels -> coordenadas normalizadas sem distorcao (NaN onde invalido)."""
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    if pts.size == 0:
        return out
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return out
    if cv2 is None:
        raise RuntimeError("opencv (cv2) e' obrigatorio para o adaptador")
    normalized = cv2.undistortPoints(
        pts[finite].reshape(-1, 1, 2).astype(np.float64),
        np.asarray(K, np.float64),
        np.asarray(dist, np.float64),
    ).reshape(-1, 2)
    out[finite] = normalized
    return out


def undistort_landmarks(landmarks_px: np.ndarray, intrinsic: Any,
                        dist: Any = None) -> np.ndarray:
    """Atalho publico: (21,2) pixels -> (21,2) normalizado sem distorcao."""
    return _undistort_points(landmarks_px, _as_intrinsic(intrinsic), _as_dist(dist))


def _project_normalized(points_3d: np.ndarray, K: np.ndarray,
                        rotation: Optional[np.ndarray] = None,
                        translation: Optional[np.ndarray] = None,
                        eps: float = 1e-9) -> np.ndarray:
    """Projeta pontos 3D no frame do Kinect para (u,v) normalizados de uma vista."""
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    if rotation is not None:
        R = np.asarray(rotation, np.float64).reshape(3, 3)
        T = np.asarray(translation, np.float64).reshape(3)
        pts = (R @ pts.T).T + T.reshape(1, 3)
    z = pts[:, 2]
    out = np.full((pts.shape[0], 2), np.nan)
    ok = np.isfinite(pts).all(axis=1) & (np.abs(z) > eps)
    out[ok, 0] = pts[ok, 0] / z[ok]
    out[ok, 1] = pts[ok, 1] / z[ok]
    return out


def project_points_px(points_3d: np.ndarray, intrinsic: Any, dist: Any = None,
                      rotation: Optional[np.ndarray] = None,
                      translation: Optional[np.ndarray] = None) -> np.ndarray:
    """Projeta pontos 3D (frame do Kinect) em pixels de uma vista calibrada.

    Aplica o modelo de distorcao (cv2.projectPoints) da calibracao -- e' a
    operacao inversa do `_undistort_points` usado na triangulacao, e a mesma
    convencao dos pixels do MediaPipe na imagem crua.
    """
    K = _as_intrinsic(intrinsic)
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3).copy()
    if rotation is not None:
        R = np.asarray(rotation, np.float64).reshape(3, 3)
        T = np.asarray(translation, np.float64).reshape(3)
        pts = (R @ pts.T).T + T.reshape(1, 3)
    if cv2 is None:                                     # pragma: no cover
        norm = _project_normalized(points_3d, K, rotation, translation)
        px = np.full_like(norm, np.nan)
        ok = np.isfinite(norm).all(axis=1)
        px[ok, 0] = norm[ok, 0] * K[0, 0] + K[0, 2]
        px[ok, 1] = norm[ok, 1] * K[1, 1] + K[1, 2]
        return px
    finite = np.isfinite(pts).all(axis=1)
    px = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    if finite.any():
        projected, _ = cv2.projectPoints(
            pts[finite].reshape(-1, 1, 3).astype(np.float64),
            np.zeros(3), np.zeros(3), K, _as_dist(dist))
        px[finite] = projected.reshape(-1, 2)
    return px


# =============================================================================
# BBox / crop / normalizacao da imagem (replica o pre-processamento oficial)
# =============================================================================
def square_bbox_from_landmarks(landmarks_px: np.ndarray,
                               image_size: Optional[Sequence[int]] = None,
                               margin: float = DEFAULT_CROP_MARGIN) -> np.ndarray:
    """BBox QUADRADA (x1, y1, x2, y2) em pixels em volta dos 21 landmarks.

    Reproduz `datasets/utils.points2d_to_bbox` do repositorio oficial (borda
    quadrada em volta dos pontos), com margem multiplicativa opcional. A caixa
    NAO e' recortada na borda da imagem: o `crop_and_pad_image` preenche com
    zeros o que cai fora (mesmo comportamento do treino oficial).
    """
    pts = np.asarray(landmarks_px, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        raise ValueError("sem landmarks finitos para montar o bbox")
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    side = max(xmax - xmin, ymax - ymin) * float(margin)
    side = max(side, 2.0)
    x1 = int(np.floor(cx - side / 2.0))
    y1 = int(np.floor(cy - side / 2.0))
    x2 = int(np.ceil(cx + side / 2.0))
    y2 = int(np.ceil(cy + side / 2.0))
    if x2 <= x1:
        x2 = x1 + 2
    if y2 <= y1:
        y2 = y1 + 2
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def crop_and_pad_image(image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    """Recorta `bbox` da imagem e preenche com zeros o que estiver fora.

    Equivalente a `datasets/utils.crop_and_pad_image` do repositorio oficial,
    inclusive o canvas do tamanho ORIGINAL do bbox (antes do resize).
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(f"imagem com shape invalido {img.shape}")
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    out_w = max(int(x2 - x1), 1)
    out_h = max(int(y2 - y1), 1)
    channels = (1,) if img.ndim == 2 else (img.shape[2],)
    out = np.zeros((out_h, out_w) + channels, dtype=img.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[sy1:sy2, sx1:sx2]
    return out


def image_to_model_tensor(image: np.ndarray, image_size: int = DEFAULT_IMAGE_SIZE,
                          image_bgr: bool = True) -> np.ndarray:
    """Imagem BGR/BGRA -> tensor (3, S, S) float32 RGB normalizado ImageNet.

    Mesma ordem de operacoes do `img_transform` oficial (ToTensor -> Resize ->
    Normalize ImageNet): resize para SxS, /255, (x - mean)/std, HWC -> CHW.
    """
    if cv2 is None:                                     # pragma: no cover
        raise RuntimeError("opencv (cv2) e' obrigatorio para o adaptador")
    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"imagem vazia {img.shape}")
    # Downscale com AREA = mesma antialias do Resize oficial; upscale LINEAR.
    interp = cv2.INTER_AREA if (h > image_size or w > image_size) else cv2.INTER_LINEAR
    resized = cv2.resize(img, (int(image_size), int(image_size)),
                         interpolation=interp)
    if resized.ndim == 3 and resized.shape[2] == 4:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
    elif image_bgr and resized.ndim == 3:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        rgb = resized
    tensor = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(tensor, (2, 0, 1)).astype(np.float32)


def joints_to_crop(landmarks_px: np.ndarray, bbox: Sequence[float],
                   image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Pixels da imagem inteira -> pixels do crop SxS (formula oficial)."""
    pts = np.asarray(landmarks_px, dtype=np.float64).reshape(-1, 2).copy()
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    pts[:, 0] = (pts[:, 0] - x1) * float(image_size) / w
    pts[:, 1] = (pts[:, 1] - y1) * float(image_size) / h
    return pts


def crop_to_joints(crop_px: np.ndarray, bbox: Sequence[float],
                   image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Pixels do crop SxS -> pixels da imagem inteira (inverso de `joints_to_crop`)."""
    pts = np.asarray(crop_px, dtype=np.float64).reshape(-1, 2).copy()
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    pts[:, 0] = pts[:, 0] * w / float(image_size) + x1
    pts[:, 1] = pts[:, 1] * h / float(image_size) + y1
    return pts


def mirror_view_data(image: np.ndarray, landmarks_px: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Espelha imagem e pixels como `transform_auxiliary_frame` (cv2.flip(frame, 1)).

    cv2.flip(frame, 1) mapeia x -> (w - 1 - x); a calibracao das auxiliares do
    projeto foi feita NA IMAGEM ESPELHADA, entao quem entrega frame cru tem de
    passar por aqui antes de chamar o adaptador.
    """
    img = np.asarray(image)
    pts = np.asarray(landmarks_px, dtype=np.float64).reshape(-1, 2).copy()
    width = img.shape[1]
    pts[:, 0] = float(width - 1) - pts[:, 0]
    return np.ascontiguousarray(img[:, ::-1]), pts


# =============================================================================
# Vistas de entrada
# =============================================================================
@dataclass
class HandMvNetView:
    """Uma vista sincronizada: imagem + landmarks 2D + calibracao.

    Attributes:
        name: rotulo da vista (ex.: "kinect", "aux0"). O PRIMEIRO da lista e'
            tratado como vista raiz do modelo (referencial da saida).
        image: frame BGR/BGRA (convencao OpenCV do projeto).
        landmarks_px: (21,2) landmarks MediaPipe em pixels da imagem, na MESMA
            convencao da calibracao (auxiliares ja' espelhadas).
        intrinsic: K 3x3 (ou [fx, fy, cx, cy]) na resolucao DESTE frame.
        dist: coeficientes de distorcao (opcional; zeros se None).
        rotation: (3,3) da calibracao auxiliar (X_aux = R X_kin + T). None para
            a vista do Kinect (que e' a raiz).
        translation: (3,) T da calibracao auxiliar. None para o Kinect.
        aux_index: indice OpenCV da auxiliar (informativo/integracao), None no
            Kinect.
        image_bgr: True = imagem BGR/BGRA (padrao do projeto).
        is_kinect: marca a vista raiz/do Kinect.
    """

    name: str
    image: np.ndarray
    landmarks_px: np.ndarray
    intrinsic: np.ndarray
    dist: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    aux_index: Optional[int] = None
    image_bgr: bool = True
    is_kinect: bool = False

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        if self.image.ndim not in (2, 3):
            raise ValueError(f"view {self.name!r}: imagem com shape {self.image.shape}")
        pts = np.asarray(self.landmarks_px, dtype=np.float64).reshape(-1, 2)
        if pts.shape != (NUM_HAND_LANDMARKS, 2):
            raise ValueError(
                f"view {self.name!r}: landmarks devem ser (21,2); veio {pts.shape}")
        self.landmarks_px = pts
        self.intrinsic = _as_intrinsic(self.intrinsic)
        self.dist = _as_dist(self.dist)
        if self.rotation is not None:
            self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        if self.translation is not None:
            self.translation = np.asarray(self.translation,
                                          dtype=np.float64).reshape(3)

    # ------------------------------------------------------------------ helpers
    @property
    def size(self) -> Tuple[int, int]:
        """(w, h) do frame."""
        return int(self.image.shape[1]), int(self.image.shape[0])

    @property
    def is_auxiliary(self) -> bool:
        return self.rotation is not None and self.translation is not None

    def extrinsic(self) -> np.ndarray:
        """Matriz 4x4 world->camera (world = frame do Kinect)."""
        E = np.eye(4, dtype=np.float64)
        if self.is_auxiliary:
            E[:3, :3] = self.rotation
            E[:3, 3] = self.translation
        return E

    def with_landmarks(self, landmarks_px: np.ndarray) -> "HandMvNetView":
        return HandMvNetView(
            name=self.name, image=self.image, landmarks_px=landmarks_px,
            intrinsic=self.intrinsic.copy(), dist=self.dist.copy(),
            rotation=None if self.rotation is None else self.rotation.copy(),
            translation=None if self.translation is None else self.translation.copy(),
            aux_index=self.aux_index, image_bgr=self.image_bgr,
            is_kinect=self.is_kinect,
        )

    def corrupt_image_for_privacy(self) -> "HandMvNetView":
        """Troca a imagem por uma sintetica (testes/smoke sem dado real)."""
        h, w = self.size[1], self.size[0]
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        return HandMvNetView(
            name=self.name, image=dummy, landmarks_px=self.landmarks_px,
            intrinsic=self.intrinsic.copy(), dist=self.dist.copy(),
            rotation=None if self.rotation is None else self.rotation.copy(),
            translation=None if self.translation is None else self.translation.copy(),
            aux_index=self.aux_index, image_bgr=self.image_bgr,
            is_kinect=self.is_kinect,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": list(self.size),
            "is_kinect": bool(self.is_kinect),
            "is_auxiliary": bool(self.is_auxiliary),
            "aux_index": self.aux_index,
            "fx": float(self.intrinsic[0, 0]),
            "fy": float(self.intrinsic[1, 1]),
            "cx": float(self.intrinsic[0, 2]),
            "cy": float(self.intrinsic[1, 2]),
            "landmarks_finitos": int(np.isfinite(self.landmarks_px).all(axis=1).sum()),
        }

    # ------------------------------------------------------------ construtores
    @classmethod
    def from_calibration_npz(
        cls,
        name: str,
        image: np.ndarray,
        landmarks_px: np.ndarray,
        npz_path: str | os.PathLike,
        *,
        is_kinect: bool = False,
        aux_index: Optional[int] = None,
        image_bgr: bool = True,
        mirror_to_calibration: bool = False,
    ) -> "HandMvNetView":
        """Monta a vista lendo a calibracao do npz (numpy puro, sem hardware).

        Se `mirror_to_calibration=True`, a imagem e os pixels sao espelhados
        ANTES do uso (para quem capturou frame cru, sem `transform_auxiliary_frame`).
        Para a vista do Kinect aceita tambem `kinect_intrinsics.npz` (so' a
        intrinseca), alem do arquivo de par stereo.
        """
        try:
            calib = load_stereo_calibration(npz_path)
        except ValueError:
            if not is_kinect:
                raise
            calib = _load_kinect_only_calibration(npz_path)
        img = np.asarray(image)
        pts = np.asarray(landmarks_px, dtype=np.float64).reshape(-1, 2)
        if mirror_to_calibration:
            img, pts = mirror_view_data(img, pts)
        size = (int(img.shape[1]), int(img.shape[0]))
        if is_kinect:
            K = scale_intrinsic(calib["kinect_matrix"], calib["kinect_size"], size)
            return cls(name=name, image=img, landmarks_px=pts, intrinsic=K,
                       dist=calib["kinect_dist"], is_kinect=True,
                       aux_index=None, image_bgr=image_bgr)
        K = scale_intrinsic(calib["auxiliary_matrix"],
                            calib["auxiliary_size"], size)
        return cls(name=name, image=img, landmarks_px=pts, intrinsic=K,
                   dist=calib["auxiliary_dist"], rotation=calib["rotation"],
                   translation=calib["translation"], aux_index=aux_index,
                   image_bgr=image_bgr)

    @classmethod
    def from_alignment(
        cls,
        alignment: Any,
        name: str,
        image: np.ndarray,
        landmarks_px: np.ndarray,
        *,
        is_kinect: bool = False,
        aux_index: Optional[int] = None,
        image_bgr: bool = True,
    ) -> "HandMvNetView":
        """Monta a vista a partir de um `CameraAlignment` JAH carregado.

        Caminho recomendado no pipeline ao vivo: usa a mesma calibracao (e o
        mesmo escalonamento de resolucao em runtime) que o resto do projeto.
        """
        img = np.asarray(image)
        pts = np.asarray(landmarks_px, dtype=np.float64).reshape(-1, 2)
        size = (int(img.shape[1]), int(img.shape[0]))
        if is_kinect:
            calibrated = getattr(alignment, "calibrated_kinect_size", None)
            K = scale_intrinsic(alignment.kinect_matrix, calibrated, size)
            return cls(name=name, image=img, landmarks_px=pts, intrinsic=K,
                       dist=alignment.kinect_dist, is_kinect=True,
                       aux_index=None, image_bgr=image_bgr)
        view = alignment.aux_view(int(aux_index)) if aux_index is not None else None
        if view is None:
            raise ValueError(f"CameraAlignment sem calibracao para aux {aux_index}")
        calibrated = view.calibrated_size
        K = scale_intrinsic(view.effective_matrix, calibrated, size)
        return cls(name=name, image=img, landmarks_px=pts, intrinsic=K,
                   dist=view.dist, rotation=view.rotation,
                   translation=view.translation, aux_index=int(aux_index),
                   image_bgr=image_bgr)


# =============================================================================
# Triangulacao multi-view (fallback e estimativa da raiz)
# =============================================================================
@dataclass
class TriangulationResult:
    """Saida da triangulacao multi-view por landmark."""

    points_3d: np.ndarray                     # (21,3) m, frame do Kinect
    valid_mask: np.ndarray                    # (21,) bool
    errors_px: Dict[str, np.ndarray]          # vista -> (21,) erro de reprojecao
    views_used: Tuple[str, ...] = ()
    views_per_point: Optional[np.ndarray] = None   # (21,) numero de vistas usadas

    @property
    def n_valid(self) -> int:
        return int(np.asarray(self.valid_mask).sum())


def _dlt_point(projections: Sequence[np.ndarray],
               normalized_2d: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Um ponto 3D por DLT (minimos quadrados) a partir de N vistas."""
    rows: List[np.ndarray] = []
    for P, xy in zip(projections, normalized_2d):
        x, y = float(xy[0]), float(xy[1])
        rows.append(x * P[2, :] - P[0, :])
        rows.append(y * P[2, :] - P[1, :])
    A = np.asarray(rows, dtype=np.float64)
    if A.shape[0] < 4 or not np.isfinite(A).all():
        return None
    try:
        _, _, vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:                        # pragma: no cover
        return None
    X = vt[-1]
    w = float(X[3])
    if abs(w) < 1e-12 or not np.isfinite(w):
        return None
    point = X[:3] / w
    return point if np.isfinite(point).all() else None


def triangulate_views(
    views: Sequence[HandMvNetView],
    *,
    min_views: int = 2,
    max_reproj_px: float = DEFAULT_MAX_REPROJ_PX,
    min_depth_m: float = 0.1,
) -> TriangulationResult:
    """Triangula os 21 landmarks por DLT nas vistas dadas (frame do Kinect).

    - undistort por vista com K/dist da propria vista;
    - DLT com todas as vistas validas daquele landmark;
    - gate de reprojecao em px: se falhar e houver 3+ vistas, refaz descartando
      a vista de MAIOR erro (fallback de outlier de uma camera);
    - profundidade positiva (Z > `min_depth_m`) em TODAS as vistas usadas.

    Retorna `TriangulationResult` com (21,3) NaN onde nao deu para triangular,
    mascara de validos e erro de reprojecao por vista (px). A matematica segue a
    mesma convencao de `CameraAlignment`/`triangulate_three_views` do projeto
    (P1=[I|0] para o Kinect, P2=[R|T] para a auxiliar, em coordenadas
    normalizadas) e o teste manual compara as duas implementacoes.
    """
    active = [v for v in views if isinstance(v, HandMvNetView)]
    if len(active) < min_views:
        names = tuple(v.name for v in active)
        return TriangulationResult(
            points_3d=np.full((NUM_HAND_LANDMARKS, 3), np.nan),
            valid_mask=np.zeros(NUM_HAND_LANDMARKS, dtype=bool),
            errors_px={v.name: np.full(NUM_HAND_LANDMARKS, np.nan) for v in active},
            views_used=names,
            views_per_point=np.zeros(NUM_HAND_LANDMARKS, dtype=int),
        )

    names: List[str] = []
    intrinsics: List[np.ndarray] = []
    projections: List[np.ndarray] = []      # (3,4) em coords normalizadas
    measured: List[np.ndarray] = []         # (21,2) normalizado
    tmp_transforms: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
    for view in active:
        names.append(view.name)
        K = np.asarray(view.intrinsic, np.float64)
        intrinsics.append(K)
        measured.append(_undistort_points(view.landmarks_px, K, view.dist))
        if view.is_auxiliary:
            P = np.hstack([np.asarray(view.rotation, np.float64),
                           np.asarray(view.translation, np.float64).reshape(3, 1)])
        else:
            P = np.hstack([np.eye(3), np.zeros((3, 1))])
        projections.append(P)
        tmp_transforms.append((view.rotation, view.translation))

    points = np.full((NUM_HAND_LANDMARKS, 3), np.nan)
    valid = np.zeros(NUM_HAND_LANDMARKS, dtype=bool)
    errors = {name: np.full(NUM_HAND_LANDMARKS, np.nan) for name in names}
    views_per_point = np.zeros(NUM_HAND_LANDMARKS, dtype=int)

    def _errors_for(point: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Erro px por vista e profundidade valida por vista para `point`."""
        errs = np.full(len(names), np.nan)
        depth_ok = np.zeros(len(names), dtype=bool)
        for k, name in enumerate(names):
            K = intrinsics[k]
            R, T = tmp_transforms[k]
            norm = _project_normalized(point.reshape(1, 3), K, R, T)[0]
            z = (point[2] if R is None else
                 float(np.asarray(R)[2] @ point + np.asarray(T)[2]))
            depth_ok[k] = np.isfinite(z) and z > float(min_depth_m)
            if not np.isfinite(norm).all():
                continue
            m = measured[k][idx]
            if not np.isfinite(m).all():
                continue
            errs[k] = float(np.hypot((norm[0] - m[0]) * K[0, 0],
                                     (norm[1] - m[1]) * K[1, 1]))
        return errs, depth_ok

    for idx in range(NUM_HAND_LANDMARKS):
        avail = [k for k in range(len(names))
                 if np.isfinite(measured[k][idx]).all()]
        if len(avail) < min_views:
            continue
        point = _dlt_point([projections[k] for k in avail],
                           [measured[k][idx] for k in avail])
        if point is None:
            continue
        errs, depth_ok = _errors_for(point, idx)
        selected = list(avail)
        used_ok = False
        if np.all(depth_ok[selected]) and np.max(errs[selected]) <= max_reproj_px:
            used_ok = True
        elif len(selected) > 3:
            # Descarta a vista de maior erro, mantendo as outras (>= min_views).
            worst = max(selected, key=lambda k: (errs[k] if np.isfinite(errs[k])
                                                 else np.inf))
            reduced = [k for k in selected if k != worst]
            point2 = _dlt_point([projections[k] for k in reduced],
                                [measured[k][idx] for k in reduced])
            if point2 is not None:
                errs2, depth2 = _errors_for(point2, idx)
                if np.all(depth2[reduced]) and np.max(errs2[reduced]) <= max_reproj_px:
                    point, errs, selected, used_ok = point2, errs2, reduced, True
        if not used_ok:
            # Melhor esforco: aceita com o erro mascarado (fica invalido, mas o
            # erro por vista e' util no diagnostico).
            for k in avail:
                errors[names[k]][idx] = errs[k]
            continue
        points[idx] = point
        valid[idx] = True
        views_per_point[idx] = len(selected)
        for k in range(len(names)):
            errors[names[k]][idx] = errs[k]

    return TriangulationResult(
        points_3d=points,
        valid_mask=valid,
        errors_px=errors,
        views_used=tuple(names),
        views_per_point=views_per_point,
    )


# =============================================================================
# Saida do adaptador
# =============================================================================
@dataclass
class HandMvNetOutput:
    """Pose 3D de UMA mao em UM frame sincronizado.

    `joints_3d` esta' em METROS no frame da camera RGB do Kinect (OpenCV: X
    direita, Y baixo, Z frente), igual a `AuxStereoFrame.points_3d` da
    triangulacao do projeto. `joints_3d_root_relative` e' a saida crua do
    modelo oficial (relativa ao landmark 0/wrist), quando o backend foi o
    modelo; no fallback ela e' `joints_3d - root_3d`.
    """

    hand: str
    timestamp: float
    backend: str                     # "handmvnet" | "handmvnet_random_weights" | "triangulation_fallback" | "unavailable"
    valid: bool = False
    reason: str = ""
    joints_3d: Optional[np.ndarray] = None
    joints_3d_root_relative: Optional[np.ndarray] = None
    root_3d: Optional[np.ndarray] = None
    root_source: str = ""
    joints_2d: Dict[str, np.ndarray] = field(default_factory=dict)
    reproj_errors: Dict[str, np.ndarray] = field(default_factory=dict)
    scores: Optional[np.ndarray] = None
    views_used: Tuple[str, ...] = ()
    latency_ms: float = 0.0
    model_weights: str = ""
    warnings: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ derivados
    @property
    def is_model(self) -> bool:
        return self.backend.startswith("handmvnet")

    def valid_joint_mask(self) -> np.ndarray:
        if self.joints_3d is None:
            return np.zeros(NUM_HAND_LANDMARKS, dtype=bool)
        return np.isfinite(self.joints_3d).all(axis=1)

    def palm_points_3d(self) -> Optional[np.ndarray]:
        """(5,3) pontos do dorso (indices 0,5,9,13,17) ou None."""
        if self.joints_3d is None:
            return None
        pts = np.asarray(self.joints_3d, dtype=np.float64)[list(PALM_LANDMARK_INDICES)]
        if pts.shape != (len(PALM_LANDMARK_INDICES), 3):
            return None
        if not np.isfinite(pts).all():
            return None
        return pts

    def palm_reproj_errors(self) -> Optional[np.ndarray]:
        """(5,) erros de reprojecao do dorso na vista raiz, se houver."""
        if not self.reproj_errors:
            return None
        root_name = self.views_used[0] if self.views_used else None
        errs = self.reproj_errors.get(root_name) if root_name else None
        if errs is None:
            errs = next(iter(self.reproj_errors.values()))
        arr = np.asarray(errs, dtype=np.float64).reshape(-1)
        if arr.size != NUM_HAND_LANDMARKS:
            return None
        palm = arr[list(PALM_LANDMARK_INDICES)]
        return palm if np.isfinite(palm).any() else None

    def as_aux_stereo_frame(self) -> Any:
        """Converte para o contrato `fusion/aux_vision.AuxStereoFrame`.

        E' o encaixe DIRETO no `MultiViewSource` da fusao: ele usa
        `palm_points_3d` e `palm_reproj_errors` (e converte Y para cima).
        Se `fusion.aux_vision` nao estiver importavel, devolve um
        `SimpleNamespace` com os mesmos atributos.
        """
        if self.joints_3d is None:
            points = np.zeros((0, 3), dtype=np.float64)
            reproj = np.zeros((0,), dtype=np.float64)
        else:
            mask = self.valid_joint_mask()
            points = np.asarray(self.joints_3d, dtype=np.float64)[mask]
            root_name = self.views_used[0] if self.views_used else None
            errs = self.reproj_errors.get(root_name) if root_name else None
            reproj = (np.asarray(errs, dtype=np.float64)[mask]
                      if errs is not None else np.full(len(points), np.nan))
        fields = dict(
            hand=self.hand,
            points_3d=points,
            reproj_errors=np.asarray(reproj, dtype=np.float64),
            timestamp=float(self.timestamp),
            valid=bool(self.valid),
            kinect_points=None,
            palm_points_3d=self.palm_points_3d(),
            palm_reproj_errors=self.palm_reproj_errors(),
        )
        try:
            from fusion.aux_vision import AuxStereoFrame  # import tardio
            return AuxStereoFrame(**fields)
        except Exception:
            return SimpleNamespace(**fields)

    def to_dict(self) -> Dict[str, Any]:
        """Versao JSON-friendly (listas no lugar de arrays)."""
        def _arr(value):
            if value is None:
                return None
            return np.asarray(value, dtype=np.float64).tolist()

        return {
            "hand": self.hand,
            "timestamp": float(self.timestamp),
            "backend": self.backend,
            "valid": bool(self.valid),
            "reason": self.reason,
            "joints_3d": _arr(self.joints_3d),
            "joints_3d_root_relative": _arr(self.joints_3d_root_relative),
            "root_3d": _arr(self.root_3d),
            "root_source": self.root_source,
            "joints_2d": {k: _arr(v) for k, v in self.joints_2d.items()},
            "reproj_errors": {k: _arr(v) for k, v in self.reproj_errors.items()},
            "scores": _arr(self.scores),
            "views_used": list(self.views_used),
            "latency_ms": float(self.latency_ms),
            "model_weights": self.model_weights,
            "warnings": list(self.warnings),
            "extra": self.extra,
        }

    def summary(self) -> str:
        """Linha curta para log/terminal."""
        reproj = [np.nanmedian(np.asarray(v, dtype=np.float64))
                  for v in self.reproj_errors.values() if v is not None]
        reproj_txt = (f" reproj_med={np.nanmedian(reproj):.1f}px"
                      if reproj and np.isfinite(np.nanmedian(reproj)) else "")
        return (f"[{self.hand}] backend={self.backend} valid={self.valid} "
                f"n={int(self.valid_joint_mask().sum())}/{NUM_HAND_LANDMARKS} "
                f"lat={self.latency_ms:.1f}ms{reproj_txt}"
                + (f" motivo={self.reason}" if self.reason else ""))


# =============================================================================
# Repositorio oficial: inspecao, shims e carregamento do modelo
# =============================================================================
def _read_git_commit(repo_root: Path) -> Optional[str]:
    """Le o commit atual direto dos arquivos do .git (sem subprocess)."""
    git = repo_root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        try:
            return (git / ref).read_text(encoding="utf-8",
                                         errors="ignore").strip()
        except OSError:
            pass
        try:                                            # clone empacotado
            for line in (git / "packed-refs").read_text(
                    encoding="utf-8", errors="ignore").splitlines():
                if line and not line.startswith("#") and line.endswith(ref):
                    return line.split()[0]
        except OSError:
            return None
    elif re.fullmatch(r"[0-9a-f]{40}", head):
        return head
    return None


def describe_official_repo(repo_root: str | os.PathLike = DEFAULT_REPO_ROOT
                           ) -> Dict[str, Any]:
    """Auditoria do repositorio oficial: presenca, commit, licenca, pesos."""
    root = Path(repo_root)
    src = root / "src"
    present = (src / "models" / "handmvnet.py").is_file()
    info: Dict[str, Any] = {
        "repo_root": str(root),
        "url": OFFICIAL_REPO_URL,
        "present": bool(present),
        "commit": _read_git_commit(root) if present else None,
        "readme": (root / "README.md").is_file(),
        "license_file": next(
            (str(p.name) for p in root.glob("LICEN*") if p.is_file()), None)
        if present else None,
        "license_status": LICENSE_STATUS,
        "weights_status": WEIGHTS_STATUS,
        "weights_in_repo": [],
        "eval_script_modules_missing": [],
        "official_eval_runs_out_of_the_box": False,
    }
    if not present:
        return info
    info["weights_in_repo"] = [str(p.relative_to(root)).replace("\\", "/")
                               for p in root.rglob("*")
                               if p.is_file() and p.suffix in (".ckpt", ".pth", ".pt")]
    missing = [name for name in ("dexycb.py", "mvhand.py")
               if not (src / "datasets" / name).is_file()]
    info["eval_script_modules_missing"] = missing
    info["official_eval_runs_out_of_the_box"] = not missing
    if not info["license_file"]:
        info["license_file"] = None
    return info


def _spec_exists(module_name: str) -> bool:
    """True se o modulo pode ser importado (sem importa-lo)."""
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        return False


def resolve_torch_device(requested: str = "auto") -> str:
    """Resolve o device de inferencia: `cpu`, `xpu` (Intel Arc) ou `cuda`.

    O ambiente do projeto tem Intel Arc (torch 2.14 + XPU) e SEM CUDA. `auto`
    escolhe cuda > xpu > cpu, nessa ordem, so' com o runtime disponivel.
    """
    requested = (requested or "auto").strip().lower()
    if requested in ("cpu", "xpu", "cuda"):
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.xpu.is_available():
            return "xpu"
    except Exception:                                   # pragma: no cover
        pass
    return "cpu"


#: Modulos que o codigo oficial importa mas que NAO sao usados no forward de
#: inferencia. Quando faltam, instalamos um shim so' para o import funcionar:
#: `lightning` (base da classe), `einops` (rearrange), `torchvision` (enums de
#: pesos ImageNet), `vis.visualizer` (visualizacao de treino) e
#: `models.joints_to_vertices` (MANO, desligado via get_vertices=False).


def _install_einops_shim() -> types.ModuleType:
    """`einops.rearrange` minimo para os 2 padroes usados no codigo oficial."""
    module = types.ModuleType("einops")

    def _parse(expr: str) -> List[Tuple[str, ...]]:
        # Tokeniza mantendo "(h d)" inteiro: split() puro quebraria o grupo.
        groups: List[Tuple[str, ...]] = []
        for term in re.findall(r"\([^)]*\)|\S+", expr.strip()):
            if term == "...":
                continue
            if term.startswith("(") and term.endswith(")"):
                groups.append(tuple(t for t in term[1:-1].split() if t and t != "..."))
            else:
                groups.append((term,))
        return groups

    def rearrange(tensor, pattern, **axes):               # noqa: N802
        if hasattr(tensor, "shape"):
            shape = tuple(int(v) for v in tensor.shape)
            is_torch = hasattr(tensor, "permute")
        else:                                           # pragma: no cover
            raise TypeError("rearrange: esperado tensor")
        if "->" not in pattern:
            raise ValueError(f"rearrange: padrao sem '->': {pattern!r}")
        left_expr, right_expr = pattern.split("->")
        left, right = _parse(left_expr), _parse(right_expr)
        sizes: Dict[str, int] = {k: int(v) for k, v in axes.items()}
        ptr = 0
        for group in left:
            if ptr >= len(shape):
                raise ValueError(f"rearrange: dims insuficientes em {pattern!r}")
            if len(group) == 1:
                name = group[0]
                if name in sizes and int(sizes[name]) != shape[ptr] \
                        and shape[ptr] % int(sizes[name]) == 0:
                    # aceita dimensao agrupada em nome simples (nao usado aqui)
                    pass
                sizes.setdefault(name, shape[ptr])
                ptr += 1
            else:
                total = shape[ptr]
                ptr += 1
                known = [n for n in group if n in sizes]
                unknown = [n for n in group if n not in sizes]
                prod = 1
                for n in known:
                    prod *= int(sizes[n])
                if len(unknown) == 1 and prod > 0 and total % prod == 0:
                    sizes[unknown[0]] = total // prod
                elif not unknown:
                    if prod != total:
                        raise ValueError(
                            f"rearrange: grupo {group} nao fecha ({prod} != {total})")
                else:
                    raise ValueError(
                        f"rearrange: nao da' para inferir tamanhos de {group} "
                        f"no padrao {pattern!r} (passe os eixos como kwargs)")
        left_flat = [n for g in left for n in g]
        right_flat = [n for g in right for n in g]
        if sorted(left_flat) != sorted(right_flat):
            raise ValueError(f"rearrange: eixos diferentes em {pattern!r}")
        # Expandir os grupos compostos em eixos separados ANTES do permute
        # (um grupo "(h d)" e' UMA dimensao no tensor de entrada).
        if any(len(g) > 1 for g in left):
            tensor = tensor.reshape(tuple(int(sizes[n]) for n in left_flat))
        if left_flat != right_flat:
            perm = tuple(left_flat.index(n) for n in right_flat)
            if is_torch:
                tensor = tensor.permute(perm)
            else:                                       # pragma: no cover
                tensor = np.transpose(tensor, perm)
        new_shape: List[int] = []
        for group in right:
            prod = 1
            for name in group:
                prod *= int(sizes[name])
            new_shape.append(prod)
        return tensor.reshape(tuple(new_shape))

    module.rearrange = rearrange
    sys.modules.setdefault("einops", module)
    return module


def _install_torchvision_shim() -> None:
    """Stub de `torchvision.models.resnet` (so' os enums de URL de pesos)."""
    if "torchvision.models.resnet" in sys.modules:
        return
    urls = {
        "ResNet18_Weights": "/models/resnet18-5c106cde.pth",
        "ResNet34_Weights": "/models/resnet34-333f7ec4.pth",
        "ResNet50_Weights": "/models/resnet50-19c8e357.pth",
    }
    tv = types.ModuleType("torchvision") if "torchvision" not in sys.modules \
        else sys.modules["torchvision"]
    models = types.ModuleType("torchvision.models")
    resnet = types.ModuleType("torchvision.models.resnet")
    for name, suffix in urls.items():
        enum = type(name, (), {"DEFAULT": SimpleNamespace(
            url="https://download.pytorch.org/models" + suffix)})
        setattr(resnet, name, enum)
    tv.models = models
    models.resnet = resnet
    sys.modules.setdefault("torchvision", tv)
    sys.modules.setdefault("torchvision.models", models)
    sys.modules.setdefault("torchvision.models.resnet", resnet)


def _install_lightning_shim(torch_module) -> None:
    """Stub de `lightning.LightningModule` para inferencia (sem Trainer)."""
    if "lightning" in sys.modules:
        return
    module = types.ModuleType("lightning")

    class LightningModule(torch_module.nn.Module):
        """Base minima: o forward oficial nao usa nenhum recurso do Lightning."""

        def save_hyperparameters(self, *args, **kwargs):
            self.hparams = {}
            return None

        def log(self, *args, **kwargs):
            return None

        def freeze(self):
            self.eval()
            return self

        def save_checkpoint(self, *args, **kwargs):    # pragma: no cover
            raise RuntimeError("shim de inferencia nao salva checkpoints")

    module.LightningModule = LightningModule
    module.seed_everything = lambda *a, **k: None
    sys.modules.setdefault("lightning", module)


def _install_simple_shim(name: str, attrs: Mapping[str, Any]) -> None:
    """Registra um modulo stub com atributos, encadeando no pacote pai."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child, module)


def _bootstrap_official_modules(repo_root: Path,
                                allow_shims: bool = True
                                ) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    """Prepara sys.path + shims e importa a classe oficial `HandMvNet`.

    Devolve (dict com modulo/classe, lista de shims instalados). Levanta
    `HandMvNetUnavailable` com motivo legivel quando nao der.
    """
    if _spec_exists("torch") is False:
        raise HandMvNetUnavailable(
            "torch nao instalado: o modo modelo exige PyTorch (o venv do "
            "projeto tem torch 2.14+xpu).")
    import torch                                           # noqa: F401 (lazy)
    src = (repo_root / "src").resolve()
    if not (src / "models" / "handmvnet.py").is_file():
        raise HandMvNetUnavailable(
            f"repositorio oficial nao encontrado em {repo_root} "
            "(esperado external/HandMvNet com src/models/handmvnet.py).")
    # Vai para o FIM do path: nao compete com os modulos do projeto.
    if str(src) not in sys.path:
        sys.path.append(str(src))

    def _deps_missing() -> List[str]:
        missing = []
        for module_name in ("lightning", "einops", "torchvision",
                            "vis.visualizer", "models.joints_to_vertices"):
            try:
                importlib.import_module(module_name)
            except Exception:
                missing.append(module_name)
        return missing

    shims: List[str] = []
    missing = _deps_missing()
    if missing and not allow_shims:
        raise HandMvNetUnavailable(
            "dependencias do codigo oficial ausentes e shims desabilitados: "
            + ", ".join(missing) + ". Instale-as ou use allow_import_shims=True.")
    if missing:
        if "lightning" in missing:
            _install_lightning_shim(torch)
            shims.append("lightning")
        if "einops" in missing:
            _install_einops_shim()
            shims.append("einops")
        if "torchvision" in missing:
            _install_torchvision_shim()
            shims.append("torchvision")
        if "vis.visualizer" in missing:
            _install_simple_shim("vis.visualizer", {
                "HandPoseVisualizer": type("HandPoseVisualizer", (), {
                    "__init__": lambda self, *a, **k: None})})
            shims.append("vis.visualizer")
        if "models.joints_to_vertices" in missing:
            class _JointsToVerticesStub:                 # noqa: D401
                """Vertices MANO desabilitados no adaptador (use get_vertices=False)."""

                def __init__(self, *args, **kwargs):
                    raise RuntimeError(
                        "JointsToVertices exige manopth + modelos MANO; o "
                        "adaptador desabilita vertices (get_vertices=False).")

            _install_simple_shim("models.joints_to_vertices", {
                "JointsToVertices": _JointsToVerticesStub})
            shims.append("models.joints_to_vertices")
    try:
        models_module = importlib.import_module("models.handmvnet")
    except Exception as exc:
        raise HandMvNetUnavailable(
            f"nao foi possivel importar models.handmvnet do repositorio "
            f"oficial: {type(exc).__name__}: {exc}") from exc
    model_class = getattr(models_module, "HandMvNet", None)
    if model_class is None:
        raise HandMvNetUnavailable("models.handmvnet importou sem a classe HandMvNet")
    return {"module": models_module, "class": model_class, "torch": torch,
            "src": str(src)}, tuple(shims)


def _load_checkpoint_dict(torch_module, path: Path) -> Mapping[str, Any]:
    """Carrega um .ckpt do Lightning (weights_only=False: so' trusted weights)."""
    try:
        checkpoint = torch_module.load(str(path), map_location="cpu",
                                       weights_only=False)
    except TypeError:                                    # torch antigo
        checkpoint = torch_module.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise HandMvNetUnavailable(f"checkpoint {path} nao e' um dict")
    return checkpoint


def _legacy_state_dict_fix(state_dict: Mapping[str, Any]) -> Tuple[Mapping[str, Any], bool]:
    """Mesmo remapeamento de chaves antigas de `eval.py` do repo oficial."""
    legacy_keys = ("pose_net.conv.0.weight", "sample_net.conv.0.weight")
    if not any(key in state_dict for key in legacy_keys):
        return state_dict, False
    from collections import OrderedDict
    fixed = OrderedDict()
    for key, value in state_dict.items():
        new_key = (str(key).replace("pose_net.conv.", "pose_net.")
                   .replace("sample_net.", "sample_nets.0."))
        fixed[new_key] = value
    return fixed, True


def _default_train_params(root_relative: bool = True,
                          device: str = "cpu") -> Dict[str, Any]:
    return {
        "debug": False,
        "root_relative": bool(root_relative),
        "device": device,
        "mask_invisible_joints": False,
        "loss_weights": {"heatmap": 10.0, "joints_2d": 1.0, "joints_3d": 1000.0},
    }


def _default_model_params(num_views: int, *, pos_enc: Sequence[str],
                          backbone: str = "resnet", backbone_type: str = "18",
                          fusion: str = "cross_attn", use_gcn: bool = False
                          ) -> Dict[str, Any]:
    if backbone == "resnet":
        if backbone_type in ("18", "34"):
            channels = [256, 128, 64]        # early_return=3: layer3/2/1
        else:
            channels = [1024]
    elif backbone == "hrnet":
        channels = [40, 80, 160, 320]
    else:
        raise ValueError(f"backbone desconhecido: {backbone}")
    return {
        "selected_views": list(range(int(num_views))),
        "num_views": int(num_views),
        "fusion": fusion,
        "fusion_layers": 5,
        "pos_enc": list(pos_enc),
        "use_gcn": bool(use_gcn),
        "backbone": backbone,
        "backbone_type": backbone_type,
        "backbone_early_return": 3,
        "backbone_channels": channels,
        "backbone_pretrained": False,
        "backbone_pretrained_path": "",
        "freeze_bn": False,
        "get_vertices": False,
    }


def _default_data_params(image_size: int = DEFAULT_IMAGE_SIZE) -> Dict[str, Any]:
    return {
        "name": "ho3d",           # qualquer um dos 3 aceitos: dexycb/ho3d/mvhand
        "batch_size": 1,
        "image_size": int(image_size),
        "heatmap_size": DEFAULT_HEATMAP_SIZE,
        "mask_size": 32,
        "depth_size": 32,
        "mano_models_dir": "src/mano",
        "num_workers": 0,
        "augment": False,
        "pad_image": True,
    }


@dataclass
class _LoadedOfficialModel:
    """Modelo oficial instanciado + metadados para inferencia."""

    model: Any
    torch: Any
    class_name: str
    train_params: Dict[str, Any]
    model_params: Dict[str, Any]
    data_params: Dict[str, Any]
    num_views: int
    image_size: int
    heatmap_size: int
    pos_enc: Tuple[str, ...]
    root_relative: bool
    device: str
    weights_source: str
    shims: Tuple[str, ...]
    load_report: Dict[str, Any] = field(default_factory=dict)

    @property
    def uses_crop_features(self) -> bool:
        return "crop" in self.pos_enc

    @property
    def param_count(self) -> int:
        try:
            return int(sum(p.numel() for p in self.model.parameters()))
        except Exception:                               # pragma: no cover
            return -1

    def summary(self) -> str:
        return (f"HandMvNet({self.model_params.get('backbone')}"
                f"/{self.model_params.get('backbone_type')}, views={self.num_views}, "
                f"pos_enc={list(self.pos_enc)}, root_relative={self.root_relative}, "
                f"params={self.param_count}, device={self.device}, "
                f"weights={self.weights_source})")

    def forward_numpy(self, crops: np.ndarray, bboxes: np.ndarray,
                      intrinsics: np.ndarray, extrinsics: np.ndarray
                      ) -> Dict[str, np.ndarray]:
        """Roda `HandMvNet.forward` com batch=1 e devolve arrays numpy."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(crops, dtype=np.float32))
        bbox_t = torch.from_numpy(np.ascontiguousarray(bboxes, dtype=np.float32))
        if self.device != "cpu":
            x = x.to(self.device)
            bbox_t = bbox_t.to(self.device)
        cam_params = {
            "intrinsic": torch.from_numpy(
                np.ascontiguousarray(intrinsics, dtype=np.float32)).to(self.device),
            "extrinsic": torch.from_numpy(
                np.ascontiguousarray(extrinsics, dtype=np.float32)).to(self.device),
        }
        with torch.inference_mode():
            out = self.model(x, bbox_t, cam_params)
        result: Dict[str, np.ndarray] = {}
        for key, value in (out or {}).items():
            if hasattr(value, "detach"):
                result[key] = value.detach().to("cpu").numpy()
            else:                                       # pragma: no cover
                result[key] = np.asarray(value)
        return result


def _build_model_instance(module_info: Dict[str, Any], train_params: Dict[str, Any],
                          model_params: Dict[str, Any], data_params: Dict[str, Any],
                          *, device: str, shims: Tuple[str, ...],
                          weights_source: str,
                          load_report: Optional[Dict[str, Any]] = None
                          ) -> _LoadedOfficialModel:
    torch = module_info["torch"]
    model_class = module_info["class"]
    model = model_class(train_params, model_params, data_params)
    model.eval()
    if device != "cpu":
        try:
            model = model.to(device)
        except Exception as exc:
            raise HandMvNetUnavailable(
                f"nao consegui mover o modelo para {device!r}: "
                f"{type(exc).__name__}: {exc}") from exc
    loaded = _LoadedOfficialModel(
        model=model,
        torch=torch,
        class_name=model_class.__name__,
        train_params=dict(train_params),
        model_params=dict(model_params),
        data_params=dict(data_params),
        num_views=int(model_params["num_views"]),
        image_size=int(data_params["image_size"]),
        heatmap_size=int(data_params["heatmap_size"]),
        pos_enc=tuple(model_params.get("pos_enc", [])),
        root_relative=bool(train_params.get("root_relative", True)),
        device=device,
        weights_source=weights_source,
        shims=tuple(shims),
        load_report=dict(load_report or {}),
    )
    return loaded


def _load_state_dict_tolerant(model: Any, state_dict: Mapping[str, Any],
                              *, strict: bool) -> Dict[str, Any]:
    """Carrega state_dict e, se precisar, filtra chaves de shape incompativel."""
    report: Dict[str, Any] = {"strict": strict, "dropped_keys": 0}
    try:
        result = model.load_state_dict(state_dict, strict=strict)
        report["missing_keys"] = list(getattr(result, "missing_keys", []) or [])
        report["unexpected_keys"] = list(getattr(result, "unexpected_keys", []) or [])
        return report
    except RuntimeError:
        model_sd = model.state_dict()
        filtered = {}
        dropped = 0
        for key, value in state_dict.items():
            if key in model_sd and tuple(model_sd[key].shape) == tuple(value.shape):
                filtered[key] = value
            else:
                dropped += 1
        report["dropped_keys"] = dropped
        result = model.load_state_dict(filtered, strict=False)
        report["missing_keys"] = list(getattr(result, "missing_keys", []) or [])
        report["unexpected_keys"] = list(getattr(result, "unexpected_keys", []) or [])
        report["filtered"] = True
        return report


def load_official_model(
    repo_root: str | os.PathLike = DEFAULT_REPO_ROOT,
    checkpoint: Optional[str | os.PathLike] = None,
    *,
    device: str = "auto",
    config_path: Optional[str | os.PathLike] = None,
    num_views_override: Optional[int] = None,
    adapt_num_views: bool = False,
    allow_import_shims: bool = True,
) -> _LoadedOfficialModel:
    """Instancia o modelo OFICIAL com um `.ckpt` (ou config, se sem checkpoint).

    Args:
        repo_root: raiz do clone oficial (`external/HandMvNet`).
        checkpoint: caminho do `.ckpt` do Lightning. Se None, levanta
            `HandMvNetUnavailable` (pesos nao publicados) -- use
            `load_random_model_for_validation` apenas para smoke test.
        device: "auto" (cuda>xpu>cpu), "cpu", "xpu" ou "cuda".
        config_path: yaml oficial usado quando o checkpoint NAO traz
            `hyper_parameters` (fallback). Sem ele usa `configs/exp/default.yaml`.
        num_views_override: forca outro numero de vistas (experimental; so'
            faz sentido com `adapt_num_views=True`).
        adapt_num_views: aceita carregar um checkpoint treinado com N vistas no
            modelo reconstruido com `num_views_override` vistas. NAO VALIDADO
            cientificamente: as camadas de atencao nao dependem do numero de
            vistas, mas a qualidade com menos vistas e' desconhecida.
        allow_import_shims: instala shims de import quando `lightning`,
            `einops`, `torchvision`, `vis.visualizer` ou `models.joints_to_vertices`
            faltam (nenhum deles participa do forward de inferencia).

    Returns:
        `_LoadedOfficialModel` pronto para `forward_numpy`.
    """
    repo = Path(repo_root)
    if checkpoint is None:
        raise HandMvNetUnavailable(
            "sem checkpoint HandMvNet: os pesos NAO sao publicados pelo "
            "repositorio oficial (ver WEIGHTS_STATUS). Coloque um .ckpt em "
            f"{repo / 'weights'} ou passe `checkpoint=`/`HANDMVNET_CHECKPOINT`.")
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise HandMvNetUnavailable(f"checkpoint nao encontrado: {ckpt_path}")

    module_info, shims = _bootstrap_official_modules(repo, allow_import_shims)
    torch = module_info["torch"]
    device_resolved = resolve_torch_device(device)
    checkpoint_data = _load_checkpoint_dict(torch, ckpt_path)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    state_dict, legacy = _legacy_state_dict_fix(state_dict)

    hyper = checkpoint_data.get("hyper_parameters") or {}
    if not hyper:
        # Checkpoint sem hyper_parameters: usa o yaml oficial (se existir).
        yaml_path = Path(config_path) if config_path else (
            repo / "configs" / "exp" / "default.yaml")
        if not yaml_path.is_file():
            raise HandMvNetUnavailable(
                "checkpoint sem `hyper_parameters` e sem config yaml valido "
                f"({yaml_path}); nao da' para reconstruir a arquitetura.")
        import yaml
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        train_params = dict(cfg.get("train", {}))
        model_params = dict(cfg.get("model", {}))
        data_params = dict(cfg.get("data", {}))
        model_params["num_views"] = len(model_params.get("selected_views", []))
    else:
        train_params = dict(hyper.get("train_params", {}) or {})
        model_params = dict(hyper.get("model_params", {}) or {})
        data_params = dict(hyper.get("data_params", {}) or {})

    # Overrides de inferencia (nao alteram o repositorio externo).
    model_params["get_vertices"] = False
    model_params["backbone_pretrained"] = False
    model_params.setdefault("backbone", "resnet")
    model_params.setdefault("backbone_type", "18")
    model_params.setdefault("fusion", "cross_attn")
    model_params.setdefault("use_gcn", True)
    model_params.setdefault("pos_enc", ["pos2d", "sin"])
    model_params.setdefault("fusion_layers", 5)
    train_params.setdefault("root_relative", True)
    train_params.setdefault("debug", False)
    train_params["device"] = device_resolved
    train_params.setdefault("mask_invisible_joints", False)
    train_params.setdefault("loss_weights", {"heatmap": 10.0, "joints_2d": 1.0,
                                             "joints_3d": 1000.0})
    data_params.setdefault("name", "ho3d")
    data_params.setdefault("batch_size", 1)
    data_params.setdefault("image_size", DEFAULT_IMAGE_SIZE)
    data_params.setdefault("heatmap_size", DEFAULT_HEATMAP_SIZE)
    data_params.setdefault("mask_size", 32)
    data_params.setdefault("depth_size", 32)
    data_params.setdefault("mano_models_dir", "src/mano")

    checkpoint_views = int(model_params.get("num_views") or 0)
    target_views = (int(num_views_override) if num_views_override is not None
                    else checkpoint_views)
    if target_views <= 0:
        raise HandMvNetUnavailable(
            "numero de vistas do checkpoint desconhecido (hyper_parameters sem "
            "`num_views`); passe `num_views_override` ou um yaml em config_path.")
    if checkpoint_views and target_views != checkpoint_views and not adapt_num_views:
        raise HandMvNetUnavailable(
            f"checkpoint treinado com {checkpoint_views} vistas, mas a inferencia "
            f"pede {target_views}. Passe `adapt_num_views=True` para adaptar "
            "(experimental) ou use um checkpoint treinado para este numero de "
            "vistas.")
    model_params["num_views"] = target_views
    model_params["selected_views"] = list(range(target_views))
    data_params["num_views"] = target_views
    data_params["selected_views"] = list(range(target_views))

    weights_source = f"checkpoint:{ckpt_path}"
    loaded = _build_model_instance(
        module_info, train_params, model_params, data_params,
        device=device_resolved, shims=shims, weights_source=weights_source)
    strict = not (adapt_num_views and target_views != checkpoint_views)
    report = _load_state_dict_tolerant(loaded.model, state_dict, strict=strict)
    report["legacy_keys_remapped"] = bool(legacy)
    report["checkpoint_views"] = checkpoint_views
    report["inference_views"] = target_views
    loaded.load_report = report
    loaded.model.eval()
    return loaded


def load_random_model_for_validation(
    repo_root: str | os.PathLike = DEFAULT_REPO_ROOT,
    *,
    num_views: int = 3,
    image_size: int = DEFAULT_IMAGE_SIZE,
    pos_enc: Sequence[str] = ("pos2d", "crop", "sin"),
    backbone: str = "resnet",
    backbone_type: str = "18",
    use_gcn: bool = False,
    root_relative: bool = True,
    device: str = "cpu",
    allow_import_shims: bool = True,
) -> _LoadedOfficialModel:
    """Modelo OFICIAL com pesos ALEATORIOS -- SO' para smoke test do plumbing.

    Nao existe checkpoint publico (ver WEIGHTS_STATUS). Este caminho monta a
    arquitetura oficial real (mesmas camadas/forward) com inicializacao Kaiming
    e roda o forward de verdade: serve para provar que o encanamento
    (crops -> tensor -> forward -> junções 3D -> mm/px) funciona e para medir
    latencia no hardware, SEM afirmar nada sobre a QUALIDADE da predicao.
    """
    module_info, shims = _bootstrap_official_modules(Path(repo_root),
                                                     allow_import_shims)
    device_resolved = resolve_torch_device(device)
    train_params = _default_train_params(root_relative=root_relative,
                                         device=device_resolved)
    model_params = _default_model_params(num_views, pos_enc=pos_enc,
                                         backbone=backbone,
                                         backbone_type=backbone_type,
                                         use_gcn=use_gcn)
    data_params = _default_data_params(image_size)
    return _build_model_instance(
        module_info, train_params, model_params, data_params,
        device=device_resolved, shims=shims,
        weights_source="random_weights_validation_only")


# =============================================================================
# Adaptador
# =============================================================================
class HandMvNetAdapter:
    """Integra HandMvNet (ou triangulacao classica) ao pipeline multiview.

    Exemplo tipico no pipeline (frames ja' sincronizados e landmarks MediaPipe):

        from vision.handmvnet_adapter import HandMvNetAdapter, HandMvNetView

        adapter = HandMvNetAdapter(checkpoint=None)      # fallback hoje
        views = [
            HandMvNetView.from_alignment(alignment, "kinect", bgra, kin_px,
                                         is_kinect=True),
            HandMvNetView.from_alignment(alignment, "aux0", aux_frame, aux_px,
                                         aux_index=0),
        ]
        out = adapter.infer(views, hand="right")
        if out.valid:
            print(out.palm_points_3d(), out.reproj_errors)

    O modo `auto` tenta o modelo oficial quando ha checkpoint+dependencias e o
    numero de vistas casa; caso contrario (ou se o forward falhar) usa a
    triangulacao e explica o motivo em `HandMvNetOutput.reason`.
    """

    def __init__(
        self,
        *,
        repo_root: str | os.PathLike = DEFAULT_REPO_ROOT,
        checkpoint: Optional[str | os.PathLike] = None,
        calibration_dir: Optional[str | os.PathLike] = None,
        device: str = "auto",
        mode: str = "auto",
        image_size: Optional[int] = None,
        crop_margin: float = DEFAULT_CROP_MARGIN,
        min_valid_points: int = 5,
        max_reproj_error_px: float = DEFAULT_MAX_REPROJ_PX,
        allow_import_shims: bool = True,
        adapt_num_views: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            repo_root: clone oficial (`external/HandMvNet`).
            checkpoint: `.ckpt`; se None, tenta env `HANDMVNET_CHECKPOINT` e
                depois procura `*.ckpt` no clone/projeto.
            calibration_dir: pasta dos npz (default `configs/calibracao`).
            device: "auto"/"cpu"/"xpu"/"cuda".
            mode: "auto" (modelo se possivel, senao fallback), "model" (exige
                modelo; se indisponivel devolve `valid=False` sem cair no
                fallback) ou "triangulation" (forca fallback).
            image_size: lado do crop (default: o do checkpoint; senao 256).
            crop_margin: margem do bbox quadrado (1.2 = 20%).
            min_valid_points: minimo de junções validas para `valid=True`.
            max_reproj_error_px: gate do fallback e das checagens do modelo.
            allow_import_shims: shims de import para o codigo oficial.
            adapt_num_views: experimental (ver `load_official_model`).
            verbose: imprime o status na criacao.
        """
        self.repo_root = Path(repo_root)
        self.calibration_dir = Path(calibration_dir or DEFAULT_CALIBRATION_DIR)
        self.device = (device or "auto").strip().lower()
        if mode not in ("auto", "model", "triangulation"):
            raise ValueError("mode deve ser 'auto', 'model' ou 'triangulation'")
        self.mode = mode
        self.crop_margin = float(crop_margin)
        self.min_valid_points = int(min_valid_points)
        self.max_reproj_error_px = float(max_reproj_error_px)
        self.allow_import_shims = bool(allow_import_shims)
        self.adapt_num_views = bool(adapt_num_views)
        self.verbose = bool(verbose)
        self._checkpoint_arg = Path(checkpoint) if checkpoint else None
        self._image_size_override = int(image_size) if image_size else None
        self._models: Dict[int, _LoadedOfficialModel] = {}
        self._model_error: Optional[str] = None
        self._status_cache: Optional[Dict[str, Any]] = None
        if self.verbose:
            print(json.dumps(self.status(), indent=2, ensure_ascii=False,
                             default=str))

    # ---------------------------------------------------------------- status
    def _resolve_checkpoint(self) -> Optional[Path]:
        if self._checkpoint_arg is not None:
            return self._checkpoint_arg if self._checkpoint_arg.is_file() else None
        env = os.environ.get("HANDMVNET_CHECKPOINT", "").strip()
        if env and Path(env).is_file():
            return Path(env)
        candidates: List[Path] = []
        for folder in (self.repo_root / "weights", self.repo_root / "checkpoints",
                       self.repo_root, _PROJECT_ROOT / "assets" / "models",
                       _PROJECT_ROOT / "weights"):
            if not folder.is_dir():
                continue
            try:
                candidates.extend(sorted(folder.rglob("*.ckpt")))
                candidates.extend(sorted(folder.glob("handmvnet*")))
            except OSError:
                continue
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _resolve_image_size(self) -> int:
        if self._image_size_override:
            return self._image_size_override
        for loaded in self._models.values():
            return int(loaded.image_size)
        return DEFAULT_IMAGE_SIZE

    def status(self, refresh: bool = False) -> Dict[str, Any]:
        """Diagnostico do modo modelo: repo, deps, checkpoint, licenca, pesos."""
        if self._status_cache is not None and not refresh:
            return dict(self._status_cache)
        repo_info = describe_official_repo(self.repo_root)
        torch_available = _spec_exists("torch")
        deps = {name: _spec_exists(name) for name in
                ("lightning", "einops", "torchvision", "plotly", "manopth",
                 "transforms3d")}
        checkpoint = self._resolve_checkpoint()
        missing_optional = [name for name, ok in deps.items() if not ok]
        reasons: List[str] = []
        if not repo_info["present"]:
            reasons.append(f"clone oficial ausente em {self.repo_root}")
        if not torch_available:
            reasons.append("torch nao instalado")
        if checkpoint is None:
            reasons.append("checkpoint .ckpt ausente (pesos nao publicados)")
        if self.mode == "triangulation":
            reasons.append("modo 'triangulation' forcado")
        model_ready = bool(repo_info["present"] and torch_available
                           and checkpoint is not None
                           and self.mode != "triangulation")
        if model_ready and missing_optional and not self.allow_import_shims:
            model_ready = False
            reasons.append("dependencias ausentes e shims desabilitados: "
                           + ", ".join(missing_optional))
        status = {
            "mode": self.mode,
            "repo_root": str(self.repo_root),
            "repo": repo_info,
            "torch_available": bool(torch_available),
            "device_requested": self.device,
            "optional_deps": deps,
            "optional_deps_missing": missing_optional,
            "import_shims_allowed": self.allow_import_shims,
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "checkpoint_present": checkpoint is not None,
            "model_ready": bool(model_ready),
            "model_error": self._model_error,
            "reasons": reasons,
            "license_status": LICENSE_STATUS,
            "weights_status": WEIGHTS_STATUS,
            "eval_script_modules_missing": repo_info["eval_script_modules_missing"],
            "checkpoint_num_views_hint": None,
        }
        if self._models:
            first = next(iter(self._models.values()))
            status["checkpoint_num_views_hint"] = first.num_views
            status["device_resolved"] = first.device
            status["model"] = {
                "summary": first.summary(),
                "shims": list(first.shims),
                "load_report": first.load_report,
            }
        self._status_cache = status
        return dict(status)

    # ------------------------------------------------------------- carregamento
    def load_model(self, num_views: int) -> Optional[_LoadedOfficialModel]:
        """Carrega/recupera o modelo oficial para `num_views` vistas.

        Devolve None (e guarda o motivo em `self._model_error`) quando o
        modelo nao estiver disponivel -- nunca levanta por falta de pesos,
        porque o adaptador tem fallback.
        """
        num_views = int(num_views)
        if num_views in self._models:
            return self._models[num_views]
        checkpoint = self._resolve_checkpoint()
        if self.mode == "triangulation":
            self._model_error = "modo 'triangulation' forcado"
            return None
        if checkpoint is None:
            self._model_error = (
                "checkpoint HandMvNet ausente (pesos nao publicados; ver "
                "WEIGHTS_STATUS). O adaptador usa o fallback geometrico.")
            return None
        try:
            loaded = load_official_model(
                self.repo_root, checkpoint, device=self.device,
                config_path=None, num_views_override=num_views,
                adapt_num_views=self.adapt_num_views,
                allow_import_shims=self.allow_import_shims)
        except HandMvNetUnavailable as exc:
            self._model_error = str(exc)
            return None
        except Exception as exc:                        # pragma: no cover
            self._model_error = f"falha ao carregar o modelo: {type(exc).__name__}: {exc}"
            return None
        self._models[num_views] = loaded
        self._model_error = None
        self._status_cache = None
        return loaded

    def attach_random_model(self, num_views: int = 3, *,
                            pos_enc: Sequence[str] = ("pos2d", "crop", "sin"),
                            device: Optional[str] = None,
                            root_relative: bool = True) -> _LoadedOfficialModel:
        """Anexa o modelo oficial com pesos ALEATORIOS (SO' smoke test).

        Nao e' caminho de producao: existe para validar o encanamento e medir
        latencia sem checkpoint. A saida do adaptador marca
        `backend="handmvnet_random_weights"` e nunca fica `valid=True` por
        qualidade (o gate de plausibilidade continua valendo).
        """
        loaded = load_random_model_for_validation(
            self.repo_root, num_views=num_views,
            image_size=self._image_size_override or DEFAULT_IMAGE_SIZE,
            pos_enc=pos_enc, root_relative=root_relative,
            device=device or self.device,
            allow_import_shims=self.allow_import_shims)
        self._models[int(num_views)] = loaded
        self._model_error = None
        self._status_cache = None
        return loaded

    def _model_for_views(self, n_views: int) -> Tuple[Optional[_LoadedOfficialModel],
                                                      str]:
        """Modelo para `n_views` ou (None, motivo) sem tentar de novo a cada frame."""
        if n_views in self._models:
            return self._models[n_views], ""
        if self._model_error:
            return None, self._model_error
        loaded = self.load_model(n_views)
        if loaded is None:
            return None, self._model_error or "modelo indisponivel"
        expected = loaded.num_views
        if expected != n_views:                         # pragma: no cover
            return None, f"modelo com {expected} vistas != {n_views} vistas recebidas"
        return loaded, ""

    # -------------------------------------------------------------- inferencia
    def infer(
        self,
        views: Sequence[HandMvNetView],
        hand: str = "right",
        *,
        timestamp: Optional[float] = None,
        root_anchor: Optional[np.ndarray] = None,
    ) -> HandMvNetOutput:
        """Estima a pose 3D de UMA mao num frame sincronizado.

        Args:
            views: >= 2 `HandMvNetView` sincronizadas; a PRIMEIRA e' a vista
                raiz (Kinect) e define o referencial da saida.
            hand: "right" ou "left" (a lateralidade em si vem do detector; o
                adaptador so' a rotula na saida).
            timestamp: segundos (default: `time.monotonic()`).
            root_anchor: (3,) posicao do landmark 0 (wrist) no frame do Kinect,
                em metros, se quem chama ja' tem uma ancora melhor (ex.: fusao).
                Sem ela o adaptador triangula a raiz das mesmas vistas.

        Returns:
            `HandMvNetOutput` (nunca levanta por falta de modelo: o motivo vai
            em `reason`/`warnings`).
        """
        t0 = time.perf_counter()
        now = float(timestamp) if timestamp is not None else time.monotonic()
        hand = str(hand).lower().strip()
        if hand not in ("right", "left"):
            raise ValueError("hand deve ser 'right' ou 'left'")
        views = [v for v in views if isinstance(v, HandMvNetView)]
        warnings_: List[str] = []
        if len(views) < 2:
            return HandMvNetOutput(
                hand=hand, timestamp=now, backend="unavailable",
                valid=False,
                reason=(f"precisa de >=2 vistas sincronizadas; recebi {len(views)}"),
                latency_ms=(time.perf_counter() - t0) * 1e3,
            )
        names = [v.name for v in views]
        if len(set(names)) != len(names):
            raise ValueError(f"nomes de vista duplicados: {names}")
        if root_anchor is not None:
            anchor = np.asarray(root_anchor, dtype=np.float64).reshape(3)
            if not np.isfinite(anchor).all():
                anchor = None
        else:
            anchor = None

        model: Optional[_LoadedOfficialModel] = None
        model_reason = ""
        if self.mode != "triangulation":
            model, model_reason = self._model_for_views(len(views))
            if model is not None and len(views) != model.num_views:
                model_reason = (f"checkpoint com {model.num_views} vistas != "
                                f"{len(views)} vistas recebidas")
                model = None
        output: Optional[HandMvNetOutput] = None
        if model is not None:
            try:
                output = self._infer_model(model, views, hand, now, anchor,
                                           warnings_)
            except Exception as exc:                    # pragma: no cover
                model_reason = (f"forward do modelo falhou: "
                                f"{type(exc).__name__}: {exc}")
                warnings_.append(model_reason)
                output = None
        if output is None:
            if self.mode == "model":
                return HandMvNetOutput(
                    hand=hand, timestamp=now, backend="unavailable",
                    valid=False,
                    reason=(model_reason or self._model_error
                            or "modelo indisponivel em mode='model'"),
                    views_used=tuple(names),
                    warnings=tuple(warnings_),
                    latency_ms=(time.perf_counter() - t0) * 1e3,
                )
            output = self._infer_triangulation(views, hand, now, warnings_,
                                               model_reason)
        output.latency_ms = (time.perf_counter() - t0) * 1e3
        if warnings_:
            output.warnings = tuple(list(output.warnings) + warnings_)
        return output

    # ---------------------------------------------------------- modelo oficial
    def _prepare_model_inputs(self, views: Sequence[HandMvNetView],
                              image_size: int
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, List[np.ndarray]]:
        """Crops normalizados (1,V,3,S,S), bboxes (1,V,4), K (1,V,4), E (1,V,4,4)."""
        crops, bboxes, intrinsics, extrinsics = [], [], [], []
        bboxes_list: List[np.ndarray] = []
        for view in views:
            bbox = square_bbox_from_landmarks(view.landmarks_px,
                                              view.size, self.crop_margin)
            bboxes_list.append(bbox)
            crop = crop_and_pad_image(view.image, bbox)
            crops.append(image_to_model_tensor(crop, image_size,
                                               image_bgr=view.image_bgr))
            bboxes.append(bbox.astype(np.float32))
            K = np.asarray(view.intrinsic, np.float64)
            intrinsics.append(np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                                       dtype=np.float32))
            extrinsics.append(view.extrinsic().astype(np.float32))
        return (
            np.stack(crops, axis=0)[None, ...].astype(np.float32),
            np.stack(bboxes, axis=0)[None, ...],
            np.stack(intrinsics, axis=0)[None, ...],
            np.stack(extrinsics, axis=0)[None, ...],
            bboxes_list,
        )

    def _infer_model(self, model: _LoadedOfficialModel,
                     views: Sequence[HandMvNetView], hand: str, now: float,
                     root_anchor: Optional[np.ndarray],
                     warnings_: List[str]) -> HandMvNetOutput:
        image_size = model.image_size
        x, bbox, intrinsic, extrinsic, bboxes_list = self._prepare_model_inputs(
            views, image_size)
        out = model.forward_numpy(x, bbox, intrinsic, extrinsic)
        if "joints_cam" not in out:
            raise HandMvNetOutError(f"forward sem 'joints_cam': {sorted(out)}")
        joints_rel = np.asarray(out["joints_cam"], dtype=np.float64).reshape(
            NUM_HAND_LANDMARKS, 3)
        # Raiz: ancora externa > triangulacao das vistas > raiz predita.
        triang: Optional[TriangulationResult] = None
        root = None
        root_source = ""
        if root_anchor is not None:
            root, root_source = np.asarray(root_anchor, np.float64), "external"
        else:
            triang = triangulate_views(views, max_reproj_px=self.max_reproj_error_px)
            root = self._root_from_triangulation(triang)
            if root is not None:
                root_source = "triangulation"
        if root is None and not model.root_relative and "root_joint" in out:
            root = np.asarray(out["root_joint"], dtype=np.float64).reshape(3)
            root_source = "model"
        joints_abs = (joints_rel + root.reshape(1, 3)
                      if root is not None else np.full_like(joints_rel, np.nan))

        # 2D por vista: coords do crop (escala image_size) -> pixels da imagem.
        joints_2d: Dict[str, np.ndarray] = {}
        joints_crop = out.get("joints_crop_img")
        if joints_crop is not None:
            arr = np.asarray(joints_crop, dtype=np.float64)
            if arr.ndim == 4:                            # (1,V,21,2)
                arr = arr[0]
            for i, view in enumerate(views):
                if i >= arr.shape[0]:
                    continue
                joints_2d[view.name] = crop_to_joints(arr[i], bboxes_list[i],
                                                      image_size)
        # Confianca 2D: pico do softmax espacial dos heatmaps oficiais.
        scores = None
        heatmap = out.get("heatmap")
        if heatmap is not None:
            hm = np.asarray(heatmap, dtype=np.float64)
            if hm.ndim == 5:
                hm = hm[0]
            try:
                flat = hm.reshape(hm.shape[0], hm.shape[1], -1)
                flat = flat - flat.max(axis=-1, keepdims=True)
                expo = np.exp(flat)
                probs = expo / np.maximum(expo.sum(axis=-1, keepdims=True), 1e-12)
                scores = probs.max(axis=-1).mean(axis=0)
            except Exception as exc:                    # pragma: no cover
                warnings_.append(f"nao deu para extrair score do heatmap: {exc}")

        reproj = self._reprojection_errors(views, joints_abs)
        backend = ("handmvnet_random_weights"
                   if model.weights_source.startswith("random")
                   else "handmvnet")
        valid, gate_reasons = self._validate_joints(joints_abs, joints_rel, root)
        reasons = list(gate_reasons)
        if backend == "handmvnet_random_weights":
            # Pesos aleatorios NUNCA sao predicao valida, mesmo que os numeros
            # passem no gate de plausibilidade (offset/depth): o encanamento e'
            # o que esta' sendo testado, nao a qualidade.
            valid = False
            reasons.append("pesos ALEATORIOS (validacao de encanamento; nao e' predicao)")
        else:
            # Com pesos reais, erro de reprojecao alto e' sinal de dominio
            # diferente do treino; nao invalida sozinho (o gate e' geometrico),
            # mas precisa aparecer no relatorio.
            root_name = views[0].name
            root_err = reproj.get(root_name)
            if root_err is not None:
                med = float(np.nanmedian(root_err))
                if np.isfinite(med) and med > 2.0 * self.max_reproj_error_px:
                    warnings_.append(
                        f"reprojecao mediana {med:.1f}px na vista raiz > "
                        f"2x o gate ({2.0 * self.max_reproj_error_px:.1f}px): "
                        "possivel dominio diferente do treino do checkpoint")
        if model.load_report.get("dropped_keys"):
            warnings_.append(f"{model.load_report['dropped_keys']} chaves de "
                             "checkpoint ignoradas por shape incompativel")
        if model.shims:
            warnings_.append("shims de import usados: " + ", ".join(model.shims))
        output = HandMvNetOutput(
            hand=hand, timestamp=now, backend=backend, valid=bool(valid),
            reason="; ".join(reasons),
            joints_3d=joints_abs,
            joints_3d_root_relative=joints_rel,
            root_3d=None if root is None else np.asarray(root, np.float64),
            root_source=root_source,
            joints_2d=joints_2d,
            reproj_errors=reproj,
            scores=scores,
            views_used=tuple(v.name for v in views),
            model_weights=model.weights_source,
            extra={
                "model": model.summary(),
                "root_relative": bool(model.root_relative),
                "pos_enc": list(model.pos_enc),
                "image_size": int(image_size),
                "bboxes": [b.tolist() for b in bboxes_list],
                "triangulation_root_valid": (
                    None if triang is None else bool(triang.valid_mask[0])),
            },
        )
        return output

    def _root_from_triangulation(self, triang: TriangulationResult
                                 ) -> Optional[np.ndarray]:
        if triang.valid_mask[0]:
            return np.asarray(triang.points_3d[0], np.float64)
        palm_idx = list(PALM_LANDMARK_INDICES)
        palm_valid = np.isfinite(triang.points_3d[palm_idx]).all(axis=1)
        if palm_valid.any():
            return np.median(triang.points_3d[np.asarray(palm_idx)[palm_valid]],
                             axis=0)
        if triang.valid_mask.any():
            return np.median(triang.points_3d[triang.valid_mask], axis=0)
        return None

    def _reprojection_errors(self, views: Sequence[HandMvNetView],
                             joints_3d: np.ndarray
                             ) -> Dict[str, np.ndarray]:
        """Erro de reprojecao (px) das junções 3D contra os landmarks de cada vista."""
        errors: Dict[str, np.ndarray] = {}
        if joints_3d is None:
            return errors
        pts = np.asarray(joints_3d, dtype=np.float64).reshape(-1, 3)
        for view in views:
            K = np.asarray(view.intrinsic, np.float64)
            measured = _undistort_points(view.landmarks_px, K, view.dist)
            proj = _project_normalized(pts, K, view.rotation, view.translation)
            err = np.hypot((proj[:, 0] - measured[:, 0]) * K[0, 0],
                           (proj[:, 1] - measured[:, 1]) * K[1, 1])
            err = np.asarray(err, dtype=np.float64)
            err[~np.isfinite(err)] = np.nan
            errors[view.name] = err
        return errors

    def _validate_joints(self, joints_abs: np.ndarray, joints_rel: np.ndarray,
                         root: Optional[np.ndarray]
                         ) -> Tuple[bool, List[str]]:
        """Gate de plausibilidade comum a modelo e fallback."""
        reasons: List[str] = []
        if root is None or not np.isfinite(root).all():
            reasons.append("raiz 3D indisponivel (sem triangulacao da wrist)")
            return False, reasons
        finite = np.isfinite(joints_abs).all(axis=1)
        if finite.sum() < self.min_valid_points:
            reasons.append(f"apenas {int(finite.sum())}/{NUM_HAND_LANDMARKS} "
                           "junções finitas")
            return False, reasons
        offsets = np.linalg.norm(np.asarray(joints_rel, np.float64), axis=1)
        palm_len = float(np.nanmax(offsets[finite])) if finite.any() else np.inf
        depth = np.asarray(joints_abs, np.float64)[:, 2]
        plausible = (finite
                     & (offsets < 0.6)                       # mao < 60 cm do punho
                     & (depth > 0.15) & (depth < 5.0)        # frente do Kinect
                     & (np.abs(np.asarray(joints_abs, np.float64)[:, 0]) < 3.0)
                     & (np.abs(np.asarray(joints_abs, np.float64)[:, 1]) < 3.0))
        n_ok = int(plausible.sum())
        if n_ok < self.min_valid_points:
            reasons.append(
                f"saida implausivel: {n_ok}/{NUM_HAND_LANDMARKS} junções "
                f"dentro do gate (maior offset={palm_len * 100:.1f}cm, "
                f"z_med={np.nanmedian(depth):.2f}m)")
            return False, reasons
        return True, reasons

    # ----------------------------------------------------- fallback geometrico
    def _infer_triangulation(self, views: Sequence[HandMvNetView], hand: str,
                             now: float, warnings_: List[str],
                             model_reason: str) -> HandMvNetOutput:
        triang = triangulate_views(
            views, max_reproj_px=self.max_reproj_error_px)
        points = np.asarray(triang.points_3d, np.float64)
        root = self._root_from_triangulation(triang)
        if root is not None:
            joints_rel = points - root.reshape(1, 3)
        else:
            joints_rel = np.full_like(points, np.nan)
        joints_2d = {v.name: np.asarray(v.landmarks_px, np.float64).copy()
                     for v in views}
        valid, reasons = self._validate_joints(points, joints_rel, root)
        if not valid and triang.n_valid < self.min_valid_points:
            reasons = [f"triangulacao com {triang.n_valid}/{NUM_HAND_LANDMARKS} "
                       "junções válidas"] + reasons
        if model_reason:
            reasons = [f"modelo nao usado: {model_reason}"] + reasons
        root_source = ("triangulation" if root is not None and
                       (triang.valid_mask[0] if len(triang.valid_mask) else False)
                       else "triangulation_palm_median" if root is not None else "")
        if triang.n_valid > 0 and np.isfinite(points[triang.valid_mask]).all():
            med = np.nanmedian(np.stack([
                np.nanmedian(np.asarray(err))
                for err in triang.errors_px.values()
                if np.isfinite(np.asarray(err)).any()] or [np.nan]))
            if np.isfinite(med):
                reasons.append(f"reproj_med={med:.1f}px")
        return HandMvNetOutput(
            hand=hand,
            timestamp=now,
            backend="triangulation_fallback",
            valid=bool(valid),
            reason="; ".join(reasons),
            joints_3d=points,
            joints_3d_root_relative=joints_rel,
            root_3d=None if root is None else np.asarray(root, np.float64),
            root_source=root_source,
            joints_2d={k: np.asarray(v, np.float64) for k, v in joints_2d.items()},
            reproj_errors={k: np.asarray(v, np.float64)
                           for k, v in triang.errors_px.items()},
            views_used=tuple(v.name for v in views),
            extra={
                "triangulation": {
                    "n_valid": int(triang.n_valid),
                    "views_used": list(triang.views_used),
                    "views_per_point": (None if triang.views_per_point is None
                                        else triang.views_per_point.tolist()),
                },
            },
        )

    # ------------------------------------------------------------ utilidades
    def describe(self) -> Dict[str, Any]:
        """Status resumido + configuracao efetiva (para log/JSON)."""
        status = self.status()
        status["image_size"] = self._resolve_image_size()
        status["crop_margin"] = self.crop_margin
        status["adapt_num_views"] = self.adapt_num_views
        return status

    def __repr__(self) -> str:                          # pragma: no cover
        return (f"HandMvNetAdapter(mode={self.mode!r}, "
                f"repo={str(self.repo_root)!r}, "
                f"checkpoint={'sim' if self._resolve_checkpoint() else 'nao'})")


# =============================================================================
# Conveniencias de leitura das calibracoes do projeto
# =============================================================================
def calibration_paths(calibration_dir: str | os.PathLike = DEFAULT_CALIBRATION_DIR
                      ) -> Dict[str, Path]:
    """Mapeia os npz de calibracao disponiveis (`aux0`, `aux1`, ...).

    `auxN` e' o arquivo de PAR estereo (Kinect + auxiliar) -- e' dele que saem
    tanto a intrinseca do Kinect quanto R/T da auxiliar. A chave `kinect`
    aponta para `kinect_intrinsics.npz` (so' intrinseca do Kinect, quando
    existir); para montar uma vista do Kinect use o par `aux0`.
    """
    folder = Path(calibration_dir)
    found: Dict[str, Path] = {}
    kinect = folder / "kinect_intrinsics.npz"
    if kinect.is_file():
        found[KINECT_VIEW_NAME] = kinect
    legacy = folder / "stereo_calibration.npz"
    if legacy.is_file():
        found["aux0"] = legacy
    for path in sorted(folder.glob("stereo_calibration_aux*.npz")):
        digits = re.search(r"aux(\d+)", path.stem)
        if digits:
            found[f"aux{int(digits.group(1))}"] = path
    return found
