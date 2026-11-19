#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Processamento OFFLINE das duas auxiliares: MediaPipe 2D -> triangulacao 3D.

Le a sessao gravada por `tools/handmvnet_video_record.py` (videos das auxiliares
+ CSV de timestamps + meta.json), roda o MediaPipe HandLandmarker OFFLINE em
cada vista, sincroniza as duas timelines (flash/clap ou inicio simultaneo),
triangula a mao nas DUAS auxiliares e grava:

    <sessao>/processed/landmarks_3d.csv      (formato longo: 1 linha por junção)
    <sessao>/processed/landmarks_3d.npz      (arrays prontos: joints/valid/reproj)
    <sessao>/processed/coverage_report.json  (cobertura/erro/sincronismo)
    <sessao>/processed/coverage_report.txt   (resumo legivel)
    <sessao>/processed/meta_processado.json  (parametros + resumo)

A matematica NAO e' reimplementada aqui:
  - triangulacao par auxiliar: `kinect_imu_groundtruth.triangulate_pair_points`
    (undistort + DLT + retorno ao frame do Kinect), composto por
    `relative_transform` das duas calibracoes com o Kinect;
  - triangulacao 3 vistas (opcional, se houver video do Kinect):
    `kinect_imu_groundtruth.triangulate_three_views`;
  - erro de reprojecao: `kinect_imu_groundtruth.erro_reprojecao_px`;
  - deteccao: o MESMO `criar_hand_landmarker`/HandLandmarker do pipeline online,
    com a lateralidade corrigida por `aux_handedness_flip` (mesma convencao);
  - as vistas ainda podem ser convertidas para `vision.handmvnet_adapter.
    HandMvNetView` (`make_handmvnet_views`) -- contrato do HandMvNet integrado
    na tarefa anterior -- para um estagio opcional de modelo (`--handmvnet auto`,
    so' roda se houver checkpoint).

Uso:
    # 1) checagem rapida da sessao (sem MediaPipe): sync, pares, calibracao
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\handmvnet_video_process.py --session CAMINHO --dry-run
    # 2) processar de verdade (par de auxiliares) -> pastas processed/
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\handmvnet_video_process.py --session CAMINHO
    # 3) opcional com video do Kinect (3 vistas); precisa de timestamps do Kinect
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\handmvnet_video_process.py --session CAMINHO \\
        --mode three --kinect-video kinect.mp4 --kinect-timestamps kinect_timestamps.csv

Formato do CSV de timestamps do Kinect (para --mode three): mesmas colunas do
recorder (`frame,wall_s,monotonic_s,...`) OU pelo menos `frame,monotonic_s`.
O pareamento Kinect<->aux e' por tempo (relogio monotonic do MESMO processo,
quando gravado junto; senao use --kinect-offset-s).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# sys.path do projeto
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1]                    # src/mestradopy
for _p in (_SRC, _SRC / "core", _SRC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                           # pragma: no cover
    pass

import cv2                                  # noqa: E402

from vision.handmvnet_adapter import (      # noqa: E402
    DEFAULT_CALIBRATION_DIR,
    PALM_LANDMARK_INDICES,
    HandMvNetView,
)
from vision.handmvnet_video_io import (     # noqa: E402
    PROCESSED_META_FILENAME,
    SCHEMA_VERSION,
    CameraTimeline,
    TimestampRow,
    estimate_sync_offset,
    fmt_seconds,
    iso_now,
    pair_by_time,
    probe_video,
    read_meta,
    read_timestamps_csv,
    timestamp_stats,
    write_meta,
)

#: Ordem canonica dos lados (mesma do pipeline; direita primeiro).
HAND_SIDE_ORDER = ("right", "left", "unknown")
NUM_LANDMARKS = 21
#: Minimo de junções validas para considerar a mao triangulada.
MIN_VALID_JOINTS = 5

CSV_COLUMNS: Tuple[str, ...] = (
    "pair", "frame0", "frame1", "frame_kinect",
    "time0_s", "time1_s", "time_sync_s",
    "hand", "track_id", "joint", "x_m", "y_m", "z_m",
    "reproj_aux0_px", "reproj_aux1_px", "reproj_kinect_px",
    "valid", "method",
)


# ===========================================================================
# Estruturas
# ===========================================================================
@dataclass
class HandDetection:
    """Uma mao detectada numa vista (2D em pixels da imagem original)."""

    side: str                       # "right"/"left"/"" (desconhecido)
    side_raw: str                   # rotulo cru do MediaPipe (sem flip)
    score: float
    landmarks_px: np.ndarray        # (21,2) pixels

    def __post_init__(self) -> None:
        self.landmarks_px = np.asarray(self.landmarks_px,
                                       dtype=np.float64).reshape(-1, 2)

    def describe(self) -> Dict[str, Any]:
        return {"side": self.side, "side_raw": self.side_raw,
                "score": float(self.score)}


@dataclass
class TriangRecord:
    """Uma linha "larga" de resultado: um par de frames + uma mao."""

    pair: int
    frame0: int
    frame1: int
    time0_s: float
    time1_s: float
    time_sync_s: float
    hand: str
    method: str
    joints_xyz: np.ndarray                      # (21,3) m, frame do Kinect
    valid: np.ndarray                           # (21,) bool
    reproj_aux0: np.ndarray                     # (21,) px
    reproj_aux1: np.ndarray                     # (21,) px
    pixels_aux0: np.ndarray                     # (21,2) px
    pixels_aux1: np.ndarray                     # (21,2) px
    undist_aux0: np.ndarray                     # (21,2) coords normalizadas
    undist_aux1: np.ndarray
    frame_kinect: int = -1
    reproj_kinect: Optional[np.ndarray] = None  # (21,) px (modo three)
    pixels_kinect: Optional[np.ndarray] = None  # (21,2)
    track_id: int = -1                         # ID temporal (nao e' o lado)


@dataclass
class FrameCursor:
    """Leitor sequencial de video que busca um frame por indice (1 passada)."""

    path: Path
    cap: Any = None
    next_index: int = 0
    error: str = ""

    @classmethod
    def open(cls, path: str | Path) -> "FrameCursor":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"nao consegui abrir o video {path}")
        return cls(path=Path(path), cap=cap)

    def read_at(self, target_index: int) -> Optional[np.ndarray]:
        """Devolve o frame `target_index` (avancando o leitor) ou None."""
        if self.cap is None:
            self.error = "cursor fechado"
            return None
        if int(target_index) < self.next_index:
            self.error = (f"frame {target_index} < posicao atual "
                          f"{self.next_index} (pareamento fora de ordem)")
            return None
        while self.next_index <= int(target_index):
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.error = (f"video acabou antes do frame {target_index} "
                              f"(parou em {self.next_index})")
                return None
            self.next_index += 1
            if self.next_index - 1 == int(target_index):
                return frame
        return None                                     # pragma: no cover

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None


# ===========================================================================
# Detector MediaPipe (offline) + estagio opcional HandMvNet
# ===========================================================================
class MediaPipeHandDetector:
    """HandLandmarker do projeto, UM detector por vista (VideoMode nao e' reentrante).

    Reusa `criar_hand_landmarker` e a correcao de lateralidade do pipeline
    online (`aux_handedness_flip`), para o resultado offline ser comparavel ao
    que o sistema faz ao vivo.
    """

    def __init__(self, *, model_path: Optional[str] = None, gpu: bool = False,
                 num_hands: int = 2, min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, verbose: bool = True):
        self.gpu = bool(gpu)
        self.num_hands = int(num_hands)
        self.conf_det = float(min_detection_confidence)
        self.conf_pres = float(min_presence_confidence)
        self.conf_track = float(min_tracking_confidence)
        self.model_path = Path(model_path) if model_path else None
        self.verbose = verbose
        self._detectors: Dict[str, Any] = {}
        self._stamps: Dict[str, int] = {}
        self._kinect_gt: Any = None
        self._mp: Any = None

    def _modulos(self):
        if self._kinect_gt is None:
            import kinect_imu_groundtruth as kinect_gt
            import mediapipe as mp
            self._kinect_gt, self._mp = kinect_gt, mp
        return self._kinect_gt, self._mp

    def _detector_for(self, view_name: str):
        if view_name in self._detectors:
            return self._detectors[view_name]
        kinect_gt, mp = self._modulos()
        model_path = self.model_path or kinect_gt.ensure_hand_model()
        detector, delegate = kinect_gt.criar_hand_landmarker(
            model_path, gpu=self.gpu,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.conf_det,
            min_hand_presence_confidence=self.conf_pres,
            min_tracking_confidence=self.conf_track)
        if self.verbose:
            print(f"[mediapipe] detector de {view_name}: delegate {delegate}",
                  flush=True)
        self._detectors[view_name] = detector
        return detector

    def detect(self, frame_bgr: np.ndarray, view_name: str,
               timestamp_ms: float, aux_index: Optional[int] = None
               ) -> List[HandDetection]:
        kinect_gt, mp = self._modulos()
        detector = self._detector_for(view_name)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) \
            if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3 else frame_bgr
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        stamp = max(int(self._stamps.get(view_name, 0)) + 1,
                    int(round(float(timestamp_ms))))
        self._stamps[view_name] = stamp
        result = detector.detect_for_video(image, stamp)
        altura, largura = frame_bgr.shape[:2]
        flip = kinect_gt.aux_handedness_flip(aux_index) if aux_index is not None else False
        deteccoes: List[HandDetection] = []
        for indice, landmarks in enumerate(
                getattr(result, "hand_landmarks", None) or []):
            pontos = np.array([(float(lm.x) * largura, float(lm.y) * altura)
                               for lm in landmarks], dtype=np.float64)
            if pontos.shape != (NUM_LANDMARKS, 2):
                continue
            deteccoes.append(HandDetection(
                side=kinect_gt.KinectHandTracker._handedness_of(result, indice, flip),
                side_raw=kinect_gt.KinectHandTracker._handedness_of(result, indice, False),
                score=_score_of(result, indice),
                landmarks_px=pontos))
        return _ordena_deteccoes(deteccoes)

    def close(self) -> None:
        for detector in self._detectors.values():
            try:
                detector.close()
            except Exception:                           # pragma: no cover
                pass
        self._detectors.clear()


def _score_of(result: Any, index: int) -> float:
    """Score (0..1) da deteccao `index` do resultado do HandLandmarker."""
    try:
        groups = result.handedness[index]
        return float(groups[0].score)
    except (AttributeError, IndexError, TypeError, ValueError):
        return 1.0


def _ordena_deteccoes(deteccoes: Iterable[HandDetection]) -> List[HandDetection]:
    """Ordena por lado (direita, esquerda, desconhecida) e depois por score."""
    return sorted(deteccoes,
                  key=lambda d: (HAND_SIDE_ORDER.index(d.side)
                                 if d.side in HAND_SIDE_ORDER else 3,
                                 -float(d.score)))


# ===========================================================================
# Helpers geometricos (delegam a matematica do projeto)
# ===========================================================================
def _modulo_kinect():
    import kinect_imu_groundtruth as kinect_gt
    return kinect_gt


def _para_frame_da_camera(view: Any, points_kin: np.ndarray) -> np.ndarray:
    """X_cam = R X_kin + T (uma multiplicacao; a triangulacao e' do projeto)."""
    pts = np.asarray(points_kin, dtype=np.float64).reshape(-1, 3)
    if view is None or getattr(view, "rotation", None) is None:
        return pts.copy()
    rot = np.asarray(view.rotation, np.float64).reshape(3, 3)
    trans = np.asarray(view.translation, np.float64).reshape(3)
    return (rot @ pts.T).T + trans.reshape(1, 3)


def _reproj_por_juncao(points_kin: np.ndarray, pixels_px: np.ndarray,
                       matrix: np.ndarray, dist: np.ndarray,
                       view: Optional[Any] = None) -> np.ndarray:
    """(21,) erro de reprojecao em px de pontos do Kinect numa vista calibrada."""
    pts = _para_frame_da_camera(view, points_kin)
    px = np.asarray(pixels_px, dtype=np.float64).reshape(-1, 2)
    erros = np.full(pts.shape[0], np.nan, dtype=np.float64)
    finitos = np.isfinite(pts).all(axis=1) & np.isfinite(px).all(axis=1)
    if not finitos.any():
        return erros
    try:
        projetados, _ = cv2.projectPoints(
            pts[finitos].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
            np.asarray(matrix, np.float64), np.asarray(dist, np.float64))
    except cv2.error:                                   # pragma: no cover
        return erros
    erros[finitos] = np.linalg.norm(
        projetados.reshape(-1, 2) - px[finitos], axis=1)
    return erros


def _undistort_normalizado(pixels_px: np.ndarray, matrix: Any,
                           dist: Any) -> np.ndarray:
    """Pixels -> coords normalizadas sem distorcao (passo explicito do relatorio)."""
    px = np.asarray(pixels_px, dtype=np.float64).reshape(-1, 1, 2)
    try:
        return cv2.undistortPoints(
            px.astype(np.float64), np.asarray(matrix, np.float64),
            np.asarray(dist, np.float64)).reshape(-1, 2)
    except cv2.error:                                   # pragma: no cover
        return np.full((px.shape[0], 2), np.nan)


def _valid_mask(points_kin: np.ndarray, reproj_0: np.ndarray,
                reproj_1: np.ndarray, *, max_reproj_px: float,
                view0: Any, view1: Any,
                reproj_kin: Optional[np.ndarray] = None) -> np.ndarray:
    """Junções validas: finitas, na frente das cameras e reprojecao abaixo do gate."""
    pts = np.asarray(points_kin, dtype=np.float64)
    mask = np.isfinite(pts).all(axis=1)
    for view, reproj in ((view0, reproj_0), (view1, reproj_1)):
        cam = _para_frame_da_camera(view, pts)
        mask &= np.isfinite(cam).all(axis=1) & (cam[:, 2] > 0.1)
        mask &= np.isfinite(reproj) & (reproj <= float(max_reproj_px))
    if reproj_kin is not None:
        mask &= np.isfinite(reproj_kin) & (reproj_kin <= float(max_reproj_px))
    return mask


def _casa_maos(dets0: Sequence[HandDetection],
               dets1: Sequence[HandDetection]
               ) -> Tuple[List[Tuple[str, HandDetection, HandDetection]], int]:
    """Casa as maos das duas vistas PELO LADO (direita/esquerda).

    Returns:
        (lista de (lado, det0, det1) na ordem canonica, n_nao_casadas)
    """
    por_lado_0: Dict[str, HandDetection] = {}
    por_lado_1: Dict[str, HandDetection] = {}
    for det in dets0:
        if det.side and det.side not in por_lado_0:
            por_lado_0[det.side] = det
    for det in dets1:
        if det.side and det.side not in por_lado_1:
            por_lado_1[det.side] = det
    pares: List[Tuple[str, HandDetection, HandDetection]] = []
    for lado in ("right", "left"):
        if lado in por_lado_0 and lado in por_lado_1:
            pares.append((lado, por_lado_0[lado], por_lado_1[lado]))
    nao_casadas = (len(por_lado_0) + len(por_lado_1)) - 2 * len(pares)
    # Caso especial: exatamente 1 deteccao em cada vista e o rotulo nao casou
    # (classificador oscilou / desconhecido) -> casa como "unknown" avisando.
    if not pares and len(dets0) == 1 and len(dets1) == 1:
        pares.append(("unknown", dets0[0], dets1[0]))
        nao_casadas = 0
    return pares, nao_casadas


def _stats(values: Iterable[float]) -> Dict[str, Any]:
    """Estatisticas robustas (n/mediana/media/p90/max) ignorando NaN."""
    arr = np.asarray([v for v in values if v is not None],
                     dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "median": None, "mean": None, "p90": None, "max": None}
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 3),
        "mean": round(float(np.mean(arr)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "max": round(float(np.max(arr)), 3),
    }


# ===========================================================================
# Contrato HandMvNet: converter um frame detectado em `HandMvNetView`
# ===========================================================================
def make_handmvnet_views(alignment: Any,
                         views: Sequence[Tuple[str, np.ndarray,
                                               np.ndarray, Optional[int]]]
                         ) -> List[HandMvNetView]:
    """Converte frames+landmarks detectados no contrato do `HandMvNetAdapter`.

    Args:
        alignment: `CameraAlignment` carregado (calibracao ja' validada).
        views: sequencia de `(nome, frame_bgr, landmarks_px (21,2), aux_index)`;
            `aux_index=None` marca a vista do Kinect (raiz).

    Returns:
        Lista de `HandMvNetView` (mesma ordem), pronta para
        `HandMvNetAdapter.infer(...)` -- util para avaliar o modelo offline nos
        MESMOS frames deste processamento (tarefa de integracao do HandMvNet).
    """
    resultado: List[HandMvNetView] = []
    for nome, frame, landmarks_px, aux_index in views:
        view = HandMvNetView.from_alignment(
            alignment, nome, frame, landmarks_px,
            is_kinect=(aux_index is None), aux_index=aux_index)
        resultado.append(view)
    return resultado


# ===========================================================================
# Timeline externa (Kinect) e plano da sessao
# ===========================================================================
def load_external_timeline(video_path: str | Path, timestamps_csv: str | Path,
                           *, name: str = "kinect", index: int = -1,
                           fps_fallback: float = 30.0) -> CameraTimeline:
    """Timeline de um video externo (ex.: Kinect) a partir de um CSV de tempos.

    Aceita o CSV do nosso recorder (colunas `frame,wall_s,monotonic_s,...`) ou
    um CSV minimo (`frame,monotonic_s` / `frame,time_s` / `frame,wall_s`).
    """
    video = Path(video_path)
    csv_path = Path(timestamps_csv)
    if not video.is_file():
        raise FileNotFoundError(f"video do Kinect nao encontrado: {video}")
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV de timestamps do Kinect nao encontrado: {csv_path}")
    rows: List[TimestampRow] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        campos = [str(c).strip().lower() for c in (reader.fieldnames or [])]
        col_mono = next((c for c in ("monotonic_s", "time_s", "t_s", "monotonic",
                                     "timestamp_s") if c in campos), None)
        col_wall = next((c for c in ("wall_s", "wall", "epoch_s") if c in campos),
                        None)
        col_frame = "frame" if "frame" in campos else None
        for linha in reader:
            try:
                mono = float(linha.get(col_mono, 0.0)) if col_mono else 0.0
                wall = float(linha.get(col_wall, 0.0)) if col_wall else 0.0
                frame = int(float(linha.get(col_frame, len(rows)))) \
                    if col_frame else len(rows)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(mono):
                continue
            rows.append(TimestampRow(frame=frame, wall_s=wall,
                                     monotonic_s=mono))
    if not rows:
        raise ValueError(f"CSV de timestamps sem linhas validas: {csv_path}")
    return CameraTimeline(name=name, index=int(index), video_path=video,
                          rows=rows, fps_hint=float(fps_fallback))


def _mirror_effective(meta: Dict[str, Any], mirror: str) -> bool:
    """Decide se o processador precisa espelhar os frames lidos.

    O recorder grava frames JA' espelhados (convencao da calibracao) e marca
    `meta["mirror"]=true`. Videos gravados por fora (raw) precisam de espelho.
    `auto` = espelha se o meta disser que NAO foi gravado espelhado.
    """
    mirror = (mirror or "auto").lower()
    if mirror == "on":
        return True
    if mirror == "off":
        return False
    return not bool(meta.get("mirror", True))


# --- escala pixels do video -> resolucao calibrada --------------------------
def _video_size(path: str | Path) -> Optional[Tuple[int, int]]:
    """(w,h) do video, ou None se nao der para abrir."""
    try:
        info = probe_video(path)
    except Exception:                                   # pragma: no cover
        return None
    if not info.get("opened"):
        return None
    width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    return (width, height) if width > 0 and height > 0 else None


def _calib_size_aux(view: Any) -> Optional[Tuple[int, int]]:
    """Resolucao em que a calibracao da auxiliar foi feita."""
    size = getattr(view, "calibrated_size", None) or getattr(view, "size", None)
    if size is None:
        return None
    try:
        return (int(size[0]), int(size[1]))
    except (TypeError, IndexError, ValueError):         # pragma: no cover
        return None


def _calib_size_kinect(alignment: Any) -> Tuple[int, int]:
    """Resolucao calibrada do RGB do Kinect (1920x1080 por default do projeto)."""
    size = getattr(alignment, "calibrated_kinect_size", None)
    if size:
        try:
            return (int(size[0]), int(size[1]))
        except (TypeError, IndexError, ValueError):     # pragma: no cover
            pass
    return tuple(int(v) for v in kinect_gt_color_size())


def kinect_gt_color_size() -> Tuple[int, int]:
    """(1920,1080) do modulo do projeto, sem importar Kinect hardware."""
    return (1920, 1080)


def _escala_pixels_calib(px: np.ndarray, video_size: Optional[Tuple[int, int]],
                         calib_size: Optional[Tuple[int, int]]
                         ) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Pixels do video -> pixels na resolucao da CALIBRACAO (intrinsica fixa).

    O `CameraAlignment`/`triangulate_pair_points` trabalham na resolucao em que
    o npz foi calibrado (ex.: 1280x720). Se a sessao foi gravada em outra
    resolucao (ex.: 640x360), o detetor devolve pixels da resolucao do video e a
    geometria so' fecha depois de reescalar. E' equivalente a capturar na
    resolucao calibrada (mesmo FOV, mesmo modelo de distorcao).
    """
    pts = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    if not video_size or not calib_size:
        return pts.copy(), (1.0, 1.0)
    sx = float(calib_size[0]) / float(video_size[0]) if video_size[0] else 1.0
    sy = float(calib_size[1]) / float(video_size[1]) if video_size[1] else 1.0
    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        return pts.copy(), (1.0, 1.0)
    out = pts.copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out, (sx, sy)


def _escala_deteccao(det: HandDetection, video_size: Optional[Tuple[int, int]],
                     calib_size: Optional[Tuple[int, int]]) -> HandDetection:
    """Copia a deteccao reescalando os pixels para a resolucao calibrada."""
    px, _ = _escala_pixels_calib(det.landmarks_px, video_size, calib_size)
    return HandDetection(side=det.side, side_raw=det.side_raw,
                         score=det.score, landmarks_px=px)


def _clocks_compativeis(timeline_a: CameraTimeline,
                        timeline_b: CameraTimeline) -> Tuple[str, Dict[str, Any]]:
    """Escolhe o relogio para parear duas timelines ('monotonic' ou 'wall')."""
    ta, tb = timeline_a.np_time(), timeline_b.np_time()
    info: Dict[str, Any] = {}
    if ta.size and tb.size:
        k = min(len(ta), len(tb), 100)
        dif_mono = float(np.median(np.abs(tb[:k] - ta[:k])))
        if np.isfinite(dif_mono) and dif_mono < 5.0:
            info = {"clock": "monotonic", "delta_mediano_inicio_s": dif_mono}
            return "monotonic", info
        wall_a, wall_b = timeline_a.times("wall"), timeline_b.times("wall")
        wal_a, wal_b = wall_a[wall_a > 1e6], wall_b[wall_b > 1e6]
        if wal_a.size and wal_b.size:
            kk = min(len(wal_a), len(wal_b), 100)
            dif_wall = float(np.median(np.abs(wal_b[:kk] - wal_a[:kk])))
            if np.isfinite(dif_wall) and dif_wall < 5.0:
                info = {"clock": "wall", "delta_mediano_inicio_s": dif_wall}
                return "wall", info
    info = {"clock": "monotonic",
            "aviso": "relogios nao comparaveis: pareamento por monotonic mesmo assim"}
    return "monotonic", info


@dataclass
class SessionPlan:
    """Plano de processamento sem rodar MediaPipe (usado no --dry-run)."""

    session_dir: Path
    meta: Dict[str, Any]
    timelines: List[CameraTimeline]
    aux_indices: Tuple[int, ...]
    alignment: Any
    views: Dict[int, Any]
    mirror_effective: bool
    mode: str
    sync: Dict[str, Any]
    pairs: List[Tuple[int, int]]
    kinect_timeline: Optional[CameraTimeline] = None
    kinect_map: Dict[int, int] = field(default_factory=dict)
    kinect_sync: Dict[str, Any] = field(default_factory=dict)
    video_sizes: Dict[int, Optional[Tuple[int, int]]] = field(default_factory=dict)
    calib_sizes: Dict[int, Optional[Tuple[int, int]]] = field(default_factory=dict)
    escala_aux: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    kinect_video_size: Optional[Tuple[int, int]] = None
    kinect_calib_size: Tuple[int, int] = (1920, 1080)
    kinect_escala: Tuple[float, float] = (1.0, 1.0)


def plan_session(session_dir: str | Path, *,
                 aux_indices: Optional[Sequence[int]] = None,
                 mode: str = "auto",
                 kinect_video: Optional[str] = None,
                 kinect_timestamps: Optional[str] = None,
                 kinect_offset_s: Optional[float] = None,
                 max_sync_dt_s: float = 0.034,
                 mirror: str = "auto",
                 verbose: bool = True) -> SessionPlan:
    """Le a sessao, carrega a calibracao, estima sincronismo e pareia frames.

    Nao abre MediaPipe nem depende de GPU: serve para o `--dry-run`, para o
    processamento em si e para diagnosticar a sessao quando algo nao fecha.
    """
    kinect_gt = _modulo_kinect()
    session = Path(session_dir)
    meta = read_meta(session)
    cameras = sorted(meta.get("cameras") or [], key=lambda c: int(c.get("index", 0)))
    if len(cameras) < 2:
        raise ValueError(
            f"meta de {session} tem {len(cameras)} camera(s); o processamento "
            "offline exige as DUAS auxiliares (aux0+aux1).")
    indices = tuple(int(c["index"]) for c in cameras)
    if aux_indices:
        indices = tuple(int(i) for i in aux_indices)
        if set(indices) != {int(c["index"]) for c in cameras}:
            raise ValueError(
                f"--aux-indices {indices} nao bate com as cameras do meta "
                f"{[int(c['index']) for c in cameras]}")
    timelines = [CameraTimeline.from_meta_entry(c, session) for c in cameras]
    for timeline in timelines:
        if not timeline.video_path.is_file():
            raise FileNotFoundError(
                f"video da camera aux{timeline.index} nao existe: "
                f"{timeline.video_path}")
        if not timeline.rows:
            stamps_esperado = timeline.video_path.with_name(
                f"timestamps_aux{timeline.index:02d}.csv")
            raise ValueError(
                f"CSV de timestamps ausente/vazio para aux{timeline.index}: "
                f"{stamps_esperado}")
    timelines.sort(key=lambda t: t.index)
    # Calibracao canonica (mesma usada pela triangulacao do projeto).
    alignment = kinect_gt.CameraAlignment(aux_indices=indices)
    views: Dict[int, Any] = {}
    problemas: List[str] = []
    for idx in indices:
        view = alignment.aux_view(int(idx))
        views[int(idx)] = view
        if view is None:
            problemas.append(f"aux{idx}: sem registro de calibracao")
        elif not view.usable:
            problemas.append(
                f"aux{idx}: calibracao stereo NAO usavel (rms={view.rms}); "
                "recalibre com stereo_calibration.py")
    if problemas:
        raise RuntimeError(
            "calibracao insuficiente para triangulacao par:\n  "
            + "\n  ".join(problemas))
    # Resolucao dos videos x resolucao da calibracao (ver `_escala_pixels_calib`).
    video_sizes: Dict[int, Optional[Tuple[int, int]]] = {}
    calib_sizes: Dict[int, Optional[Tuple[int, int]]] = {}
    escala_aux: Dict[int, Tuple[float, float]] = {}
    for timeline in timelines:
        idx = int(timeline.index)
        video_sizes[idx] = _video_size(timeline.video_path)
        calib_sizes[idx] = _calib_size_aux(views[idx])
        if video_sizes[idx] is None:
            continue
        _, escala = _escala_pixels_calib(np.zeros((1, 2)), video_sizes[idx],
                                         calib_sizes[idx])
        escala_aux[idx] = escala
        if abs(escala[0] - 1.0) > 1e-9 or abs(escala[1] - 1.0) > 1e-9:
            if verbose:
                print(f"[plano] AVISO: aux{idx} gravada em "
                      f"{video_sizes[idx]} mas calibrada em {calib_sizes[idx]}; "
                      f"pixels serao reescalados x{escala[0]:.3g}/"
                      f"x{escala[1]:.3g}", flush=True)
    # Sincronismo entre as DUAS auxiliares (eventos flash/manual do CSV/meta).
    eventos = list(meta.get("sync_events") or [])
    eventos += [dict(e, kind="flash") for t in timelines for e in t.flash_events()]
    eventos += [dict(e, kind="manual_sync") for t in timelines
                for e in t.manual_sync_events()]
    sync = estimate_sync_offset(timelines[0], timelines[1], events=eventos,
                                max_event_dt_s=0.75)
    pares = pair_by_time(timelines[0].np_time(), timelines[1].np_time(),
                         sync["offset_s"], max_sync_dt_s)
    # Modo: pair (2 auxiliares) ou three (Kinect + aux0 + aux1).
    modo = (mode or "auto").lower()
    if modo == "auto":
        modo = "three" if kinect_video else "pair"
    kin_timeline = None
    kin_map: Dict[int, int] = {}
    kin_sync: Dict[str, Any] = {}
    kinect_video_size: Optional[Tuple[int, int]] = None
    kinect_calib_size: Tuple[int, int] = kinect_gt_color_size()
    kinect_escala: Tuple[float, float] = (1.0, 1.0)
    if modo == "three":
        if not kinect_video or not kinect_timestamps:
            raise ValueError(
                "--mode three exige --kinect-video e --kinect-timestamps "
                "(o Kinect e' a raiz do frame 3D).")
        kin_timeline = load_external_timeline(
            kinect_video, kinect_timestamps, name="kinect", index=-1)
        kinect_video_size = _video_size(kin_timeline.video_path)
        kinect_calib_size = _calib_size_kinect(alignment)
        if kinect_video_size is not None:
            _, kinect_escala = _escala_pixels_calib(
                np.zeros((1, 2)), kinect_video_size, kinect_calib_size)
            if (abs(kinect_escala[0] - 1.0) > 1e-9
                    or abs(kinect_escala[1] - 1.0) > 1e-9) and verbose:
                print(f"[plano] AVISO: Kinect gravado em {kinect_video_size} "
                      f"mas calibrado em {kinect_calib_size}; pixels "
                      f"reescalados x{kinect_escala[0]:.3g}/"
                      f"x{kinect_escala[1]:.3g}", flush=True)
        clock, clock_info = _clocks_compativeis(timelines[0], kin_timeline)
        if clock == "wall":
            times_a = timelines[0].times("wall")
            times_b = kin_timeline.times("wall")
        else:
            times_a = timelines[0].np_time()
            times_b = kin_timeline.np_time()
        if kinect_offset_s is not None:
            offset_kin = float(kinect_offset_s)
            method = "manual_offset"
            residual = float("nan")
        else:
            sync_tmp = estimate_sync_offset(timelines[0], kin_timeline,
                                            events=[])
            offset_kin = sync_tmp["offset_s"]
            method = f"tempo_{clock}"
            residual = sync_tmp["residual_s"]
        kin_map = dict(pair_by_time(times_a, times_b, offset_kin,
                                    max_sync_dt_s))
        kin_sync = {"offset_s": round(float(offset_kin), 6),
                    "method": method, "residual_s": residual,
                    "clock": clock, **clock_info,
                    "pairs_kinect_aux0": len(kin_map)}
    if verbose:
        print(f"[plano] sessao={session.name} auxiliares={indices} "
              f"modo={modo} espelhar={_mirror_effective(meta, mirror)}")
        print(f"[plano] pares aux0-aux1: {len(pares)}/{min(len(timelines[0].rows), len(timelines[1].rows))} "
              f"| sync={sync['method']} offset={sync['offset_s'] * 1000:.1f}ms "
              f"residual={sync['residual_s'] * 1000 if np.isfinite(sync['residual_s']) else float('nan'):.1f}ms")
        if modo == "three":
            print(f"[plano] Kinect: {len(kin_map)} pares Kinect-aux0 "
                  f"(offset {kin_sync.get('offset_s', 0) * 1000:.1f}ms, "
                  f"metodo {kin_sync.get('method')})")
        for timeline in timelines:
            stats = timestamp_stats(timeline)
            print(f"[plano] aux{timeline.index}: {timeline.frame_count} frames, "
                  f"fps~{stats['fps']:.1f}, buracos={stats['buracos']}, "
                  f"flash={len(timeline.flash_events())} "
                  f"manual={len(timeline.manual_sync_events())}")
        if sync.get("warning"):
            print(f"[plano] AVISO: {sync['warning']}")
        if not pares:
            print("[plano] AVISO: nenhum par de frames dentro da tolerancia; "
                  "aumente --max-sync-dt-ms ou confira o flash/offset")
    return SessionPlan(session_dir=session, meta=meta, timelines=timelines,
                       aux_indices=indices, alignment=alignment, views=views,
                       mirror_effective=_mirror_effective(meta, mirror),
                       mode=modo, sync=sync, pairs=pares,
                       kinect_timeline=kin_timeline, kinect_map=kin_map,
                       kinect_sync=kin_sync, video_sizes=video_sizes,
                       calib_sizes=calib_sizes, escala_aux=escala_aux,
                       kinect_video_size=kinect_video_size,
                       kinect_calib_size=kinect_calib_size,
                       kinect_escala=kinect_escala)


# ===========================================================================
# Processamento
# ===========================================================================
def process_session(session_dir: str | Path, *,
                    out_dir: Optional[str | Path] = None,
                    aux_indices: Optional[Sequence[int]] = None,
                    mode: str = "auto",
                    kinect_video: Optional[str] = None,
                    kinect_timestamps: Optional[str] = None,
                    kinect_offset_s: Optional[float] = None,
                    kinect_mirror: bool = False,
                    max_sync_dt_s: float = 0.034,
                    max_reproj_px: float = 10.0,
                    every: int = 1,
                    start_s: Optional[float] = None,
                    max_seconds: Optional[float] = None,
                    mirror: str = "auto",
                    detector: Any = None,
                    detector_factory: Optional[Callable[[], Any]] = None,
                    detector_options: Optional[Dict[str, Any]] = None,
                    handmvnet: str = "off",
                    handmvnet_checkpoint: Optional[str] = None,
                    handmvnet_adapt_num_views: bool = False,
                    verbose: bool = True) -> Dict[str, Any]:
    """Processa uma sessao gravada e grava CSV/NPZ/relatorio (retorna o report).

    `detector`/`detector_factory` existem para a suite sintetica injetar um
    detector fake (frames conhecidos); em producao fica `None` e o
    `MediaPipeHandDetector` real e' usado.
    """
    t_inicio = time.monotonic()
    plan = plan_session(session_dir, aux_indices=aux_indices, mode=mode,
                        kinect_video=kinect_video,
                        kinect_timestamps=kinect_timestamps,
                        kinect_offset_s=kinect_offset_s,
                        max_sync_dt_s=max_sync_dt_s, mirror=mirror,
                        verbose=verbose)
    session = plan.session_dir
    destino = Path(out_dir) if out_dir else session / "processed"
    destino.mkdir(parents=True, exist_ok=True)
    kinect_gt = _modulo_kinect()
    indice0, indice1 = int(plan.timelines[0].index), int(plan.timelines[1].index)
    view0, view1 = plan.views[indice0], plan.views[indice1]
    timelines = plan.timelines
    tempo0, tempo1 = timelines[0].np_time(), timelines[1].np_time()

    detector_proprio = False
    if detector is None:
        detector = (detector_factory() if detector_factory
                    else MediaPipeHandDetector(**(detector_options or {}),
                                               verbose=verbose))
        detector_proprio = True
    cursor0 = cursor1 = cursor_kin = None
    records: List[TriangRecord] = []
    handmvnet_records: List[Dict[str, Any]] = []
    hnet_status: Dict[str, Any] = {"requested": handmvnet}
    stats: Dict[str, Any] = {
        "pairs_total": len(plan.pairs),
        "pairs_considered": 0,
        "pairs_skipped_filter": 0,
        "frames_unreadable": 0,
        "pairs_with_both_detected": 0,
        "pairs_no_hands": 0,
        "hands_detected_aux0": 0,
        "hands_detected_aux1": 0,
        "kinect_missing": 0,
        "hand_unmatched": 0,
        "triangulation_attempts": 0,
        "triangulation_ok": 0,
        "triangulation_failed": 0,
        "methods": {},
        "sides": {"right": 0, "left": 0, "unknown": 0},
    }
    t_processado_inicio = time.monotonic()
    proc_index = 0
    try:
        cursor0 = FrameCursor.open(timelines[0].video_path)
        cursor1 = FrameCursor.open(timelines[1].video_path)
        if plan.mode == "three" and plan.kinect_timeline is not None:
            cursor_kin = FrameCursor.open(plan.kinect_timeline.video_path)
        # Estagio opcional HandMvNet (so' com checkpoint): configurado UMA vez.
        adapter = None
        if (handmvnet or "off").lower() != "off":
            adapter = _prepara_handmvnet(session, plan, handmvnet,
                                         handmvnet_checkpoint,
                                         handmvnet_adapt_num_views, hnet_status,
                                         verbose=verbose)
        for pair_idx, (i0, i1) in enumerate(plan.pairs):
            # --- filtros de tempo/amostragem -----------------------------
            if start_s is not None and float(tempo0[i0]) - float(tempo0[0]) < float(start_s):
                stats["pairs_skipped_filter"] += 1
                continue
            if max_seconds is not None and (time.monotonic() - t_processado_inicio) >= float(max_seconds):
                if verbose:
                    print(f"[process] limite --max-seconds {max_seconds:g}s atingido",
                          flush=True)
                break
            stats["pairs_considered"] += 1
            proc_index += 1
            if every and every > 1 and (proc_index - 1) % int(every):
                stats["pairs_skipped_filter"] += 1
                continue
            frame0 = cursor0.read_at(i0)
            frame1 = cursor1.read_at(i1)
            if frame0 is None or frame1 is None:
                stats["frames_unreadable"] += 1
                if verbose and stats["frames_unreadable"] <= 3:
                    print(f"[process] AVISO: nao consegui ler par ({i0},{i1}): "
                          f"{cursor0.error or cursor1.error}", flush=True)
                continue
            if plan.mirror_effective:
                frame0 = cv2.flip(frame0, 1)
                frame1 = cv2.flip(frame1, 1)
            stamp0 = float(tempo0[i0]) * 1000.0
            stamp1 = float(tempo1[i1]) * 1000.0
            dets0_raw = detector.detect(frame0, f"aux{indice0}", stamp0,
                                        aux_index=indice0) or []
            dets1_raw = detector.detect(frame1, f"aux{indice1}", stamp1,
                                        aux_index=indice1) or []
            stats["hands_detected_aux0"] += len(dets0_raw)
            stats["hands_detected_aux1"] += len(dets1_raw)
            # Pixels do video -> pixels na resolucao CALIBRADA (a geometria do
            # projeto trabalha na resolucao do npz; ver _escala_pixels_calib).
            dets0 = [_escala_deteccao(d, plan.video_sizes.get(indice0),
                                      plan.calib_sizes.get(indice0))
                     for d in dets0_raw]
            dets1 = [_escala_deteccao(d, plan.video_sizes.get(indice1),
                                      plan.calib_sizes.get(indice1))
                     for d in dets1_raw]
            bruto0_por_lado = {d.side: d for d in dets0_raw if d.side}
            bruto1_por_lado = {d.side: d for d in dets1_raw if d.side}
            casadas, nao_casadas = _casa_maos(dets0, dets1)
            stats["hand_unmatched"] += int(nao_casadas)
            if not casadas:
                stats["pairs_no_hands"] += 1
                continue
            stats["pairs_with_both_detected"] += 1
            kin_frame = None
            j_kin = -1
            deps_kin: List[HandDetection] = []
            kin_bruto_por_lado: Dict[str, HandDetection] = {}
            if plan.mode == "three":
                j_kin = plan.kinect_map.get(i0)
                if j_kin is None or cursor_kin is None:
                    stats["kinect_missing"] += 1
                    continue
                frame_kin = cursor_kin.read_at(j_kin)
                if frame_kin is None:
                    stats["kinect_missing"] += 1
                    continue
                if kinect_mirror:
                    frame_kin = cv2.flip(frame_kin, 1)
                kin_times = plan.kinect_timeline.np_time()
                deps_kin_raw = detector.detect(
                    frame_kin, "kinect", float(kin_times[j_kin]) * 1000.0,
                    aux_index=None) or []
                kin_bruto_por_lado = {d.side: d for d in deps_kin_raw if d.side}
                deps_kin = [_escala_deteccao(
                    d, plan.kinect_video_size, plan.kinect_calib_size)
                    for d in deps_kin_raw]
                kin_frame = frame_kin
            for lado, det0, det1 in casadas:
                if plan.mode == "three":
                    det_kin = next((d for d in deps_kin if d.side == lado), None)
                    if det_kin is None:
                        stats["kinect_missing"] += 1
                        continue
                    record = _triangula_three(kinect_gt, plan, indice0, indice1,
                                              pair_idx, i0, i1, j_kin,
                                              tempo0[i0], tempo1[i1],
                                              lado, det0, det1, det_kin,
                                              max_reproj_px=max_reproj_px)
                else:
                    record = _triangula_pair(kinect_gt, view0, view1, pair_idx,
                                             i0, i1, tempo0[i0], tempo1[i1],
                                             lado, det0, det1,
                                             max_reproj_px=max_reproj_px)
                stats["triangulation_attempts"] += 1
                if record is None:
                    stats["triangulation_failed"] += 1
                    continue
                stats["triangulation_ok"] += 1
                stats["sides"][lado if lado in stats["sides"] else "unknown"] += 1
                stats["methods"][record.method] = stats["methods"].get(record.method, 0) + 1
                records.append(record)
                # --- estagio opcional HandMvNet (mesmos frames/deteccoes) ---
                if adapter is not None:
                    try:
                        det0_bruto = bruto0_por_lado.get(lado, det0)
                        det1_bruto = bruto1_por_lado.get(lado, det1)
                        views_hn = [(f"aux{indice0}", frame0,
                                     det0_bruto.landmarks_px, indice0)]
                        if plan.mode == "three" and kin_frame is not None:
                            det_kin_bruto = kin_bruto_por_lado.get(lado)
                            if det_kin_bruto is not None:
                                views_hn.insert(
                                    0, ("kinect", kin_frame,
                                        det_kin_bruto.landmarks_px, None))
                            else:
                                views_hn = []
                        views_hn.append((f"aux{indice1}", frame1,
                                         det1_bruto.landmarks_px, indice1))
                        if views_hn:
                            hn_out = adapter.infer(
                                make_handmvnet_views(plan.alignment, views_hn),
                                hand=lado)
                            handmvnet_records.append({
                                "pair": pair_idx, "hand": lado,
                                "output": hn_out})
                    except Exception as exc:            # pragma: no cover
                        hnet_status.setdefault("erros", []).append(
                            f"{type(exc).__name__}: {exc}")
    finally:
        for cursor in (cursor0, cursor1, cursor_kin):
            if cursor is not None:
                try:
                    cursor.release()
                except Exception:                       # pragma: no cover
                    pass
        if detector_proprio:
            try:
                detector.close()
            except Exception:                           # pragma: no cover
                pass

    _atribui_tracks(records)
    _suaviza_lados_por_track(records)
    tempo_total = time.monotonic() - t_inicio
    report = _monta_report(
        plan=plan, destino=destino, records=records, stats=stats,
        tempo_total_s=tempo_total, max_reproj_px=max_reproj_px,
        every=every, start_s=start_s, max_seconds=max_seconds,
        handmvnet_records=handmvnet_records, hnet_status=hnet_status,
        parametros={
            "mode": plan.mode, "mirror_effective": plan.mirror_effective,
            "max_sync_dt_ms": round(float(max_sync_dt_s) * 1000.0, 2),
            "max_reproj_px": float(max_reproj_px), "every": int(every),
            "start_s": start_s, "max_seconds": max_seconds,
            "handmvnet": handmvnet, "kinect_mirror": bool(kinect_mirror),
            "pixel_scaling": {
                f"aux{int(i)}": list(plan.escala_aux.get(int(i), (1.0, 1.0)))
                for i in plan.aux_indices},
            "kinect_pixel_scaling": list(plan.kinect_escala),
            "video_sizes": {f"aux{int(k)}": (None if v is None else list(v))
                            for k, v in plan.video_sizes.items()},
            "kinect_video_size": (None if plan.kinect_video_size is None
                                  else list(plan.kinect_video_size)),
            "calib_sizes": {f"aux{int(k)}": (None if v is None else list(v))
                            for k, v in plan.calib_sizes.items()},
            "handmvnet_checkpoint": handmvnet_checkpoint,
            "kinect_video": str(plan.kinect_timeline.video_path
                                if plan.kinect_timeline else ""),
            "kinect_sync": plan.kinect_sync,
        })
    caminhos = _escreve_saidas(destino, records, report, plan,
                               handmvnet_records)
    report["outputs"] = caminhos
    _escreve_report(report, destino)
    _escreve_meta_processado(destino, report, plan)
    if verbose:
        print(report_resumo_txt(report))
        print(f"\nSaidas em: {destino}")
    return report


# ---------------------------------------------------------------------------
# Triangulacao por par / 3 vistas (usa a matematica do projeto)
# ---------------------------------------------------------------------------
def _triangula_pair(kinect_gt: Any, view0: Any, view1: Any, pair_idx: int,
                    i0: int, i1: int, time0: float, time1: float, lado: str,
                    det0: HandDetection, det1: HandDetection,
                    *, max_reproj_px: float) -> Optional[TriangRecord]:
    points_kin, erro_mediano = kinect_gt.triangulate_pair_points(
        view0, view1, det0.landmarks_px, det1.landmarks_px)
    if points_kin is None:
        return None
    pontos = np.asarray(points_kin, np.float64).reshape(-1, 3)
    reproj0 = _reproj_por_juncao(
        pontos, det0.landmarks_px, view0.effective_matrix, view0.dist, view0)
    reproj1 = _reproj_por_juncao(
        pontos, det1.landmarks_px, view1.effective_matrix, view1.dist, view1)
    valid = _valid_mask(pontos, reproj0, reproj1, max_reproj_px=max_reproj_px,
                        view0=view0, view1=view1)
    return TriangRecord(
        pair=int(pair_idx), frame0=int(i0), frame1=int(i1),
        time0_s=float(time0), time1_s=float(time1),
        time_sync_s=float(0.5 * (time0 + time1)), hand=str(lado),
        method="pair_aux",
        joints_xyz=pontos, valid=valid, reproj_aux0=reproj0,
        reproj_aux1=reproj1, pixels_aux0=det0.landmarks_px.copy(),
        pixels_aux1=det1.landmarks_px.copy(),
        undist_aux0=_undistort_normalizado(
            det0.landmarks_px, view0.effective_matrix, view0.dist),
        undist_aux1=_undistort_normalizado(
            det1.landmarks_px, view1.effective_matrix, view1.dist))


def _triangula_three(kinect_gt: Any, plan: SessionPlan, indice0: int,
                     indice1: int, pair_idx: int, i0: int, i1: int,
                     i_kin: int, time0: float, time1: float, lado: str,
                     det0: HandDetection, det1: HandDetection,
                     det_kin: HandDetection, *, max_reproj_px: float
                     ) -> Optional[TriangRecord]:
    points_kin, mask, info = kinect_gt.triangulate_three_views(
        plan.alignment, det_kin.landmarks_px, det0.landmarks_px,
        det1.landmarks_px, aux_indices=plan.aux_indices, min_valid_points=5,
        max_reproj_error_px=float(max_reproj_px))
    if points_kin is None or mask is None:
        return None
    pontos = np.asarray(points_kin, np.float64).reshape(-1, 3)
    view0, view1 = plan.views[indice0], plan.views[indice1]
    reproj0 = _reproj_por_juncao(
        pontos, det0.landmarks_px, view0.effective_matrix, view0.dist, view0)
    reproj1 = _reproj_por_juncao(
        pontos, det1.landmarks_px, view1.effective_matrix, view1.dist, view1)
    reproj_kin = _reproj_por_juncao(
        pontos, det_kin.landmarks_px, plan.alignment.kinect_matrix,
        plan.alignment.kinect_dist, None)
    valid = (_valid_mask(pontos, reproj0, reproj1, max_reproj_px=max_reproj_px,
                         view0=view0, view1=view1, reproj_kin=reproj_kin)
             & np.asarray(mask, dtype=bool))
    kin_times = plan.kinect_timeline.np_time() if plan.kinect_timeline else None
    return TriangRecord(
        pair=int(pair_idx), frame0=int(i0), frame1=int(i1),
        frame_kinect=int(i_kin), time0_s=float(time0), time1_s=float(time1),
        time_sync_s=float(0.5 * (time0 + time1)), hand=str(lado),
        method=str(info.get("method", "three_view")),
        joints_xyz=pontos, valid=valid, reproj_aux0=reproj0,
        reproj_aux1=reproj1, reproj_kinect=reproj_kin,
        pixels_aux0=det0.landmarks_px.copy(),
        pixels_aux1=det1.landmarks_px.copy(),
        pixels_kinect=det_kin.landmarks_px.copy(),
        undist_aux0=_undistort_normalizado(
            det0.landmarks_px, view0.effective_matrix, view0.dist),
        undist_aux1=_undistort_normalizado(
            det1.landmarks_px, view1.effective_matrix, view1.dist))


# ---------------------------------------------------------------------------
# HandMvNet opcional
# ---------------------------------------------------------------------------
def _prepara_handmvnet(session: Path, plan: SessionPlan, modo: str,
                       checkpoint: Optional[str], adapt_num_views: bool,
                       status_out: Dict[str, Any], *,
                       verbose: bool = True
                       ) -> Optional[Any]:
    """Configura o adaptador HandMvNet SE houver checkpoint (senao registra motivo)."""
    from vision.handmvnet_adapter import HandMvNetAdapter
    status_out.update({"attempted": True, "used": False})
    if plan.mode != "three":
        status_out["reason"] = ("estagio HandMvNet requer --mode three (vista "
                                "raiz = Kinect); no modo par o modelo nao tem "
                                "referencial absoluto")
        if verbose:
            print(f"[handmvnet] pulado: {status_out['reason']}", flush=True)
        return None
    adapter = HandMvNetAdapter(checkpoint=checkpoint, mode="model",
                               adapt_num_views=adapt_num_views, verbose=False)
    status = adapter.status(refresh=True)
    n_views = 3
    if not status.get("model_ready"):
        status_out["reason"] = (
            "modelo HandMvNet indisponivel: "
            + "; ".join(status.get("reasons") or ["motivo desconhecido"]))
        if verbose:
            print(f"[handmvnet] pulado: {status_out['reason']}", flush=True)
        return None
    loaded = adapter.load_model(n_views)
    if loaded is None:
        status_out["reason"] = adapter._model_error or "falha ao carregar"
        return None
    status_out.update({"used": True,
                       "model": loaded.summary(),
                       "weights": loaded.weights_source})
    if verbose:
        print(f"[handmvnet] estagio ativo: {loaded.summary()}", flush=True)
    return adapter


# ---------------------------------------------------------------------------
# Identidade temporal (track ID) sem depender do rotulo left/right
# ---------------------------------------------------------------------------
TRACK_MAX_DIST_M = 0.35
TRACK_MAX_GAP_S = 1.5


def _ponto_referencia(record: TriangRecord) -> Optional[np.ndarray]:
    """Ponto 3D para associacao: palma (9), senao media dos joints validos."""
    xyz = np.asarray(record.joints_xyz, np.float64)
    valid = np.asarray(record.valid, dtype=bool)
    if xyz.shape != (NUM_LANDMARKS, 3):
        return None
    if valid[9] and np.isfinite(xyz[9]).all():
        return xyz[9].copy()
    pontos = xyz[valid & np.isfinite(xyz).all(axis=1)]
    if len(pontos) == 0:
        return None
    return np.median(pontos, axis=0)


def _atribui_tracks(records: Sequence[TriangRecord]) -> None:
    """Atribui track_id por continuidade 3D (um ID por mao fisica).

    Nao usa o rotulo do MediaPipe. Dois registros do mesmo par nao podem
    receber o mesmo ID; tracks expirados por tempo/distancia sao fechados.
    """
    trilhas: Dict[int, Dict[str, Any]] = {}
    proximo_id = 0
    usados_no_par: set[int] = set()
    ultimo_par = None
    for record in records:
        if record.pair != ultimo_par:
            usados_no_par.clear()
            ultimo_par = record.pair
        ponto = _ponto_referencia(record)
        if ponto is None:
            record.track_id = -1
            continue
        melhor_id = None
        melhor_dist = float("inf")
        for tid, estado in trilhas.items():
            if tid in usados_no_par:
                continue
            dt = float(record.time_sync_s) - float(estado["t"])
            if dt < 0 or dt > TRACK_MAX_GAP_S:
                continue
            dist = float(np.linalg.norm(ponto - estado["p"]))
            if dist < melhor_dist:
                melhor_dist = dist
                melhor_id = tid
        if melhor_id is None or melhor_dist > TRACK_MAX_DIST_M:
            melhor_id = proximo_id
            proximo_id += 1
            trilhas[melhor_id] = {"p": ponto, "t": record.time_sync_s,
                                  "n": 0, "hand": record.hand}
        record.track_id = int(melhor_id)
        usados_no_par.add(int(melhor_id))
        estado = trilhas[int(melhor_id)]
        estado["p"] = ponto
        estado["t"] = record.time_sync_s
        estado["n"] += 1
        if record.hand in ("left", "right"):
            estado.setdefault("lados", {})
            estado["lados"][record.hand] = estado["lados"].get(record.hand, 0) + 1


def _suaviza_lados_por_track(records: Sequence[TriangRecord]) -> None:
    """Troca rotulos left/right instaveis pela maioria do track fisico.

    O MediaPipe oscila entre left/right em vista lateral. O track_id e' dado
    pela continuidade 3D; aqui cada track recebe o lado majoritario (>=60%).
    Sem maioria clara, mantem o rotulo original. Nao altera os pixels/3D.
    """
    por_track: Dict[int, Dict[str, int]] = {}
    for r in records:
        if int(r.track_id) < 0:
            continue
        lado = r.hand if r.hand in ("left", "right") else "unknown"
        por_track.setdefault(int(r.track_id), {}).setdefault(lado, 0)
        por_track[int(r.track_id)][lado] += 1
    majoritarios: Dict[int, str] = {}
    for tid, contagem in por_track.items():
        total = sum(contagem.values())
        if total <= 0:
            continue
        lado, n = max(contagem.items(), key=lambda kv: kv[1])
        if lado in ("left", "right") and n / total >= 0.60:
            majoritarios[tid] = lado
    for r in records:
        tid = int(r.track_id)
        if tid >= 0 and tid in majoritarios and r.hand != majoritarios[tid]:
            r.hand = majoritarios[tid]


def _resumo_tracks(records: Sequence[TriangRecord]) -> Dict[str, Any]:
    ids = [int(r.track_id) for r in records if int(r.track_id) >= 0]
    unicos = sorted(set(ids))
    counts = {str(tid): ids.count(tid) for tid in unicos}
    lados: Dict[str, Dict[str, int]] = {}
    for r in records:
        if int(r.track_id) < 0:
            continue
        lado = r.hand if r.hand in ("left", "right") else "unknown"
        lados.setdefault(str(int(r.track_id)), {}).setdefault(lado, 0)
        lados[str(int(r.track_id))][lado] += 1
    trocas = sum(1 for a, b in zip(records, records[1:])
                 if a.hand != b.hand)
    return {"n_tracks": len(unicos), "ids": unicos,
            "records_por_track": counts, "lados_por_track": lados,
            "hand_switches": int(trocas)}


# ---------------------------------------------------------------------------
# Relatorio e saidas
# ---------------------------------------------------------------------------
def _monta_report(*, plan: SessionPlan, destino: Path,
                  records: Sequence[TriangRecord], stats: Dict[str, Any],
                  tempo_total_s: float, max_reproj_px: float, every: int,
                  start_s: Optional[float], max_seconds: Optional[float],
                  handmvnet_records: Sequence[Dict[str, Any]],
                  hnet_status: Dict[str, Any],
                  parametros: Dict[str, Any]) -> Dict[str, Any]:
    """Monta o dict do relatorio de cobertura/erro."""
    n_pares = len(plan.pairs)
    n_gravados = len(records)
    validos_por_juncao = np.zeros(NUM_LANDMARKS, dtype=np.int64)
    reproj0: List[float] = []
    reproj1: List[float] = []
    reproj_kin: List[float] = []
    erros_hand: Dict[str, List[float]] = {}
    for record in records:
        validos_por_juncao += np.asarray(record.valid, dtype=np.int64)
        m = np.asarray(record.valid, dtype=bool)
        if m.any():
            reproj0.extend(np.asarray(record.reproj_aux0)[m].tolist())
            reproj1.extend(np.asarray(record.reproj_aux1)[m].tolist())
            if record.reproj_kinect is not None:
                reproj_kin.extend(np.asarray(record.reproj_kinect)[m].tolist())
        erros_hand.setdefault(record.hand, []).append(
            float(np.nanmedian(record.reproj_aux0)))
    calibricao_views: Dict[str, Any] = {}
    for idx in plan.aux_indices:
        view = plan.views.get(int(idx))
        if view is None:
            continue
        calibricao_views[f"aux{int(idx)}"] = {
            "rms_px": (None if getattr(view, "rms", None) is None
                       else round(float(view.rms), 3)),
            "camera_name": getattr(view, "camera_name", None),
            "calibrated_size": list(getattr(view, "calibrated_size", []) or []),
            "runtime_size": list(getattr(view, "runtime_size", []) or []),
        }
    timelines = {f"aux{t.index}": {**t.summary(), **timestamp_stats(t)}
                 for t in plan.timelines}
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "script": "handmvnet_video_process.py",
        "generated_at": iso_now(),
        "session": {
            "dir": str(plan.session_dir),
            "name": plan.meta.get("session_name"),
            "recorded_at": plan.meta.get("created_at"),
            "status_gravacao": plan.meta.get("status"),
            "resolucao": plan.meta.get("resolution_requested"),
            "fps_pedido": plan.meta.get("fps_requested"),
            "mirror_gravado": plan.meta.get("mirror"),
        },
        "mode": plan.mode,
        "mirror_effective": bool(plan.mirror_effective),
        "aux_indices": [int(i) for i in plan.aux_indices],
        "calibration": calibricao_views,
        "timelines": timelines,
        "sync": {
            "offset_s": round(float(plan.sync["offset_s"]), 6),
            "offset_ms": round(float(plan.sync["offset_s"]) * 1000.0, 2),
            "method": plan.sync["method"],
            "residual_ms": (None if not np.isfinite(plan.sync["residual_s"])
                            else round(float(plan.sync["residual_s"]) * 1000.0, 2)),
            "n_events": int(plan.sync["n_events"]),
            "warning": plan.sync.get("warning", ""),
            "kinect": plan.kinect_sync,
        },
        "counts": dict(stats, pairs_total=n_pares,
                       pairs_written=int(n_gravados),
                       coverage_pairs_pct=(
                           100.0 * n_gravados / n_pares if n_pares else 0.0)),
        "tracking": _resumo_tracks(records),
        "quality": {
            "reproj_aux0_px": _stats(reproj0),
            "reproj_aux1_px": _stats(reproj1),
            "reproj_kinect_px": _stats(reproj_kin),
            "reproj_por_lado_aux0_px": {
                lado: _stats(valores) for lado, valores in erros_hand.items()},
            "joint_coverage": [
                {"joint": int(j), "n": int(validos_por_juncao[j]),
                 "pct": (round(100.0 * float(validos_por_juncao[j])
                               / max(n_gravados, 1), 1))}
                for j in range(NUM_LANDMARKS)],
            "palm_joints_coverage": {
                f"joint_{int(j)}": int(validos_por_juncao[int(j)])
                for j in PALM_LANDMARK_INDICES},
        },
        "parameters": {
            "max_reproj_px": float(max_reproj_px), "every": int(every),
            "start_s": start_s, "max_seconds": max_seconds,
            **{k: v for k, v in parametros.items() if v is not None},
        },
        "runtime_s": round(float(tempo_total_s), 3),
        "handmvnet": dict(hnet_status,
                          n_records=len(handmvnet_records)),
        "outputs": {},
        "limitations": [
            "MediaPipe roda OFFLINE nos frames gravados: o tracking nao ve o "
            "frame anterior em tempo real, mas o HandLandmarker em modo VIDEO "
            "processa os frames em ordem (equivalente ao online).",
            "Sincronismo por flash/clap: 1 evento da' alinhamento de ~1 frame; "
            "para bancada de movimento marque 2-3 eventos ou use o mesmo inicio.",
            "Sem Kinect por padrão: a pose 3D e' relativa ao par auxiliar; o "
            "frame de saida e' o do Kinect (composicao das calibracoes), mas "
            "nao ha' correcao com a profundidade do Kinect.",
            "Erro de reprojecao abaixo do gate nao garante pose correta: "
            "confira a cobertura por junção e a consistencia temporal.",
        ],
    }
    return report


def _escreve_saidas(destino: Path, records: Sequence[TriangRecord],
                    report: Dict[str, Any], plan: SessionPlan,
                    handmvnet_records: Sequence[Dict[str, Any]]
                    ) -> Dict[str, str]:
    """Grava CSV longo + NPZ + (opcional) saidas do HandMvNet."""
    caminhos: Dict[str, str] = {}
    csv_path = destino / "landmarks_3d.csv"
    _escreve_csv_long(csv_path, records)
    caminhos["csv"] = str(csv_path)
    npz_path = destino / "landmarks_3d.npz"
    _escreve_npz_landmarks(npz_path, records, report)
    caminhos["npz"] = str(npz_path)
    if handmvnet_records:
        hn_dir = destino / "handmvnet"
        hn_dir.mkdir(parents=True, exist_ok=True)
        hn_csv = hn_dir / "handmvnet_landmarks_3d.csv"
        _escreve_csv_handmvnet(hn_csv, handmvnet_records)
        caminhos["handmvnet_csv"] = str(hn_csv)
        arrays: Dict[str, np.ndarray] = {
            "joints_xyz": np.stack([np.asarray(r["output"].joints_3d, np.float64)
                                    for r in handmvnet_records]),
            "pair": np.array([r["pair"] for r in handmvnet_records],
                             dtype=np.int64),
            "hand": np.array([r["hand"] for r in handmvnet_records],
                             dtype="U7"),
            "valid": np.array([r["output"].valid for r in handmvnet_records],
                              dtype=bool),
            "backend": np.array([r["output"].backend for r in handmvnet_records],
                                dtype="U32"),
            "meta_json": np.array(json.dumps(report.get("handmvnet", {}),
                                             ensure_ascii=False)),
        }
        npz_hn = destino / "handmvnet_landmarks_3d.npz"
        np.savez_compressed(npz_hn, **arrays)
        caminhos["handmvnet_npz"] = str(npz_hn)
    caminhos["report_json"] = str(destino / "coverage_report.json")
    caminhos["report_txt"] = str(destino / "coverage_report.txt")
    caminhos["meta_processado"] = str(destino / PROCESSED_META_FILENAME)
    return caminhos


def _escreve_csv_long(path: Path, records: Sequence[TriangRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for record in records:
            for juntao in range(NUM_LANDMARKS):
                x, y, z = (float(v) for v in record.joints_xyz[juntao])
                r0 = float(record.reproj_aux0[juntao])
                r1 = float(record.reproj_aux1[juntao])
                rk = (float(record.reproj_kinect[juntao])
                      if record.reproj_kinect is not None else float("nan"))
                writer.writerow([
                    int(record.pair), int(record.frame0), int(record.frame1),
                    int(record.frame_kinect),
                    f"{record.time0_s:.6f}", f"{record.time1_s:.6f}",
                    f"{record.time_sync_s:.6f}", record.hand,
                    int(record.track_id), int(juntao),
                    f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                    "" if not np.isfinite(r0) else f"{r0:.3f}",
                    "" if not np.isfinite(r1) else f"{r1:.3f}",
                    "" if not np.isfinite(rk) else f"{rk:.3f}",
                    int(bool(record.valid[juntao])), record.method])


def _escreve_csv_handmvnet(path: Path, handmvnet_records: Sequence[Dict[str, Any]]
                           ) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair", "hand", "backend", "valid", "latency_ms",
                         "model_weights", "reason", "joint", "x_m", "y_m", "z_m"])
        for item in handmvnet_records:
            out = item["output"]
            for juntao in range(NUM_LANDMARKS):
                x, y, z = (float(v) for v in
                           np.asarray(out.joints_3d, np.float64)[juntao])
                writer.writerow([
                    int(item["pair"]), item["hand"], out.backend,
                    int(bool(out.valid)), f"{out.latency_ms:.2f}",
                    out.model_weights, out.reason, int(juntao),
                    f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])


def _escreve_npz_landmarks(path: Path, records: Sequence[TriangRecord],
                           report: Dict[str, Any]) -> None:
    n = len(records)

    def _stack(nome: str, shape_tail: Tuple[int, ...], dtype=np.float64):
        if n == 0:
            return np.zeros((0,) + shape_tail, dtype=dtype)
        valores = [getattr(r, nome) for r in records]
        return np.stack([np.asarray(v, dtype=dtype) for v in valores])

    arrays: Dict[str, np.ndarray] = {
        "pair": np.array([r.pair for r in records], dtype=np.int64),
        "frame0": np.array([r.frame0 for r in records], dtype=np.int64),
        "frame1": np.array([r.frame1 for r in records], dtype=np.int64),
        "frame_kinect": np.array([r.frame_kinect for r in records],
                                 dtype=np.int64),
        "time0_s": np.array([r.time0_s for r in records], dtype=np.float64),
        "time1_s": np.array([r.time1_s for r in records], dtype=np.float64),
        "time_sync_s": np.array([r.time_sync_s for r in records],
                                dtype=np.float64),
        "hand": np.array([r.hand for r in records], dtype="U7"),
        "track_id": np.array([r.track_id for r in records], dtype=np.int64),
        "method": np.array([r.method for r in records], dtype="U24"),
        "joints_xyz": _stack("joints_xyz", (NUM_LANDMARKS, 3)),
        "valid": _stack("valid", (NUM_LANDMARKS,), dtype=bool),
        "reproj_aux0": _stack("reproj_aux0", (NUM_LANDMARKS,)),
        "reproj_aux1": _stack("reproj_aux1", (NUM_LANDMARKS,)),
        "pixels_aux0": _stack("pixels_aux0", (NUM_LANDMARKS, 2)),
        "pixels_aux1": _stack("pixels_aux1", (NUM_LANDMARKS, 2)),
        "undist_aux0": _stack("undist_aux0", (NUM_LANDMARKS, 2)),
        "undist_aux1": _stack("undist_aux1", (NUM_LANDMARKS, 2)),
        "meta_json": np.array(json.dumps(report, ensure_ascii=False,
                                         default=str)),
    }
    if n and any(r.reproj_kinect is not None for r in records):
        kin_reproj = np.full((n, NUM_LANDMARKS), np.nan)
        kin_pixels = np.full((n, NUM_LANDMARKS, 2), np.nan)
        for i, r in enumerate(records):
            if r.reproj_kinect is not None:
                kin_reproj[i] = np.asarray(r.reproj_kinect, np.float64)
            if r.pixels_kinect is not None:
                kin_pixels[i] = np.asarray(r.pixels_kinect, np.float64)
        arrays["reproj_kinect"] = kin_reproj
        arrays["pixels_kinect"] = kin_pixels
    np.savez_compressed(path, **arrays)


def resumo_curto(report: Dict[str, Any]) -> str:
    """Uma linha para log (usada no main e no teste)."""
    counts = report["counts"]
    sync = report["sync"]
    cobertura = counts.get("coverage_pairs_pct", 0.0)
    med = report["quality"]["reproj_aux0_px"]["median"]
    return (f"{report['mode']}: pares={counts['pairs_total']} "
            f"escritos={counts['pairs_written']} ({cobertura:.1f}%) "
            f"sync={sync['method']}@{sync['offset_ms']:+.1f}ms "
            f"reproj0_med={med}px")


def report_resumo_txt(report: Dict[str, Any]) -> str:
    """Versao multilinha do relatorio para o terminal."""
    quality = report["quality"]
    counts = report["counts"]
    linhas = [
        "=" * 78,
        f"PROCESSAMENTO OFFLINE: {report['session'].get('name')} "
        f"({report['mode']})",
        "=" * 78,
        f"auxiliares        : {report['aux_indices']} | espelhado="
        f"{report['mirror_effective']}",
    ]
    for nome, calib in report["calibration"].items():
        linhas.append(f"  {nome:<6} calib RMS={calib.get('rms_px')} px "
                      f"camera={calib.get('camera_name')}")
    sync = report["sync"]
    linhas.append(
        f"sync              : {sync['method']} offset={sync['offset_ms']:+.1f}ms "
        f"residual={sync.get('residual_ms')}ms eventos={sync['n_events']}")
    if sync.get("warning"):
        linhas.append(f"  AVISO sync: {sync['warning']}")
    linhas.append(
        f"pares             : {counts['pairs_total']} totais, "
        f"{counts['pairs_written']} escritos "
        f"({counts.get('coverage_pairs_pct', 0):.1f}% de cobertura)")
    linhas.append(
        f"detecção          : aux0={counts['hands_detected_aux0']} maos, "
        f"aux1={counts['hands_detected_aux1']} maos, "
        f"sem mao em ambos={counts['pairs_no_hands']}, nao casadas="
        f"{counts['hand_unmatched']}"
        + (f", Kinect ausente={counts['kinect_missing']}"
           if report["mode"] == "three" else ""))
    linhas.append(
        f"triangulação      : {counts['triangulation_ok']}/"
        f"{counts['triangulation_attempts']} ok "
        f"({counts['triangulation_failed']} falhas) metodos="
        f"{counts['methods']}")
    linhas.append(
        f"reproj aux0/aux1  : mediana {quality['reproj_aux0_px']['median']} / "
        f"{quality['reproj_aux1_px']['median']} px | p90 "
        f"{quality['reproj_aux0_px']['p90']} / "
        f"{quality['reproj_aux1_px']['p90']} px")
    if quality["reproj_kinect_px"]["n"]:
        linhas.append(f"reproj Kinect     : mediana "
                      f"{quality['reproj_kinect_px']['median']} px")
    cobertura = [item["pct"] for item in quality["joint_coverage"]]
    linhas.append(f"cobertura/junção  : min {min(cobertura) if cobertura else 0}% "
                  f"mediana {float(np.median(cobertura)) if cobertura else 0:.1f}%")
    tr = report.get("tracking", {})
    linhas.append(f"tracks            : {tr.get('n_tracks', 0)} "
                  f"IDs={tr.get('ids', [])} | trocas de lado="
                  f"{tr.get('hand_switches', 0)}")
    hn = report.get("handmvnet", {})
    if hn.get("used"):
        linhas.append(f"HandMvNet         : {hn.get('model')}")
    elif hn.get("attempted"):
        linhas.append(f"HandMvNet         : pulado ({hn.get('reason')})")
    linhas.append(f"tempo             : {report['runtime_s']:.1f}s")
    linhas.append("=" * 78)
    return "\n".join(linhas)


def _escreve_report(report: Dict[str, Any], destino: Path) -> None:
    with open(destino / "coverage_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    with open(destino / "coverage_report.txt", "w", encoding="utf-8") as fh:
        fh.write(report_resumo_txt(report) + "\n")


def _escreve_meta_processado(destino: Path, report: Dict[str, Any],
                             plan: SessionPlan) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "script": "handmvnet_video_process.py",
        "generated_at": iso_now(),
        "session_dir": str(plan.session_dir),
        "parameters": report["parameters"],
        "mode": report["mode"],
        "sync": report["sync"],
        "counts": report["counts"],
        "quality": report["quality"],
        "outputs": report.get("outputs", {}),
    }
    write_meta(destino, payload, filename=PROCESSED_META_FILENAME)


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processa offline os videos das 2 auxiliares "
                    "(MediaPipe 2D -> triangulacao 3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--session", required=True,
                        help="pasta da sessao gravada (com meta.json + videos)")
    parser.add_argument("--out", default=None,
                        help="pasta de saida (default: <session>/processed)")
    parser.add_argument("--aux-indices", nargs="+", type=int, default=None,
                        help="indices das auxiliares (default: os do meta)")
    parser.add_argument("--mode", default="auto", choices=("auto", "pair", "three"),
                        help="pair = so' as 2 auxiliares; three = Kinect+aux0+aux1")
    parser.add_argument("--kinect-video", default=None,
                        help="video RGB do Kinect (obrigatorio no --mode three)")
    parser.add_argument("--kinect-timestamps", default=None,
                        help="CSV de timestamps do Kinect (frame,monotonic_s)")
    parser.add_argument("--kinect-offset-s", type=float, default=None,
                        help="offset manual Kinect<->aux0 (s) se os relogios "
                             "nao forem comparaveis")
    parser.add_argument("--kinect-mirror", action="store_true",
                        help="espelhar o video do Kinect (default: nao; o RGB "
                             "do Kinect no projeto nao e' espelhado)")
    parser.add_argument("--max-sync-dt-ms", type=float, default=34.0,
                        help="tolerancia de pareamento temporal (ms)")
    parser.add_argument("--max-reproj-px", type=float, default=10.0,
                        help="gate de erro de reprojecao por junção (px)")
    parser.add_argument("--every", type=int, default=1,
                        help="processa 1 a cada N pares de frames")
    parser.add_argument("--start-s", type=float, default=None,
                        help="comeca a processar depois de X s de sessao")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="limite de tempo de processamento (s)")
    parser.add_argument("--mirror", default="auto", choices=("auto", "on", "off"),
                        help="espelhar frames antes do MediaPipe (auto = do meta)")
    parser.add_argument("--mediapipe-gpu", action="store_true",
                        help="tenta o delegate GPU do MediaPipe (opcional)")
    parser.add_argument("--num-hands", type=int, default=2,
                        help="maximo de maos por vista (use 1 se so' uma mao "
                             "aparece; evita deteccao duplicada)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--handmvnet", default="off", choices=("off", "auto"),
                        help="estagio opcional HandMvNet (so' com checkpoint; "
                             "requer --mode three)")
    parser.add_argument("--handmvnet-checkpoint", default=None,
                        help="path do .ckpt HandMvNet (ou env HANDMVNET_CHECKPOINT)")
    parser.add_argument("--handmvnet-adapt-num-views", action="store_true",
                        help="EXPERIMENTAL: checkpoint de N vistas com 3 vistas")
    parser.add_argument("--dry-run", action="store_true",
                        help="so' le meta/timestamps/calibracao e mostra o plano")
    parser.add_argument("--json", action="store_true",
                        help="imprime o relatorio JSON no fim")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            plan = plan_session(
                args.session, aux_indices=args.aux_indices, mode=args.mode,
                kinect_video=args.kinect_video,
                kinect_timestamps=args.kinect_timestamps,
                kinect_offset_s=args.kinect_offset_s,
                max_sync_dt_s=args.max_sync_dt_ms / 1000.0,
                mirror=args.mirror)
            info = {
                "session": str(plan.session_dir),
                "mode": plan.mode,
                "aux_indices": list(plan.aux_indices),
                "mirror_effective": plan.mirror_effective,
                "sync": plan.sync,
                "pairs": len(plan.pairs),
                "kinect_sync": plan.kinect_sync,
                "timelines": {f"aux{t.index}": {
                    **t.summary(), **timestamp_stats(t)}
                    for t in plan.timelines},
            }
            if args.json:
                print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
            return 0
        detector_options = {
            "gpu": bool(args.mediapipe_gpu),
            "num_hands": max(1, int(args.num_hands)),
            "min_detection_confidence": args.min_detection_confidence,
            "min_presence_confidence": args.min_presence_confidence,
            "min_tracking_confidence": args.min_tracking_confidence,
        }
        report = process_session(
            args.session, out_dir=args.out, aux_indices=args.aux_indices,
            mode=args.mode, kinect_video=args.kinect_video,
            kinect_timestamps=args.kinect_timestamps,
            kinect_offset_s=args.kinect_offset_s,
            kinect_mirror=bool(args.kinect_mirror),
            max_sync_dt_s=args.max_sync_dt_ms / 1000.0,
            max_reproj_px=args.max_reproj_px, every=max(args.every, 1),
            start_s=args.start_s, max_seconds=args.max_seconds,
            mirror=args.mirror, detector_options=detector_options,
            handmvnet=args.handmvnet,
            handmvnet_checkpoint=args.handmvnet_checkpoint,
            handmvnet_adapt_num_views=args.handmvnet_adapt_num_views)
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:                           # pragma: no cover
        print("\ninterrompido", file=sys.stderr)
        return 130
    except Exception as exc:                            # pragma: no cover
        print(f"FALHA: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
