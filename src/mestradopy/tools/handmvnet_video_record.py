#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gravador OFFLINE das DUAS webcams auxiliares (aux0/aux1), sem MediaPipe.

Motivo (novo pipeline offline, 17/11/2026): o MediaPipe/HandMvNet online
disputa CPU/GPU com o Kinect e derruba o FPS justamente na aquisicao. Aqui a
aquisicao e' "burra": grava os dois streams direto em disco, com timestamp por
frame, e o processamento (MediaPipe 2D + triangulacao) roda DEPOIS, offline,
em `tools/handmvnet_video_process.py`. O Kinect continua online em outro
programa como referencia, mas NAO e' usado aqui.

O que e' gravado (por sessao, em `--dest`):
    <dest>/<sessao>/meta.json
    <dest>/<sessao>/aux00.<mp4|avi>            video da auxiliar 0
    <dest>/<sessao>/aux01.<mp4|avi>            video da auxiliar 1
    <dest>/<sessao>/timestamps_aux00.csv       timestamp/brilho/flash por frame
    <dest>/<sessao>/timestamps_aux01.csv

Codecs: tenta H.264 (`.mp4`), cai para MPEG-4 (`mp4v`) e por fim MJPEG
(`.avi`) -- o codec REAL vai no meta. Estimativa: 720p30 H.264 ~2-4 GB/h;
MJPEG ~15-40 GB/h (o tool AVISA quando cai para MJPEG).

Sincronismo: da' um FLASH/CALP na frente das duas cameras durante a gravacao.
O salto de brilho e' detectado por camera e gravado no CSV/meta
(`sync_events`); tambem da' para marcar manualmente com a tecla **S** (ex.:
palma) e o processador alinha as timelines por esses eventos.

Uso:
    # ver o que sera' usado (nao abre camera)
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\handmvnet_video_record.py --dry-run
    # gravar (ESC/Q encerra; S marca sincronismo manual)
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\handmvnet_video_record.py --dest F:\\handmvnet_videos --duration 120
    # sem janela (bancada remota/servidor)
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\handmvnet_video_record.py --no-window --duration 60

Depois processar:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\handmvnet_video_process.py --session F:\\handmvnet_videos\\<sessao>
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# sys.path do projeto (flat imports, mesmo esquema dos outros tools)
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

from vision.handmvnet_video_io import (     # noqa: E402
    CODECS,
    CODEC_AUTO_ORDER,
    DEFAULT_MIN_FREE_GB,
    SCHEMA_VERSION,
    CameraTimeline,
    FlashDetector,
    InsufficientSpaceError,
    TimestampRow,
    brightness_of,
    default_session_name,
    ensure_dir_and_space,
    estimate_storage_gb_per_hour,
    fmt_seconds,
    free_space_gb,
    iso_now,
    open_video_writer,
    probe_video,
    resolve_default_video_dir,
    safe_slug,
    write_meta,
)

#: Mesma convencao de `kinect_imu_groundtruth.AUXILIARY_MIRROR_HORIZONTAL`: a
#: calibracao da auxiliar foi feita na imagem ESPELHADA, entao o video gravado
#: tem de estar espelhado para o processamento bater com os npz.
MIRRORED_BY_DEFAULT = True


# ===========================================================================
# Configuracao e resultados
# ===========================================================================
@dataclass
class RecordConfig:
    """Parametros de uma sessao de gravacao (testavel sem hardware)."""

    dest: Path
    session_name: str
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    codecs: Tuple[str, ...] = CODEC_AUTO_ORDER
    codec_requested: str = "auto"
    mirror: bool = MIRRORED_BY_DEFAULT
    duration_s: Optional[float] = None
    max_frames: Optional[int] = None
    min_free_gb: float = DEFAULT_MIN_FREE_GB
    flash_delta: float = 35.0
    flash_min_brightness: float = 170.0
    show_window: bool = True
    countdown_s: float = 0.0
    verbose: bool = True
    needed_gb_estimate: float = 0.0
    notes: Tuple[str, ...] = ()


@dataclass
class CameraSource:
    """Uma camera a gravar: captura + identidade + calibracao (diagnostico)."""

    index: int
    name: str
    capture: Any
    calibration: Optional[Dict[str, Any]] = None
    uvc_report: Optional[Dict[str, Any]] = None
    size_hint: Optional[Tuple[int, int]] = None


@dataclass
class CameraRecording:
    """Resultado de UMA camera (preenchido pela thread)."""

    index: int
    name: str
    video_path: Optional[Path] = None
    timestamps_path: Optional[Path] = None
    codec: str = ""
    frames: int = 0
    dropped_reads: int = 0
    error: str = ""
    rows: List[TimestampRow] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    resolution: Optional[Tuple[int, int]] = None
    fps_measured: float = float("nan")
    latest_frame: Optional[np.ndarray] = None
    latest_time: float = 0.0
    latest_brightness: float = float("nan")
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def preview(self) -> Tuple[Optional[np.ndarray], float, float]:
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            return frame, self.latest_time, self.latest_brightness


@dataclass
class RecordResult:
    """Resultado da sessao inteira."""

    session_dir: Path
    meta: Dict[str, Any]
    recordings: List[CameraRecording]
    elapsed_s: float
    status: str
    stopped_reason: str

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "duration", "max_frames", "key")


# ===========================================================================
# Thread de gravacao de UMA camera
# ===========================================================================
class _CameraWorker(threading.Thread):
    """Le a camera, espelha, grava video + CSV de timestamps, detecta flash.

    Uma thread por camera: uma camera que trava/derruba o driver nao para a
    outra (o loop principal monitora e encerra a sessao com motivo).
    """

    def __init__(self, source: CameraSource, config: RecordConfig,
                 session_dir: Path,
                 stop_event: threading.Event,
                 sync_request: List[int]):
        super().__init__(name=f"rec-aux{source.index}", daemon=True)
        self.source = source
        self.config = config
        self.session_dir = session_dir
        self.stop_event = stop_event
        self.sync_request = sync_request
        self.recording = CameraRecording(index=source.index, name=source.name)
        self._handled_sync = 0

    # ------------------------------------------------------------------ run
    def run(self) -> None:                              # pragma: no cover (thread)
        rec = self.recording
        stem = self.session_dir / f"aux{int(self.source.index):02d}"
        timestamps_path = self.session_dir / f"timestamps_aux{int(self.source.index):02d}.csv"
        writer = None
        csv_file = None
        try:
            writer, video_path, codec = open_video_writer(
                stem, self.config.fps,
                (self.config.width, self.config.height), self.config.codecs)
            rec.video_path = video_path
            rec.codec = codec
            rec.timestamps_path = timestamps_path
            csv_file = open(timestamps_path, "w", newline="", encoding="utf-8")
            import csv as _csv
            csv_writer = _csv.writer(csv_file)
            csv_writer.writerow([
                "frame", "wall_s", "wall_iso", "monotonic_s", "perf_s",
                "pos_msec", "brightness", "flash", "manual_sync"])
            flash = FlashDetector(delta=self.config.flash_delta,
                                  min_brightness=self.config.flash_min_brightness)
            capture = self.source.capture
            self._gravar_loop(writer, csv_writer, csv_file, flash,
                              self.config, rec, capture)
        except Exception as exc:                        # pragma: no cover
            rec.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if writer is not None:
                    writer.release()
            except Exception:                           # pragma: no cover
                pass
            try:
                if csv_file is not None:
                    csv_file.flush()
                    csv_file.close()
            except Exception:                           # pragma: no cover
                pass
            rec.fps_measured = (rec.frames / max(rec.latest_time - self._t_start, 1e-9)
                                if rec.frames and getattr(self, "_t_start", None)
                                else float("nan"))

    def _gravar_loop(self, writer, csv_writer, csv_file, flash: FlashDetector,
                     config: RecordConfig, rec: CameraRecording,
                     capture: Any) -> None:
        self._t_start = time.monotonic()
        ultimo_flush = self._t_start
        falhas_seguidas = 0
        while not self.stop_event.is_set():
            if config.max_frames is not None and rec.frames >= int(config.max_frames):
                break
            try:
                ok, frame = capture.read()
            except Exception as exc:                    # pragma: no cover
                rec.error = f"read() levantou {type(exc).__name__}: {exc}"
                break
            t_mono = time.monotonic()
            t_wall = time.time()
            if not ok or frame is None:
                falhas_seguidas += 1
                rec.dropped_reads += 1
                if falhas_seguidas >= 40:
                    rec.error = "camera parou de entregar frames (40 leituras vazias)"
                    break
                time.sleep(0.004)
                continue
            falhas_seguidas = 0
            try:
                pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            except Exception:                           # pragma: no cover
                pos_msec = float("nan")
            if config.mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rec.resolution = (int(width), int(height))
            acendeu, brilho = flash.update(frame)
            manual = False
            pedido = int(self.sync_request[0])
            if pedido > self._handled_sync:
                self._handled_sync = pedido
                manual = True
            try:
                writer.write(frame)
            except Exception as exc:                    # pragma: no cover
                rec.error = f"escrita de video falhou: {type(exc).__name__}: {exc}"
                break
            row = TimestampRow(
                frame=rec.frames, wall_s=t_wall, monotonic_s=t_mono,
                perf_s=time.perf_counter(), pos_msec=(pos_msec
                                                     if np.isfinite(pos_msec) else 0.0),
                brightness=brilho, flash=acendeu, manual_sync=manual,
                wall_iso=iso_now(),
            )
            rec.rows.append(row)
            csv_writer.writerow(row.csv_row())
            if acendeu:
                rec.events.append({"camera": int(rec.index), "frame": int(rec.frames),
                                   "t_monotonic": float(t_mono),
                                   "wall_s": float(t_wall), "kind": "flash",
                                   "brightness": float(brilho)})
            if manual:
                rec.events.append({"camera": int(rec.index), "frame": int(rec.frames),
                                   "t_monotonic": float(t_mono),
                                   "wall_s": float(t_wall), "kind": "manual_sync",
                                   "brightness": float(brilho)})
            rec.frames += 1
            with rec.lock:
                rec.latest_frame = frame
                rec.latest_time = t_mono
                rec.latest_brightness = brilho
            agora = time.monotonic()
            if agora - ultimo_flush >= 0.5:
                csv_file.flush()
                ultimo_flush = agora


# ===========================================================================
# Sessao de gravacao (nucleo testavel sem hardware)
# ===========================================================================
def _preview_panel(recordings: Sequence[CameraRecording], config: RecordConfig,
                   elapsed_s: float, free_gb: float, status: str,
                   height: int = 360) -> Optional[np.ndarray]:
    """Monta a janela de preview (2 cameras lado a lado + HUD)."""
    frames: List[np.ndarray] = []
    for rec in recordings:
        frame, _t, brilho = rec.preview()
        if frame is None:
            frame = np.zeros((max(config.height, 2), max(config.width, 2), 3),
                             dtype=np.uint8)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.ndim == 3 and \
            frame.shape[2] == 4 else frame
        escala = height / max(bgr.shape[0], 1)
        largura = max(int(bgr.shape[1] * escala), 2)
        painel = cv2.resize(bgr, (largura, height))
        if rec.events:
            cv2.putText(painel, "FLASH/SYNC", (12, height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with rec.lock:
            n_flash = sum(1 for e in rec.events if e["kind"] == "flash")
        cv2.putText(painel, f"aux{rec.index}  {rec.frames} fr  "
                            f"flash={n_flash}  brilho={brilho:.0f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        frames.append(painel)
    if not frames:
        return None
    vis = np.hstack(frames)
    linhas = [
        f"GRAVANDO {config.session_name}  {fmt_seconds(elapsed_s)}  "
        f"livre={free_gb:.1f}GB  {status}",
        "S = sincronismo manual (palma/flash) | Q/ESC = encerrar",
        "Codecs: " + ",".join(config.codecs) + f" | espelhado={config.mirror}",
    ]
    for i, texto in enumerate(linhas):
        cv2.putText(vis, texto, (12, height + 26 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    return vis


def record_session(sources: Sequence[CameraSource], config: RecordConfig,
                   *, stop_event: Optional[threading.Event] = None
                   ) -> RecordResult:
    """Grava uma sessao com 2+ cameras (nucleo reutilizavel/testavel).

    Nao abre cameras: recebe `CameraSource.capture` (qualquer objeto com
    `read()`/`get()`/`release()`), entao a suite sintetica injeta cameras fake
    com frames conhecidos (inclusive o flash).

    Fluxo: valida destino/espaco -> cria pasta da sessao -> dispara uma thread
    por camera -> loop principal (janela/teclas/duracao/espaco) -> encerra ->
    meta.json com resumo, eventos de sincronismo e arquivos.
    """
    if len(sources) < 2:
        raise ValueError("gravacao offline exige >= 2 cameras auxiliares")
    dest, free_start = ensure_dir_and_space(
        config.dest, config.min_free_gb, config.needed_gb_estimate)
    session_dir = dest / safe_slug(config.session_name, default="sessao")
    session_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = stop_event or threading.Event()
    sync_request = [0]
    workers = [_CameraWorker(src, config, session_dir, stop_flag, sync_request)
               for src in sources]
    recordings = [w.recording for w in workers]
    t_inicio = time.monotonic()
    status, motivo = "ok", ""

    janela = "Gravacao auxiliares - HandMvNet offline"
    if config.show_window:
        try:
            cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(janela, 1280, 460)
            try:
                cv2.setWindowProperty(janela, cv2.WND_PROP_TOPMOST, 1)
            except Exception:                       # pragma: no cover
                pass
        except Exception:                           # pragma: no cover
            config.show_window = False

    for worker in workers:
        worker.start()

    ultima_checagem_espaco = t_inicio
    try:
        while True:
            agora = time.monotonic()
            elapsed = agora - t_inicio
            if stop_flag.is_set():
                motivo = motivo or "stop_event"
                break
            # --- fim por duracao / frames / erro / espaco --------------------
            if config.duration_s is not None and elapsed >= float(config.duration_s):
                status, motivo = "duration", f"duracao {config.duration_s:g}s"
                break
            if config.max_frames is not None and all(
                    rec.frames >= int(config.max_frames) for rec in recordings):
                status, motivo = "max_frames", f"max_frames={config.max_frames}"
                break
            erro = next((rec.error for rec in recordings if rec.error), "")
            if erro:
                status, motivo = "camera_error", erro
                break
            if all(not w.is_alive() for w in workers):
                status, motivo = "all_workers_stopped", "todas as threads terminaram"
                break
            if agora - ultima_checagem_espaco >= 2.0:
                ultima_checagem_espaco = agora
                livre = free_space_gb(dest)
                if livre < float(config.min_free_gb):
                    status = "space_low"
                    motivo = (f"espaco livre caiu para {livre:.2f} GB "
                              f"(< {config.min_free_gb:.2f} GB)")
                    break
            # --- janela/teclas ------------------------------------------------
            if config.show_window:
                vis = _preview_panel(recordings, config, elapsed,
                                     free_space_gb(dest), status)
                if vis is not None:
                    try:
                        cv2.imshow(janela, vis)
                        tecla = cv2.waitKey(1) & 0xFF
                        if tecla in (27, ord("q"), ord("Q")):
                            status, motivo = "key", "encerrado pelo usuario (Q/ESC)"
                            break
                        if tecla in (ord("s"), ord("S")):
                            sync_request[0] += 1
                            print(f"[sync] marcador manual #{sync_request[0]} "
                                  "registrado nas duas cameras", flush=True)
                    except Exception:               # pragma: no cover
                        config.show_window = False
                else:
                    time.sleep(0.01)
            else:
                if stop_flag.wait(0.01):
                    motivo = motivo or "stop_event"
                    break
    finally:
        stop_flag.set()
        for worker in workers:
            worker.join(timeout=2.0)
        for rec in recordings:
            try:
                rec.capture.release()
            except Exception:                       # pragma: no cover
                pass
        if config.show_window:
            try:
                cv2.destroyWindow(janela)
            except Exception:                       # pragma: no cover
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:                       # pragma: no cover
                pass

    elapsed = time.monotonic() - t_inicio
    free_end = free_space_gb(dest)
    meta = _build_meta(sources, recordings, config, session_dir, free_start,
                       free_end, elapsed, status, motivo)
    write_meta(session_dir, meta)
    result = RecordResult(session_dir=session_dir, meta=meta,
                          recordings=recordings, elapsed_s=elapsed,
                          status=status, stopped_reason=motivo)
    if config.verbose:
        print(meta_resumo(result))
    return result


def _build_meta(sources: Sequence[CameraSource], recordings: Sequence[CameraRecording],
                config: RecordConfig, session_dir: Path, free_start: float,
                free_end: float, elapsed: float, status: str,
                motivo: str) -> Dict[str, Any]:
    eventos: List[Dict[str, Any]] = []
    cameras: List[Dict[str, Any]] = []
    for src, rec in zip(sources, recordings):
        eventos.extend(rec.events)
        video_rel = None
        bytes_video = 0
        if rec.video_path is not None and rec.video_path.exists():
            video_rel = str(rec.video_path.relative_to(session_dir))
            bytes_video = int(rec.video_path.stat().st_size)
        timestamps_rel = None
        if rec.timestamps_path is not None and rec.timestamps_path.exists():
            timestamps_rel = str(rec.timestamps_path.relative_to(session_dir))
        codec_key = rec.codec or "?"
        expected = estimate_storage_gb_per_hour(
            config.width, config.height, config.fps,
            codec_key if codec_key in CODECS else "h264")
        cameras.append({
            "index": int(rec.index),
            "name": rec.name,
            "video": video_rel,
            "timestamps": timestamps_rel,
            "codec": CODECS.get(codec_key, {}).get("label", codec_key),
            "fourcc": CODECS.get(codec_key, {}).get("fourcc", "?"),
            "resolution": list(rec.resolution or (config.width, config.height)),
            "size_requested": [int(config.width), int(config.height)],
            "fps_requested": float(config.fps),
            "fps_measured": (float(rec.frames / elapsed) if elapsed > 0 else None),
            "frames": int(rec.frames),
            "dropped_reads": int(rec.dropped_reads),
            "error": rec.error,
            "bytes_video": bytes_video,
            "storage_estimate_gb_per_hour": [round(expected[0], 2),
                                             round(expected[1], 2)],
            "mirror": bool(config.mirror),
            "calibration": dict(src.calibration or {}),
            "uvc": dict(src.uvc_report or {}),
        })
    eventos.sort(key=lambda e: (float(e.get("t_monotonic", 0.0)),
                                int(e.get("camera", -1))))
    flashes = [e for e in eventos if e.get("kind") == "flash"]
    manuais = [e for e in eventos if e.get("kind") == "manual_sync"]
    estimativas = {key: [round(v, 2) for v in estimate_storage_gb_per_hour(
        config.width, config.height, config.fps, key)]
        for key in CODECS}
    return {
        "schema_version": SCHEMA_VERSION,
        "script": "handmvnet_video_record.py",
        "created_at": iso_now(),
        "finished_at": iso_now(),
        "session_name": config.session_name,
        "session_dir": str(session_dir),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "opencv": getattr(cv2, "__version__", "?"),
        "mirror": bool(config.mirror),
        "resolution_requested": [int(config.width), int(config.height)],
        "fps_requested": float(config.fps),
        "codec_requested": config.codec_requested,
        "codec_order": list(config.codecs),
        "duration_requested_s": config.duration_s,
        "max_frames": config.max_frames,
        "elapsed_s": round(float(elapsed), 3),
        "status": status,
        "stopped_reason": motivo,
        "free_gb_start": round(float(free_start), 2),
        "free_gb_end": round(float(free_end), 2),
        "min_free_gb": float(config.min_free_gb),
        "storage_estimate_gb_per_hour": estimativas,
        "cameras": cameras,
        "sync_events": eventos,
        "sync_summary": {
            "flash": len(flashes),
            "manual_sync": len(manuais),
            "cameras_with_flash": sorted({int(e["camera"]) for e in flashes}),
            "cameras_with_manual": sorted({int(e["camera"]) for e in manuais}),
        },
        "notes": list(config.notes),
    }


def meta_resumo(result: RecordResult) -> str:
    """Resumo legivel da sessao (usado no terminal e no --dry-run)."""
    meta = result.meta
    linhas = [
        "=" * 78,
        f"GRAVACAO {'OK' if result.ok else 'COM PROBLEMA'}: "
        f"{meta.get('session_name')}",
        "=" * 78,
        f"pasta      : {result.session_dir}",
        f"status     : {meta.get('status')} ({meta.get('stopped_reason') or '-'})",
        f"duracao    : {fmt_seconds(result.elapsed_s)} | "
        f"espaco livre inicio/fim: {meta.get('free_gb_start')} / "
        f"{meta.get('free_gb_end')} GB",
    ]
    for cam in meta.get("cameras", []):
        linhas.append(
            f"  aux{cam['index']:<2} {cam['name'][:26]:<26} "
            f"{cam['frames']:>6} fr  {cam.get('resolution')}  "
            f"{cam.get('codec', '?'):<14} {cam.get('bytes_video', 0) / 1e6:8.1f} MB")
    sync = meta.get("sync_summary", {})
    linhas.append(f"sincronismo: flash={sync.get('flash', 0)} "
                  f"manual={sync.get('manual_sync', 0)}"
                  + ("  (ok: evento detectado nas duas cameras)"
                     if len(sync.get("cameras_with_flash", [])) >= 2
                     or len(sync.get("cameras_with_manual", [])) >= 2
                     else "  AVISO: de' um flash/clap nas DUAS cameras!"))
    for nota in meta.get("notes", []):
        linhas.append(f"  nota: {nota}")
    linhas.append("=" * 78)
    return "\n".join(linhas)


# ===========================================================================
# Hardware: abertura/comandos das cameras reais
# ===========================================================================
def _abre_camera(index: int, config: RecordConfig, *, exposicao: Optional[float],
                 ganho: Optional[float], mjpeg: bool = True,
                 travar_exposicao: bool = True) -> CameraSource:
    """Abre uma webcam UVC via OpenCV/DirectShow e aplica modo/exposicao."""
    import camera_uvc
    import kinect_imu_groundtruth as kinect_gt

    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"camera auxiliar indice {index} nao abriu (DirectShow). "
            "Confira o USB/nome em `tools/problemas`/camera_names e tente "
            "outro --aux-indices.")
    camera_uvc.define_modo(cap, config.width, config.height, mjpeg=mjpeg)
    try:
        cap.set(cv2.CAP_PROP_FPS, float(config.fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:                               # pragma: no cover
        pass
    uvc_report: Dict[str, Any] = {}
    try:
        if travar_exposicao:
            uvc_report = camera_uvc.trava_exposicao(
                cap, exposicao=exposicao, ganho=ganho, fps=config.fps,
                rotulo=f"aux{index}") or {}
        else:
            uvc_report = {"fourcc": camera_uvc.le_fourcc(cap),
                          "fps_real": camera_uvc.mede_fps(cap)}
    except Exception as exc:                        # pragma: no cover
        uvc_report = {"erro": f"{type(exc).__name__}: {exc}"}
    largura = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    altura = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    nome = f"aux{index}"
    try:
        import camera_names
        nome = camera_names.nome_da_camera(int(index)) or kinect_gt.nome_auxiliar(int(index))
    except Exception:                               # pragma: no cover
        pass
    calib: Dict[str, Any] = {}
    try:
        npz = kinect_gt.aux_stereo_file(int(index))
        if npz is not None:
            data = np.load(npz)
            calib = {
                "file": str(npz),
                "camera_name": (str(data["camera_name"]).strip()
                                if "camera_name" in data.files else None),
                "rms": (float(data["rms"]) if "rms" in data.files else None),
                "auxiliary_size": ([int(v) for v in data["auxiliary_size"]]
                                   if "auxiliary_size" in data.files else None),
            }
    except Exception as exc:                        # pragma: no cover
        calib = {"erro": f"{type(exc).__name__}: {exc}"}
    return CameraSource(index=int(index), name=nome, capture=cap,
                        calibration=calib, uvc_report=uvc_report,
                        size_hint=(largura, altura))


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grava video offline das 2 webcams auxiliares (sem MediaPipe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--dest", default=None,
                        help="pasta de destino (default: env "
                             "HANDMVNET_VIDEO_DIR > F:\\handmvnet_videos > "
                             "outputs/handmvnet_videos)")
    parser.add_argument("--session-name", default=None,
                        help="nome da sessao (default: aux2v_AAAAMMDD_HHMMSS)")
    parser.add_argument("--aux-indices", nargs="+", type=int, default=None,
                        help="indices OpenCV das auxiliares (default: "
                             "indices_auxiliares_automaticos()[0:2])")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", default="auto",
                        choices=("auto", "h264", "mp4v", "mjpg"),
                        help="codec do video (auto = h264 > mp4v > mjpg)")
    parser.add_argument("--duration", type=float, default=None,
                        help="duracao maxima da sessao (s; default: sem limite)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="limite de frames POR camera (smoke/teste)")
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
                        help="reserva minima de espaco livre no disco de destino")
    parser.add_argument("--flash-delta", type=float, default=35.0,
                        help="salto de brilho (0-255) que conta como flash")
    parser.add_argument("--flash-min-brightness", type=float, default=170.0,
                        help="brilho minimo do frame de flash")
    parser.add_argument("--no-window", action="store_true",
                        help="nao abrir janela de preview (bancada remota)")
    parser.add_argument("--countdown", type=float, default=0.0,
                        help="contagem regressiva antes de comecar a gravar (s)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="NAO espelhar os frames (default espelha, igual a "
                             "convencao da calibracao)")
    parser.add_argument("--no-exposure-lock", action="store_true",
                        help="nao congelar exposicao/ganho (padrao congela)")
    parser.add_argument("--exposicao", type=str, default=None,
                        help="exposicao: -6 para todas, ou \"0:-6,1:-7\" "
                             "por camera")
    parser.add_argument("--ganho", type=str, default=None,
                        help="ganho: valor unico ou \"0:0,1:1\" por camera")
    parser.add_argument("--no-mjpeg", action="store_true",
                        help="nao pedir MJPG na abertura (pode cair para 8-10 fps)")
    parser.add_argument("--dry-run", action="store_true",
                        help="so' mostra cameras/calibracao/espaco/estimativa")
    parser.add_argument("--json", action="store_true",
                        help="imprime o meta JSON completo no fim")
    return parser


def _resumo_dry_run(indices: Sequence[int], config: RecordConfig,
                    dest: Path, calib_paths: Dict[int, Optional[Path]]) -> Dict[str, Any]:
    estimativas = {key: [round(v, 2) for v in estimate_storage_gb_per_hour(
        config.width, config.height, config.fps, key)] for key in CODECS}
    info = {
        "dest": str(dest),
        "dest_existe": Path(dest).exists(),
        "free_gb": round(free_space_gb(dest), 2),
        "min_free_gb": config.min_free_gb,
        "session_name": config.session_name,
        "resolution": [config.width, config.height],
        "fps": config.fps,
        "mirror": config.mirror,
        "duration_s": config.duration_s,
        "max_frames": config.max_frames,
        "codec_requested": config.codec_requested,
        "storage_estimate_gb_per_hour": estimativas,
        "needed_gb_estimate": round(config.needed_gb_estimate, 3),
        "cameras": [],
    }
    for idx in indices:
        npz = calib_paths.get(int(idx))
        calib: Dict[str, Any] = {"file": str(npz) if npz else None,
                                 "stereo_usavel": False}
        if npz is not None:
            try:
                data = np.load(npz)
                calib.update({
                    "camera_name": (str(data["camera_name"]).strip()
                                    if "camera_name" in data.files else None),
                    "rms": (float(data["rms"]) if "rms" in data.files else None),
                    "auxiliary_size": ([int(v) for v in data["auxiliary_size"]]
                                       if "auxiliary_size" in data.files else None),
                })
                calib["stereo_usavel"] = bool(
                    calib.get("rms") is not None and calib["rms"] <= 5.0)
            except Exception as exc:
                calib["erro"] = f"{type(exc).__name__}: {exc}"
        info["cameras"].append({"index": int(idx), "calibration": calib})
    return info


def _imprime_dry_run(info: Dict[str, Any]) -> None:
    print("=" * 78)
    print("DRY-RUN: configuracao da gravacao offline das auxiliares")
    print("=" * 78)
    print(f"destino  : {info['dest']}")
    print(f"espaco   : {info['free_gb']} GB livres (reserva "
          f"{info['min_free_gb']} GB; estimativa das 2 cameras: "
          f"{info['needed_gb_estimate']} GB)")
    print(f"sessao   : {info['session_name']}  "
          f"{info['resolution'][0]}x{info['resolution'][1]} @ {info['fps']:g}fps "
          f"espelhado={info['mirror']}")
    print("estimativa de armazenamento (2 cameras, 1 h):")
    for codec, (lo, hi) in info["storage_estimate_gb_per_hour"].items():
        print(f"    {codec:<5} {lo * 2:6.1f} - {hi * 2:6.1f} GB/h "
              f"({CODECS[codec]['label']})")
    print("cameras:")
    for cam in info["cameras"]:
        calib = cam["calibration"]
        estado = "SEM CALIBRACAO" if not calib.get("file") else (
            f"RMS={calib.get('rms')} px "
            f"{'USAVEL' if calib.get('stereo_usavel') else 'NAO USUAL'}")
        print(f"    aux{cam['index']}  {estado}")
        if calib.get("camera_name"):
            print(f"          npz camera_name={calib['camera_name']} "
                  f"size={calib.get('auxiliary_size')}")
    print("para gravar, rode sem --dry-run (ESC/Q encerra; S = sync manual).")
    print("=" * 78)


def _controle_por_camera(texto, indice):
    """Converte "-6" ou "0:-6,1:-7" no valor daquela camera."""
    if texto is None:
        return None
    from camera_uvc import parse_controle_por_camera
    c = parse_controle_por_camera(texto)
    if c.get("todos") is not None:
        return float(c["todos"])
    return c.get("indices", {}).get(int(indice))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dest = Path(args.dest) if args.dest else resolve_default_video_dir()
    session_name = args.session_name or default_session_name()
    mirror = not args.no_mirror
    if args.codec == "auto":
        codecs = CODEC_AUTO_ORDER
    else:
        codecs = (args.codec,)

    # --- indices + calibracao (importa o core do projeto) -------------------
    import kinect_imu_groundtruth as kinect_gt
    indices = ([int(i) for i in args.aux_indices] if args.aux_indices
               else list(kinect_gt.indices_auxiliares_automaticos())[:2])
    if len(indices) < 2:
        indices = list(kinect_gt.AUX_CAMERA_INDICES)[:2]
    if len(indices) < 2:
        print("ERRO: nao encontrei 2 cameras auxiliares; passe --aux-indices 0 1",
              file=sys.stderr)
        return 2
    calib_paths: Dict[int, Optional[Path]] = {}
    for idx in indices:
        try:
            calib_paths[int(idx)] = kinect_gt.aux_stereo_file(int(idx))
        except Exception:                           # pragma: no cover
            calib_paths[int(idx)] = None

    needed_gb = 0.0
    if args.duration:
        # Estimativa conservadora: pior codec da cadeia (MJPEG) para a reserva.
        lo_hi = estimate_storage_gb_per_hour(args.width, args.height, args.fps,
                                             args.codec if args.codec != "auto"
                                             else "mjpg")
        needed_gb = float(lo_hi[1])
    config = RecordConfig(
        dest=dest, session_name=session_name, width=max(int(args.width), 32),
        height=max(int(args.height), 32), fps=max(float(args.fps), 1.0),
        codecs=tuple(codecs), codec_requested=str(args.codec), mirror=mirror,
        duration_s=args.duration, max_frames=args.max_frames,
        min_free_gb=float(args.min_free_gb), flash_delta=float(args.flash_delta),
        flash_min_brightness=float(args.flash_min_brightness),
        show_window=not args.no_window, countdown_s=float(args.countdown),
        verbose=True, needed_gb_estimate=needed_gb,
        notes=("gravacao offline das auxiliares (sem MediaPipe online)",))

    if args.dry_run:
        info = _resumo_dry_run(indices, config, dest, calib_paths)
        if args.json:
            print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        else:
            _imprime_dry_run(info)
        return 0

    # --- abre as cameras reais ---------------------------------------------
    sources: List[CameraSource] = []
    try:
        for idx in indices:
            print(f"[camera] abrindo auxiliar {idx}...", flush=True)
            source = _abre_camera(
                int(idx), config,
                exposicao=_controle_por_camera(args.exposicao, idx),
                ganho=_controle_por_camera(args.ganho, idx),
                mjpeg=not args.no_mjpeg,
                travar_exposicao=not args.no_exposure_lock)
            largura_real, altura_real = source.size_hint or (0, 0)
            print(f"[camera] aux{idx}: {source.name!r} {largura_real}x{altura_real}"
                  + (f" calib={calib_paths[int(idx)]}" if calib_paths.get(int(idx))
                     else " SEM CALIBRACAO STEREO"), flush=True)
            if int(largura_real) != config.width or int(altura_real) != config.height:
                print(f"[camera] ATENCAO: pedido {config.width}x{config.height}, "
                      f"camera entregou {largura_real}x{altura_real} -- a "
                      "calibracao e' escalada no processamento, mas o ideal e' "
                      "gravar na resolucao calibrada", flush=True)
            sources.append(source)
        if config.countdown_s > 0 and config.show_window:
            _countdown_preview(sources, config)
        result = record_session(sources, config)
    except InsufficientSpaceError as exc:
        print(f"ERRO DE ESPACO: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    finally:
        for source in sources:
            try:
                source.capture.release()
            except Exception:                       # pragma: no cover
                pass
    if args.json:
        print(json.dumps(result.meta, indent=2, ensure_ascii=False, default=str))
    if not result.ok:
        return 1
    print("\nProcesse a sessao com:")
    print(f"  .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\"
          f"handmvnet_video_process.py --session \"{result.session_dir}\"")
    return 0


def _countdown_preview(sources: Sequence[CameraSource],
                       config: RecordConfig) -> None:
    """Contagem regressiva com preview (aquecimento das cameras antes de gravar)."""
    janela = "Contagem regressiva"
    try:
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(janela, 1280, 460)
    except Exception:                               # pragma: no cover
        return
    t0 = time.monotonic()
    try:
        while True:
            restante = config.countdown_s - (time.monotonic() - t0)
            if restante <= 0:
                break
            paineis = []
            for source in sources:
                ok, frame = source.capture.read()
                if not ok or frame is None:
                    continue
                if config.mirror:
                    frame = cv2.flip(frame, 1)
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if \
                    frame.ndim == 3 and frame.shape[2] == 4 else frame
                escala = 360 / max(bgr.shape[0], 1)
                paineis.append(cv2.resize(
                    bgr, (max(int(bgr.shape[1] * escala), 2), 360)))
            if paineis:
                vis = np.hstack(paineis)
                cv2.putText(vis, f"Comecando em {restante:.0f}s...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 220, 255), 2)
                cv2.imshow(janela, vis)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
    finally:
        try:
            cv2.destroyWindow(janela)
        except Exception:                           # pragma: no cover
            pass


if __name__ == "__main__":
    raise SystemExit(main())
