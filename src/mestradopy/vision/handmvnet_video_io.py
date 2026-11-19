"""Biblioteca comum do pipeline OFFLINE de video das duas auxiliares.

Existe para que `tools/handmvnet_video_record.py` (aquisicao) e
`tools/handmvnet_video_process.py` (processamento posterior) compartilhem o
MESMO formato de arquivos e a MESMA matematica de sincronizacao -- sem
duplicar codigo:

  - formato de sessao em disco: `meta.json` + 1 video + 1 CSV de timestamps por
    camera auxiliar (`aux00.*`, `timestamps_aux00.csv`, ...);
  - CSV de timestamps por frame (relogio de parede, `time.monotonic`,
    `time.perf_counter`, brightness, flags de flash/sync manual);
  - deteccao de FLASH (evento de sincronizacao) por brilho;
  - casamento temporal de frames entre as duas cameras (`pair_by_time`);
  - estimativa do offset de relogio/sincronismo entre as cameras;
  - estimativa de armazenamento e checagem de espaco livre em disco;
  - abertura de `cv2.VideoWriter` com cadeia de fallback de codec
    (H.264 -> MP4V -> MJPG), porque no Windows o H.264 depende do codec
    instalado (openh264/FFmpeg build) e o MJPEG sempre funciona.

Nada aqui abre camera nem depende de Kinect/MediaPipe: e' IO puro
(numpy/cv2/stdlib), o que deixa a suite sintetica testar o fluxo inteiro sem
hardware. A convencao de referenciais e de pixels e' a do projeto
(`kinect_imu_groundtruth.py`).
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:                                     # pragma: no cover
    cv2 = None

# =============================================================================
# Constantes / formato em disco
# =============================================================================
#: Versao do formato de sessao (mudou layout de arquivos/meta -> incrementar).
SCHEMA_VERSION = 1
META_FILENAME = "meta.json"
PROCESSED_META_FILENAME = "meta_processado.json"

#: Colunas do CSV de timestamps por frame (uma linha por frame gravado).
TIMESTAMP_COLUMNS: Tuple[str, ...] = (
    "frame",            # indice do frame no video (0-based)
    "wall_s",           # epoch (time.time) no instante da leitura
    "wall_iso",         # mesma coisa legivel
    "monotonic_s",      # time.monotonic (mesmo processo; sincroniza cameras)
    "perf_s",           # time.perf_counter (diagnostico)
    "pos_msec",         # CAP_PROP_POS_MSEC do driver (pode ser 0/mentir)
    "brightness",       # brilho medio do frame 0..255 (para o flash)
    "flash",            # 1 no frame em que o flash foi detectado
    "manual_sync",      # 1 no frame marcado pelo usuario (tecla S)
)

#: Codecs suportados, em ordem de preferencia do modo "auto".
#: `fourcc` + extensao + rotulo + faixa de bits por pixel (bps) usada na
#: estimativa de armazenamento. H.264 ~0,14-0,28 bpp em webcam 720p30.
CODECS: Dict[str, Dict[str, Any]] = {
    "h264": {"fourcc": "avc1", "ext": ".mp4", "label": "H.264",
             "bps_range": (0.14, 0.28)},
    "mp4v": {"fourcc": "mp4v", "ext": ".mp4", "label": "MPEG-4 (mp4v)",
             "bps_range": (0.20, 0.40)},
    "mjpg": {"fourcc": "MJPG", "ext": ".avi", "label": "MJPEG",
             "bps_range": (1.5, 3.0)},
}
CODEC_AUTO_ORDER: Tuple[str, ...] = ("h264", "mp4v", "mjpg")

#: Diretorio default de gravacao: env explicita > F:\handmvnet_videos (quando o
#: disco F: existe, como pedido na tarefa) > outputs/handmvnet_videos do repo.
DEFAULT_VIDEO_DIR_ENV = "HANDMVNET_VIDEO_DIR"
_FALLBACK_VIDEO_DIR = Path(__file__).resolve().parents[3] / "outputs" / "handmvnet_videos"
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MIN_FREE_GB = 2.0
#: Frequencia (s) da checagem de espaco livre durante a gravacao.
FREE_SPACE_CHECK_PERIOD_S = 2.0


class InsufficientSpaceError(RuntimeError):
    """Disco sem espaco livre suficiente para (continuar) a gravacao."""


# =============================================================================
# Caminhos / sessao
# =============================================================================
def resolve_default_video_dir() -> Path:
    """Melhor destino default de gravacao (env > F:\\handmvnet_videos > repo).

    NUNCA cria o diretorio: quem grava chama `ensure_dir_and_space`.
    """
    env = os.environ.get(DEFAULT_VIDEO_DIR_ENV, "").strip()
    if env:
        return Path(env)
    drive_f = Path("F:/")
    if drive_f.exists():
        return Path("F:/handmvnet_videos")
    return _FALLBACK_VIDEO_DIR


def default_session_name(prefix: str = "aux2v") -> str:
    """Nome de sessao com data/hora (ordenavel no explorer)."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_dir_and_space(destination: str | os.PathLike,
                         min_free_gb: float = DEFAULT_MIN_FREE_GB,
                         needed_gb: float = 0.0) -> Tuple[Path, float]:
    """Cria o diretorio de destino e confere espaco livre.

    `needed_gb` e' a estimativa para a sessao inteira (quando a duracao e'
    conhecida); exige `free >= min_free_gb + needed_gb`. Levanta
    `InsufficientSpaceError` com numeros no texto (nada de "disco cheio" mudo).

    Returns:
        (destination, free_gb)
    """
    dest = Path(destination)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InsufficientSpaceError(
            f"nao consegui criar/abrir o diretorio de destino {dest}: {exc}"
        ) from exc
    free_gb = free_space_gb(dest)
    minimo = float(min_free_gb) + float(max(needed_gb, 0.0))
    if free_gb < minimo:
        raise InsufficientSpaceError(
            f"espaco livre insuficiente em {dest}: {free_gb:.2f} GB livres, "
            f"precisa de >= {minimo:.2f} GB "
            f"(reserva {min_free_gb:.2f} GB"
            + (f" + estimativa da sessao {needed_gb:.2f} GB" if needed_gb else "")
            + "). Escolha outro disco/filepath (`--dest`) ou libere espaco.")
    return dest, free_gb


def free_space_gb(path: str | os.PathLike) -> float:
    """Espaco livre (GB) no volume que contem `path` (0 se nao der para medir)."""
    alvo = Path(path)
    try:
        # `path` pode ainda nao existir: sobe ate' o ancestral existente.
        while not alvo.exists() and alvo.parent != alvo:
            alvo = alvo.parent
        return float(shutil.disk_usage(str(alvo)).free) / (1024.0 ** 3)
    except OSError:                                     # pragma: no cover
        return 0.0


def estimate_storage_gb(seconds: float, width: int, height: int, fps: float,
                        codec: str = "h264") -> Tuple[float, float]:
    """Estimativa (GB_min, GB_max) de armazenamento para a duracao/parametros.

    Base: bits por pixel (bpp) tipicos de webcam 720p (ver `CODECS`).
    H.264 720p30 ~0,14-0,28 bpp -> ~1,5-3,0 GB/h (na ordem dos 2-4 GB/h
    medidos na pratica com movimento). MJPEG cru e' muito maior (15-40 GB/h
    a 720p30) -- por isso o default e' tentar H.264 e avisar quando cai.
    """
    codec_key = _codec_key(codec)
    bps_min, bps_max = CODECS[codec_key]["bps_range"]
    pixels_per_s = max(int(width), 1) * max(int(height), 1) * max(float(fps), 1.0)
    segundos = max(float(seconds), 1.0)
    bytes_min = pixels_per_s * bps_min * segundos / 8.0
    bytes_max = pixels_per_s * bps_max * segundos / 8.0
    return (bytes_min / (1024.0 ** 3), bytes_max / (1024.0 ** 3))


def _codec_key(codec: str) -> str:
    return (codec or "h264").strip().lower()


def estimate_storage_gb_per_hour(width: int, height: int, fps: float,
                                 codec: str = "h264") -> Tuple[float, float]:
    """Atalho: estimativa para 1 hora."""
    return estimate_storage_gb(3600.0, width, height, fps, codec)


def iso_now() -> str:
    """Data/hora local ISO-8601 com milissegundos (meta legivel)."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


# =============================================================================
# Meta / CSV
# =============================================================================
def write_meta(session_dir: str | os.PathLike, meta: Dict[str, Any],
               filename: str = META_FILENAME) -> Path:
    """Grava o meta JSON da sessao (atualiza `meta["updated_at"]`)."""
    path = Path(session_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["updated_at"] = iso_now()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    return path


def read_meta(session_dir: str | os.PathLike,
              filename: str = META_FILENAME) -> Dict[str, Any]:
    """Le o meta JSON da sessao; levanta erro legivel quando nao existe."""
    path = Path(session_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"meta da sessao nao encontrado: {path} (grave primeiro com "
            "tools/handmvnet_video_record.py ou passe --session correto)")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"meta invalido em {path}: {exc}") from exc


@dataclass
class TimestampRow:
    """Uma linha do CSV de timestamps (um frame gravado)."""

    frame: int
    wall_s: float
    monotonic_s: float
    perf_s: float = 0.0
    pos_msec: float = 0.0
    brightness: float = float("nan")
    flash: bool = False
    manual_sync: bool = False
    wall_iso: str = ""

    def csv_row(self) -> List[Any]:
        iso = self.wall_iso
        if not iso and float(self.wall_s) > 1e8:
            # So' tenta formatar epoch REAL (1970+); timestamps sinteticos
            # pequenos (testes) ficam sem ISO em vez de derrubar a gravacao.
            try:
                iso = datetime.fromtimestamp(
                    float(self.wall_s)).astimezone().isoformat(
                    timespec="milliseconds")
            except (OSError, OverflowError, ValueError):  # pragma: no cover
                iso = ""
        return [
            int(self.frame),
            f"{float(self.wall_s):.6f}",
            iso,
            f"{float(self.monotonic_s):.6f}",
            f"{float(self.perf_s):.6f}",
            f"{float(self.pos_msec):.3f}",
            f"{float(self.brightness):.2f}",
            int(bool(self.flash)),
            int(bool(self.manual_sync)),
        ]

    @classmethod
    def from_mapping(cls, row: Dict[str, Any]) -> "TimestampRow":
        def _f(key: str, default: float = 0.0) -> float:
            try:
                value = row.get(key, default)
                return float(value)
            except (TypeError, ValueError):
                return default

        return cls(
            frame=int(row.get("frame", 0)),
            wall_s=_f("wall_s"),
            monotonic_s=_f("monotonic_s"),
            perf_s=_f("perf_s"),
            pos_msec=_f("pos_msec"),
            brightness=_f("brightness", float("nan")),
            flash=bool(int(row.get("flash", 0) or 0)),
            manual_sync=bool(int(row.get("manual_sync", 0) or 0)),
            wall_iso=str(row.get("wall_iso", "") or ""),
        )


def write_timestamps_csv(path: str | os.PathLike,
                         rows: Sequence[TimestampRow]) -> Path:
    """Grava o CSV de timestamps de UMA camera."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(TIMESTAMP_COLUMNS)
        for row in rows:
            writer.writerow(row.csv_row())
    return out


def read_timestamps_csv(path: str | os.PathLike) -> List[TimestampRow]:
    """Le o CSV de timestamps de UMA camera (tolera arquivo truncado)."""
    rows: List[TimestampRow] = []
    with open(Path(path), "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for linha in reader:
            try:
                rows.append(TimestampRow.from_mapping(linha))
            except (TypeError, ValueError):
                continue
    return rows


# =============================================================================
# Video: probe e writer com fallback de codec
# =============================================================================
def probe_video(path: str | os.PathLike) -> Dict[str, Any]:
    """Le metadados do video (frames, fps, tamanho) -- usado no processamento.

    O `frame_count` do `CAP_PROP_FRAME_COUNT` pode mentir em MP4; o numero
    efetivo e' confirmado contra o CSV de timestamps no processador.
    """
    info: Dict[str, Any] = {"path": str(path), "opened": False}
    if cv2 is None:                                     # pragma: no cover
        return info
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return info
        info.update({
            "opened": True,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        })
        if info["fps"] and info["fps"] > 0:
            info["duration_s"] = info["frame_count"] / info["fps"]
    finally:
        cap.release()
    return info


def open_video_writer(base_path: str | os.PathLike, fps: float,
                      size: Tuple[int, int],
                      codecs: Sequence[str] = CODEC_AUTO_ORDER
                      ) -> Tuple[Any, Path, str]:
    """Abre `cv2.VideoWriter` tentando os codecs em ordem, com fallback.

    `base_path` e' o caminho SEM extensao (ela vem do codec). Devolve
    `(writer, path_real, codec_key)`. Se nenhum codec abrir, levanta
    RuntimeError listando o que foi tentado.

    H.264 (`avc1`) depende do build do OpenCV/FFmpeg do Windows; quando nao
    abre, `isOpened()` e' False e caimos para MP4V e depois MJPG (AVI),
    sempre registrando nos metadados qual foi usado de verdade.
    """
    if cv2 is None:                                     # pragma: no cover
        raise RuntimeError("opencv (cv2) e' obrigatorio para gravar video")
    caminho = Path(base_path)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(size[0]), int(size[1])
    tentados: List[str] = []
    for codec in codecs:
        key = _codec_key(codec)
        if key not in CODECS:
            continue
        fourcc = cv2.VideoWriter_fourcc(*CODECS[key]["fourcc"])
        out_path = caminho.with_suffix(CODECS[key]["ext"])
        try:
            writer = cv2.VideoWriter(str(out_path), fourcc, float(fps),
                                     (width, height))
            aberto = bool(writer.isOpened())
        except Exception:                       # pragma: no cover (driver codec)
            writer, aberto = None, False
        if aberto:
            return writer, out_path, key
        try:
            if writer is not None:
                writer.release()
        except Exception:                       # pragma: no cover
            pass
        try:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except OSError:                         # pragma: no cover
            pass
        tentados.append(f"{CODECS[key]['label']} ({CODECS[key]['fourcc']})")
    raise RuntimeError(
        "nenhum codec de video abriu. Tentados: " + ", ".join(tentados)
        + ". O MJPEG costuma ser o fallback universal (AVI); se nem ele abrir, "
        "verifique se o OpenCV tem suporte a video gravado (cv2.VideoWriter).")


# =============================================================================
# Flash (evento de sincronizacao visual)
# =============================================================================
def brightness_of(frame: np.ndarray) -> float:
    """Brilho medio 0..255 de um frame BGR/BGRA/escala de cinza."""
    if frame is None:
        return float("nan")
    if cv2 is None:                                     # pragma: no cover
        arr = np.asarray(frame)
        return float(np.mean(arr))
    img = np.asarray(frame)
    if img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(cv2.mean(gray)[0])


class FlashDetector:
    """Detecta o FLASH de sincronizacao por salto de brilho (borda de subida).

    O protocolo de bancada: o experimentador da' um flash/clap na frente das
    duas cameras. Visualmente o frame inteiro fica branco: o brilho medio salta
    de ~40-140 para ~200-255. Detecta a BORDA (nao fica re-detectando os frames
    seguintes do flash) com `cooldown_frames` de carencia.

    A linha do tempo dos eventos e' o `time.monotonic` do MESMO processo para as
    duas cameras (ver recorder), entao a comparacao entre cameras e' direta.
    """

    def __init__(self, delta: float = 35.0, min_brightness: float = 170.0,
                 cooldown_frames: int = 8, ema_alpha: float = 0.05):
        self.delta = float(delta)
        self.min_brightness = float(min_brightness)
        self.cooldown_frames = int(cooldown_frames)
        self.ema_alpha = float(ema_alpha)
        self._baseline: Optional[float] = None
        self._cooldown = 0
        self._em_flash = False
        self.events = 0

    def update(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Atualiza com um frame; devolve `(acendeu_agora, brilho_medio)`."""
        brilho = brightness_of(frame)
        if not np.isfinite(brilho):
            return False, brilho
        if self._baseline is None:
            self._baseline = brilho
            return False, brilho
        if self._cooldown > 0:
            self._cooldown -= 1
        muito_claro = brilho >= self.min_brightness
        salto = (brilho - self._baseline) >= self.delta
        flash = bool(muito_claro and salto and self._cooldown == 0)
        if flash:
            self.events += 1
            self._cooldown = self.cooldown_frames
            self._em_flash = True
        elif not muito_claro:
            self._em_flash = False
        # Baseline adapta devagar, exceto durante o flash (senao a base sobe).
        if not self._em_flash:
            self._baseline = ((1.0 - self.ema_alpha) * self._baseline
                              + self.ema_alpha * brilho)
        return flash, brilho


# =============================================================================
# Sincronismo / casamento de frames
# =============================================================================
@dataclass
class CameraTimeline:
    """Vista/timeline de UMA camera gravada: video + CSV de timestamps."""

    name: str
    index: int
    video_path: Path
    rows: List[TimestampRow] = field(default_factory=list)
    fps_hint: float = 30.0

    @classmethod
    def from_meta_entry(cls, entry: Dict[str, Any], session_dir: str | os.PathLike
                        ) -> "CameraTimeline":
        """Constroi a timeline a partir da entrada `cameras[]` do meta."""
        base = Path(session_dir)
        cam = dict(entry)
        video = Path(cam.get("video") or "")
        if not video.is_absolute():
            video = base / video
        stamps = Path(cam.get("timestamps") or "")
        if not stamps.is_absolute():
            stamps = base / stamps
        rows = read_timestamps_csv(stamps) if stamps.is_file() else []
        return cls(name=str(cam.get("name") or f"aux{cam.get('index', '?')}"),
                   index=int(cam.get("index", -1)), video_path=video,
                   rows=rows, fps_hint=float(cam.get("fps_measured")
                                             or cam.get("fps") or 30.0))

    @property
    def frame_count(self) -> int:
        return len(self.rows)

    def times(self, clock: str = "monotonic") -> np.ndarray:
        """Array de tempos (s) no relogio pedido ('monotonic' ou 'wall')."""
        if clock == "wall":
            return np.asarray([row.wall_s for row in self.rows], dtype=np.float64)
        return np.asarray([row.monotonic_s for row in self.rows],
                          dtype=np.float64)

    def np_time(self) -> np.ndarray:
        return self.times("monotonic")

    def flash_events(self) -> List[Dict[str, Any]]:
        return [{"frame": row.frame, "t_monotonic": row.monotonic_s,
                 "wall_s": row.wall_s, "kind": "flash"}
                for row in self.rows if row.flash]

    def manual_sync_events(self) -> List[Dict[str, Any]]:
        return [{"frame": row.frame, "t_monotonic": row.monotonic_s,
                 "wall_s": row.wall_s, "kind": "manual"}
                for row in self.rows if row.manual_sync]

    def summary(self) -> Dict[str, Any]:
        tempos = self.np_time()
        return {
            "name": self.name,
            "index": self.index,
            "video": str(self.video_path),
            "frames_csv": len(self.rows),
            "first_t": float(tempos[0]) if tempos.size else None,
            "last_t": float(tempos[-1]) if tempos.size else None,
            "duration_s": (float(tempos[-1] - tempos[0])
                           if tempos.size >= 2 else 0.0),
            "fps_estimado": (float((len(tempos) - 1) / (tempos[-1] - tempos[0]))
                             if tempos.size >= 2 and tempos[-1] > tempos[0]
                             else None),
            "n_flash": len(self.flash_events()),
            "n_manual_sync": len(self.manual_sync_events()),
        }


def _events_by_camera(events: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    agrupado: Dict[int, List[Dict[str, Any]]] = {}
    for evento in events or []:
        try:
            cam = int(evento.get("camera"))
            t = float(evento.get("t_monotonic"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(t):
            continue
        agrupado.setdefault(cam, []).append(dict(evento, t_monotonic=t))
    for cam in agrupado:
        agrupado[cam].sort(key=lambda e: e["t_monotonic"])
    return agrupado


def estimate_sync_offset(timeline_a: CameraTimeline, timeline_b: CameraTimeline,
                         *,
                         events: Optional[Iterable[Dict[str, Any]]] = None,
                         max_event_dt_s: float = 0.75,
                         max_lockstep_offset_s: float = 2.0) -> Dict[str, Any]:
    """Estima `offset` tal que `t_b + offset ~= t_a` (frames simultaneos).

    Ordem de confianca:
      1. eventos de sincronismo (flash/clap) registrados nas DUAS cameras:
         `offset = mediana(t_b_evento - t_a_evento)`;
      2. sem eventos: assume que as duas gravacoes comecaram juntas
         (lockstep do MESMO processo) e usa a mediana de `t_b - t_a` nos
         primeiros frames -- valido so' quando as cameras foram abertas pelo
         mesmo recorder;
      3. senao inventa `offset=0` com `method="none"` e `warning` (o
         processamento continua, mas o pareamento pode errar por 1 frame).

    Devolve dict: offset_s, method ("flash"|"manual"|"lockstep_start"|"none"),
    residual_s (mediana de |t_b_evento+offset - t_a_evento|), n_events, warning.
    """
    por_camera = _events_by_camera(events or [])
    ev_a = por_camera.get(int(timeline_a.index), [])
    ev_b = por_camera.get(int(timeline_b.index), [])
    # Prefere flash a manual se houver ambos; se so' um tipo aparece nas duas,
    # usa esse tipo nos dois lados.
    def _por_tipo(lista: List[Dict[str, Any]], tipo: str) -> List[Dict[str, Any]]:
        return [e for e in lista if str(e.get("kind", "")).startswith(tipo)]

    for tipo, method in (("flash", "flash"), ("manual", "manual")):
        a_tipo, b_tipo = _por_tipo(ev_a, tipo), _por_tipo(ev_b, tipo)
        if not a_tipo or not b_tipo:
            continue
        difs: List[float] = []
        usados = 0
        for evento_a in a_tipo:
            t_a = float(evento_a["t_monotonic"])
            candidatos = [float(e["t_monotonic"]) for e in b_tipo]
            if not candidatos:
                continue
            t_b = min(candidatos, key=lambda t: abs(t - t_a))
            if abs(t_b - t_a) <= float(max_event_dt_s):
                difs.append(t_a - t_b)
                usados += 1
        if difs:
            offset = float(np.median(difs))       # t_b + offset ~= t_a
            residual = float(np.median(np.abs(np.asarray(difs) - offset)))
            return {"offset_s": offset, "method": method,
                    "residual_s": residual, "n_events": usados, "warning": ""}

    # Sem eventos nas duas cameras: alinhamento por inicio (mesmo processo).
    tempos_a = timeline_a.np_time()
    tempos_b = timeline_b.np_time()
    if tempos_a.size and tempos_b.size:
        k = min(len(tempos_a), len(tempos_b), 150)
        difs = tempos_a[:k] - tempos_b[:k]
        offset = float(np.median(difs))       # t_b + offset ~= t_a
        residual = float(np.median(np.abs(difs - offset)))
        if abs(offset) <= float(max_lockstep_offset_s):
            return {"offset_s": offset, "method": "lockstep_start",
                    "residual_s": residual, "n_events": int(k), "warning": ""}
    return {"offset_s": 0.0, "method": "none", "residual_s": float("nan"),
            "n_events": 0,
            "warning": ("sem flash/clap nas duas cameras e sem inicio "
                        "simultaneo confiavel: pareamento por tempo pode errar "
                        "1+ frame; marque um flash no inicio da sessao")}


def pair_by_time(times_a: np.ndarray, times_b: np.ndarray,
                 offset_s: float = 0.0, max_dt_s: float = 0.034
                 ) -> List[Tuple[int, int]]:
    """Casa cada frame de A com o frame mais proximo de B (1:1, monotonico).

    `offset_s` e' tal que `t_b + offset_s ~= t_a`. Um par so' entra quando
    `|t_a - (t_b + offset_s)| <= max_dt_s`; frames de A sem par ficam de fora
    (o relatorio de cobertura mostra quantos).
    """
    a = np.asarray(times_a, dtype=np.float64).reshape(-1)
    b = np.asarray(times_b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return []
    b_align = b + float(offset_s)
    pares: List[Tuple[int, int]] = []
    j = 0
    ultimo_j = -1
    for i, t in enumerate(a):
        while j + 1 < b_align.size and abs(b_align[j + 1] - t) <= abs(
                b_align[j] - t):
            j += 1
        if abs(b_align[j] - t) <= float(max_dt_s) and j > ultimo_j:
            pares.append((i, j))
            ultimo_j = j
    return pares


def timestamp_stats(timeline: CameraTimeline) -> Dict[str, Any]:
    """Estatisticas de fps/intervalos de uma timeline (para o relatorio)."""
    tempos = timeline.np_time()
    if tempos.size < 2:
        return {"frames": int(tempos.size), "fps": float("nan"),
                "dt_mediano_ms": float("nan"), "dt_p95_ms": float("nan"),
                "buracos": 0}
    dts = np.diff(tempos)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    if dts.size == 0:
        return {"frames": int(tempos.size), "fps": float("nan"),
                "dt_mediano_ms": float("nan"), "dt_p95_ms": float("nan"),
                "buracos": 0}
    mediano = float(np.median(dts))
    buracos = int(np.sum(dts > max(2.5 * mediano, 0.1)))
    return {
        "frames": int(tempos.size),
        "fps": float(1.0 / mediano) if mediano > 0 else float("nan"),
        "dt_mediano_ms": mediano * 1000.0,
        "dt_p95_ms": float(np.percentile(dts, 95)) * 1000.0,
        "buracos": buracos,
    }


# =============================================================================
# Utilidades de texto
# =============================================================================
def fmt_seconds(seconds: float) -> str:
    """'HH:MM:SS' para logs/terminais."""
    if seconds is None or not np.isfinite(float(seconds)) or seconds < 0:
        return "--:--:--"
    total = int(round(float(seconds)))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def safe_slug(text: str, default: str = "session") -> str:
    """Nome de arquivo seguro (sem caracteres proibidos do Windows)."""
    limpo = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("._-")
    return limpo or default
