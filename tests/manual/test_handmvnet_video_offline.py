#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Suite manual do pipeline OFFLINE de video das 2 auxiliares (sem hardware).

Cobre as tres pecas da tarefa:
  1. `vision/handmvnet_video_io.py` -- formato de sessao, timestamps, flash,
     sincronismo, armazenamento/espaco livre (unidades puras);
  2. `tools/handmvnet_video_record.py` -- gravacao das 2 cameras com cameras
     FAKE (frames sinteticos + flash), incluindo meta.json/CSV, deteccao de
     flash e aborto por falta de espaco;
  3. `tools/handmvnet_video_process.py` -- processamento offline: detector
     FAKE devolvendo os landmarks conhecidos (simula MediaPipe perfeito),
     sincronismo por flash, triangulacao par (e 3 vistas com Kinect fake),
     CSV/NPZ/relatorio e erro de reconstrucao contra o ground truth sintetico.

Nada abre camera real nem roda MediaPipe: os frames vem de uma mao 3D sintetica
projetada com as calibracoes REAIS de `configs/calibracao/` (stereo dos pares
com o Kinect), entao a geometria exercitada e' a mesma da bancada.

Rodar (na raiz do repositorio):

    .\\.venv\\Scripts\\python.exe tests\\manual\\test_handmvnet_video_offline.py

Codigo de saida 0 = todos os checks passaram.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# sys.path do projeto
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mestradopy"
for _p in (_SRC, _SRC / "core", _SRC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                           # pragma: no cover
    pass

import cv2                                  # noqa: E402

import kinect_imu_groundtruth as kinect_gt  # noqa: E402
from vision.handmvnet_adapter import (      # noqa: E402
    project_points_px,
    scale_intrinsic,
    load_stereo_calibration,
)
from vision.handmvnet_synthetic import (    # noqa: E402
    make_synthetic_hand,
    render_hand_image,
)
from vision.handmvnet_video_io import (     # noqa: E402
    CODECS,
    DEFAULT_MIN_FREE_GB,
    FlashDetector,
    InsufficientSpaceError,
    TimestampRow,
    brightness_of,
    ensure_dir_and_space,
    estimate_storage_gb,
    estimate_storage_gb_per_hour,
    free_space_gb,
    open_video_writer,
    pair_by_time,
    probe_video,
    read_meta,
    read_timestamps_csv,
    timestamp_stats,
    write_meta,
    write_timestamps_csv,
)
import handmvnet_video_record as recmod          # noqa: E402
import handmvnet_video_process as procmod        # noqa: E402
from handmvnet_video_process import (            # noqa: E402
    HandDetection,
    plan_session,
    process_session,
    make_handmvnet_views,
)

FALHAS: list[str] = []
AVISOS: list[str] = []
CHECKS = 0
T0 = time.perf_counter()

CALIB_DIR = _REPO / "configs" / "calibracao"
NPZ_AUX0 = CALIB_DIR / "stereo_calibration.npz"
NPZ_AUX1 = CALIB_DIR / "stereo_calibration_aux1.npz"
#: Resolucao pequena para o teste ser rapido (a calibracao e' escalada).
W, H = 320, 180
FPS = 60.0
N_FRAMES = 24
FLASH_FRAMES = (8,)
MISSING_AUX1 = (10, 11, 12)     # mao so' vista na aux0 nesses frames
MISSING_AUX0 = (6,)             # mao so' vista na aux1
MISSING_BOTH = (15, 16)         # sem mao em nenhuma vista
NOISE_PX = 0.2


# ===========================================================================
# Infra da suite
# ===========================================================================
def check(nome: str, condicao: bool, detalhe: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if condicao:
        print(f"  OK   {nome}" + (f" [{detalhe}]" if detalhe else ""))
        return True
    print(f"  FALHA {nome}" + (f" [{detalhe}]" if detalhe else ""))
    FALHAS.append(nome)
    return False


def warn(nome: str, motivo: str) -> None:
    print(f"  AVISO {nome}: {motivo}")
    AVISOS.append(f"{nome}: {motivo}")


def secao(titulo: str) -> None:
    print(f"\n--- {titulo} ---")


def _eq(a, b, tol: float = 1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
                            atol=tol, rtol=0.0, equal_nan=True))


# ===========================================================================
# Calibracao + geracao dos frames/landmarks sinteticos
# ===========================================================================
def carrega_calibracoes() -> dict:
    cal0 = load_stereo_calibration(NPZ_AUX0)
    cal1 = load_stereo_calibration(NPZ_AUX1)
    return {
        "K0": scale_intrinsic(cal0["auxiliary_matrix"], cal0["auxiliary_size"], (W, H)),
        "d0": cal0["auxiliary_dist"],
        "R0": cal0["rotation"], "T0": cal0["translation"],
        "K1": scale_intrinsic(cal1["auxiliary_matrix"], cal1["auxiliary_size"], (W, H)),
        "d1": cal1["auxiliary_dist"],
        "R1": cal1["rotation"], "T1": cal1["translation"],
        "Kk": scale_intrinsic(cal0["kinect_matrix"], cal0["kinect_size"], (W, H)),
        "dk": cal0["kinect_dist"],
        "npz0": str(NPZ_AUX0), "npz1": str(NPZ_AUX1),
    }


def gera_mao(indice: int, n: int, *, estatica: bool = False) -> np.ndarray:
    """(21,3) mao sintetica (frame do Kinect) movendo-se (ou parada)."""
    fase = 0.0 if estatica else indice / max(n - 1, 1)
    centro = (0.30 + 0.05 * math.sin(2.0 * math.pi * fase),
              0.20 - 0.02 * math.cos(2.0 * math.pi * fase),
              0.80 + 0.02 * math.sin(4.0 * math.pi * fase))
    rot = ((10.0 + 15.0 * math.sin(2.0 * math.pi * fase)),
           -15.0 + 10.0 * fase, 8.0)
    return make_synthetic_hand(center=centro, rotation_deg=rot)


def _ruido(px: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(px, float).reshape(-1, 2) + rng.normal(
        0.0, NOISE_PX, size=(len(px), 2))


def espelha_pixels(px: np.ndarray, largura: int = W) -> np.ndarray:
    """Pixels da imagem CRUA -> pixels da imagem ESPELHADA (cv2.flip(frame,1))."""
    pts = np.asarray(px, float).reshape(-1, 2).copy()
    pts[:, 0] = float(largura - 1) - pts[:, 0]
    return pts


def gera_fixture(n: int = N_FRAMES, *, estatica: bool = False) -> dict:
    """Frames "stream" (como a webcam entrega), gt e px fisicos/deteccoes.

    Convencao do projeto: o driver entrega o stream ESPELHADO e o gravador
    aplica `transform_auxiliary_frame` (flip) para restaurar a imagem fisica,
    que e' a convencao da calibracao. Entao:
      - `frames_raw`: imagem do STREAM (fisica, mas espelhada) que a camera fake
        entrega; o gravador flipa -> imagem fisica;
      - `px_raw`: landmarks na imagem FISICA (o que o MediaPipe devolve DEPOIS
        do flip e o que o processador espera do detector);
      - `px_mirror`: landmarks no stream entregue (usados so' para desenhar a
        imagem que a camera fake entrega).
    """
    calib = carrega_calibracoes()
    rng = np.random.default_rng(7)
    gt: list[np.ndarray] = []
    frames_raw = {"aux0": [], "aux1": [], "kinect": []}
    px_raw = {"aux0": [], "aux1": [], "kinect": []}
    px_mirror = {"aux0": [], "aux1": []}
    tempos_mono = {"aux0": [], "aux1": [], "kinect": []}
    t0 = 100.0
    for i in range(n):
        mao = gera_mao(i, n, estatica=estatica)
        gt.append(mao)
        p0 = _ruido(project_points_px(mao, calib["K0"], calib["d0"],
                                      calib["R0"], calib["T0"]), rng)
        p1 = _ruido(project_points_px(mao, calib["K1"], calib["d1"],
                                      calib["R1"], calib["T1"]), rng)
        pk = _ruido(project_points_px(mao, calib["Kk"], calib["dk"]), rng)
        px_raw["aux0"].append(p0)
        px_raw["aux1"].append(p1)
        px_raw["kinect"].append(pk)
        px_mirror["aux0"].append(espelha_pixels(p0))
        px_mirror["aux1"].append(espelha_pixels(p1))
        # A camera entrega o stream espelhado; o gravador flipa de volta.
        img0 = render_hand_image((W, H), px_mirror["aux0"][-1])
        img1 = render_hand_image((W, H), px_mirror["aux1"][-1])
        imgk = render_hand_image((W, H), pk)
        if i in FLASH_FRAMES:
            img0 = np.full((H, W, 3), 255, np.uint8)
            img1 = np.full((H, W, 3), 255, np.uint8)
            imgk = np.full((H, W, 3), 255, np.uint8)
        frames_raw["aux0"].append(img0)
        frames_raw["aux1"].append(img1)
        frames_raw["kinect"].append(imgk)
        tempos_mono["aux0"].append(t0 + i / FPS)
        tempos_mono["aux1"].append(t0 + i / FPS)
        tempos_mono["kinect"].append(t0 + 0.005 + i / FPS)
    return {
        "calib": calib, "gt": np.stack(gt),
        "frames_raw": frames_raw, "px_raw": px_raw, "px_mirror": px_mirror,
        "tempos": tempos_mono,
    }


def _deteccoes_por_frame(px_por_frame: dict[int, np.ndarray],
                         lados_por_frame: dict[int, str],
                         times_ms: np.ndarray) -> dict:
    """Tabela (tempo_ms -> [HandDetection]) para o detector fake."""
    dets: dict[int, list[HandDetection]] = {}
    for indice, px in px_por_frame.items():
        lado = lados_por_frame.get(indice, "right")
        dets[indice] = [HandDetection(side=lado, side_raw=lado.title(),
                                      score=0.99,
                                      landmarks_px=np.asarray(px, float).copy())]
    return {"times_ms": np.asarray(times_ms, float), "dets": dets}


class DetectorFake:
    """Detector injetado: devolve landmarks conhecidos pelo timestamp do frame.

    O processador passa `timestamp_ms` vindo do CSV de timestamps; a tabela e'
    montada com os mesmos valores, entao cada frame recebe exatamente os
    landmarks que foram projetados na geracao do video (MediaPipe "perfeito").
    """

    def __init__(self, tabelas: dict):
        self.tabelas = tabelas
        self.consultas = 0
        self.falhas_tempo = 0

    def detect(self, frame_bgr, view_name, timestamp_ms, aux_index=None):
        tabela = self.tabelas.get(view_name)
        if tabela is None:
            return []
        self.consultas += 1
        tempos = tabela["times_ms"]
        if tempos.size == 0:
            return []
        idx = int(np.argmin(np.abs(tempos - float(timestamp_ms))))
        if abs(float(tempos[idx]) - float(timestamp_ms)) > 2.0:
            self.falhas_tempo += 1
            return []
        return [_copia_deteccao(d) for d in tabela["dets"].get(idx, [])]

    def close(self):
        pass


def _copia_deteccao(det: HandDetection) -> HandDetection:
    return HandDetection(side=det.side, side_raw=det.side_raw,
                         score=det.score, landmarks_px=det.landmarks_px.copy())


# ===========================================================================
# 1) Unidades de video_io
# ===========================================================================
def teste_video_io() -> None:
    secao("1) video_io: armazenamento, timestamps, flash, sincronismo")
    lo, hi = estimate_storage_gb(3600.0, 1280, 720, 30.0, "h264")
    check("estimativa H.264 720p30 1h entre 1 e 6 GB", 1.0 <= lo <= hi <= 6.0,
          f"{lo:.2f}-{hi:.2f} GB")
    mj_lo, mj_hi = estimate_storage_gb_per_hour(1280, 720, 30.0, "mjpg")
    check("MJPEG e' bem maior que H.264 (>4x)", mj_lo > 4.0 * lo,
          f"mjpg={mj_lo:.1f} GB/h vs h264={lo:.2f} GB/h")
    check("estimativa para duracao zero nao explode", np.isfinite(
        estimate_storage_gb(0.0, 640, 480, 30, "h264")[0]))

    with tempfile.TemporaryDirectory(prefix="videoio_") as tmp:
        tmp_path = Path(tmp)
        dest, livre = ensure_dir_and_space(tmp_path, min_free_gb=0.001)
        check("ensure_dir_and_space cria pasta e mede espaco",
              dest.is_dir() and livre > 0.0, f"livre={livre:.2f} GB")
        check("free_space_gb > 0 no temp", free_space_gb(tmp_path) > 0.0)
        try:
            ensure_dir_and_space(tmp_path, min_free_gb=1e9)
            estourou = False
        except InsufficientSpaceError as exc:
            estourou = "insuficiente" in str(exc).lower()
        check("espaco insuficiente levanta erro com mensagem clara", estourou)

        # CSV de timestamps roundtrip.
        rows = [TimestampRow(frame=i, wall_s=1000.0 + i / 60.0,
                             monotonic_s=100.0 + i / 60.0,
                             perf_s=10.0 + i, pos_msec=i * 16.7,
                             brightness=40.0 + i, flash=(i == 3),
                             manual_sync=(i == 5),
                             wall_iso=f"t{i}")
                for i in range(8)]
        csv_path = write_timestamps_csv(tmp_path / "timestamps_aux00.csv", rows)
        voltou = read_timestamps_csv(csv_path)
        check("CSV de timestamps roundtrip (8 linhas)",
              len(voltou) == 8 and voltou[3].flash and voltou[5].manual_sync)
        check("CSV preserva tempos com 1e-3 ms", _eq(
            [r.monotonic_s for r in voltou],
            [r.monotonic_s for r in rows], tol=1e-4))

        # Flash: cinza -> branco -> cinza detecta UMA vez.
        detector = FlashDetector(delta=35.0, min_brightness=170.0,
                                 cooldown_frames=3)
        cinza = np.full((H, W, 3), 60, np.uint8)
        branco = np.full((H, W, 3), 250, np.uint8)
        eventos = [detector.update(f)[0] for f in
                   [cinza, cinza, branco, branco, branco, cinza, branco]]
        check("FlashDetector: uma borda por flash", sum(eventos) == 2,
              str(eventos))
        check("brilho medio do cinza ~60 e do branco ~250",
              abs(brightness_of(cinza) - 60.0) < 2.0
              and abs(brightness_of(branco) - 250.0) < 2.0)

        # pareamento temporal com offset conhecido (b esta' 10 ms ADIANTADO).
        ta = np.arange(30, dtype=float) / 30.0
        tb = ta + 0.010
        pares = pair_by_time(ta, tb, offset_s=-0.010, max_dt_s=0.02)
        check("pair_by_time: 30 pares 1:1", len(pares) == 30
              and all(i == j for i, j in pares), str(len(pares)))
        pares_ruins = pair_by_time(np.array([0.0, 1.0]), np.array([5.0]),
                                   offset_s=0.0, max_dt_s=0.02)
        check("pair_by_time: rejeita fora da tolerancia", pares_ruins == [])

        # estimativa de offset por eventos flash.
        from vision.handmvnet_video_io import CameraTimeline
        timeline_a = CameraTimeline(name="aux0", index=0,
                                    video_path=tmp_path / "a.avi",
                                    rows=rows)
        timeline_b = CameraTimeline(name="aux1", index=1,
                                    video_path=tmp_path / "b.avi",
                                    rows=[TimestampRow(frame=r.frame,
                                                       wall_s=r.wall_s,
                                                       monotonic_s=r.monotonic_s + 0.012)
                                          for r in rows])
        eventos_sync = [
            {"camera": 0, "frame": 3, "t_monotonic": rows[3].monotonic_s,
             "kind": "flash"},
            {"camera": 1, "frame": 3, "t_monotonic": rows[3].monotonic_s + 0.012,
             "kind": "flash"},
        ]
        est = procmod.estimate_sync_offset(timeline_a, timeline_b,
                                           events=eventos_sync)
        check("offset por flash ~ -12 ms", abs(est["offset_s"] + 0.012) < 1e-6
              and est["method"] == "flash", str(est))
        est2 = procmod.estimate_sync_offset(timeline_a, timeline_b, events=[])
        check("offset por inicio (sem eventos) ~ -12 ms",
              abs(est2["offset_s"] + 0.012) < 1e-6
              and est2["method"] == "lockstep_start", str(est2))
        stats = timestamp_stats(timeline_a)
        check("timestamp_stats: frames/fps coerentes",
              stats["frames"] == 8 and abs(stats["fps"] - 60.0) < 2.0,
              str(stats))


# ===========================================================================
# 2) Gravador com cameras FAKE
# ===========================================================================
class CameraFake:
    """Camera fake: entrega frames pre-gravados no ritmo do fps."""

    def __init__(self, frames, fps: float = FPS):
        self.frames = list(frames)
        self.fps = float(fps)
        self.i = 0
        self.released = False
        self._dt = 1.0 / max(self.fps, 1.0)

    def read(self):
        if self.released or self.i >= len(self.frames):
            time.sleep(0.001)
            return False, None
        frame = self.frames[self.i]
        self.i += 1
        time.sleep(self._dt)
        return True, frame.copy()

    def get(self, prop):
        try:
            if prop == cv2.CAP_PROP_POS_MSEC:
                return float(self.i - 1) * (1000.0 / max(self.fps, 1.0))
        except Exception:                           # pragma: no cover
            pass
        return 0.0

    def release(self):
        self.released = True


def _grava_sessao(tmp: Path, fixture: dict, *,
                  nome: str = "sintetica") -> recmod.RecordResult:
    """Grava a sessao sintetica no disco usando o gravador real (cameras fake)."""
    fontes = [
        recmod.CameraSource(index=0, name="fake-sigma",
                            capture=CameraFake(fixture["frames_raw"]["aux0"])),
        recmod.CameraSource(index=1, name="fake-trust",
                            capture=CameraFake(fixture["frames_raw"]["aux1"])),
    ]
    config = recmod.RecordConfig(
        dest=tmp / "videos", session_name=nome, width=W, height=H, fps=FPS,
        codecs=("mjpg",), codec_requested="mjpg", mirror=True,
        max_frames=N_FRAMES, show_window=False, verbose=False,
        min_free_gb=0.001, flash_delta=35.0, flash_min_brightness=170.0)
    return recmod.record_session(fontes, config)


def teste_gravador() -> None:
    secao("2) Gravador offline (cameras fake + flash + meta/CSV)")
    fixture = gera_fixture(estatica=True)
    with tempfile.TemporaryDirectory(prefix="rec_") as tmp:
        tmp_path = Path(tmp)
        resultado = _grava_sessao(tmp_path, fixture, nome="gravacao_teste")
        check("gravacao termina com status ok", resultado.ok,
              f"{resultado.status} / {resultado.stopped_reason}")
        meta = resultado.meta
        check("meta: schema/versao e script",
              meta["schema_version"] >= 1
              and meta["script"] == "handmvnet_video_record.py")
        check("meta: 2 cameras com video+timestamps",
              len(meta["cameras"]) == 2
              and all(c["video"] and c["timestamps"] for c in meta["cameras"]))
        check("meta: resolucao/fps/mirror registrados",
              meta["resolution_requested"] == [W, H]
              and abs(meta["fps_requested"] - FPS) < 1e-6
              and meta["mirror"] is True)
        session = resultado.session_dir
        for cam in meta["cameras"]:
            idx = cam["index"]
            video = session / cam["video"]
            stamps = session / cam["timestamps"]
            info = probe_video(video)
            check(f"aux{idx}: video legivel ({video.suffix})",
                  info["opened"] and (info["width"], info["height"]) == (W, H),
                  f"{info.get('frame_count')} frames")
            rows = read_timestamps_csv(stamps)
            check(f"aux{idx}: CSV de timestamps com {N_FRAMES} linhas",
                  len(rows) == N_FRAMES, str(len(rows)))
            check(f"aux{idx}: meta frames == CSV linhas",
                  int(cam["frames"]) == len(rows)
                  and int(cam["frames"]) >= N_FRAMES - 1, str(cam["frames"]))
        sync = meta["sync_summary"]
        flash_por_cam = meta.get("sync_events") or []
        cams_flash = sorted({int(e["camera"]) for e in flash_por_cam
                             if e.get("kind") == "flash"})
        check("flash detectado nas DUAS cameras", cams_flash == [0, 1],
              str(cams_flash))
        check("meta: eventos de flash no frame esperado (~8)",
              all(abs(int(e["frame"]) - FLASH_FRAMES[0]) <= 1
                  for e in flash_por_cam if e.get("kind") == "flash"),
              str([e["frame"] for e in flash_por_cam]))
        check("meta: espaco livre inicio/fim medido",
              float(meta["free_gb_start"]) > 0
              and float(meta["free_gb_end"]) > 0)
        check("meta: estimativas de armazenamento por codec",
              set(meta["storage_estimate_gb_per_hour"]) == set(CODECS))

        # Espaco insuficiente aborta ANTES de criar/abrir cameras.
        fontes = [recmod.CameraSource(index=0, name="x", capture=CameraFake([])),
                  recmod.CameraSource(index=1, name="y", capture=CameraFake([]))]
        config_ruim = recmod.RecordConfig(
            dest=tmp_path / "videos", session_name="sem_espaco", width=W,
            height=H, fps=FPS, codecs=("mjpg",), max_frames=5,
            show_window=False, verbose=False, min_free_gb=1e9)
        try:
            recmod.record_session(fontes, config_ruim)
            estourou = False
        except InsufficientSpaceError as exc:
            estourou = "insuficiente" in str(exc).lower()
        check("gravador aborta com espaco insuficiente", estourou)

        # Meta falsa para o processador (dry-run precisa do meta).
        meta_path = session / "meta.json"
        check("meta.json gravado no disco", meta_path.is_file())


# ===========================================================================
# 3) Processador offline (sessao sintetica montada diretamente)
# ===========================================================================
def _meta_sessao(session: Path, *, indices=(0, 1), fps: float = FPS,
                 mirror: bool = True, n_frames: int = N_FRAMES,
                 tempos: dict | None = None) -> dict:
    """Meta no formato do gravador para a sessao sintetica montada direto."""
    tempos = tempos or {}
    cameras = []
    for idx in indices:
        cameras.append({
            "index": int(idx), "name": f"sintetica-aux{idx}",
            "video": f"aux{int(idx):02d}.avi",
            "timestamps": f"timestamps_aux{int(idx):02d}.csv",
            "codec": "MJPEG", "fourcc": "MJPG",
            "resolution": [W, H], "size_requested": [W, H],
            "fps_requested": fps, "fps_measured": fps,
            "frames": int(n_frames), "mirror": mirror,
            "calibration": {"file": str(NPZ_AUX0 if idx == 0 else NPZ_AUX1)},
        })
    eventos = []
    for indice in FLASH_FRAMES:
        for idx in indices:
            base = tempos.get(f"aux{int(idx)}")
            t_flash = (float(base[indice]) if base is not None
                       else 100.0 + indice / fps)
            eventos.append({
                "camera": int(idx), "frame": int(indice),
                "t_monotonic": t_flash, "kind": "flash",
                "brightness": 250.0})
    return {
        "schema_version": 1, "script": "handmvnet_video_record.py",
        "created_at": "2026-11-17T10:00:00", "session_name": session.name,
        "session_dir": str(session), "host": "test", "mirror": bool(mirror),
        "resolution_requested": [W, H], "fps_requested": fps,
        "codec_requested": "mjpg", "elapsed_s": N_FRAMES / fps,
        "status": "max_frames", "stopped_reason": "teste",
        "free_gb_start": 10.0, "free_gb_end": 10.0,
        "cameras": cameras, "sync_events": eventos,
        "sync_summary": {"flash": len(eventos),
                         "cameras_with_flash": list(indices)},
    }


def _escreve_video(caminho: Path, frames: list[np.ndarray]) -> None:
    writer, real, codec = open_video_writer(caminho.with_suffix(""), FPS,
                                            (W, H), codecs=("mjpg",))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    assert real.suffix == ".avi"


def monta_sessao_sintetica(tmp: Path, fixture: dict, *,
                           nome: str = "sessao_sintetica") -> dict:
    """Cria a sessao (videos espelhados + CSV + meta) SEM o gravador.

    Timestamps determinísticos (i/FPS) para o pareamento ser exato (i==j);
    videos ja' espelhados (meta.mirror=True) como o gravador real faria.
    """
    session = tmp / nome
    session.mkdir(parents=True, exist_ok=True)
    tempos = fixture["tempos"]
    # frames gravados = espelho do frame cru (recorder com mirror=True)
    frames_esp = {vista: [cv2.flip(f, 1) for f in fixture["frames_raw"][vista]]
                  for vista in ("aux0", "aux1")}
    _escreve_video(session / "aux00.avi", frames_esp["aux0"])
    _escreve_video(session / "aux01.avi", frames_esp["aux1"])
    for vista, idx in (("aux0", 0), ("aux1", 1)):
        rows = [TimestampRow(frame=i, wall_s=1000.0 + i / FPS,
                             monotonic_s=float(tempos[vista][i]),
                             perf_s=10.0 + i, pos_msec=1000.0 * i / FPS,
                             brightness=brightness_of(frames_esp[vista][i]),
                             flash=(i in FLASH_FRAMES))
                for i in range(N_FRAMES)]
        write_timestamps_csv(session / f"timestamps_aux{idx:02d}.csv", rows)
    meta = _meta_sessao(session, tempos={"aux0": tempos["aux0"],
                                         "aux1": tempos["aux1"]})
    write_meta(session, meta)
    return {"session": session, "meta": meta}


def _tabelas_detector(fixture: dict) -> dict:
    """Tabelas de deteccao (timestamp -> landmarks) para o DetectorFake.

    Os landmarks sao os da imagem FISICA (depois do flip do gravador), que e'
    o que o MediaPipe devolve no processamento (frames ja' na convencao da
    calibracao).
    """
    tempos = fixture["tempos"]
    px = fixture["px_raw"]
    tabelas = {}
    for vista, idx in (("aux0", 0), ("aux1", 1)):
        px_por_frame: dict[int, np.ndarray] = {}
        lados: dict[int, str] = {}
        for i in range(N_FRAMES):
            if i in FLASH_FRAMES or i in MISSING_BOTH:
                continue
            if vista == "aux1" and i in MISSING_AUX1:
                continue
            if vista == "aux0" and i in MISSING_AUX0:
                continue
            px_por_frame[i] = px[vista][i]
            lados[i] = "right"
        tabelas[f"aux{idx}"] = _deteccoes_por_frame(
            px_por_frame, lados, np.asarray(tempos[vista], float) * 1000.0)
    return tabelas


def teste_processador_par() -> None:
    secao("3) Processador offline: modo par (2 auxiliares)")
    fixture = gera_fixture()
    with tempfile.TemporaryDirectory(prefix="proc_") as tmp:
        tmp_path = Path(tmp)
        sessao = monta_sessao_sintetica(tmp_path, fixture,
                                        nome="sessao_par")["session"]
        # --- dry-run/plan ----------------------------------------------------
        plano = plan_session(sessao, mode="pair", verbose=False)
        check("plano: 24 pares aux0-aux1 (timestamps exatos)",
              len(plano.pairs) == N_FRAMES, str(len(plano.pairs)))
        check("plano: sincronismo por flash", plano.sync["method"] == "flash",
              str(plano.sync))
        check("plano: offset ~0", abs(plano.sync["offset_s"]) < 1e-6,
              f"{plano.sync['offset_s'] * 1000:.3f} ms")
        check("plano: espelho nao duplicado (meta.mirror=True)",
              plano.mirror_effective is False)
        check("plano: calibracao aux usavel (rms<5px)",
              all(0.0 < float(plano.views[i].rms) <= 5.0 for i in (0, 1)),
              str({i: plano.views[i].rms for i in (0, 1)}))

        # --- processamento ---------------------------------------------------
        detector = DetectorFake(_tabelas_detector(fixture))
        report = process_session(
            sessao, mode="pair", detector=detector, verbose=False,
            max_reproj_px=10.0)
        counts = report["counts"]
        check("processo: modo pair e metodos 'pair_aux'",
              report["mode"] == "pair"
              and counts["methods"] == {"pair_aux": counts["triangulation_ok"]}
              and counts["triangulation_ok"] > 0,
              str(counts["methods"]))
        esperado_sem_par = len(set(FLASH_FRAMES) | set(MISSING_AUX1)
                               | set(MISSING_AUX0) | set(MISSING_BOTH))
        escritos = counts["pairs_written"]
        check(f"cobertura: ~{N_FRAMES - esperado_sem_par} frames com mao nas 2 vistas",
              abs(escritos - (N_FRAMES - esperado_sem_par)) <= 2,
              f"escritos={escritos}, esperado~{N_FRAMES - esperado_sem_par}")
        check("cobertura: pares sem mao contabilizados",
              counts["pairs_no_hands"] >= len(MISSING_BOTH) - 1,
              str(counts["pairs_no_hands"]))
        # --- arquivos --------------------------------------------------------
        destino = sessao / "processed"
        for nome in ("landmarks_3d.csv", "landmarks_3d.npz",
                     "coverage_report.json", "coverage_report.txt",
                     "meta_processado.json"):
            check(f"saida {nome} existe", (destino / nome).is_file())
        npz = np.load(destino / "landmarks_3d.npz")
        n = int(npz["joints_xyz"].shape[0])
        check("npz: joints_xyz (N,21,3) finito (validos)",
              npz["joints_xyz"].shape[1:] == (21, 3)
              and np.isfinite(npz["joints_xyz"][npz["valid"]]).all(),
              str(npz["joints_xyz"].shape))
        check("npz: reproj_aux0/aux1 (N,21)",
              npz["reproj_aux0"].shape == (n, 21)
              and npz["reproj_aux1"].shape == (n, 21))
        check("npz: pixels/undist (N,21,2)",
              npz["pixels_aux0"].shape == (n, 21, 2)
              and npz["undist_aux0"].shape == (n, 21, 2))
        check("npz: meta_json JSON-serializavel",
              len(json.loads(str(npz["meta_json"]))) > 0)
        # CSV longo: 21 linhas por registro + header.
        with open(destino / "landmarks_3d.csv", "r", encoding="utf-8") as fh:
            linhas = fh.read().splitlines()
        check("CSV: 1 header + 21*x linhas",
              len(linhas) == 1 + 21 * n, f"{len(linhas)} linhas, n={n}")
        check("CSV: colunas esperadas no header",
              "time_sync_s" in linhas[0] and "reproj_aux0_px" in linhas[0]
              and "valid" in linhas[0])
        # --- qualidade contra o ground truth ---------------------------------
        erros = []
        for k in range(n):
            valid = np.asarray(npz["valid"][k], bool)
            if valid.sum() < 5:
                continue
            frame0 = int(npz["frame0"][k])
            gt = fixture["gt"][frame0]
            erro = np.linalg.norm(npz["joints_xyz"][k][valid] - gt[valid],
                                  axis=1)
            erros.extend(erro.tolist())
        erros = np.asarray(erros) * 1000.0
        check("reconstrucao 3D: erro mediano < 5 mm",
              erros.size > 100 and float(np.median(erros)) < 5.0,
              f"mediana={float(np.median(erros)):.2f}mm n={erros.size}")
        check("reconstrucao: p90 < 10 mm",
              float(np.percentile(erros, 90)) < 10.0,
              f"p90={float(np.percentile(erros, 90)):.2f}mm")
        quality = report["quality"]
        check("relatorio: reprojecao mediana < 2 px",
              quality["reproj_aux0_px"]["median"] is not None
              and quality["reproj_aux0_px"]["median"] < 2.0,
              str(quality["reproj_aux0_px"]))
        check("relatorio: cobertura por junção (21 itens)",
              len(quality["joint_coverage"]) == 21
              and quality["joint_coverage"][0]["pct"] > 50.0,
              str(quality["joint_coverage"][0]))
        check("relatorio: contagem right/left coerente",
              report["counts"]["sides"]["right"] == n,
              str(report["counts"]["sides"]))
        # --- handmvnet auto no modo par: pula com motivo ---------------------
        npz.close()
        report_hn = process_session(
            sessao, mode="pair", detector=detector, verbose=False,
            out_dir=tmp_path / "proc_hn", handmvnet="auto", max_seconds=30.0)
        hn = report_hn["handmvnet"]
        check("HandMvNet auto no modo par: pulado com motivo",
              not hn["used"] and "three" in str(hn.get("reason", "")).lower(),
              str(hn.get("reason")))

        # --- make_handmvnet_views: contrato do adaptador ---------------------
        alignment = kinect_gt.CameraAlignment(aux_indices=(0, 1))
        frame_aux0 = cv2.flip(fixture["frames_raw"]["aux0"][0], 1)
        det = HandDetection(side="right", side_raw="Right", score=0.99,
                            landmarks_px=fixture["px_raw"]["aux0"][0])
        views = make_handmvnet_views(alignment, [
            ("aux0", frame_aux0, det.landmarks_px, 0),
            ("aux1", cv2.flip(fixture["frames_raw"]["aux1"][0], 1),
             fixture["px_raw"]["aux1"][0], 1),
        ])
        aux_view0 = alignment.aux_view(0)
        K_esperado = scale_intrinsic(aux_view0.effective_matrix,
                                     aux_view0.calibrated_size, (W, H))
        check("make_handmvnet_views: 2 HandMvNetView com K escalada ao frame",
              len(views) == 2
              and bool(np.allclose(views[0].intrinsic, K_esperado, atol=1e-9)),
              str([v.describe() for v in views])[:120])
        check("make_handmvnet_views: landmarks preservados",
              bool(np.allclose(views[0].landmarks_px, det.landmarks_px)))

        # --- filtros (every/start) nao quebram ------------------------------
        report_filtro = process_session(
            sessao, mode="pair", detector=detector, verbose=False,
            out_dir=tmp_path / "proc_every", every=2, max_seconds=30.0)
        check("every=2 processa menos frames que every=1",
              report_filtro["counts"]["pairs_written"]
              < report["counts"]["pairs_written"],
              f"{report_filtro['counts']['pairs_written']} < "
              f"{report['counts']['pairs_written']}")

        # --- erros claros ----------------------------------------------------
        try:
            process_session(tmp_path / "nao_existe", detector=detector,
                            verbose=False)
            falhou = False
        except (FileNotFoundError, ValueError):
            falhou = True
        check("sessao inexistente levanta erro claro", falhou)
        meta_sem_video = dict(meta=read_meta(sessao))
        meta_sem_video["meta"]["cameras"][0]["video"] = "nao_existe.avi"
        sessao_ruim = tmp_path / "sessao_ruim"
        sessao_ruim.mkdir()
        write_meta(sessao_ruim, meta_sem_video["meta"])
        try:
            plan_session(sessao_ruim, mode="pair", verbose=False)
            falhou_video = False
        except FileNotFoundError:
            falhou_video = True
        check("video ausente no meta levanta FileNotFoundError", falhou_video)


# ===========================================================================
# 4) Processador no modo 3 vistas (Kinect fake)
# ===========================================================================
def _escreve_kinect_fake(session: Path, fixture: dict) -> dict:
    """Video + CSV de timestamps do Kinect fake (relogio do mesmo processo)."""
    video = session / "kinect.avi"
    _escreve_video(video, fixture["frames_raw"]["kinect"])
    tempos = fixture["tempos"]["kinect"]
    rows = []
    for i in range(N_FRAMES):
        rows.append(TimestampRow(frame=i, wall_s=2000.0 + i / FPS,
                                 monotonic_s=float(tempos[i]),
                                 brightness=brightness_of(
                                     fixture["frames_raw"]["kinect"][i])))
    csv_path = write_timestamps_csv(session / "kinect_timestamps.csv", rows)
    return {"video": video, "timestamps": csv_path,
            "times_ms": np.asarray(tempos, float) * 1000.0}


def teste_processador_tres_vistas() -> None:
    secao("4) Processador offline: modo 3 vistas (Kinect + aux0 + aux1)")
    fixture = gera_fixture()
    with tempfile.TemporaryDirectory(prefix="proc3_") as tmp:
        tmp_path = Path(tmp)
        sessao = monta_sessao_sintetica(tmp_path, fixture,
                                        nome="sessao_tres")["session"]
        kin = _escreve_kinect_fake(sessao, fixture)
        tabelas = _tabelas_detector(fixture)
        # deteccoes do Kinect fake (sem espelho: RGB do Kinect e' cru)
        px_por_frame, lados = {}, {}
        for i in range(N_FRAMES):
            if i in FLASH_FRAMES or i in MISSING_BOTH:
                continue
            px_por_frame[i] = fixture["px_raw"]["kinect"][i]
            lados[i] = "right"
        tabelas["kinect"] = _deteccoes_por_frame(
            px_por_frame, lados, kin["times_ms"])
        detector = DetectorFake(tabelas)
        plano = plan_session(
            sessao, mode="three", kinect_video=str(kin["video"]),
            kinect_timestamps=str(kin["timestamps"]), verbose=False)
        check("plano 3 vistas: mapa Kinect->aux0 cobre ~24 frames",
              len(plano.kinect_map) >= N_FRAMES - 2,
              str(len(plano.kinect_map)))
        check("plano 3 vistas: modo 'three'", plano.mode == "three")
        report = process_session(
            sessao, mode="three", detector=detector, verbose=False,
            kinect_video=str(kin["video"]),
            kinect_timestamps=str(kin["timestamps"]),
            max_reproj_px=10.0, out_dir=tmp_path / "processed3")
        npz = np.load(tmp_path / "processed3" / "landmarks_3d.npz")
        n = int(npz["joints_xyz"].shape[0])
        check("3 vistas: registros escritos", n >= 12, f"n={n}")
        check("3 vistas: reproj_kinect presente (N,21)",
              "reproj_kinect" in npz.files
              and npz["reproj_kinect"].shape == (n, 21))
        metodos = report["counts"]["methods"]
        check("3 vistas: metodo canonico do projeto usado",
              any(("three_view" in m) or ("pair_" in m) for m in metodos),
              str(metodos))
        erros = []
        for k in range(n):
            valid = np.asarray(npz["valid"][k], bool)
            frame0 = int(npz["frame0"][k])
            if valid.sum() < 5:
                continue
            gt = fixture["gt"][frame0]
            erros.extend((np.linalg.norm(npz["joints_xyz"][k][valid] - gt[valid],
                                         axis=1) * 1000.0).tolist())
        erros = np.asarray(erros)
        check("3 vistas: erro mediano < 5 mm", erros.size > 100
              and float(np.median(erros)) < 5.0,
              f"mediana={float(np.median(erros)) if erros.size else float('nan'):.2f}mm")
        check("3 vistas: relatorio com sincronismo do Kinect",
              report["sync"]["kinect"].get("pairs_kinect_aux0", 0) >= N_FRAMES - 2,
              str(report["sync"]["kinect"]))
        npz.close()
        # modo three sem video do Kinect -> erro claro
        try:
            plan_session(sessao, mode="three", verbose=False)
            falhou = False
        except ValueError as exc:
            falhou = "kinect" in str(exc).lower()
        check("modo three sem Kinect levanta erro claro", falhou)


# ===========================================================================
# 5) Cadeia completa: gravador -> processador
# ===========================================================================
def teste_cadeia_completa() -> None:
    secao("5) Cadeia completa: gravacao (fake) -> processamento offline")
    fixture = gera_fixture(estatica=True)
    with tempfile.TemporaryDirectory(prefix="chain_") as tmp:
        tmp_path = Path(tmp)
        resultado = _grava_sessao(tmp_path, fixture, nome="cadeia")
        sessao = resultado.session_dir
        meta = read_meta(sessao)
        # timeline dos CSVs gravados + deteccoes na convencao dos CSVs
        tabelas = {}
        tabelas["kinect"] = None
        for cam in meta["cameras"]:
            idx = int(cam["index"])
            rows = read_timestamps_csv(sessao / cam["timestamps"])
            vista = f"aux{idx}"
            px_por_frame, lados = {}, {}
            for i in range(N_FRAMES):
                if i in FLASH_FRAMES or i in MISSING_BOTH:
                    continue
                px_por_frame[i] = fixture["px_raw"][vista][i]
                lados[i] = "right"
            tempos_ms = np.asarray([r.monotonic_s for r in rows], float) * 1000.0
            tabelas[vista] = _deteccoes_por_frame(px_por_frame, lados, tempos_ms)
        detector = DetectorFake(tabelas)
        report = process_session(
            sessao, mode="pair", detector=detector, verbose=False,
            max_reproj_px=10.0, out_dir=tmp_path / "proc_chain")
        check("cadeia: processou a sessao gravada",
              report["counts"]["pairs_written"] > 0,
              f"pares={report['counts']['pairs_total']} "
              f"escritos={report['counts']['pairs_written']}")
        check("cadeia: sincronismo por flash da gravacao",
              report["sync"]["method"] == "flash", str(report["sync"]))
        check("cadeia: mirror do meta respeitado (sem espelhar 2x)",
              report["mirror_effective"] is False)
        npz = np.load(tmp_path / "proc_chain" / "landmarks_3d.npz")
        n = int(npz["joints_xyz"].shape[0])
        erros = []
        gt = fixture["gt"][0]          # mao parada no modo estatico
        for k in range(n):
            valid = np.asarray(npz["valid"][k], bool)
            if valid.sum() < 5:
                continue
            erros.extend((np.linalg.norm(npz["joints_xyz"][k][valid] - gt[valid],
                                         axis=1) * 1000.0).tolist())
        check("cadeia: mao parada reconstruida (mediana < 8 mm)",
              len(erros) > 30 and float(np.median(erros)) < 8.0,
              f"mediana={float(np.median(erros)) if erros else float('nan'):.2f}mm")
        npz.close()


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    print("=" * 78)
    print("PIPELINE OFFLINE DE VIDEO (2 auxiliares) -- teste sintetico")
    print("=" * 78)
    print(f"repo: {_REPO}")
    try:
        teste_video_io()
        teste_gravador()
        teste_processador_par()
        teste_processador_tres_vistas()
        teste_cadeia_completa()
    except Exception:                                   # pragma: no cover
        traceback.print_exc()
        FALHAS.append("EXCECAO NAO TRATADA (ver traceback)")
    duracao = time.perf_counter() - T0
    print("\n" + "=" * 78)
    if AVISOS:
        print(f"avisos: {len(AVISOS)}")
        for aviso in AVISOS:
            print(f"  - {aviso}")
    if FALHAS:
        print(f"HANDMVNET_VIDEO_OFFLINE_FALHOU: {len(FALHAS)}/{CHECKS} checks "
              f"({duracao:.1f}s): " + " | ".join(FALHAS))
        print("=" * 78)
        return 1
    print(f"HANDMVNET_VIDEO_OFFLINE_OK: {CHECKS}/{CHECKS} checks ({duracao:.1f}s)")
    print("  gravacao fake + processamento offline (MediaPipe fake) + "
          "triangulacao par/3 vistas validados contra ground truth sintetico.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
