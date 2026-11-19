#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preview AO VIVO das duas webcams auxiliares (hÃ­brido ATHENA-style).

IMPORTANTE: o ATHENA oficial Ã© um pipeline OFFLINE (vÃ­deos frame-locked,
fase 1 + fase 2 em lote). Este script NÃƒO Ã© o ATHENA completo: ele faz o
preview em tempo real com o mesmo detector (MediaPipe Hands) + a nossa
calibraÃ§Ã£o/triangulaÃ§Ã£o do par auxiliar. Serve para ver o tracking das duas
cÃ¢meras ao vivo e diagnosticar lateralidade/cobertura antes de gravar.

Layout: duas cÃ¢meras em cima (direita=verde, esquerda=laranja) e dois grÃ¡ficos
3D embaixo (um por mÃ£o). O canvas padrÃ£o Ã© 1280x720 para caber na tela.

Uso:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\athena_live_preview.py
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\athena_live_preview.py --exposicao "0:-6,1:-6"

Teclas: ESC/Q sai.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1]
for _p in (_SRC, _SRC / "core", _SRC / "tools", _SRC / "vision"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kinect_imu_groundtruth as gt  # noqa: E402
from camera_uvc import (  # noqa: E402
    define_modo, parse_controle_por_camera, trava_exposicao)
from handmvnet_video_process import (  # noqa: E402
    HandDetection, MediaPipeHandDetector, _casa_maos, _casa_maos_geometrica,
)

# BGR. Direita=verde, esquerda=laranja/azul, desconhecida=cinza.
CORES_MAO = {
    "right": (0, 230, 0),
    "left": (0, 165, 255),
    "unknown": (200, 200, 200),
}
CORES_3D = {
    "right": (0, 230, 0),
    "left": (0, 165, 255),
    "unknown": (200, 200, 200),
}


def _controle(texto: str | None, idx: int):
    if not texto:
        return None
    c = parse_controle_por_camera(texto)
    if c.get("todos") is not None:
        return float(c["todos"])
    return c.get("indices", {}).get(int(idx))


def _abre(idx: int, exposicao, ganho, largura=1280, altura=720):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"nao abri aux{idx}")
    define_modo(cap, largura, altura, mjpeg=True)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if exposicao is not None or ganho is not None:
        try:
            trava_exposicao(cap, exposicao=exposicao, ganho=ganho,
                            rotulo=f"aux{idx}")
        except Exception as exc:  # noqa: BLE001
            print(f"[aviso] exposicao aux{idx}: {exc}")
    return cap


def _draw_mao(frame, det: HandDetection, cor) -> None:
    pts = np.asarray(det.landmarks_px, np.float64)
    for a, b in gt.HAND_CONNECTIONS:
        pa = tuple(np.round(pts[a]).astype(int))
        pb = tuple(np.round(pts[b]).astype(int))
        cv2.line(frame, pa, pb, cor, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(np.round(p).astype(int)), 3, cor, -1,
                   cv2.LINE_AA)
    rotulo = {"right": "R", "left": "L"}.get(det.side, "?")
    p0 = tuple(np.round(pts[0]).astype(int))
    cv2.putText(frame, f"{rotulo} {det.score:.2f}", (p0[0] + 6, p0[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2, cv2.LINE_AA)


def _projeta(pontos_3d: np.ndarray, centro: np.ndarray, escala: float,
             largura: int, altura: int) -> np.ndarray:
    """ProjeÃ§Ã£o ortogrÃ¡fica 3D->2D com yaw/pitch fixos (eixo Y para cima)."""
    p = np.asarray(pontos_3d, np.float64).reshape(-1, 3)
    x = p[:, 0] - centro[0]
    y = -(p[:, 1] - centro[1])
    z = p[:, 2] - centro[2]
    yaw = np.deg2rad(35.0)
    pitch = np.deg2rad(18.0)
    xr = x * np.cos(yaw) - z * np.sin(yaw)
    zr = x * np.sin(yaw) + z * np.cos(yaw)
    yr = y * np.cos(pitch) - zr * np.sin(pitch)
    u = largura * 0.5 + xr * escala
    v = altura * 0.56 - yr * escala
    return np.stack([u, v], axis=1)


def _draw_3d_panel(tamanho, pontos: np.ndarray, lado: str,
                   trail: deque, nota: str = "") -> np.ndarray:
    w, h = tamanho
    painel = np.full((h, w, 3), 24, np.uint8)
    cor = CORES_3D.get(lado, CORES_3D["unknown"])
    titulo = {"right": "MAO DIREITA (3D)", "left": "MAO ESQUERDA (3D)"}.get(
        lado, "MAO 3D")
    cv2.putText(painel, titulo, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                cor, 2, cv2.LINE_AA)
    if nota:
        cv2.putText(painel, nota, (12, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1,
                    cv2.LINE_AA)
    p = np.asarray(pontos, np.float64).reshape(-1, 3)
    ok = np.isfinite(p).all(axis=1)
    if ok.sum() < 5:
        cv2.putText(painel, "sem 3D valido", (12, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1,
                    cv2.LINE_AA)
        return painel
    validos = p[ok]
    centro = np.nanmedian(validos, axis=0)
    dist = np.linalg.norm(validos - centro, axis=1)
    raio = max(0.06, float(np.nanpercentile(dist, 95)) * 1.35)
    escala = min(w, h) * 0.34 / raio
    centro_px = _projeta(centro.reshape(1, 3), centro, escala, w, h)[0]
    # eixos X/Y/Z
    eixo = 0.12
    for nome, vec, c in (
            ("X", np.array([eixo, 0, 0]), (60, 60, 230)),
            ("Y", np.array([0, -eixo, 0]), (60, 220, 60)),
            ("Z", np.array([0, 0, eixo]), (230, 120, 60))):
        q = _projeta((centro + vec).reshape(1, 3), centro, escala, w, h)[0]
        cv2.line(painel, tuple(centro_px.astype(int)),
                 tuple(q.astype(int)), c, 1, cv2.LINE_AA)
        cv2.putText(painel, nome, (int(q[0]) + 3, int(q[1]) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    # trilha do punho
    if len(trail) >= 2:
        tp = np.asarray(list(trail), np.float64)
        pp = _projeta(tp, centro, escala, w, h)
        cv2.polylines(painel, [pp.astype(np.int32)], False, cor, 1,
                      cv2.LINE_AA)
    # esqueleto
    proj = _projeta(p, centro, escala, w, h)
    for a, b in gt.HAND_CONNECTIONS:
        if ok[a] and ok[b]:
            cv2.line(painel, tuple(proj[a].astype(int)),
                     tuple(proj[b].astype(int)), cor, 2, cv2.LINE_AA)
    for j in range(21):
        if ok[j]:
            cv2.circle(painel, tuple(proj[j].astype(int)), 3, cor, -1,
                       cv2.LINE_AA)
    palm = p[9]
    cv2.putText(painel, f"X={palm[0]:+.3f} Y={palm[1]:+.3f} Z={palm[2]:.3f} m",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235),
                1, cv2.LINE_AA)
    return painel


def _video_panel(frame, largura, altura, idx, n_maos, vis="R:-- L:--"):
    painel = cv2.resize(frame, (largura, altura), interpolation=cv2.INTER_AREA)
    cv2.putText(painel, f"aux{idx} | maos={n_maos} | {vis} | "
                f"R=verde L=laranja",
                (10, altura - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (230, 230, 230), 1, cv2.LINE_AA)
    return painel


def main() -> int:
    ap = argparse.ArgumentParser(description="Preview hibrido das 2 auxiliares")
    ap.add_argument("--aux", type=int, nargs=2, default=(0, 1),
                    help="indices OpenCV das duas auxiliares")
    ap.add_argument("--exposicao", type=str, default=None,
                    help="-6 para todas ou '0:-6,1:-6'")
    ap.add_argument("--ganho", type=str, default=None)
    ap.add_argument("--largura", type=int, default=1280)
    ap.add_argument("--altura", type=int, default=720)
    ap.add_argument("--max-width", type=int, default=1280,
                    help="largura total do canvas (video+3D) para caber na tela")
    ap.add_argument("--panel-height", type=int, default=340)
    ap.add_argument("--num-hands", type=int, default=2,
                    help="2 para as duas maos; 1 para sessoes de uma mao")
    ap.add_argument("--min-detection-confidence", type=float, default=0.5)
    ap.add_argument("--min-presence-confidence", type=float, default=0.5)
    ap.add_argument("--min-tracking-confidence", type=float, default=0.5)
    ap.add_argument("--pareamento", choices=("lateral", "geometrico"),
                    default="geometrico")
    ap.add_argument("--duracao", type=float, default=0.0,
                    help="encerra automaticamente apos N s (0 = sem limite)")
    ap.add_argument("--dry-run", action="store_true",
                    help="so' valida calibracao/indices, nao abre camera")
    args = ap.parse_args()
    indices = (int(args.aux[0]), int(args.aux[1]))
    alignment = gt.CameraAlignment(aux_indices=indices)
    views = {i: alignment.aux_view(i) for i in indices}
    for i, v in views.items():
        if not v.usable:
            raise SystemExit(f"calibracao aux{i} nao utilizavel")
    print(f"auxiliares: {indices} | vista aux0={views[indices[0]].size} "
          f"aux1={views[indices[1]].size}")
    if args.dry_run:
        for i in indices:
            print(f"aux{i}: camera={getattr(views[i], 'camera_name', None)} "
                  f"rms={getattr(views[i], 'rms', None)}")
        return 0
    caps = []
    try:
        for i in indices:
            caps.append(_abre(i, _controle(args.exposicao, i),
                              _controle(args.ganho, i),
                              args.largura, args.altura))
        detector = MediaPipeHandDetector(
            num_hands=args.num_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_presence_confidence=args.min_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            verbose=False)
        col_w = max(320, int(args.max_width) // 2)
        video_h = max(220, int(args.panel_height))
        plot_h = max(220, int(args.panel_height))
        canvas_w = col_w * 2
        canvas_h = video_h + plot_h
        cv2.namedWindow("ATHENA-style live preview - 2 auxiliares",
                        cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ATHENA-style live preview - 2 auxiliares",
                         canvas_w, canvas_h)
        trails: Dict[str, deque] = {"right": deque(maxlen=80),
                                    "left": deque(maxlen=80)}
        t0 = time.time(); n = 0
        while args.duracao <= 0 or (time.time() - t0) < args.duracao:
            frames = []
            for cap in caps:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("camera parou de entregar frame")
                frames.append(gt.transform_auxiliary_frame(frame))
            stamp = int(time.monotonic() * 1000)
            dets: Dict[int, List[HandDetection]] = {}
            for k, i in enumerate(indices):
                dets[i] = detector.detect(frames[k], f"aux{i}", stamp,
                                          aux_index=i) or []
            if args.pareamento == "geometrico":
                casadas, _ = _casa_maos_geometrica(
                    dets[indices[0]], dets[indices[1]],
                    views[indices[0]], views[indices[1]], gt)
            else:
                casadas, _ = _casa_maos(dets[indices[0]], dets[indices[1]])
            graficos = {"right": np.full((21, 3), np.nan),
                        "left": np.full((21, 3), np.nan)}
            # visibilidade por lado em cada camera (para diagnosticar oclusao)
            vis = {
                "right": tuple(any(d.side == "right" for d in dets[i])
                               for i in indices),
                "left": tuple(any(d.side == "left" for d in dets[i])
                              for i in indices),
            }
            status = {}
            for lado, d0, d1 in casadas:
                pontos, err = gt.triangulate_pair_points(
                    views[indices[0]], views[indices[1]],
                    d0.landmarks_px, d1.landmarks_px)
                if pontos is not None:
                    pts = np.asarray(pontos, np.float64).reshape(-1, 3)
                    graficos[lado] = pts
                    palm = pts[9]
                    if np.isfinite(palm).all():
                        trails.setdefault(lado, deque(maxlen=80)).append(palm)
                    vtxt = "ambas" if all(vis[lado]) else "1 vista"
                    status[lado] = (f"{lado}: reproj={err:.1f}px "
                                     f"Z={palm[2]:.3f}m | {vtxt}")
                else:
                    status[lado] = f"{lado}: 3D falhou"
            for lado in ("right", "left"):
                if lado in status:
                    continue
                v0, v1 = vis[lado]
                if v0 or v1:
                    qual = "aux0" if v0 and not v1 else (
                        "aux1" if v1 and not v0 else "ambas")
                    status[lado] = f"{lado}: SEM 3D (oclusao/{qual})"
                else:
                    status[lado] = f"{lado}: ausente"
            # desenha cada deteccao na sua camera, com a cor do lado
            for k, i in enumerate(indices):
                for d in dets[i]:
                    _draw_mao(frames[k], d, CORES_MAO.get(d.side,
                                                          CORES_MAO["unknown"]))
                for j, lado in enumerate(("right", "left")):
                    txt = status.get(lado, f"{lado}: ausente")
                    cv2.putText(frames[k], txt, (10, 24 + 22 * j),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                CORES_MAO.get(lado, (255, 255, 255)), 1,
                                cv2.LINE_AA)
            # monta canvas 2x2
            canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
            def _vis_txt(k):
                return " ".join(
                    f"{lado[0].upper()}:{'OK' if vis[lado][k] else '--'}"
                    for lado in ("right", "left"))
            canvas[0:video_h, 0:col_w] = _video_panel(
                frames[0], col_w, video_h, indices[0],
                len(dets[indices[0]]), _vis_txt(0))
            canvas[0:video_h, col_w:2 * col_w] = _video_panel(
                frames[1], col_w, video_h, indices[1],
                len(dets[indices[1]]), _vis_txt(1))
            for j, lado in enumerate(("right", "left")):
                v0, v1 = vis[lado]
                nota = (f"visao: aux0={'OK' if v0 else '--'} "
                        f"aux1={'OK' if v1 else '--'}")
                painel3d = _draw_3d_panel((col_w, plot_h),
                                          graficos[lado], lado,
                                          trails.get(lado, deque()), nota)
                canvas[video_h:video_h + plot_h,
                       j * col_w:(j + 1) * col_w] = painel3d
            n += 1
            fps = n / max(1e-6, time.time() - t0)
            cv2.putText(canvas, f"PREVIEW HIBRIDO (MediaPipe+triangulacao) "
                        f"| {fps:.1f} fps | ESC/Q",
                        (10, canvas_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow("ATHENA-style live preview - 2 auxiliares", canvas)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

