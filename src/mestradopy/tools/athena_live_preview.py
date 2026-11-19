#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preview AO VIVO das duas webcams auxiliares (híbrido ATHENA-style).

IMPORTANTE: o ATHENA oficial é um pipeline OFFLINE (vídeos frame-locked,
fase 1 + fase 2 em lote). Este script NÃO é o ATHENA completo: ele faz o
preview em tempo real com o mesmo detector (MediaPipe Hands) + a nossa
calibração/triangulação do par auxiliar. Serve para ver o tracking das duas
câmeras ao vivo e diagnosticar lateralidade/cobertura antes de gravar.

Uso:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\athena_live_preview.py
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\athena_live_preview.py --exposicao "0:-6,1:-6"

Teclas: ESC/Q sai; D liga/desliga profundidade do Kinect (não usada aqui).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1]
for _p in (_SRC, _SRC / "core", _SRC / "tools", _SRC / "vision"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kinect_imu_groundtruth as gt  # noqa: E402
from camera_uvc import define_modo, parse_controle_por_camera, trava_exposicao  # noqa: E402
from handmvnet_video_process import (  # noqa: E402
    HandDetection, MediaPipeHandDetector, _casa_maos, _casa_maos_geometrica,
)


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


def _draw_mao(frame, det: HandDetection, cor=(0, 255, 0)):
    pts = np.asarray(det.landmarks_px, np.float64)
    for a, b in gt.HAND_CONNECTIONS:
        pa = tuple(np.round(pts[a]).astype(int))
        pb = tuple(np.round(pts[b]).astype(int))
        cv2.line(frame, pa, pb, cor, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(np.round(p).astype(int)), 3, (0, 0, 255), -1,
                   cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preview hibrido das 2 auxiliares")
    ap.add_argument("--aux", type=int, nargs=2, default=(0, 1),
                    help="indices OpenCV das duas auxiliares")
    ap.add_argument("--exposicao", type=str, default=None,
                    help="-6 para todas ou '0:-6,1:-6'")
    ap.add_argument("--ganho", type=str, default=None)
    ap.add_argument("--largura", type=int, default=1280)
    ap.add_argument("--altura", type=int, default=720)
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
            for lado, d0, d1 in casadas:
                pontos, err = gt.triangulate_pair_points(
                    views[indices[0]], views[indices[1]],
                    d0.landmarks_px, d1.landmarks_px)
                if pontos is not None:
                    palm = np.asarray(pontos, np.float64).reshape(-1, 3)[9]
                    texto = (f"{lado}: X={palm[0]:.3f} Y={palm[1]:.3f} "
                             f"Z={palm[2]:.3f} m | reproj={err:.1f}px")
                else:
                    texto = f"{lado}: 3D falhou"
                cv2.putText(frames[0], texto, (12, 32 + 24 * len(casadas)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                            cv2.LINE_AA)
            for k, i in enumerate(indices):
                for d in dets[i]:
                    _draw_mao(frames[k], d)
                cv2.putText(frames[k], f"aux{i} | maos={len(dets[i])} | "
                            f"{'espelhada' if gt.AUXILIARY_MIRROR_HORIZONTAL else 'crua'}",
                            (12, frames[k].shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                            cv2.LINE_AA)
            h = min(f.shape[0] for f in frames)
            paineis = []
            for f in frames:
                escala = h / f.shape[0]
                paineis.append(cv2.resize(f, (int(f.shape[1] * escala), h)))
            canvas = np.hstack(paineis)
            n += 1
            fps = n / max(1e-6, time.time() - t0)
            cv2.putText(canvas, f"PREVIEW HIBRIDO (MediaPipe + triangulacao) "
                        f"| {fps:.1f} fps | ESC/Q sai",
                        (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("ATHENA-style live preview - aux0 | aux1", canvas)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
