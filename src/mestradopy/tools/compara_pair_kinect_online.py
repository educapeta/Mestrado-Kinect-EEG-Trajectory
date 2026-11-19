#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comparador ONLINE: triangulaÃ§Ã£o aux0â†”aux1 vs Kinect RGB/depth.

Abre Kinect (RGB+depth) e as duas webcams auxiliares, detecta as mÃ£os com
MediaPipe, triangula o par aux0â†”aux1 e compara com o 3D obtido pelo depth do
Kinect (corrigido depthâ†’cor). Mostra mÃ©tricas ao vivo e grava CSV.

Uso:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\compara_pair_kinect_online.py ^
        --exposicao "0:-7,1:-7" --duracao 30

MÃ©tricas por frame/lado:
  - erro 3D parÃ—Kinect (mediana/p90 em mm)
  - reprojeÃ§Ã£o do 3D do par na aux0/aux1 e no Kinect (px)
  - cobertura 2D/3D
  - FPS

IMPORTANTE: o Kinect depth nÃ£o Ã© verdade absoluta; ele tem ruÃ­do, falhas de
depth e a correÃ§Ã£o depthâ†’cor. O CSV registra tambÃ©m quantos joints do Kinect
falharam para vocÃª nÃ£o ler o erro sem contexto.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    HandDetection, MediaPipeHandDetector, _casa_maos,
    _casa_maos_geometrica)
from compara_tracking_mao import _corrige_frame_cor, _kinect_3d_21  # noqa: E402

COR = {"right": (0, 230, 0), "left": (0, 165, 255), "unknown": (210, 210, 210)}


def _controle(texto: Optional[str], idx: int):
    if not texto:
        return None
    c = parse_controle_por_camera(texto)
    if c.get("todos") is not None:
        return float(c["todos"])
    return c.get("indices", {}).get(int(idx))


def _abre(idx: int, exposicao, ganho, w=1280, h=720):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"nao abri aux{idx}")
    define_modo(cap, w, h, mjpeg=True)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        trava_exposicao(cap, exposicao=exposicao, ganho=ganho,
                        rotulo=f"aux{idx}")
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] aux{idx} exposicao: {exc}")
    return cap


def _draw_mao(img, pts, cor, label=""):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    for a, b in gt.HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                 cor, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p.astype(int)), 2, cor, -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, tuple((pts[0] + [6, -6]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA)


def _proj_px(pontos, K, dist):
    p = np.asarray(pontos, np.float64).reshape(-1, 3)
    ok = np.isfinite(p).all(axis=1)
    out = np.full((len(p), 2), np.nan)
    if ok.any():
        r, _ = cv2.projectPoints(p[ok].reshape(-1, 1, 3),
                                 np.zeros(3), np.zeros(3),
                                 np.asarray(K, np.float64),
                                 np.asarray(dist, np.float64))
        out[ok] = r.reshape(-1, 2)
    return out


BONES = tuple(gt.HAND_CONNECTIONS)


def _comprimentos_ossos(xyz):
    """Comprimento (m) de cada osso adjacente; NaN se ponta invalida."""
    p = np.asarray(xyz, np.float64).reshape(-1, 3)
    out = np.full(len(BONES), np.nan)
    for k, (a, b) in enumerate(BONES):
        if (np.isfinite(p[a]).all() and np.isfinite(p[b]).all()):
            out[k] = float(np.linalg.norm(p[a] - p[b]))
    return out


def _resumo_rigidez(por_osso):
    """[{osso, n, media_mm, std_mm, cv_pct, mad_mm}] a partir de listas."""
    linhas = []
    for k, (a, b) in enumerate(BONES):
        vals = np.asarray(por_osso.get(k, []), np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 3:
            continue
        media = float(np.median(vals))
        mad = float(np.median(np.abs(vals - media)))
        linhas.append({"osso": f"{a}-{b}", "n": int(vals.size),
                       "media_mm": round(media * 1000.0, 3),
                       "std_mm": round(float(np.std(vals)) * 1000.0, 3),
                       "cv_pct": round(100.0 * float(np.std(vals)) / max(media, 1e-9), 3),
                       "mad_mm": round(mad * 1000.0, 3)})
    return linhas


def _proj_aux(view, pontos):
    """X_aux = R X_kin + T; depois projeta com K/dist da auxiliar."""
    p = np.asarray(pontos, np.float64).reshape(-1, 3).copy()
    R = np.asarray(view.rotation, np.float64).reshape(3, 3)
    T = np.asarray(view.translation, np.float64).reshape(3)
    p_aux = (R @ p.T).T + T.reshape(1, 3)
    return _proj_px(p_aux, view.effective_matrix, view.dist)


def _resumo(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                   np.float64)
    if a.size == 0:
        return None, None
    return round(float(np.median(a)), 2), round(float(np.percentile(a, 90)), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Comparador online par aux vs Kinect")
    ap.add_argument("--aux", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--exposicao", type=str, default=None,
                    help="-7 para todas ou '0:-7,1:-7'")
    ap.add_argument("--ganho", type=str, default=None)
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--min-detection-confidence", type=float, default=0.5)
    ap.add_argument("--min-presence-confidence", type=float, default=0.5)
    ap.add_argument("--min-tracking-confidence", type=float, default=0.5)
    ap.add_argument("--pareamento", choices=("lateral", "geometrico"),
                    default="geometrico")
    ap.add_argument("--duracao", type=float, default=0.0)
    ap.add_argument("--csv", type=Path,
                    default=Path("outputs/visao/comparacao_pair_kinect.csv"))
    ap.add_argument("--dry-run", action="store_true",
                    help="so' valida calibracao, sem abrir cameras")
    args = ap.parse_args()
    indices = (int(args.aux[0]), int(args.aux[1]))
    alignment = gt.CameraAlignment(aux_indices=indices)
    views = {i: alignment.aux_view(i) for i in indices}
    for i in indices:
        if not views[i].usable:
            raise SystemExit(f"calibracao aux{i} nao utilizavel")
    print(f"calibracao aux0={views[indices[0]].rms}px aux1={views[indices[1]].rms}px")
    if args.dry_run:
        return 0

    caps = []
    tracker = None
    csv_file = None
    writer = None
    try:
        for i in indices:
            caps.append(_abre(i, _controle(args.exposicao, i),
                              _controle(args.ganho, i)))
        tracker = gt.KinectHandTracker(aux_indices=indices)
        detector = MediaPipeHandDetector(
            num_hands=args.num_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_presence_confidence=args.min_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            verbose=False)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame", "t", "lado", "fps", "valido",
            "erro3d_med_mm", "erro3d_p90_mm",
            "reproj_aux0_med_px", "reproj_aux1_med_px",
            "reproj_kinect_med_px", "n3d", "n_kin",
            "kin_depth_fail", "kin_map_fail", "obs"])
        t0 = time.time(); n = 0
        last_fps_t = t0
        fps_atual = 0.0
        last_palm: Dict[str, np.ndarray] = {}
        rigidez: Dict[str, Dict[str, Dict[int, list]]] = {
            side: {"pair": {}, "kin": {}} for side in ("right", "left")}
        cv2.namedWindow("Comparador ONLINE: aux pair x Kinect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Comparador ONLINE: aux pair x Kinect", 1280, 760)
        while args.duracao <= 0 or (time.time() - t0) < args.duracao:
            color, depth = tracker.frames()
            if color is None:
                time.sleep(0.01)
                continue
            if color.ndim == 3 and color.shape[2] == 4:
                kin_bgr = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            else:
                kin_bgr = np.asarray(color).copy()
            frames = []
            for cap in caps:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("camera auxiliar parou")
                frames.append(gt.transform_auxiliary_frame(frame))
            now = time.time()
            fps_atual = 1.0 / max(1e-6, now - last_fps_t)
            last_fps_t = now
            stamp = int(time.monotonic() * 1000)
            dets: Dict[int, List[HandDetection]] = {}
            for k, i in enumerate(indices):
                dets[i] = detector.detect(frames[k], f"aux{i}", stamp,
                                          aux_index=i) or []
            _, kin_hands = tracker.detect_hands_all(color, depth)
            kin_por_lado = {h.get("side"): h for h in kin_hands if h.get("side")}
            if args.pareamento == "geometrico":
                casadas, _ = _casa_maos_geometrica(
                    dets[indices[0]], dets[indices[1]],
                    views[indices[0]], views[indices[1]], gt)
            else:
                casadas, _ = _casa_maos(dets[indices[0]], dets[indices[1]])
            for lado, d0, d1 in casadas:
                if lado not in ("right", "left"):
                    continue
                pair_xyz, pair_err = gt.triangulate_pair_points(
                    views[indices[0]], views[indices[1]],
                    d0.landmarks_px, d1.landmarks_px)
                pair_xyz = (None if pair_xyz is None else
                            np.asarray(pair_xyz, np.float64).reshape(-1, 3))
                kin = kin_por_lado.get(lado)
                kin_xyz = kin_cor = None
                kin_px = None
                n_kin_fail = n_kin_map = 0
                if kin is not None and pair_xyz is not None:
                    kin_xyz, info = _kinect_3d_21(tracker, kin.get("landmarks") or [], lado)
                    kin_cor = _corrige_frame_cor(kin_xyz)
                    n_kin_fail = sum(1 for x in info if x["depth_fail"])
                    n_kin_map = sum(1 for x in info if x["map_fail"])
                    kin_px = _proj_px(kin_cor, alignment.kinect_matrix,
                                       alignment.kinect_dist)
                # reprojecao do par
                r0 = _proj_aux(views[indices[0]], pair_xyz)
                r1 = _proj_aux(views[indices[1]], pair_xyz)
                e0 = (np.linalg.norm(r0 - d0.landmarks_px, axis=1)
                      if pair_xyz is not None else np.array([]))
                e1 = (np.linalg.norm(r1 - d1.landmarks_px, axis=1)
                      if pair_xyz is not None else np.array([]))
                ekg = None
                if kin_px is not None and pair_xyz is not None:
                    ekg = (np.linalg.norm(kin_px - np.asarray(
                        [(lm.x * tracker.color_width,
                          lm.y * tracker.color_height)
                         for lm in (kin.get("landmarks") or [])], np.float64),
                        axis=1))
                if pair_xyz is not None:
                    for k, val in enumerate(_comprimentos_ossos(pair_xyz)):
                        if np.isfinite(val):
                            rigidez[lado]["pair"].setdefault(k, []).append(float(val))
                if kin_cor is not None:
                    for k, val in enumerate(_comprimentos_ossos(kin_cor)):
                        if np.isfinite(val):
                            rigidez[lado]["kin"].setdefault(k, []).append(float(val))
                erro3d = np.array([])
                if pair_xyz is not None and kin_cor is not None:
                    ok = (np.isfinite(pair_xyz).all(axis=1)
                          & np.isfinite(kin_cor).all(axis=1))
                    if ok.any():
                        erro3d = np.linalg.norm(pair_xyz[ok] - kin_cor[ok],
                                                axis=1) * 1000.0
                def _m(a):
                    a = np.asarray(a, np.float64)
                    a = a[np.isfinite(a)]
                    return None if a.size == 0 else float(np.median(a))
                vals = [float(v) for v in erro3d if np.isfinite(v)]
                med, p90 = _resumo(vals)
                # desenho
                if pair_xyz is not None:
                    for a, b in gt.HAND_CONNECTIONS:
                        if (np.isfinite(pair_xyz[a]).all()
                                and np.isfinite(pair_xyz[b]).all()):
                            # projeta no Kinect
                            ppx = _proj_px(pair_xyz[[a, b]],
                                           alignment.kinect_matrix,
                                           alignment.kinect_dist)
                            if np.isfinite(ppx).all():
                                cv2.line(kin_bgr, tuple(ppx[0].astype(int)),
                                         tuple(ppx[1].astype(int)),
                                         COR[lado], 2, cv2.LINE_AA)
                if kin is not None:
                    kp = np.array([(lm.x * tracker.color_width,
                                    lm.y * tracker.color_height)
                                   for lm in (kin.get("landmarks") or [])],
                                  np.float64)
                    _draw_mao(kin_bgr, kp, (0, 255, 0), f"Kinect {lado}")
                for k, i in enumerate(indices):
                    for d in dets[i]:
                        _draw_mao(frames[k], d.landmarks_px,
                                  COR.get(d.side, COR["unknown"]),
                                  f"{d.side}")
                    cv2.putText(frames[k], f"aux{i}", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                1, cv2.LINE_AA)
                texto = (f"{lado}: erro3D med={med} p90={p90} mm | "
                         f"reproj aux0={_m(e0)} aux1={_m(e1)} px")
                cv2.putText(kin_bgr, texto, (10, 30 + (0 if lado == "right" else 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            COR[lado], 1, cv2.LINE_AA)
                if writer is not None:
                    writer.writerow([
                        n, f"{time.time():.6f}", lado,
                        f"{fps_atual:.2f}",
                        int(pair_xyz is not None), med, p90,
                        _m(e0), _m(e1), _m(ekg),
                        int(np.isfinite(pair_xyz).all(axis=1).sum())
                        if pair_xyz is not None else 0,
                        int(kin is not None),
                        n_kin_fail, n_kin_map,
                        "" if pair_xyz is not None else "triangulacao_falhou"])
                cv2.putText(kin_bgr, f"par vs Kinect | FPS "
                            f"{fps_atual:.1f}",
                            (10, kin_bgr.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255),
                            1, cv2.LINE_AA)
            h = 360
            paineis = []
            for f in [kin_bgr, frames[0], frames[1]]:
                esc = h / f.shape[0]
                paineis.append(cv2.resize(f, (int(f.shape[1] * esc), h)))
            canvas = np.hstack(paineis)
            cv2.imshow("Comparador ONLINE: aux pair x Kinect", canvas)
            n += 1
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        for cap in caps:
            cap.release()
        try:
            if tracker is not None:
                tracker.close()
        except Exception:
            pass
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()
    # resumo de rigidez (comprimento de osso constante)
    linhas_rig = []
    for side in ("right", "left"):
        for fonte in ("pair", "kin"):
            for item in _resumo_rigidez(rigidez[side][fonte]):
                linhas_rig.append({"lado": side, "fonte": fonte, **item})
    rig_csv = args.csv.with_name(args.csv.stem + "_rigidez.csv")
    if linhas_rig:
        rig_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(rig_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(linhas_rig[0].keys()))
            w.writeheader(); w.writerows(linhas_rig)
        print("RIGIDEZ CSV:", rig_csv)
        for side in ("right", "left"):
            for fonte in ("pair", "kin"):
                cv = [x["cv_pct"] for x in linhas_rig
                      if x["lado"] == side and x["fonte"] == fonte]
                if cv:
                    print(f"  rigidez {side}/{fonte}: CV mediano "
                          f"{np.median(cv):.2f}% (n ossos={len(cv)})")
    print("CSV:", args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

