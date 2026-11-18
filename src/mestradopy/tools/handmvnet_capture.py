#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Captura sincronizada Kinect + aux0 + aux1 para treinar o HandMvNet.

Grava DIRETO no disco escolhido (ex.: F:\\handmvnet), com:
    sessao_YYYYMMDD_HHMMSS/
        frames/000000_0.jpg  (Kinect RGB)
        frames/000000_1.jpg  (aux0)
        frames/000000_2.jpg  (aux1)
        labels.npz
        meta.json

Por que frames recortados? O dataset de treino usa a imagem cheia + bbox e
faz o crop 256x256 no carregamento. Para economizar disco, a captura grava
frames em JPEG qualidade configuravel e so a cada N frames.

Exemplos:
    # grava em F: (padrao se existir), 5 amostras/s, sem janela
    python src/mestradopy/tools/handmvnet_capture.py --saida F:\\handmvnet

    # grava em F: mostrando preview
    python src/mestradopy/tools/handmvnet_capture.py --saida F:\\handmvnet --mostrar

Teclas no preview: q/ESC encerra; espaco pausa.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parents[1]
for _p in (_SRC, _SRC / "core", _SRC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from kinect_imu_groundtruth import (  # noqa: E402
    CameraAlignment, KinectHandTracker, triangulate_three_views,
    indices_auxiliares_automaticos, nome_auxiliar)
from vision.handmvnet_dataset import square_bbox  # noqa: E402
from compara_tracking_multiview import (  # noqa: E402
    open_aux_cameras, read_aux_frames, get_landmarks_pixels)


def default_output() -> Path:
    for letra in ("F:", "E:", "D:"):
        p = Path(letra + "\\")
        try:
            if p.exists() and shutil.disk_usage(p).free > 5 * 1024**3:
                return p / "handmvnet"
        except OSError:
            pass
    return Path("data/handmvnet")


def choose_side(value: str):
    return {"direita": "right", "esquerda": "left"}.get(value.lower(), None)


def calibration_arrays(alignment, aux_indices):
    K = np.asarray(alignment.kinect_matrix, np.float64)
    extr = [np.eye(4, dtype=np.float64)]
    intr = [K]
    for idx in aux_indices:
        view = alignment.aux_view(int(idx))
        if view is None or not getattr(view, "usable", False):
            raise RuntimeError(f"aux{idx} sem calibracao utilizavel")
        intr.append(np.asarray(view.effective_matrix, np.float64))
        R = np.asarray(view.rotation, np.float64).reshape(3, 3)
        T = np.asarray(view.translation, np.float64).reshape(3, 1)
        E = np.eye(4, dtype=np.float64)
        E[:3, :3] = R; E[:3, 3:] = T
        extr.append(E)
    return np.stack(intr), np.stack(extr)


def save_labels(path: Path, data):
    np.savez_compressed(path, **data)


def main():
    ap = argparse.ArgumentParser(description="Captura multi-view para HandMvNet")
    ap.add_argument("--saida", type=Path, default=default_output(),
                    help="disco/pasta de destino (padrao: F:\\handmvnet)")
    ap.add_argument("--aux", type=int, nargs="+", default=None,
                    help="indices das auxiliares (padrao automatico)")
    ap.add_argument("--cada", type=int, default=5,
                    help="salva 1 amostra a cada N frames")
    ap.add_argument("--max-amostras", type=int, default=0,
                    help="0 = sem limite")
    ap.add_argument("--min-free-gb", type=float, default=2.0,
                    help="aborta se o disco ficar abaixo disso")
    ap.add_argument("--qualidade-jpeg", type=int, default=85)
    ap.add_argument("--lado", choices=["auto", "esquerda", "direita"],
                    default="auto")
    ap.add_argument("--aux-width", type=int, default=1280)
    ap.add_argument("--aux-height", type=int, default=720)
    ap.add_argument("--aux-fps", type=int, default=30)
    ap.add_argument("--mostrar", action="store_true")
    ap.add_argument("--sem-janela", action="store_true")
    args = ap.parse_args()

    aux_indices = args.aux or list(indices_auxiliares_automaticos())
    if not aux_indices:
        raise SystemExit("nenhuma camera auxiliar encontrada; use --aux 0 1")
    aux_indices = list(aux_indices)[:2]
    out = args.saida
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = out / f"sessao_{stamp}"
    (sess / "frames").mkdir(parents=True, exist_ok=True)
    print("destino:", sess)
    print("auxiliares:", [(i, nome_auxiliar(i)) for i in aux_indices])

    tracker = KinectHandTracker(aux_indices=aux_indices)
    alignment = CameraAlignment(aux_indices=aux_indices)
    K, E = calibration_arrays(alignment, aux_indices)
    aux_caps = open_aux_cameras(aux_indices, args.aux_width,
                                args.aux_height, args.aux_fps)
    if len(aux_caps) < len(aux_indices):
        raise SystemExit("nao abriu todas as auxiliares")

    side = choose_side(args.lado)
    labels = {k: [] for k in
              ("joints_2d", "joints_3d", "bboxes", "valid", "timestamps")}
    view_names = ["kinect"] + [f"aux{i}" for i in aux_indices]
    saved = 0; frame_idx = 0; paused = False
    t0 = time.time()
    try:
        while True:
            color_frame, depth_frame = tracker.frames()
            if color_frame is None:
                time.sleep(0.01)
                continue
            aux_frames = read_aux_frames(aux_caps)
            det = get_landmarks_pixels(
                tracker, color_frame, depth_frame, aux_frames, alignment,
                want_side=side)
            kin = det["kinect"]
            aux_px = [det["aux"].get(int(i)) for i in aux_indices]
            if kin is None or any(p is None for p in aux_px):
                continue
            pts3d, mask, info = triangulate_three_views(
                alignment, kin, aux_px[0], aux_px[1],
                aux_indices=tuple(aux_indices), min_valid_points=8,
                max_reproj_error_px=15.0)
            if pts3d is None or mask is None or int(mask.sum()) < 8:
                continue
            if frame_idx % max(1, args.cada) == 0:
                free_gb = shutil.disk_usage(out).free / 1024**3
                if free_gb < args.min_free_gb:
                    print(f"ABORTANDO: disco livre {free_gb:.2f} GB < "
                          f"{args.min_free_gb} GB")
                    break
                i = saved
                c = color_frame.copy()
                cv2.imwrite(str(sess / "frames" / f"{i:06d}_0.jpg"), c,
                            [cv2.IMWRITE_JPEG_QUALITY, args.qualidade_jpeg])
                for k, idx in enumerate(aux_indices, start=1):
                    cv2.imwrite(
                        str(sess / "frames" / f"{i:06d}_{k}.jpg"),
                        aux_frames[int(idx)].copy(),
                        [cv2.IMWRITE_JPEG_QUALITY, args.qualidade_jpeg])
                px = [np.asarray(kin, np.float32)] + [
                    np.asarray(p, np.float32) for p in aux_px]
                bboxes = [square_bbox(p, 1.25, image_shape=img.shape[:2])
                          for p, img in zip(px, [color_frame] + [
                              aux_frames[int(i)] for i in aux_indices])]
                labels["joints_2d"].append(np.stack(px))
                labels["joints_3d"].append(np.asarray(pts3d, np.float32) * 1000.0)
                labels["bboxes"].append(np.stack(bboxes))
                labels["valid"].append(np.stack([
                    np.isfinite(p).all(axis=1) for p in px]))
                labels["timestamps"].append(time.time())
                saved += 1
                print(f"amostra {saved:05d} frames={frame_idx} "
                      f"validos={int(mask.sum())} metodo={info.get('method')} "
                      f"livre={free_gb:.1f}GB", flush=True)
                if args.max_amostras and saved >= args.max_amostras:
                    break
            if args.mostrar and not args.sem_janela:
                cv2.imshow("captura HandMvNet", color_frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
            frame_idx += 1
            if frame_idx % 30 == 0 and time.time() - t0 > 10:
                print(f"frames={frame_idx} amostras={saved}", flush=True)
    finally:
        for cap in aux_caps.values():
            cap.release()
        if args.mostrar:
            cv2.destroyAllWindows()
    n = len(labels["timestamps"])
    if n == 0:
        print("NENHUMA amostra salva.")
        return 1
    np.savez_compressed(
        sess / "labels.npz",
        joints_2d=np.stack(labels["joints_2d"]),
        joints_3d=np.stack(labels["joints_3d"]),
        bboxes=np.stack(labels["bboxes"]),
        intrinsics=K, extrinsics=E,
        valid=np.stack(labels["valid"]),
        hand=np.array(["direita" if side == "right" else "esquerda"
                       if side == "left" else "auto"] * n, dtype=object),
        timestamps=np.asarray(labels["timestamps"]),
    )
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "views": view_names, "aux_indices": list(aux_indices),
        "n_samples": n, "jpeg_quality": args.qualidade_jpeg,
        "cada_n_frames": args.cada, "side": args.lado,
        "calibration_dir": str(Path("configs/calibracao").resolve()),
        "note": "3D = triangulacao MediaPipe multi-view (pseudo-label); "
                "validar antes de tratar como ground truth absoluto",
    }
    (sess / "meta.json").write_text(json.dumps(meta, indent=2),
                                    encoding="utf-8")
    print(f"\nOK: {n} amostras em {sess}")
    print("Para treinar:")
    print(f"  python src\\mestradopy\\tools\\handmvnet_train.py "
          f"--data \"{sess}\" --device xpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
