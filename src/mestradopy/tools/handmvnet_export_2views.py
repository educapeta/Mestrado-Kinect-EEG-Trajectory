#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exporta uma sessao de video das duas auxiliares para o formato de dataset
do HandMvNet (`HandMvNetSessionDataset`).

Entrada:
    <sessao>/aux00.mp4, aux01.mp4
    <sessao>/processed/landmarks_3d.npz (ou --npz ...)
    <sessao>/meta.json

Saida:
    <destino>/frames/000000_0.jpg, 000000_1.jpg, ...
    <destino>/labels.npz
    <destino>/meta.json

O labels.npz segue o formato do projeto:
    joints_2d  (N,V,21,2) pixels na imagem do video
    joints_3d  (N,21,3)   mm, frame do Kinect
    bboxes     (N,V,4)    x1,y1,x2,y2
    intrinsics (V,3,3)    K na resolucao do video
    extrinsics (V,4,4)    mundo(Kinect)->camera = [R|T]
    valid      (N,V,21)
    hand       (N,)       "direita"/"esquerda"/"desconhecida"
    track_id   (N,)       ID temporal do processamento
    timestamps (N,)

Uso:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\handmvnet_export_2views.py ^
        --session F:\\handmvnet_videos\\teste_sync3 ^
        --npz F:\\handmvnet_videos\\teste_sync3\\processed_track1b\\landmarks_3d.npz ^
        --dest outputs\\handmvnet_dataset\\teste_sync3_2v
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1]                     # src/mestradopy
for _p in (_SRC, _SRC / "core", _SRC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vision.handmvnet_dataset import IMAGE_SIZE, square_bbox  # noqa: E402
from handmvnet_video_process import FrameCursor, plan_session  # noqa: E402

LADO_DATASET = {"right": "direita", "left": "esquerda", "unknown": "desconhecida"}


def _escala_px_calib_para_video(px: np.ndarray, video_size, calib_size
                               ) -> np.ndarray:
    out = np.asarray(px, np.float64).reshape(-1, 2).copy()
    if not video_size or not calib_size:
        return out
    out[:, 0] *= float(video_size[0]) / float(calib_size[0])
    out[:, 1] *= float(video_size[1]) / float(calib_size[1])
    return out


def _k_video(view: Any, video_size, calib_size) -> np.ndarray:
    K = np.asarray(view.effective_matrix, np.float64).reshape(3, 3).copy()
    if video_size and calib_size:
        sx = float(video_size[0]) / float(calib_size[0])
        sy = float(video_size[1]) / float(calib_size[1])
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
    return K.astype(np.float32)


def _extrinseca(view: Any) -> np.ndarray:
    R = np.asarray(view.rotation, np.float64).reshape(3, 3)
    T = np.asarray(view.translation, np.float64).reshape(3)
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R.astype(np.float32)
    E[:3, 3] = T.astype(np.float32)
    return E


def exportar(session: Path, npz_path: Optional[Path], destino: Path,
             *, margin: float = 1.2, min_valid: int = 5,
             max_samples: Optional[int] = None, overwrite: bool = False,
             verbose: bool = True) -> Dict[str, Any]:
    session = Path(session)
    destino = Path(destino)
    npz_path = Path(npz_path) if npz_path else session / "processed" / "landmarks_3d.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"nao encontrei {npz_path}")
    if destino.exists() and any(destino.iterdir()) and not overwrite:
        raise FileExistsError(f"{destino} nao esta vazio; use --overwrite ou outro destino")
    plan = plan_session(session, verbose=False)
    indices = [int(i) for i in plan.aux_indices]
    if len(indices) != 2:
        raise ValueError(f"o exportador exige 2 auxiliares; recebi {indices}")
    views = [plan.views[i] for i in indices]
    for v in views:
        if not v.usable:
            raise ValueError(f"calibracao da aux{v.index} nao esta utilizavel")
    videos = [plan.timelines[k].video_path for k in range(2)]
    frames_dir = destino / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(npz_path, allow_pickle=True)
    n_total = len(d["joints_xyz"])
    if max_samples:
        n_total = min(n_total, int(max_samples))
    cursors = [FrameCursor.open(p) for p in videos]
    joints_2d: List[np.ndarray] = []
    joints_3d: List[np.ndarray] = []
    bboxes: List[np.ndarray] = []
    valid: List[np.ndarray] = []
    hands: List[str] = []
    tracks: List[int] = []
    stamps: List[float] = []
    pares: List[int] = []
    frames0: List[int] = []
    frames1: List[int] = []
    for i in range(n_total):
        v = np.asarray(d["valid"][i], dtype=bool)
        if int(v.sum()) < max(1, int(min_valid)):
            continue
        f0 = int(d["frame0"][i]); f1 = int(d["frame1"][i])
        img0 = cursors[0].read_at(f0)
        img1 = cursors[1].read_at(f1)
        if img0 is None or img1 is None:
            if verbose:
                print(f"[export] frame ilegivel em {i}: {cursors[0].error or cursors[1].error}")
            continue
        j2 = np.zeros((2, 21, 2), np.float32)
        bbs = np.zeros((2, 4), np.float32)
        ok = np.zeros((2, 21), bool)
        px_list = [np.asarray(d["pixels_aux0"][i], np.float64),
                   np.asarray(d["pixels_aux1"][i], np.float64)]
        for k in range(2):
            px = _escala_px_calib_para_video(px_list[k], plan.video_sizes[indices[k]],
                                             plan.calib_sizes[indices[k]])
            img = img0 if k == 0 else img1
            h, w = img.shape[:2]
            j2[k] = px.astype(np.float32)
            bbs[k] = square_bbox(px, margin=margin, image_shape=(h, w))
            ok[k] = v
        cv2.imwrite(str(frames_dir / f"{len(joints_2d):06d}_0.jpg"), img0,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(frames_dir / f"{len(joints_2d):06d}_1.jpg"), img1,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        joints_2d.append(j2); bboxes.append(bbs); valid.append(ok)
        xyz = np.asarray(d["joints_xyz"][i], np.float64) * 1000.0
        joints_3d.append(xyz.astype(np.float32))
        hands.append(LADO_DATASET.get(str(d["hand"][i]), "desconhecida"))
        tracks.append(int(d["track_id"][i]) if "track_id" in d else -1)
        stamps.append(float(d["time_sync_s"][i]))
        pares.append(int(d["pair"][i]))
        frames0.append(f0); frames1.append(f1)
    for c in cursors:
        c.release()
    if not joints_2d:
        raise RuntimeError("nenhuma amostra exportada")
    n = len(joints_2d)
    intrinsics = np.stack([_k_video(views[k], plan.video_sizes[indices[k]],
                                    plan.calib_sizes[indices[k]])
                           for k in range(2)])
    extrinsics = np.stack([_extrinseca(v) for v in views])
    np.savez_compressed(
        destino / "labels.npz",
        joints_2d=np.stack(joints_2d).astype(np.float32),
        joints_3d=np.stack(joints_3d).astype(np.float32),
        bboxes=np.stack(bboxes).astype(np.float32),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        valid=np.stack(valid).astype(bool),
        hand=np.array(hands, dtype=object),
        track_id=np.array(tracks, dtype=np.int64),
        timestamps=np.array(stamps, dtype=np.float64),
        pair=np.array(pares, dtype=np.int64),
        frame0=np.array(frames0, dtype=np.int64),
        frame1=np.array(frames1, dtype=np.int64),
    )
    meta = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_session": str(session),
        "source_npz": str(npz_path),
        "views": 2,
        "view_names": [f"aux{i}" for i in indices],
        "aux_indices": indices,
        "n_samples": n,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "video_size": [list(plan.video_sizes.get(i) or []) for i in indices],
        "calib_sizes": [list(plan.calib_sizes.get(i) or []) for i in indices],
        "calibration": [{"aux": int(i), "rms_px": float(getattr(v, "rms", 0.0) or 0.0),
                         "camera_name": getattr(v, "camera_name", None)}
                        for i, v in zip(indices, views)],
        "mirror_video": bool(plan.meta.get("mirror", True)),
        "note": ("labels 3D vieram de MediaPipe+triangulacao do par auxiliar; "
                 "track_id e' a identidade temporal; hand e' o rotulo suavizado."),
    }
    (destino / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"[export] {n} amostras, 2 vistas -> {destino}")
        print(f"[export] labels: {(destino / 'labels.npz')}")
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exporta sessao de 2 webcams para dataset HandMvNet")
    p.add_argument("--session", required=True, type=Path)
    p.add_argument("--npz", type=Path, default=None,
                   help="landmarks_3d.npz alternativo (ex.: processed_track1b)")
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--margin", type=float, default=1.2)
    p.add_argument("--min-valid", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    exportar(args.session, args.npz, args.dest, margin=args.margin,
             min_valid=args.min_valid, max_samples=args.max_samples,
             overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
