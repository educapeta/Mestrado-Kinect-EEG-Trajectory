# -*- coding: utf-8 -*-
"""Dataset e gerador sintético para treinar HandMvNet com o nosso rig.

Formato de uma sessão de captura:

    sessao/
      labels.npz
      frames/
        000000_0.jpg    # vista 0 = Kinect RGB
        000000_1.jpg    # vista 1 = aux0
        000000_2.jpg    # vista 2 = aux1
        ...

`labels.npz` contém:
    joints_2d   (N, V, 21, 2) float32  pixels na imagem cheia
    joints_3d   (N, 21, 3)    float32  mm, frame do Kinect
    bboxes      (N, V, 4)     float32  x1,y1,x2,y2 na imagem cheia
    intrinsics  (V, 3, 3)     float32
    extrinsics  (V, 4, 4)     float32  mundo(Kinect)->camera
    valid       (N, V, 21)    bool
    hand        (N,)          object/str  "direita"/"esquerda"
    timestamps  (N,)          float64

O dataset devolve os campos que o HandMvNet oficial espera em
`batch["data"]`/`batch["cam_params"]` e usa 256x256 e heatmap 32x32.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 256
HEATMAP_SIZE = 32
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_VIEW_NAMES = ("kinect", "aux0", "aux1")


def square_bbox(px: np.ndarray, margin: float = 1.2,
                image_shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """Caixa quadrada alrededor dos landmarks, com margem."""
    px = np.asarray(px, dtype=np.float32).reshape(-1, 2)
    good = np.isfinite(px).all(axis=1)
    if good.sum() < 2:
        raise ValueError("landmarks insuficientes para bbox")
    p = px[good]
    x1, y1 = p.min(axis=0); x2, y2 = p.max(axis=0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    lado = max(x2 - x1, y2 - y1) * float(margin)
    lado = max(lado, 8.0)
    box = np.array([cx - lado / 2, cy - lado / 2,
                    cx + lado / 2, cy + lado / 2], dtype=np.float32)
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        box = np.array([max(0, box[0]), max(0, box[1]),
                        min(w, box[2]), min(h, box[3])], dtype=np.float32)
    return box


def crop_pad_resize(image: np.ndarray, bbox: Sequence[float],
                    size: int = IMAGE_SIZE, rgb: bool = True) -> np.ndarray:
    """Crop com zero-pad e resize, no mesmo espirito do dataset oficial."""
    img = np.asarray(image)
    if rgb and img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]  # BGR -> RGB
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    out = np.zeros((max(1, y2 - y1), max(1, x2 - x1), 3), dtype=np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    ex1, ey1 = min(w, x2), min(h, y2)
    if ex1 > sx1 and ey1 > sy1:
        dx, dy = sx1 - x1, sy1 - y1
        out[dy:dy + (ey1 - sy1), dx:dx + (ex1 - sx1)] = img[sy1:ey1, sx1:ex1]
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_LINEAR)


def crop_to_image(px: np.ndarray, bbox: Sequence[float],
                  size: int = IMAGE_SIZE) -> np.ndarray:
    pts = np.asarray(px, dtype=np.float32).reshape(-1, 2).copy()
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    pts[:, 0] = (pts[:, 0] - x1) * size / max(1.0, x2 - x1)
    pts[:, 1] = (pts[:, 1] - y1) * size / max(1.0, y2 - y1)
    return pts


def make_heatmap(pts_crop: np.ndarray, size: int = HEATMAP_SIZE,
                 sigma: float = 2.0) -> np.ndarray:
    hms = np.zeros((21, size, size), dtype=np.float32)
    scale = size / float(IMAGE_SIZE)
    for j, (x, y) in enumerate(np.asarray(pts_crop)):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cx, cy = float(x) * scale, float(y) * scale
        ul = (int(cx - 3 * sigma), int(cy - 3 * sigma))
        br = (int(cx + 3 * sigma + 1), int(cy + 3 * sigma + 1))
        if ul[0] >= size or ul[1] >= size or br[0] < 0 or br[1] < 0:
            continue
        gx = np.arange(max(0, ul[0]), min(size, br[0]))
        gy = np.arange(max(0, ul[1]), min(size, br[1]))
        if gx.size == 0 or gy.size == 0:
            continue
        xx, yy = np.meshgrid(gx, gy)
        hms[j, gy[0]:gy[-1] + 1, gx[0]:gx[-1] + 1] = np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    return hms


def _intr4(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32)
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                    dtype=np.float32)


class HandMvNetSessionDataset(Dataset):
    """Dataset de uma sessão de captura multi-view."""

    def __init__(self, root: str | Path, indices: Optional[Sequence[int]] = None,
                 image_size: int = IMAGE_SIZE, augment: bool = False,
                 view_names: Sequence[str] = DEFAULT_VIEW_NAMES):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text(
            encoding="utf-8")) if (self.root / "meta.json").is_file() else {}
        self.labels = np.load(self.root / "labels.npz", allow_pickle=True)
        self.frame_dir = self.root / "frames"
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.view_names = tuple(view_names)
        n = len(self.labels["joints_2d"])
        self.indices = list(range(n)) if indices is None else list(indices)
        if not self.indices:
            raise ValueError("sessao sem amostras")
        self.n_views = int(self.labels["joints_2d"].shape[1])
        if self.n_views != len(self.view_names):
            self.view_names = tuple(f"view{i}" for i in range(self.n_views))
        self._check_files()

    def _check_files(self):
        missing = []
        for i in self.indices[:5]:
            for v in range(self.n_views):
                p = self.frame_dir / f"{i:06d}_{v}.jpg"
                if not p.is_file():
                    missing.append(p.name)
        if missing:
            raise FileNotFoundError("frames ausentes: " + ", ".join(missing))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = self.indices[idx]
        rgb, crops, bboxes, joints_crop, heatmaps = [], [], [], [], []
        joints_2d = np.asarray(self.labels["joints_2d"][i], dtype=np.float32)
        bboxes_np = np.asarray(self.labels["bboxes"][i], dtype=np.float32)
        for v in range(self.n_views):
            img = cv2.imread(str(self.frame_dir / f"{i:06d}_{v}.jpg"),
                             cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"frame {i:06d}_{v}.jpg")
            b = bboxes_np[v].copy()
            if self.augment:
                # jitter leve da caixa; mantem o mesmo frame
                j = np.random.uniform(-0.03, 0.03) * max(2.0, b[2] - b[0])
                b = b + np.array([j, j, j, j], dtype=np.float32)
            crop = crop_pad_resize(img, b, self.image_size)
            x = crop.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            rgb.append(torch.from_numpy(x.transpose(2, 0, 1)).float())
            bboxes.append(torch.from_numpy(b).float())
            jc = crop_to_image(joints_2d[v], b, self.image_size)
            joints_crop.append(torch.from_numpy(jc).float())
            heatmaps.append(torch.from_numpy(make_heatmap(jc)).float())
        joints_3d = np.asarray(self.labels["joints_3d"][i], dtype=np.float32)
        root = joints_3d[0:1]
        joints_cam = (joints_3d - root).astype(np.float32)
        intr = np.stack([_intr4(k) for k in self.labels["intrinsics"]])
        extr = np.asarray(self.labels["extrinsics"], dtype=np.float32)
        valid = np.asarray(self.labels["valid"][i], dtype=bool)
        mask = ~valid  # 1 = invisivel (convencao do dataset oficial)
        return {
            "data": {
                "rgb": torch.stack(rgb),                       # V,3,S,S
                "bboxes": torch.stack(bboxes),                 # V,4
                "joints_crop_img": torch.stack(joints_crop),   # V,21,2
                "heatmap": torch.stack(heatmaps),               # V,21,32,32
                "joints_img_mask": torch.from_numpy(mask),
                "joints_cam": torch.from_numpy(joints_cam),     # 21,3 mm
                "root_joint": torch.from_numpy(root),           # 1,3 mm
            },
            "cam_params": {
                "intrinsic": torch.from_numpy(intr),            # V,4
                "extrinsic": torch.from_numpy(extr),            # V,4,4
            },
            "root_idx": torch.tensor([0], dtype=torch.long),
            "sample_idx": torch.tensor(int(i)),
        }


def make_synthetic_session(root: str | Path, n: int = 8,
                           n_views: int = 3, seed: int = 42) -> Path:
    """Gera uma sessão sintética mínima para smoke test do treino."""
    root = Path(root)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = 640, 480
    K = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]],
                 dtype=np.float32)
    intrinsics = np.stack([K] * n_views)
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * n_views)
    j2 = np.zeros((n, n_views, 21, 2), np.float32)
    j3 = np.zeros((n, 21, 3), np.float32)
    bboxes = np.zeros((n, n_views, 4), np.float32)
    valid = np.zeros((n, n_views, 21), bool)
    hands = np.array(["direita"] * n, dtype=object)
    ts = np.arange(n, dtype=np.float64)
    for i in range(n):
        base = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05),
                         0.50 + rng.uniform(-0.05, 0.05)], np.float32)
        pts = base + rng.normal(0, 0.015, (21, 3)).astype(np.float32)
        j3[i] = pts * 1000.0
        img = np.full((h, w, 3), 80, np.uint8)
        for v in range(n_views):
            z = pts[:, 2]
            xy = np.stack([K[0, 0] * pts[:, 0] / z + K[0, 2],
                           K[1, 1] * pts[:, 1] / z + K[1, 2]], axis=1)
            j2[i, v] = xy
            b = square_bbox(xy, margin=1.4, image_shape=(h, w))
            bboxes[i, v] = b
            valid[i, v] = True
            cv2.circle(img, (int(np.mean(xy[:, 0])), int(np.mean(xy[:, 1]))),
                       8, (0, 255, 0), -1)
            cv2.imwrite(str(root / "frames" / f"{i:06d}_{v}.jpg"), img)
    np.savez_compressed(
        root / "labels.npz",
        joints_2d=j2, joints_3d=j3, bboxes=bboxes,
        intrinsics=intrinsics, extrinsics=extrinsics,
        valid=valid, hand=hands, timestamps=ts)
    (root / "meta.json").write_text(json.dumps({
        "subject": "synthetic", "session": "smoke", "views": n_views,
        "view_names": list(DEFAULT_VIEW_NAMES[:n_views]),
        "source": "make_synthetic_session",
    }, indent=2), encoding="utf-8")
    return root
