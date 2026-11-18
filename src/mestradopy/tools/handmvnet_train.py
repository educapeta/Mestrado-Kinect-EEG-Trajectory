#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Treino/fine-tune minimo do HandMvNet com o nosso dataset multi-view.

Como ainda NAO existe checkpoint oficial, o modo padrao treina do zero usando
a arquitetura oficial (pesos aleatorios) e o nosso dataset. Quando um `.ckpt`
oficial for liberado, use `--checkpoint` para fine-tune.

Exemplos:
    # smoke: gera dataset sintetico e roda 1 epoca
    python src/mestradopy/tools/handmvnet_train.py --smoke

    # treino real
    python src/mestradopy/tools/handmvnet_train.py ^
        --data data/handmvnet/sessao_01 ^
        --data-val data/handmvnet/sessao_02 ^
        --epochs 40 --device xpu --batch-size 2

    # fine-tune com checkpoint oficial de 3 vistas
    python src/mestradopy/tools/handmvnet_train.py --data ... ^
        --checkpoint external/HandMvNet/weights/handmvnet.ckpt --freeze-backbone
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parents[1]
for p in (_SRC, _SRC / "core", _SRC / "tools", _ROOT / "external" / "HandMvNet" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vision.handmvnet_adapter import (  # noqa: E402
    DEFAULT_REPO_ROOT, load_official_model, load_random_model_for_validation,
    resolve_torch_device)
from vision.handmvnet_dataset import (  # noqa: E402
    HandMvNetSessionDataset, make_synthetic_session)


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def collate(samples):
    """Default collate, mas devolve um dict de batch."""
    return torch.utils.data.default_collate(samples)


def compute_loss(out, labels, torch_module=torch):
    """Loss de heatmap + 2D + 3D relativo ao punho (o minimo do oficial)."""
    hm = torch_module.nn.functional.mse_loss(out["heatmap"], labels["heatmap"])
    j2 = torch_module.nn.functional.l1_loss(out["joints_crop_img"],
                                            labels["joints_crop_img"])
    j3 = torch_module.nn.functional.l1_loss(out["joints_cam"],
                                            labels["joints_cam"] / 1000.0)
    total = 10.0 * hm + 1.0 * j2 + 1000.0 * j3
    return total, {"heatmap": float(hm.detach()), "j2": float(j2.detach()),
                   "j3": float(j3.detach())}


def evaluate(model, loader, device, torch_module):
    model.eval()
    errs = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            out = model(batch["data"]["rgb"], batch["data"]["bboxes"],
                        batch["cam_params"])
            pred = out["joints_cam"] * 1000.0
            gt = batch["data"]["joints_cam"]
            errs.append(float(torch_module.linalg.norm(
                pred - gt, dim=-1).mean().item()))
    model.train()
    return float(np.mean(errs)) if errs else float("nan")


def make_loaders(data: Path, val: Path | None, batch_size: int,
                 torch_module):
    full = HandMvNetSessionDataset(data)
    if val is None:
        n = len(full)
        nval = max(1, int(round(0.2 * n)))
        idx = np.arange(n)
        rng = np.random.default_rng(42); rng.shuffle(idx)
        tr = HandMvNetSessionDataset(data, indices=sorted(idx[nval:].tolist()))
        va = HandMvNetSessionDataset(data, indices=sorted(idx[:nval].tolist()))
    else:
        tr = full
        va = HandMvNetSessionDataset(val)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                       collate_fn=collate, num_workers=0),
            DataLoader(va, batch_size=batch_size, shuffle=False,
                       collate_fn=collate, num_workers=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--data-val", type=Path, default=None)
    ap.add_argument("--saida", type=Path,
                    default=Path("outputs/visao/handmvnet_scratch"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backbone", default="resnet")
    ap.add_argument("--backbone-type", default="18")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = resolve_torch_device(args.device)
    args.saida.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        data = args.data or (args.saida / "synthetic_session")
        make_synthetic_session(data, n=12)
        args.data = data
        args.epochs = 1
        print("smoke dataset:", data)

    if args.data is None:
        raise SystemExit("passe --data ou --smoke")

    # Modelo oficial. Sem checkpoint => inicializacao aleatoria (treino do zero).
    if args.checkpoint is not None:
        loaded = load_official_model(
            DEFAULT_REPO_ROOT, args.checkpoint, device=device,
            num_views_override=3, adapt_num_views=True)
        print("fine-tune de checkpoint:", args.checkpoint)
    else:
        loaded = load_random_model_for_validation(
            DEFAULT_REPO_ROOT, num_views=3, image_size=256,
            backbone=args.backbone, backbone_type=args.backbone_type,
            use_gcn=False, device=device)
        print("treino do zero: arquitetura oficial, pesos aleatorios")
    model = loaded.model.to(device)
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if "backbone" in name:
                p.requires_grad_(False)

    train_loader, val_loader = make_loaders(
        args.data, args.data_val, args.batch_size, loaded.torch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    best = {"epoch": -1, "val_mpjpe_mm": float("inf"), "state": None}
    hist = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            out = model(batch["data"]["rgb"], batch["data"]["bboxes"],
                        batch["cam_params"])
            loss, parts = compute_loss(out, batch["data"], loaded.torch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(parts)
        val_mpjpe = evaluate(model, val_loader, device, loaded.torch)
        row = {"epoch": epoch, "loss": float(np.mean([x["heatmap"] for x in losses])),
               "val_mpjpe_mm": val_mpjpe}
        hist.append(row)
        print(f"ep {epoch:03d} loss_hm={row['loss']:.4f} "
              f"val_mpjpe_mm={val_mpjpe:.2f}", flush=True)
        if val_mpjpe < best["val_mpjpe_mm"]:
            best = {"epoch": epoch, "val_mpjpe_mm": val_mpjpe,
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        torch.save({"state_dict": best["state"], "val_mpjpe_mm": best["val_mpjpe_mm"],
                    "epoch": best["epoch"], "weights": "scratch_or_finetune"},
                   args.saida / "handmvnet_checkpoint.pt")
    (args.saida / "train_history.json").write_text(json.dumps({
        "history": hist, "best": {k: v for k, v in best.items() if k != "state"},
        "checkpoint": str(args.saida / "handmvnet_checkpoint.pt"),
    }, indent=2), encoding="utf-8")
    print("salvo em", args.saida)


if __name__ == "__main__":
    main()
