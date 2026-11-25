#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tuning focado do EEGCCT no BCIC IV-2b.

Não é uma busca ampla: testa poucas configurações de modelo/treino, salva
resultados incrementalmente e usa validação interna (nunca o teste) para
early stopping.

Uso:
  .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\bench_eegcct_tuning.py `
      --device xpu --epochs 50 --output outputs/eeg/eegcct_tuning
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _SRC, _SRC / "eeg", _SRC / "tools", _SRC / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eeg.eegcct.model import (EEGCCTConfig, count_parameters)  # noqa: E402
from eeg.eegcct.train_utils import (EarlyStopping, cross_session_bci2b_detailed,  # noqa: E402
                                    segmentation_augmentation, set_seed,
                                    train_epoch, validate)
from sklearn.metrics import balanced_accuracy_score, confusion_matrix  # noqa: E402


def make_configs() -> Dict[str, dict]:
    return {
        "base": {
            "model": dict(),
            "train": dict(lr=3e-5, weight_decay=1e-4, batch_size=16,
                          augmentation=True),
        },
        "lr_1e-4": {
            "model": dict(),
            "train": dict(lr=1e-4, weight_decay=1e-4, batch_size=16,
                          augmentation=True),
        },
        "lr_3e-4": {
            "model": dict(),
            "train": dict(lr=3e-4, weight_decay=1e-4, batch_size=16,
                          augmentation=True),
        },
        "no_aug": {
            "model": dict(),
            "train": dict(lr=1e-4, weight_decay=1e-4, batch_size=16,
                          augmentation=False),
        },
        "small": {
            "model": dict(dim=32, num_layers=2, num_heads=4, mlp_size=32,
                          in_planes=32),
            "train": dict(lr=1e-4, weight_decay=1e-3, batch_size=16,
                          augmentation=True),
        },
        "large": {
            "model": dict(dim=96, num_layers=4, num_heads=8, mlp_size=128,
                          in_planes=96),
            "train": dict(lr=3e-5, weight_decay=1e-4, batch_size=16,
                          augmentation=True),
        },
    }


def evaluate(model, X, y, device) -> Tuple[float, float, list]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device).float()).cpu().numpy()
    pred = logits.argmax(axis=1)
    ba = float(balanced_accuracy_score(y, pred))
    acc = float((pred == y).mean())
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return ba, acc, cm


def run_experiment(exp: Dict, model_cfg: dict, train_cfg: dict,
                   epochs: int, seed: int, device: str) -> Dict:
    set_seed(seed)
    Xtr, ytr = exp["X_train"], exp["y_train"]
    Xte, yte = exp["X_test"], exp["y_test"]
    # validação interna estratificada (20%)
    idx = np.arange(len(ytr))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=seed,
                                      stratify=ytr)
    ds_tr = TensorDataset(torch.from_numpy(Xtr[tr_idx]).float(),
                          torch.from_numpy(ytr[tr_idx]).long())
    ds_va = TensorDataset(torch.from_numpy(Xtr[va_idx]).float(),
                          torch.from_numpy(ytr[va_idx]).long())
    dl_tr = DataLoader(ds_tr, batch_size=train_cfg["batch_size"], shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=train_cfg["batch_size"], shuffle=False)

    cfg = EEGCCTConfig(**model_cfg)
    model = cfg.create_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"])
    crit = torch.nn.CrossEntropyLoss()
    es = EarlyStopping(patience=10, min_delta=1e-4)
    best = None
    for ep in range(epochs):
        aug = segmentation_augmentation if train_cfg["augmentation"] else None
        train_epoch(model, dl_tr, opt, crit, device, augmentation_fn=aug,
                    X_aug=Xtr[tr_idx], y_aug=ytr[tr_idx],
                    batch_size=train_cfg["batch_size"])
        _, val_acc = validate(model, dl_va, crit, device)
        if best is None or val_acc > best:
            best = val_acc
        if es(-val_acc, model):
            break
    es.restore_best(model)
    ba, acc, cm = evaluate(model, Xte, yte, device)
    return {"subject": int(exp["subject"]),
            "test_session": int(exp["test_session"]),
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "val_acc_best": round(float(best), 4),
            "test_ba": round(ba, 4), "test_acc": round(acc, 4),
            "confusion": cm,
            "params": count_parameters(model),
            "epochs_ran": ep + 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/raw/Datasets/BCICIV_2b_gdf"))
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/eeg/eegcct_tuning"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="base,lr_1e-4,lr_3e-4,no_aug,small,large")
    ap.add_argument("--quick", action="store_true",
                    help="3 sujeitos, 2 epochs, para smoke test")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.epochs = min(args.epochs, 2)
    dev = torch.device(args.device)
    print("device", dev, "epochs", args.epochs, flush=True)
    experiments = cross_session_bci2b_detailed(args.data_dir, fs=128)
    if args.quick:
        experiments = [e for e in experiments if e["subject"] in (1, 4)]
    print(f"experimentos: {len(experiments)}", flush=True)
    configs = make_configs()
    wanted = [x.strip() for x in args.configs.split(",") if x.strip()]
    all_res = {}
    for name in wanted:
        if name not in configs:
            print("config desconhecida", name); continue
        mc = configs[name]["model"]; tc = configs[name]["train"]
        print(f"\n=== {name} ===", flush=True)
        rows = []
        for i, exp in enumerate(experiments, 1):
            r = run_experiment(exp, mc, tc, args.epochs, args.seed, dev)
            r["config"] = name
            rows.append(r)
            print(f"  {i}/{len(experiments)} subj={r['subject']} "
                  f"test={r['test_session']} BA={r['test_ba']:.3f}", flush=True)
        bas = [r["test_ba"] for r in rows]
        out = {"config": name, "n": len(rows),
               "mean_ba": round(float(np.mean(bas)), 4),
               "std_ba": round(float(np.std(bas, ddof=1)), 4),
               "rows": rows}
        all_res[name] = out
        (args.output / f"config_{name}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.output / "resumo.json").write_text(
            json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {name}: BA={out['mean_ba']:.4f} ± {out['std_ba']:.4f}", flush=True)
    print("done", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
