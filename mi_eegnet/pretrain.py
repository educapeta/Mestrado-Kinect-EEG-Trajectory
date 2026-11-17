#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-treino LOSO no BCIC-IV 2b (3 canais, 2 classes)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import carrega_bci2b, carrega_config
from .model import build_model, count_parameters
from .train_utils import train_model, set_seed


def escolhe_device(pedido="auto"):
    if pedido == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"
    return pedido


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("mi_eegnet/configs/default.yaml"))
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = carrega_config(args.config)
    set_seed(cfg["seed"])
    out = Path(args.saida or cfg["paths"]["out"]) / "pretrain"
    out.mkdir(parents=True, exist_ok=True)
    device = escolhe_device(args.device)
    print("device:", device, flush=True)

    ds = cfg["preprocessing"]
    X, y, subj = carrega_bci2b(Path(cfg["paths"]["bci2b"]),
                                fs=ds["fs"], l_freq=ds["l_freq"],
                                h_freq=ds["h_freq"],
                                window_sec=ds["window_sec"],
                                cue_offset=ds["bci2b_cue_offset_sec"])
    print("2b:", X.shape, "sujeitos:", np.unique(subj), flush=True)
    model0 = build_model(cfg)
    print("parametros:", count_parameters(model0), flush=True)

    hist_rows = []
    fold_scores = []
    best = {"ba": -1.0, "subject": None, "state": None}
    c = cfg["pretrain"]
    epochs = args.epochs or c["epochs"]
    for s in np.unique(subj):
        tr = subj != s
        va = subj == s
        model = build_model(cfg)
        res = train_model(
            X[tr], y[tr], X[va], y[va], model,
            epochs=epochs, patience=c["patience"], lr=c["lr"],
            weight_decay=c["weight_decay"], batch_size=c["batch_size"],
            augment=True, noise_sigma=c["noise_sigma"],
            channel_dropout=c["channel_dropout"], seed=cfg["seed"],
            device=device, verbose=True)
        # avaliacao no sujeito deixado de fora
        from .train_utils import predict_windows, soft_vote, trial_metrics
        probs = predict_windows(res.model, res.scaler, X[va], device)
        g = np.arange(int(va.sum()))
        _, pm, pred = soft_vote(probs, g)
        m = trial_metrics(y[va], pred)
        fold_scores.append(m["balanced_accuracy"])
        print(f"[sujeito {s:02d}] BA={m['balanced_accuracy']:.3f} "
              f"acc={m['accuracy']:.3f}", flush=True)
        for h in res.history:
            h = dict(h); h["subject"] = int(s); hist_rows.append(h)
        if m["balanced_accuracy"] > best["ba"]:
            best = {"ba": m["balanced_accuracy"], "subject": int(s),
                    "state": {k: v.detach().cpu() for k, v in
                              res.model.state_dict().items()},
                    "scaler_mean": res.scaler.mean,
                    "scaler_std": res.scaler.std}
        torch.save(res.model.state_dict(),
                   out / f"eegnet_fold_s{int(s):02d}.pt")

    pd.DataFrame(hist_rows).to_csv(out / "history.csv", index=False)
    if best["state"] is not None:
        torch.save({
            "state_dict": best["state"],
            "subject": best["subject"],
            "val_ba": best["ba"],
            "model_config": cfg["model"],
            "scaler_mean": best["scaler_mean"],
            "scaler_std": best["scaler_std"],
        }, out / "eegnet_pretrained.pt")
    (out / "summary.json").write_text(json.dumps({
        "mean_val_ba": float(np.mean(fold_scores)),
        "std_val_ba": float(np.std(fold_scores)),
        "folds": [float(v) for v in fold_scores],
        "best_subject": best["subject"],
        "best_val_ba": best["ba"],
        "params": count_parameters(build_model(cfg)),
    }, indent=2), encoding="utf-8")
    print(f"\nLOSO BA: {np.mean(fold_scores):.3f} ± "
          f"{np.std(fold_scores):.3f}", flush=True)
    print("Modelo:", out / "eegnet_pretrained.pt")


if __name__ == "__main__":
    main()
