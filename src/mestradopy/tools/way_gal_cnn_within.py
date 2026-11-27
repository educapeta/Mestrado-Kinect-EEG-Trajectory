#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEGNet within-participant no WAY-EEG-GAL.

Espelha o protocolo SAND do repositorio: um modelo por participante, split por
SESSAO/SERIE (nao por trial), z-score com stats do treino, validacao interna
para early stopping e TESTE separado que nunca entra no treino, no scaler ou na
escolha de epoca.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from sklearn.metrics import mean_absolute_error, r2_score

_SRC = Path(__file__).resolve().parents[2]
for _p in (_SRC, _SRC / "mestradopy"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mestradopy.eeg.mi_eegnet.model import EEGNetConfigurable  # noqa: E402
from mestradopy.eeg.way_gal.loader import (  # noqa: E402
    WAYGALLoader, extract_me_windows, extract_targets)


def escolhe_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def pcc(y, p):
    out = []
    for j in range(y.shape[1]):
        if np.std(y[:, j]) > 0 and np.std(p[:, j]) > 0:
            out.append(float(np.corrcoef(y[:, j], p[:, j])[0, 1]))
        else:
            out.append(0.0)
    return out


def split_series(series, seed, frac_test=0.2, frac_val=0.15):
    """Split deterministico por serie/sessao."""
    uniq = np.unique(series)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(n * frac_test)))
    n_val = max(1, int(round(n * frac_val)))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    test = set(uniq[:n_test].tolist())
    val = set(uniq[n_test:n_test + n_val].tolist())
    train = [x for x in uniq if x not in test and x not in val]
    if not train:
        raise RuntimeError("split sem treino")
    return train, sorted(val), sorted(test)


def train_one(Xtr, ytr, Xva, yva, Xte, seed, device, epochs,
              batch_size=32, lr=1e-3, patience=5):
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd

    ym = ytr.mean(axis=0, keepdims=True)
    ys = ytr.std(axis=0, keepdims=True) + 1e-6
    ytr_n = (ytr - ym) / ys
    yva_n = (yva - ym) / ys

    torch.manual_seed(seed)
    model = EEGNetConfigurable(
        n_channels=32, n_times=Xtr.shape[2], n_classes=3,
        F1=8, D=2, F2=16, kernel_length=32, dropout=0.5,
        classifier_hidden=32)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Yt = torch.tensor(ytr_n, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    Yv = torch.tensor(yva_n, dtype=torch.float32, device=device)

    best = float("inf")
    best_state = None
    bad = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), Yv).item())
        if val_loss < best - 1e-6:
            best = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32,
                                  device=device)).cpu().numpy()
    pred = pred * ys + ym
    return pred, {"best_val_mse": best, "epochs_run": epoch + 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/raw/Datasets/WAY-EEG-GAL"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/eeg/way_gal/cnn_within"))
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--n-times", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="usar apenas P1 para testar o codigo")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = escolhe_device() if args.device == "auto" else args.device

    loader = WAYGALLoader(args.data_dir)
    data = loader.load_all_participants()
    if args.smoke:
        data = {"P1": data["P1"]}
    trials = [t for s in data.values() for ss in s.values() for t in ss]
    windows = extract_me_windows(trials, fs=500.0)
    targets = extract_targets(windows)
    n_use = int(round(args.window_sec * 500.0))
    pares = [(w, t) for w, t in zip(windows, targets)
             if w.eeg.shape[0] >= n_use]
    windows = [p[0] for p in pares]
    targets = [p[1] for p in pares]
    X = np.empty((len(windows), 32, args.n_times), dtype=np.float32)
    for i, w in enumerate(windows):
        X[i] = signal.resample(np.asarray(w.eeg[:n_use], np.float64).T,
                               args.n_times, axis=1)
    y = np.stack([t.wrist_pos_disp for t in targets]).astype(np.float32)
    groups = np.array([t.trial.participant for t in targets])
    series = np.array([t.trial.series for t in targets])

    per_participant = {}
    y_true_all, y_pred_all = [], []
    for p in sorted(np.unique(groups)):
        m = groups == p
        Xp, yp, sp = X[m], y[m], series[m]
        tr, va, te = split_series(sp, seed=42 + int(p))
        mtr = np.isin(sp, tr)
        mva = np.isin(sp, va)
        mte = np.isin(sp, te)
        print(f"P{int(p)}: train={mtr.sum()} val={mva.sum()} test={mte.sum()} "
              f"series test={te}", flush=True)
        pred, meta = train_one(
            Xp[mtr], yp[mtr], Xp[mva], yp[mva], Xp[mte],
            seed=42 + int(p), device=device, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr)
        yt = yp[mte]
        metrics = {
            "pcc_per_axis": pcc(yt, pred),
            "mae_per_axis": [float(mean_absolute_error(yt[:, j], pred[:, j]))
                             for j in range(3)],
            "r2_per_axis": [float(r2_score(yt[:, j], pred[:, j]))
                            for j in range(3)],
            "n_test": int(mte.sum()), "meta": meta,
            "series_test": [int(x) for x in te],
        }
        metrics["pcc_mean"] = float(np.mean(metrics["pcc_per_axis"]))
        metrics["mae_mean"] = float(np.mean(metrics["mae_per_axis"]))
        metrics["baseline_mae_mean"] = float(np.mean([
            mean_absolute_error(yt[:, j],
                                np.full_like(yt[:, j], yp[mtr][:, j].mean()))
            for j in range(3)]))
        per_participant[f"P{int(p)}"] = metrics
        y_true_all.append(yt); y_pred_all.append(pred)
        print(f"  PCC={metrics['pcc_per_axis']} mean={metrics['pcc_mean']:.3f} "
              f"baseline_MAE={metrics['baseline_mae_mean']:.3f}", flush=True)

    if y_true_all:
        yt = np.concatenate(y_true_all, axis=0)
        yp = np.concatenate(y_pred_all, axis=0)
        overall = {
            "pcc_per_axis": pcc(yt, yp),
            "mae_per_axis": [float(mean_absolute_error(yt[:, j], yp[:, j]))
                             for j in range(3)],
            "r2_per_axis": [float(r2_score(yt[:, j], yp[:, j]))
                            for j in range(3)],
            "n_trials": int(len(yt)),
        }
        overall["pcc_mean"] = float(np.mean(overall["pcc_per_axis"]))
        overall["mae_mean"] = float(np.mean(overall["mae_per_axis"]))
        overall["participants_pcc"] = [
            v["pcc_mean"] for v in per_participant.values()]
        overall["participants_pcc_mean"] = float(
            np.mean(overall["participants_pcc"]))
        overall["participants_pcc_std"] = float(
            np.std(overall["participants_pcc"]))
    else:
        overall = {}

    result = {
        "protocol": "within-participant; split por serie/sessao; "
                    "train/val/test separados; early stopping na val",
        "overall": overall,
        "per_participant": per_participant,
        "device": device, "epochs": args.epochs,
    }
    (args.output_dir / "cnn_within_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(overall, indent=2, ensure_ascii=False), flush=True)
    print("salvo em", args.output_dir / "cnn_within_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
