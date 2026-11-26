#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Treino neural de regressao de pose no WAY-EEG-GAL.

Modelo: EEGNet configuravel (32 canais) + cabeca de regressao 3D.
Validacao: GroupKFold por participante. Nenhum trial do participante de teste
entra no treino, no scaler ou na escolha de epoca.
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
from sklearn.model_selection import GroupKFold, KFold

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


def resample_windows(windows, fs, n_times):
    X = np.empty((len(windows), 32, n_times), dtype=np.float32)
    for i, w in enumerate(windows):
        x = np.asarray(w.eeg, np.float64).T  # (32, N)
        X[i] = signal.resample(x, n_times, axis=1)
    return X


def pcc(y, p):
    out = []
    for j in range(y.shape[1]):
        if np.std(y[:, j]) > 0 and np.std(p[:, j]) > 0:
            out.append(float(np.corrcoef(y[:, j], p[:, j])[0, 1]))
        else:
            out.append(0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/raw/Datasets/WAY-EEG-GAL"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/eeg/way_gal/cnn"))
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--n-times", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="usa apenas P1 S1 para validar o codigo")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = escolhe_device() if args.device == "auto" else args.device
    torch.manual_seed(42)
    np.random.seed(42)

    print("Carregando WAY-EEG-GAL...", flush=True)
    loader = WAYGALLoader(args.data_dir)
    if args.smoke:
        data = {"P1": {1: loader.load_participant_series("P1", 1)}}
    else:
        data = loader.load_all_participants()
    trials = [t for s in data.values() for ss in s.values() for t in ss]
    windows = extract_me_windows(trials, fs=500.0)
    targets = extract_targets(windows)
    n_use = int(round(args.window_sec * 500.0))
    pares = [(w, t) for w, t in zip(windows, targets)
             if w.eeg.shape[0] >= n_use]
    windows = [p[0] for p in pares]
    targets = [p[1] for p in pares]
    # Reduz a janela ao inicio do movimento (nao usa hold/release no EEG).
    for w in windows:
        w.eeg = w.eeg[:n_use]
    y = np.stack([t.wrist_pos_disp for t in targets]).astype(np.float32)
    groups = np.array([t.trial.participant for t in targets])
    print(f"trials={len(windows)} X={len(windows)}x32x{n_use} y={y.shape}", flush=True)

    X = resample_windows(windows, 500.0, args.n_times)
    splitter = (KFold(n_splits=2, shuffle=True, random_state=42)
                if args.smoke else GroupKFold(n_splits=5))
    y_true = np.zeros_like(y)
    y_pred = np.zeros_like(y)
    fold_meta = []

    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        Xtr_raw, Xte_raw = X[tr], X[te]
        mu = Xtr_raw.mean(axis=(0, 2), keepdims=True)
        sd = Xtr_raw.std(axis=(0, 2), keepdims=True) + 1e-6
        Xtr = (Xtr_raw - mu) / sd
        Xte = (Xte_raw - mu) / sd

        ytr = y[tr]
        ym = ytr.mean(axis=0, keepdims=True)
        ys = ytr.std(axis=0, keepdims=True) + 1e-6
        ytr_n = (ytr - ym) / ys

        model = EEGNetConfigurable(
            n_channels=32, n_times=args.n_times, n_classes=3,
            F1=8, D=2, F2=16, kernel_length=32, dropout=0.5,
            classifier_hidden=32)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
        Yt = torch.tensor(ytr_n, dtype=torch.float32, device=device)
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(Xt), device=device)
            total = 0.0
            for i in range(0, len(Xt), args.batch_size):
                idx = perm[i:i + args.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx]), Yt[idx])
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(idx)
            if (epoch + 1) % 5 == 0:
                print(f"  fold {fold+1} epoch {epoch+1}/{args.epochs} "
                      f"loss={total/len(Xt):.4f}", flush=True)

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32,
                                      device=device)).cpu().numpy()
        pred = pred * ys + ym
        y_pred[te] = pred
        y_true[te] = y[te]
        fold_meta.append({
            "fold": fold + 1,
            "test_participants": sorted(map(int, np.unique(groups[te]))),
            "n_train": int(len(tr)), "n_test": int(len(te)),
        })
        print(f"fold {fold+1}: test={sorted(map(int, np.unique(groups[te])))}",
              flush=True)

    pccs = pcc(y_true, y_pred)
    overall = {
        "pcc_per_axis": pccs,
        "pcc_mean": float(np.mean(pccs)),
        "mae_per_axis": [float(mean_absolute_error(y_true[:, j], y_pred[:, j]))
                         for j in range(3)],
        "mae_mean": float(np.mean([
            mean_absolute_error(y_true[:, j], y_pred[:, j]) for j in range(3)])),
        "r2_per_axis": [float(r2_score(y_true[:, j], y_pred[:, j]))
                        for j in range(3)],
        "n_trials": int(len(y)),
        "n_features": int(args.n_times * 32),
        "epochs": int(args.epochs),
        "device": device,
    }
    # baseline: media do treino em cada fold
    base_pred = np.zeros_like(y)
    for tr, te in splitter.split(X, y, groups):
        base_pred[te] = y[tr].mean(axis=0, keepdims=True)
    base_pcc = pcc(y, base_pred)
    overall["baseline_pcc_mean"] = float(np.mean(base_pcc))
    overall["baseline_mae_mean"] = float(np.mean([
        mean_absolute_error(y[:, j], base_pred[:, j]) for j in range(3)]))

    result = {"overall": overall, "folds": fold_meta,
              "protocol": "GroupKFold por participante; scaler fit no treino; "
                          "sem escolha de epoca no teste"}
    (args.output_dir / "cnn_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(overall, indent=2, ensure_ascii=False), flush=True)
    print("salvo em", args.output_dir / "cnn_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
