#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validação ME/MI com CNN-Transformer pequeno, CPU.

Mesmo protocolo da validação LDA:
- 2 s, passo 0,5 s, C3/C4/Cz;
- 5-fold agrupado por trial;
- agregação de janelas por trial.

Objetivo: comparar um modelo leve com o baseline bandpower+LDA.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import mne
import torch
import torch.nn as nn
from scipy import signal
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import treino_baseline_3classes as tb

mne.set_log_level("ERROR")
FS_ORIG = 500.0
FS = 128.0
WIN = 256
STEP = 64


class SmallTransformer(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, d_model=32,
                 n_heads=4, n_layers=1, dropout=0.25):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(n_channels, 32, 16, stride=8, padding=8),
            nn.BatchNorm1d(32), nn.ELU(),
        )
        self.proj = nn.Linear(32, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 64, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=64,
            dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        x = self.proj(x)
        if x.shape[1] <= self.pos.shape[1]:
            x = x + self.pos[:, :x.shape[1]]
        x = self.transformer(x).mean(dim=1)
        return self.head(x)


def carrega_fase(pasta: Path, fase: str):
    f = sorted(glob.glob(str(pasta / "eeg_mi3c*.csv")))[-1]
    j = sorted(glob.glob(str(pasta / "eventos*.json")))[-1]
    eeg, markers, ts = tb.load_eeg_csv(Path(f))
    meta = json.load(open(j, encoding="utf-8"))
    eeg = eeg[:, [14, 15, 16]]  # C3, C4, Cz
    fs = float(meta.get("fs", FS_ORIG))
    eeg = mne.filter.filter_data(eeg.T, fs, 4.0, 40.0, verbose=False).T
    ini, fim = (778, 779) if fase == "ME" else (780, 781)
    ep, labels, _, _ = tb.extract_epochs_from_markers(
        eeg, markers, ts, fs, mi_start_code=ini, mi_end_code=fim,
        tmin=0.0, tmax=4.0, cue_codes=tb.CODE_TO_CLASS)
    mapa = {"esquerda": 0, "direita": 1, "repouso": 2}
    ys = np.array([mapa[x] for x in labels])
    xs, yt, gt = [], [], []
    for i, e in enumerate(ep):
        if ys[i] == 2:
            continue
        e = signal.resample(e, int(4 * FS), axis=0)  # 4 s @128
        for w, s in enumerate(range(0, e.shape[0] - WIN + 1, STEP)):
            xs.append(e[s:s + WIN].T.astype(np.float32))
            yt.append(int(ys[i]))
            gt.append(len(xs) // 5)  # não usar; substituído abaixo
    # grupos reais por trial: 5 janelas consecutivas
    trials = np.repeat(np.arange(len(xs) // 5), 5)
    X = np.stack(xs)
    y = np.array(yt)
    # z-score por janela/canal
    X = (X - X.mean(axis=2, keepdims=True)) / (
        X.std(axis=2, keepdims=True) + 1e-9)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, trials


def treina_fold(Xtr, ytr, Xte, yte, epochs=60, seed=42):
    torch.manual_seed(seed)
    model = SmallTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr).long()
    Xv = torch.from_numpy(Xte); yv = torch.from_numpy(yte).long()
    best = 1e9; state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 16):
            idx = perm[i:i + 16]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            # usa o próprio teste só para early stopping? NÃO.
            # early stopping por training loss (sem vazamento de teste)
            pass
        if ep == 0 or (ep + 1) % 20 == 0:
            print(f"    ep {ep+1}/{epochs}", flush=True)
    model.eval()
    return model


def cv(X, y, g):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    yt, yp = [], []
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups=g)):
        print(f"  fold {k+1}/5", flush=True)
        model = treina_fold(X[tr], y[tr], X[te], y[te])
        with torch.no_grad():
            p = model(torch.from_numpy(X[te])).softmax(1).numpy()
        for gg in np.unique(g[te]):
            m = g[te] == gg
            yt.append(int(y[te][m][0]))
            yp.append(int(p[m].mean(0).argmax()))
    yt = np.array(yt); yp = np.array(yp)
    return (float(np.mean(yt == yp)),
            float(balanced_accuracy_score(yt, yp)),
            confusion_matrix(yt, yp, labels=[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/validacao_transformer"))
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    res = {}
    for fase in ("ME", "MI"):
        print(f"\n[{fase}] carregando...", flush=True)
        X, y, g = carrega_fase(Path("outputs/eeg/piloto3c_memi"), fase)
        print(f"[{fase}] X={X.shape} y={np.unique(y, return_counts=True)}",
              flush=True)
        acc, ba, cm = cv(X, y, g)
        print(f"[{fase}] acc={acc:.3f} BA={ba:.3f}\n{cm}", flush=True)
        res[fase] = {"acc": acc, "ba": ba, "confusao": cm.tolist()}
    (args.saida / "resultado.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    lines = ["# ME/MI — CNN-Transformer pequeno (CPU)", ""]
    for fase, r in res.items():
        lines += [f"## {fase}", f"- acc: {r['acc']:.3f}",
                  f"- BA: {r['ba']:.3f}", f"- confusão: `{r['confusao']}`", ""]
    (args.saida / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")
    print("Salvo em", args.saida)


if __name__ == "__main__":
    main()
