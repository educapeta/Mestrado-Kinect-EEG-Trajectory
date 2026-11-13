#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark BCI IV 2a: 22 canais vs 3 canais, 100 Hz, 4 s.

Protocolo controlado:
- mesma arquitetura CNN-Transformer, muda só a entrada (22 ou 3 canais);
- BCI IV 2a, sessões T, 100 Hz, janela MI 4 s, z-score por época/canal;
- validação separada do teste (sem escolher época pelo teste).

Avaliações:
- subject-independent: A01-A05 treino, A06 validação, A07-A09 teste;
- subject-dependent: por sujeito, runs 1-4 treino, run 5 validação, run 6 teste.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import mne
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

mne.set_log_level("ERROR")
FS = 100.0
CODE_BCI = {769: "esquerda", 770: "direita", 771: "pes", 772: "lingua"}
CLASSES = ["esquerda", "direita", "pes", "lingua"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (
        x.std(axis=-1, keepdims=True) + 1e-9)


def carrega_bci(pasta: Path, n_canais: int, cache: Path):
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if d["X"].shape[1] == n_canais:
            return d["X"], d["y"], d["subj"], d["run"]
    xs, ys, subjs, runs = [], [], [], []
    for f in sorted(pasta.glob("*T.gdf")):
        m = re.search(r"A(\d\d)T", f.name)
        if not m:
            continue
        subj = int(m.group(1))
        raw = mne.io.read_raw_gdf(str(f), preload=True, verbose=False)
        raw.filter(0.5, 40.0, verbose=False)
        raw.resample(FS, verbose=False)
        # 22 primeiros canais = EEG; EOG são os 3 últimos
        picks = raw.ch_names[:n_canais]
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        code_to_class = {}
        for code, cls in CODE_BCI.items():
            chave = str(code)
            if chave in event_id:
                code_to_class[int(event_id[chave])] = cls
        ev = events[np.isin(events[:, 2], list(code_to_class))]
        ep = mne.Epochs(raw, ev, event_id={cls: c for c, cls in
                                          code_to_class.items()},
                        tmin=0.0, tmax=4.0, baseline=None, preload=True,
                        picks=picks, verbose=False)
        x = zscore(ep.get_data()).astype(np.float32)
        y = np.array([C2I[code_to_class[int(c)]] for c in ep.events[:, 2]])
        sub = np.full(len(y), subj, np.int16)
        run = (np.arange(len(y)) // 48).astype(np.int16)
        xs.append(x); ys.append(y); subjs.append(sub); runs.append(run)
        print(f"[bci] {f.name}: {x.shape}")
    X = np.concatenate(xs); y = np.concatenate(ys)
    subj = np.concatenate(subjs); run = np.concatenate(runs)
    np.savez_compressed(cache, X=X, y=y, subj=subj, run=run)
    return X, y, subj, run


class CNNTransformer(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, d_model=64,
                 n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, 25, stride=2, padding=12),
            nn.BatchNorm1d(32), nn.ELU(),
            nn.Conv1d(32, 64, 15, stride=2, padding=7),
            nn.BatchNorm1d(64), nn.ELU(),
            nn.Conv1d(64, 128, 7, stride=2, padding=3),
            nn.BatchNorm1d(128), nn.ELU(),
        )
        self.proj = nn.Linear(128, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 64, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.cnn(x).transpose(1, 2)
        x = self.proj(x)
        if x.shape[1] <= self.pos.shape[1]:
            x = x + self.pos[:, :x.shape[1]]
        x = self.transformer(x).mean(dim=1)
        return self.head(x)


def treina(Xtr, ytr, Xval, yval, epochs=30, seed=42, lr=8e-4, batch=64):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNNTransformer(n_channels=Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=batch, shuffle=True)
    val = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
    melhor_ba = -1.0; melhor_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(val.tensors[0]).argmax(1).numpy()
        yv = val.tensors[1].numpy()
        recalls = [np.mean(pv[yv == c] == c) for c in np.unique(yv)]
        ba = float(np.mean(recalls))
        if ba > melhor_ba:
            melhor_ba = ba
            melhor_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"  ep {ep:03d} val_BA={ba:.3f} best={melhor_ba:.3f}")
    if melhor_state:
        model.load_state_dict(melhor_state)
    return model, melhor_ba


def avalia(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).argmax(1).numpy()
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(pred == y)), float(np.mean(recalls)), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bci", type=Path, default=Path("BCICIV_2a_gdf"))
    ap.add_argument("--canais", type=int, choices=[3, 22], default=22)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--saida", type=Path, default=Path("_bci2a_benchmark"))
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    cache = args.cache or Path(f"_bci2a_cache_{args.canais}ch_100hz.npz")
    X, y, subj, run = carrega_bci(args.bci, args.canais, cache)
    print("Dados:", X.shape)

    # subject-independent
    tr = np.isin(subj, [1, 2, 3, 4, 5])
    va = subj == 6
    te = np.isin(subj, [7, 8, 9])
    model_si, vba = treina(X[tr], y[tr], X[va], y[va], epochs=args.epochs)
    acc_si, ba_si, _ = avalia(model_si, X[te], y[te])
    print(f"[SI] val_BA={vba:.3f} test_acc={acc_si:.3f} test_BA={ba_si:.3f}")

    # subject-dependent: treina runs 0-3, valida run 4, testa run 5
    accs, bas = [], []
    for s in np.unique(subj):
        m = subj == s
        trs = m & (run <= 3)
        vas = m & (run == 4)
        tes = m & (run == 5)
        if trs.sum() < 20 or vas.sum() < 10 or tes.sum() < 10:
            continue
        mdl, _ = treina(X[trs], y[trs], X[vas], y[vas], epochs=args.epochs)
        a, b, _ = avalia(mdl, X[tes], y[tes])
        accs.append(a); bas.append(b)
        print(f"[S{s:02d}] acc={a:.3f} BA={b:.3f}")
    print(f"[SD] acc={np.mean(accs):.3f} BA={np.mean(bas):.3f}")

    (args.saida / f"benchmark_{args.canais}canais.md").write_text(
        f"# BCI IV 2a CNN-Transformer — {args.canais} canais, 100 Hz, 4 s\n\n"
        f"- Subject-independent: acc={acc_si:.3f}, BA={ba_si:.3f}\n"
        f"- Subject-dependent (runs 5-6 holdout): acc={np.mean(accs):.3f}, "
        f"BA={np.mean(bas):.3f}\n",
        encoding="utf-8")
    print("Resultado salvo em", args.saida)


if __name__ == "__main__":
    main()
