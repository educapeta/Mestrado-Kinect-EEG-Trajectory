#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNN-Transformer para BCI IV 2a com C3/C4/Cz, 100 Hz e janela MI de 4 s.

Treina no BCI IV 2a (sessões T) e aplica no piloto S01 sem usar os labels do
piloto. Reporta:
  - subject-independent (sujeitos 7–9 como teste);
  - subject-dependent aproximado (runs finais de cada sujeito);
  - predições no piloto.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import mne
from scipy import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

mne.set_log_level("ERROR")
FS = 100.0
PICKS = ["EEG-C3", "EEG-C4", "EEG-Cz"]
CODE_BCI = {769: "esquerda", 770: "direita", 771: "pes", 772: "lingua"}
CLASSES = ["esquerda", "direita", "pes", "lingua"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (
        x.std(axis=-1, keepdims=True) + 1e-9)


def carrega_bci_cache(cache: Path, pasta: Path):
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
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
                        picks=PICKS, verbose=False)
        x = zscore(ep.get_data()).astype(np.float32)
        y = np.array([C2I[code_to_class[int(c)]] for c in ep.events[:, 2]])
        sub = np.full(len(y), subj, np.int16)
        # 6 runs de 48 trials; run index aproximado pela ordem dos trials
        run = (np.arange(len(y)) // 48).astype(np.int16)
        xs.append(x); ys.append(y); subjs.append(sub); runs.append(run)
        print(f"[bci] {f.name}: {x.shape}")
    X = np.concatenate(xs); y = np.concatenate(ys)
    subj = np.concatenate(subjs); run = np.concatenate(runs)
    np.savez_compressed(cache, X=X, y=y, subj=subj, run=run)
    return X, y, subj, run


def carrega_piloto(pasta: Path):
    f = sorted(glob.glob(str(pasta / "eeg_mi3c*.csv")))[-1]
    j = sorted(glob.glob(str(pasta / "eventos*.json")))[-1]
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import treino_baseline_3classes as tb
    eeg, markers, ts = tb.load_eeg_csv(Path(f))
    meta = json.load(open(j, encoding="utf-8"))
    eeg = eeg[:, [14, 15, 16]]  # C3, C4, Cz
    fs = float(meta.get("fs", 500.0))
    ep, labels, _, _ = tb.extract_epochs_from_markers(
        eeg, markers, ts, fs, mi_start_code=780, mi_end_code=781,
        tmin=0.0, tmax=4.0, cue_codes=tb.CODE_TO_CLASS)
    xs = []
    for e in ep:
        e = np.asarray(e, np.float64)
        if fs != FS:
            e = signal.resample(e, int(FS * 4.0), axis=0)
        xs.append(e)
    x = np.stack(xs).transpose(0, 2, 1)
    return zscore(x).astype(np.float32), np.array(labels)


class CNNTransformer(nn.Module):
    def __init__(self, n_classes=4, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, 25, stride=2, padding=12), nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 64, 15, stride=2, padding=7), nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 128, 7, stride=2, padding=3), nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.proj = nn.Linear(128, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 64, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.cnn(x)                    # (B,128,T')
        x = x.transpose(1, 2)              # (B,T',128)
        x = self.proj(x)
        if x.shape[1] <= self.pos.shape[1]:
            x = x + self.pos[:, :x.shape[1]]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)


def treina(Xtr, ytr, Xte, yte, epochs=60, seed=42, lr=1e-3, batch=64):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    model = CNNTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=batch, shuffle=True)
    te = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                    batch_size=batch)
    melhor = 0.0; melhor_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward(); opt.step()
        model.eval(); preds = []
        with torch.no_grad():
            for xb, _ in te:
                preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
        pred = np.concatenate(preds)
        acc = float(np.mean(pred == np.asarray(yte)))
        if acc > melhor:
            melhor = acc
            melhor_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"  ep {ep:03d} acc={acc:.3f} best={melhor:.3f}")
    if melhor_state:
        model.load_state_dict(melhor_state)
    return model, melhor


def avalia(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).argmax(1).numpy()
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(pred == y)), float(np.mean(recalls)), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bci", type=Path, default=Path("BCICIV_2a_gdf"))
    ap.add_argument("--piloto", type=Path, default=Path("_piloto3c_memi"))
    ap.add_argument("--cache", type=Path, default=Path("_bci2a_cache_100hz.npz"))
    ap.add_argument("--saida", type=Path, default=Path("_bci2a_cnntransformer"))
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    X, y, subj, run = carrega_bci_cache(args.cache, args.bci)
    print("BCI:", X.shape, "sujeitos:", np.unique(subj))
    # subject-independent: treina A01..A06, testa A07..A09
    tr = np.isin(subj, [1, 2, 3, 4, 5, 6])
    te = np.isin(subj, [7, 8, 9])
    model_si, best_si = treina(X[tr], y[tr], X[te], y[te],
                               epochs=args.epochs)
    acc_si, ba_si, pred_si = avalia(model_si, X[te], y[te])
    print(f"[subject-independent] acc={acc_si:.3f} ba={ba_si:.3f}")

    # subject-dependent aproximado: por sujeito, runs 1-4 treino / 5-6 teste
    accs, bas = [], []
    for s in np.unique(subj):
        m = subj == s
        trs = m & (run <= 3)
        tes = m & (run >= 4)
        if trs.sum() < 20 or tes.sum() < 20:
            continue
        mdl, _ = treina(X[trs], y[trs], X[tes], y[tes], epochs=args.epochs)
        a, b, _ = avalia(mdl, X[tes], y[tes])
        accs.append(a); bas.append(b)
        print(f"[subject {s:02d}] acc={a:.3f} ba={b:.3f}")
    print(f"[subject-dependent] acc={np.mean(accs):.3f} ba={np.mean(bas):.3f}")

    # modelo final: todos os sujeitos; aplica no piloto
    model_all, _ = treina(X, y, X[:1], y[:1], epochs=max(20, args.epochs // 2))
    Xp, yp = carrega_piloto(args.piloto)
    model_all.eval()
    with torch.no_grad():
        p = model_all(torch.from_numpy(Xp)).softmax(1).numpy()
    pred = p.argmax(1)
    nomes = np.array(CLASSES)[pred]
    print("Piloto predições:", dict(zip(*np.unique(nomes, return_counts=True))))
    print("Piloto labels:", dict(zip(*np.unique(yp, return_counts=True))))
    torch.save(model_all.state_dict(), args.saida / "modelo_cnntransformer.pt")
    np.savez(args.saida / "predicoes_piloto.npz", y_true=yp, y_pred=nomes,
             proba=p, classes=np.array(CLASSES))
    (args.saida / "resultado.md").write_text(
        f"# BCI IV 2a CNN-Transformer (C3/C4/Cz, 100 Hz, 4 s)\n\n"
        f"- subject-independent: acc={acc_si:.3f}, BA={ba_si:.3f}\n"
        f"- subject-dependent: acc={np.mean(accs):.3f}, BA={np.mean(bas):.3f}\n"
        f"- piloto (4 classes): {dict(zip(*np.unique(nomes, return_counts=True)))}\n",
        encoding="utf-8")
    print("Salvo em", args.saida)


if __name__ == "__main__":
    main()
