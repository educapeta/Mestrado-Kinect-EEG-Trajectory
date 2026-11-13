#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablação 3 canais × 3 classes + fine-tune ME -> teste MI.

BCI IV 2a:
  classes 3: esquerda, direita, outro (pés+língua)
  canais: C3,C4,Cz; 100 Hz; janela 4 s; z-score por época/canal.

Piloto:
  ME (marcadores 778/779) para fine-tune;
  MI (780/781) apenas para teste;
  separação por bloco: ME bloco 1 -> MI bloco 2 e vice-versa.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FS = 100.0
CLASSES_BCI = ["esquerda", "direita", "outro"]
CLASSES_PILOTO = ["esquerda", "direita", "repouso"]


def zscore(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (
        x.std(axis=-1, keepdims=True) + 1e-9)


class CNNTransformer(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, d_model=64,
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


def treina(model, Xtr, ytr, Xval, yval, epochs=30, lr=8e-4, batch=64):
    torch.manual_seed(42); np.random.seed(42)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=batch, shuffle=True)
    Xv = torch.from_numpy(Xval); yv = yval
    melhor = -1.0; melhor_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).argmax(1).numpy()
        recalls = [np.mean(pv[yv == c] == c) for c in np.unique(yv)]
        ba = float(np.mean(recalls))
        if ba > melhor:
            melhor = ba
            melhor_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"    ep {ep:03d} val_BA={ba:.3f} best={melhor:.3f}")
    if melhor_state:
        model.load_state_dict(melhor_state)
    return model, melhor


def avalia(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).argmax(1).numpy()
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(pred == y)), float(np.mean(recalls)), pred


def carrega_bci_3class(cache: Path):
    d = np.load(cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y4 = d["y"].astype(int)
    y3 = np.where(y4 <= 1, y4, 2)  # pés/língua -> outro
    return X, y3, d["subj"], d["run"]


def carrega_piloto(pasta: Path, fase: str):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import treino_baseline_3classes as tb
    f = sorted(glob.glob(str(pasta / "eeg_mi3c*.csv")))[-1]
    j = sorted(glob.glob(str(pasta / "eventos*.json")))[-1]
    eeg, markers, ts = tb.load_eeg_csv(Path(f))
    meta = json.load(open(j, encoding="utf-8"))
    eeg = eeg[:, [14, 15, 16]]  # C3, C4, Cz
    fs = float(meta.get("fs", 500.0))
    ini, fim = (778, 779) if fase == "ME" else (780, 781)
    ep, labels, tri, blocks = tb.extract_epochs_from_markers(
        eeg, markers, ts, fs, mi_start_code=ini, mi_end_code=fim,
        tmin=0.0, tmax=4.0, cue_codes=tb.CODE_TO_CLASS)
    xs = []
    for e in ep:
        e = np.asarray(e, np.float64)
        if fs != FS:
            from scipy import signal
            e = signal.resample(e, int(FS * 4), axis=0)
        xs.append(e)
    X = zscore(np.stack(xs).transpose(0, 2, 1)).astype(np.float32)
    y = np.array([CLASSES_PILOTO.index(l) for l in labels])
    return X, y, np.array(tri), np.array(blocks)


def resumo(y, pred):
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(y, pred):
        cm[int(a), int(b)] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bci", type=Path,
                    default=Path("_bci2a_cache_100hz.npz"))
    ap.add_argument("--piloto", type=Path, default=Path("_piloto3c_memi"))
    ap.add_argument("--saida", type=Path, default=Path("_ablation_me_mi"))
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    Xb, yb, subj, run = carrega_bci_3class(args.bci)
    print("BCI 3 classes:", Xb.shape, dict(zip(*np.unique(yb, return_counts=True))))

    # 1. benchmark BCI 3 canais x 3 classes
    tr = np.isin(subj, [1, 2, 3, 4, 5])
    va = subj == 6
    te = np.isin(subj, [7, 8, 9])
    modelo = CNNTransformer(3, 3)
    modelo, _ = treina(modelo, Xb[tr], yb[tr], Xb[va], yb[va],
                       epochs=args.epochs)
    acc, ba, _ = avalia(modelo, Xb[te], yb[te])
    print(f"[BCI 3c/3cls] test acc={acc:.3f} BA={ba:.3f}")

    # 2. pré-treino final em todo BCI (sem usar piloto)
    modelo_all = CNNTransformer(3, 3)
    modelo_all, _ = treina(modelo_all, Xb[tr | va], yb[tr | va],
                           Xb[va], yb[va], epochs=args.epochs)

    Xme, yme, _, bme = carrega_piloto(args.piloto, "ME")
    Xmi, ymi, _, bmi = carrega_piloto(args.piloto, "MI")
    print("ME:", Xme.shape, "MI:", Xmi.shape, "blocos ME/MI:",
          np.unique(bme), np.unique(bmi))

    resultados = []
    # 3. ME bloco 1 -> MI bloco 2; e ME bloco 2 -> MI bloco 1
    for btr, bte in [(0, 1), (1, 0)]:
        mtr = bme == btr
        mte = bmi == bte
        Xt, yt = Xme[mtr], yme[mtr]
        Xtest, ytest = Xmi[mte], ymi[mte]
        # validação interna 20% estratificada do ME de treino
        rng = np.random.default_rng(42)
        idx_val = []
        for c in np.unique(yt):
            ii = np.where(yt == c)[0]
            rng.shuffle(ii)
            idx_val.extend(ii[:max(1, len(ii) // 5)].tolist())
        idx_val = np.array(sorted(set(idx_val)))
        mask_tr = np.ones(len(yt), bool); mask_tr[idx_val] = False

        zero = copy.deepcopy(modelo_all)
        acc0, ba0, pred0 = avalia(zero, Xtest, ytest)

        ft = copy.deepcopy(modelo_all)
        ft, _ = treina(ft, Xt[mask_tr], yt[mask_tr],
                       Xt[idx_val], yt[idx_val], epochs=25, lr=5e-4)
        acc1, ba1, pred1 = avalia(ft, Xtest, ytest)
        print(f"[ME bloco {btr+1} -> MI bloco {bte+1}] "
              f"zero acc={acc0:.3f} BA={ba0:.3f} | "
              f"fine acc={acc1:.3f} BA={ba1:.3f}")
        resultados.append((btr, bte, acc0, ba0, acc1, ba1,
                           resumo(ytest, pred0), resumo(ytest, pred1)))

    linhas = ["# Ablação 3 canais × 3 classes + fine-tune ME → MI", "",
              f"- BCI 3c/3cls subject-independent: acc={acc:.3f}, BA={ba:.3f}",
              f"- ME: {Xme.shape} | MI: {Xmi.shape}", "",
              "## Fine-tune por bloco", ""]
    for btr, bte, a0, b0, a1, b1, cm0, cm1 in resultados:
        linhas += [f"### ME bloco {btr+1} → MI bloco {bte+1}", "",
                   f"- zero-shot: acc={a0:.3f}, BA={b0:.3f}",
                   f"- fine-tune: acc={a1:.3f}, BA={b1:.3f}",
                   "", "Confusão zero (linha=verdade, coluna=pred):", "",
                   "```", str(cm0), "```", "",
                   "Confusão fine-tune:", "", "```", str(cm1), "```", ""]
    (args.saida / "resultado_ablation_me_mi.md").write_text(
        "\n".join(linhas), encoding="utf-8")
    print("Relatório:", args.saida / "resultado_ablation_me_mi.md")


if __name__ == "__main__":
    main()
