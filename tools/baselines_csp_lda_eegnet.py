#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baselines 3.1 e 3.2 no piloto:

3.1 CSP + LDA (MNE)
3.2 EEGNet compacto (F1=8, D=2, F2=16) com early stopping

Protocolo:
- 2 s, passo 0,5 s, C3/C4/Cz;
- somente esquerda/direita;
- 5-fold agrupado por trial (todas as janelas do mesmo trial no mesmo fold);
- agregação das janelas por trial.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _finite(Z):
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

sys.path.insert(0, str(Path(__file__).parent))
import eegnet_faria_benchmark as ef
from validacao_me_transformer import carrega_fase

torch.set_num_threads(4)


def agrega(y, pred, groups):
    yt, yp = [], []
    for g in np.unique(groups):
        m = groups == g
        yt.append(int(y[m][0]))
        yp.append(int(np.mean(pred[m], axis=0).argmax()))
    return np.array(yt), np.array(yp)


def cv_csp(X, y, groups):
    yt, yp = [], []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in sgkf.split(X, y, groups):
        clf = Pipeline([
            ("csp", CSP(n_components=2, reg="ledoit_wolf", log=True)),
            ("finite", FunctionTransformer(_finite)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr",
                                               shrinkage="auto")),
        ])
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        a, b = agrega(y[te], p, groups[te])
        yt.extend(a); yp.extend(b)
    yt = np.array(yt); yp = np.array(yp)
    return (float(np.mean(yt == yp)),
            float(balanced_accuracy_score(yt, yp)),
            confusion_matrix(yt, yp, labels=[0, 1]))


def treina_eegnet(Xtr, ytr, gtr, Xte, yte, epochs=120, seed=42):
    torch.manual_seed(seed)
    model = ef.EEGNet(3, 256, 2, kernel_length=64,
                      F1=8, D=2, F2=16, dropout_rate=0.5).to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    # separa validação por trial dentro do treino
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    tr_idx, va_idx = next(sgkf.split(Xtr, ytr, groups=gtr))
    Xt = torch.from_numpy(Xtr[tr_idx])
    yt = torch.from_numpy(ytr[tr_idx]).long()
    Xv = torch.from_numpy(Xtr[va_idx])
    yv = torch.from_numpy(ytr[va_idx]).long()
    best = 1e9; state = None; patience = 15; bad = 0
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
            vloss = loss_fn(model(Xv), yv).item()
        if vloss < best - 1e-4:
            best = vloss; bad = 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if state:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(Xte)).softmax(1).numpy()
    return p


def cv_eegnet(X, y, groups):
    yt, yp = [], []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        print(f"  EEGNet fold {k+1}/5", flush=True)
        p = treina_eegnet(X[tr], y[tr], groups[tr], X[te], y[te])
        a, b = agrega(y[te], p, groups[te])
        yt.extend(a); yp.extend(b)
    yt = np.array(yt); yp = np.array(yp)
    return (float(np.mean(yt == yp)),
            float(balanced_accuracy_score(yt, yp)),
            confusion_matrix(yt, yp, labels=[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=Path("_baselines_csp_eegnet"))
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    res = {}
    for fase in ("ME", "MI"):
        print(f"\n[{fase}] carregando janelas...", flush=True)
        X, y, g = carrega_fase(Path("_piloto3c_memi"), fase)
        print("shape", X.shape, "classes", np.unique(y, return_counts=True),
              flush=True)
        print("[CSP+LDA]", flush=True)
        a1 = cv_csp(X, y, g)
        print(f"  acc={a1[0]:.3f} BA={a1[1]:.3f}\n{a1[2]}", flush=True)
        print("[EEGNet compacto]", flush=True)
        a2 = cv_eegnet(X, y, g)
        print(f"  acc={a2[0]:.3f} BA={a2[1]:.3f}\n{a2[2]}", flush=True)
        res[fase] = {
            "csp_lda": {"acc": a1[0], "ba": a1[1], "cm": a1[2].tolist()},
            "eegnet": {"acc": a2[0], "ba": a2[1], "cm": a2[2].tolist()},
        }
    (args.saida / "resultados.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    lines = ["# Baselines CSP+LDA e EEGNet compacto (CPU)", ""]
    for fase, r in res.items():
        lines += [f"## {fase}", "",
                  f"- CSP+LDA: acc={r['csp_lda']['acc']:.3f}, "
                  f"BA={r['csp_lda']['ba']:.3f}, "
                  f"cm={r['csp_lda']['cm']}",
                  f"- EEGNet: acc={r['eegnet']['acc']:.3f}, "
                  f"BA={r['eegnet']['ba']:.3f}, "
                  f"cm={r['eegnet']['cm']}", ""]
    (args.saida / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")
    print("Salvo em", args.saida)


if __name__ == "__main__":
    main()
