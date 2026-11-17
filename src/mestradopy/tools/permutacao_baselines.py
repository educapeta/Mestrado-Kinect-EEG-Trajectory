#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Permutação dos baselines CSP+LDA e EEGNet compacto (CPU).

- Labels permutados no nível do trial;
- mesmo 5-fold agrupado por trial em todas as permutações;
- CSP+LDA: 1.000 permutações;
- EEGNet: número menor (retreina a rede) — configurável.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mne.decoding import CSP

sys.path.insert(0, str(Path(__file__).parent))
from validacao_me_transformer import carrega_fase
from baselines_csp_lda_eegnet import treina_eegnet, agrega


def _finite(Z):
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def csp_pred(Xtr, ytr, Xte):
    clf = Pipeline([
        ("csp", CSP(n_components=2, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(_finite)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr",
                                           shrinkage="auto")),
    ])
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)


def eegnet_pred(Xtr, ytr, gtr, Xte, yte):
    p = treina_eegnet(Xtr, ytr, gtr, Xte, yte)
    return p


def permutacao(X, y, g, n_perm, seed=123):
    rng = np.random.default_rng(seed)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y, g))
    y_trial = y[::5].copy()  # 5 janelas por trial, ordem preservada
    n_trials = len(y_trial)
    bas = []
    for p in range(n_perm):
        yp_trial = y_trial[rng.permutation(n_trials)]
        yp = np.repeat(yp_trial, 5)
        yt, pp = [], []
        for tr, te in folds:
            prob = csp_pred(X[tr], yp[tr], X[te])
            a, b = agrega(yp[te], prob, g[te])
            yt.extend(a); pp.extend(b)
        bas.append(float(balanced_accuracy_score(yt, pp)))
        if p == 0 or (p + 1) % 100 == 0:
            print(f"    perm {p+1}/{n_perm} BA={bas[-1]:.3f}", flush=True)
    return np.array(bas)


def permutacao_eegnet(X, y, g, n_perm, seed=123):
    rng = np.random.default_rng(seed)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y, g))
    y_trial = y[::5].copy()
    n_trials = len(y_trial)
    bas = []
    for p in range(n_perm):
        yp_trial = y_trial[rng.permutation(n_trials)]
        yp = np.repeat(yp_trial, 5)
        yt, pp = [], []
        for tr, te in folds:
            prob = eegnet_pred(X[tr], yp[tr], g[tr], X[te], yp[te])
            a, b = agrega(yp[te], prob, g[te])
            yt.extend(a); pp.extend(b)
        bas.append(float(balanced_accuracy_score(yt, pp)))
        if p == 0 or (p + 1) % 10 == 0:
            print(f"    EEGNet perm {p+1}/{n_perm} BA={bas[-1]:.3f}",
                  flush=True)
    return np.array(bas)


def cv_observed(X, y, g, modelo, n_perm, seed=123):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    yt, yp = [], []
    for tr, te in sgkf.split(X, y, g):
        if modelo == "csp":
            prob = csp_pred(X[tr], y[tr], X[te])
        else:
            prob = eegnet_pred(X[tr], y[tr], g[tr], X[te], y[te])
        a, b = agrega(y[te], prob, g[te])
        yt.extend(a); yp.extend(b)
    ba = float(balanced_accuracy_score(yt, yp))
    print(f"  observado {modelo}: BA={ba:.3f}", flush=True)
    perm = (permutacao if modelo == "csp" else permutacao_eegnet)(
        X, y, g, n_perm, seed)
    p = (1 + np.sum(perm >= ba)) / (1 + len(perm))
    return ba, perm, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/permutacao_baselines"))
    ap.add_argument("--perm-csp", type=int, default=1000)
    ap.add_argument("--perm-eegnet", type=int, default=100,
                    help="EEGNet re-treina a rede; use menos permutacoes")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    res = {}
    for fase in ("ME", "MI"):
        print(f"\n[{fase}] carregando...", flush=True)
        X, y, g = carrega_fase(Path("outputs/eeg/piloto3c_memi"), fase)
        print("shape", X.shape, flush=True)
        out = {}
        for modelo, nperm in (("csp", args.perm_csp),
                              ("eegnet", args.perm_eegnet)):
            ba, perm, p = cv_observed(X, y, g, modelo, nperm)
            out[modelo] = {
                "ba": ba,
                "perm_mean": float(perm.mean()),
                "perm_p95": float(np.percentile(perm, 95)),
                "p": float(p),
                "n_perm": int(nperm),
            }
            print(f"  {modelo}: BA={ba:.3f} p={p:.4f} "
                  f"perm_mean={perm.mean():.3f} p95={np.percentile(perm,95):.3f}",
                  flush=True)
        res[fase] = out
    (args.saida / "resultados.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    lines = ["# Permutação — CSP+LDA e EEGNet compacto", ""]
    for fase, r in res.items():
        lines += [f"## {fase}", ""]
        for m, v in r.items():
            lines += [f"### {m}", "",
                      f"- BA observado: {v['ba']:.3f}",
                      f"- permutações: {v['n_perm']}",
                      f"- perm média: {v['perm_mean']:.3f}",
                      f"- perm p95: {v['perm_p95']:.3f}",
                      f"- p: {v['p']:.4f}", ""]
    (args.saida / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")
    print("Salvo em", args.saida)


if __name__ == "__main__":
    main()
