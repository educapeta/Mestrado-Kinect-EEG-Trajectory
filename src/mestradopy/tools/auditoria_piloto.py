#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auditoria independente do piloto (120 trials, folds canônicos).

Roda:
  - features simples: log-variância por canal e bandpower mu/beta;
  - CSP+LDA (referência do projeto);
  - BA por trial, com agregação das 5 janelas;
  - permutação de labels no nível do trial.
Salva JSON/Markdown em outputs/eeg/auditoria_piloto.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mne.decoding import CSP

_SRC = Path(__file__).resolve().parents[1]
for _p in (_SRC, _SRC / "eeg", _SRC / "core", _SRC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eeg.mi_eegnet.data import carrega_own  # noqa: E402

FS = 128
BANDS = {"mu": (8, 13), "beta": (13, 30)}
OUT = Path("outputs/eeg/auditoria_piloto")


def _finite(z):
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _pipe():
    return Pipeline([
        ("csp", CSP(n_components=2, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(_finite)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def carrega():
    Xs, ys, gs, ses = [], [], [], []
    off = 0
    for si, p in enumerate([Path("outputs/eeg/piloto_me_mi_s01"),
                            Path("outputs/eeg/piloto_me_mi_s02")]):
        X, y, g, _ = carrega_own(p, fs=FS, l_freq=4.0, h_freq=40.0,
                                 window_sec=2.0, step_sec=0.5,
                                 trial_sec=4.0, fase="MI")
        Xs.append(X); ys.append(y); gs.append(g + off)
        ses.append(np.full(len(y), si)); off += int(g.max()) + 1
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def feats(X: np.ndarray) -> np.ndarray:
    """(N,3,256) -> log-var (3) + bandpower mu/beta (6)."""
    var = np.log(np.var(X, axis=2) + 1e-9)
    f, p = welch(X, fs=FS, nperseg=min(128, X.shape[2]), axis=2)
    logbp = []
    for lo, hi in BANDS.values():
        m = (f >= lo) & (f <= hi)
        logbp.append(np.log(p[:, :, m].mean(axis=2) + 1e-9))
    return np.concatenate([var] + logbp, axis=1)


def avalia(X, y, g, folds, tipo):
    yt, yp = [], []
    fold_ba = []
    for f in folds:
        tr = np.isin(g, f["train_trials"]); te = np.isin(g, f["test_trials"])
        if tipo == "csp":
            m = _pipe(); m.fit(X[tr], y[tr]); prob = m.predict_proba(X[te])
            pred = prob.argmax(axis=1)
        else:
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            m.fit(feats(X[tr]), y[tr]); pred = m.predict(feats(X[te]))
        a, b = [], []
        for tid in np.unique(g[te]):
            q = g[te] == tid
            a.append(int(y[te][q][0])); b.append(int(np.bincount(pred[q]).argmax()))
        fold_ba.append(balanced_accuracy_score(a, b))
        yt.extend(a); yp.extend(b)
    return {"fold_ba": [round(float(x), 4) for x in fold_ba],
            "mean_ba": round(float(np.mean(fold_ba)), 4),
            "std_ba": round(float(np.std(fold_ba, ddof=1)), 4),
            "n_trials": len(yt),
            "confusao": confusion_matrix(yt, yp, labels=[0, 1]).tolist()}


def permuta(X, y, g, folds, n=200, seed=42):
    rng = np.random.default_rng(seed)
    obs = avalia(X, y, g, folds, "feat")["mean_ba"]
    null = []
    for _ in range(n):
        yp = rng.permutation(y)
        null.append(avalia(X, yp, g, folds, "feat")["mean_ba"])
    null = np.asarray(null)
    p = (1 + np.sum(null >= obs)) / (1 + n)
    return {"observado": obs, "n_perm": n,
            "nulo_mediana": round(float(np.median(null)), 4),
            "nulo_p95": round(float(np.percentile(null, 95)), 4),
            "p": round(float(p), 4)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, g = carrega()
    folds = json.loads(Path("docs/validation/folds_piloto_groupkf.json")
                       .read_text(encoding="utf-8"))["folds"]
    print("X", X.shape, "trials", len(np.unique(g)))
    res = {"simple": avalia(X, y, g, folds, "feat"),
           "csp": avalia(X, y, g, folds, "csp")}
    res["permutacao_simples"] = permuta(X, y, g, folds, n=200)
    # artefato: std por trial e classe
    std = np.array([np.median(np.std(X[g == tid], axis=2)) for tid in np.unique(g)])
    yy = np.array([y[g == tid][0] for tid in np.unique(g)])
    res["artefato"] = {"std_mediano_classe0": round(float(np.median(std[yy == 0])), 3),
                       "std_mediano_classe1": round(float(np.median(std[yy == 1])), 3)}
    (OUT / "resultado.json").write_text(json.dumps(res, indent=2,
                                                   ensure_ascii=False), encoding="utf-8")
    linhas = ["# Auditoria independente do piloto", "",
              f"- janelas: {X.shape[0]}; trials: {len(np.unique(g))}",
              "", "| método | BA trial | dp | folds | confusão |",
              "|---|---:|---:|---|---|"]
    for nome, r in (("features simples", res["simple"]), ("CSP+LDA", res["csp"])):
        linhas.append(f"| {nome} | {r['mean_ba']:.3f} | {r['std_ba']:.3f} | "
                      f"{r['fold_ba']} | {r['confusao']} |")
    p = res["permutacao_simples"]
    linhas += ["", "## Permutação (features simples)", "",
               f"- observado: {p['observado']:.3f}",
               f"- nulo mediana: {p['nulo_mediana']:.3f}",
               f"- nulo p95: {p['nulo_p95']:.3f}",
               f"- p unilateral: {p['p']:.4f}",
               "", "## Artefato", "",
               f"- std mediano classe 0: {res['artefato']['std_mediano_classe0']}",
               f"- std mediano classe 1: {res['artefato']['std_mediano_classe1']}"]
    (OUT / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
