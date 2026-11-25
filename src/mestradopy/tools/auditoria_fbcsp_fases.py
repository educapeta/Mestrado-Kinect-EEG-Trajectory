#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FBCSP nas fases baseline, ME e MI do piloto.

Objetivo: verificar se o sinal que o FBCSP aprende vem da MI ou de
artefato/estratégia de ME/baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

_SRC = Path(__file__).resolve().parents[1]
for _p in (_SRC, _SRC / "eeg", _SRC / "tools", _SRC / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import feedback_adaptativo as fa  # noqa: E402
from eeg.fbcsp.fbcsp import FBCSP  # noqa: E402

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
         (24, 28), (28, 32), (32, 36), (36, 40)]
FS = 128
OUT = Path("outputs/eeg/auditoria_piloto")


def janelas_2s(X, label, group, out):
    n = int(2 * FS)
    if X.shape[1] < n:
        X = np.pad(X, ((0, 0), (0, n - X.shape[1])))
    X = X[:, :4 * FS]
    for start in range(0, X.shape[1] - n + 1, int(0.5 * FS)):
        out["X"].append(X[:, start:start + n])
        out["y"].append(int(label))
        out["g"].append(int(group))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    trials = []
    for p in [Path("outputs/eeg/piloto_me_mi_s01"),
              Path("outputs/eeg/piloto_me_mi_s02")]:
        t, _ = fa.carrega_sessao_eeg(p)
        trials.extend(t)
    folds = json.loads(Path("docs/validation/folds_piloto_groupkf.json")
                       .read_text(encoding="utf-8"))["folds"]
    res = {}
    for fase in ("base", "me", "mi"):
        d = {"X": [], "y": [], "g": []}
        for i, t in enumerate(trials):
            X = np.asarray(t["X_base" if fase == "base" else
                             ("X_me" if fase == "me" else "X_mi")], float)
            if fase == "base":
                # baseline tem 1 s; um único recorte
                d["X"].append(X[:, :FS]); d["y"].append(int(t["classe"])); d["g"].append(i)
            else:
                janelas_2s(X, t["classe"], i, d)
        X = np.stack(d["X"]); y = np.asarray(d["y"]); g = np.asarray(d["g"])
        folds_ = []
        for f in folds:
            tr = np.isin(g, f["train_trials"]); te = np.isin(g, f["test_trials"])
            if not tr.any() or not te.any():
                continue
            m = FBCSP(bands=BANDS, fs=FS, n_csp_components=4,
                      random_state=42).fit(X[tr], y[tr])
            pred = m.predict(X[te])
            a, b = [], []
            for tid in np.unique(g[te]):
                q = g[te] == tid
                a.append(int(y[te][q][0]))
                b.append(int(np.bincount(pred[q]).argmax()))
            folds_.append(balanced_accuracy_score(a, b))
        res[fase] = {"n_windows": int(len(y)), "n_trials": len(trials),
                     "fold_ba": [round(float(x), 4) for x in folds_],
                     "mean_ba": round(float(np.mean(folds_)), 4),
                     "std_ba": round(float(np.std(folds_, ddof=1)), 4)}
        print(fase, res[fase], flush=True)
    (OUT / "fbcsp_fases.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    linhas = ["# FBCSP por fase — baseline, ME e MI", "",
              "| fase | janelas | BA trial | dp | folds |", "|---|---:|---:|---:|---|"]
    for fase, r in res.items():
        linhas.append(f"| {fase} | {r['n_windows']} | {r['mean_ba']:.4f} | "
                      f"{r['std_ba']:.4f} | {r['fold_ba']} |")
    linhas += ["", "Baseline usa 1 s (janela disponível); ME e MI usam 2 s."]
    (OUT / "fbcsp_fases.md").write_text("\n".join(linhas) + "\n",
                                         encoding="utf-8")
    print("\n".join(linhas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
