#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comparacao auditada do FBCSP no piloto (folds canonicos, BA por trial)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

_SRC = Path(__file__).resolve().parents[1]
for _p in (_SRC, _SRC / "eeg", _SRC / "tools", _SRC / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eeg.mi_eegnet.data import carrega_own  # noqa: E402
from eeg.fbcsp.fbcsp import FBCSP  # noqa: E402

OUT = Path("outputs/eeg/auditoria_piloto")
ALL_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
             (24, 28), (28, 32), (32, 36), (36, 40)]
CONFIGS = {
    "full_9bandas_4csp": (ALL_BANDS, 4),
    "reduzido_4bandas_2csp": ([(8, 12), (12, 16), (16, 20), (20, 24)], 2),
    "reduzido_6bandas_2csp": ([(8, 12), (12, 16), (16, 20), (20, 24),
                                (24, 28), (28, 32)], 2),
}


def carrega():
    Xs, ys, gs = [], [], []
    off = 0
    for p in [Path("outputs/eeg/piloto_me_mi_s01"),
              Path("outputs/eeg/piloto_me_mi_s02")]:
        X, y, g, _ = carrega_own(p, fs=128, l_freq=4.0, h_freq=40.0,
                                 window_sec=2.0, step_sec=0.5,
                                 trial_sec=4.0, fase="MI")
        Xs.append(X); ys.append(y); gs.append(g + off)
        off += int(g.max()) + 1
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def avalia(X, y, g, folds, bands, n_csp):
    yt, yp, ba_folds = [], [], []
    for f in folds:
        tr = np.isin(g, f["train_trials"])
        te = np.isin(g, f["test_trials"])
        m = FBCSP(bands=bands, fs=128, n_csp_components=n_csp,
                  random_state=42)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        a, b = [], []
        for tid in np.unique(g[te]):
            q = g[te] == tid
            a.append(int(y[te][q][0]))
            b.append(int(np.bincount(pred[q]).argmax()))
        ba_folds.append(balanced_accuracy_score(a, b))
        yt.extend(a); yp.extend(b)
    return {"fold_ba": [round(float(x), 4) for x in ba_folds],
            "mean_ba": round(float(np.mean(ba_folds)), 4),
            "std_ba": round(float(np.std(ba_folds, ddof=1)), 4),
            "n_trials": len(yt),
            "confusao": confusion_matrix(yt, yp, labels=[0, 1]).tolist()}


def permuta(X, y, g, folds, bands, n_csp, n=100, seed=42):
    rng = np.random.default_rng(seed)
    obs = avalia(X, y, g, folds, bands, n_csp)["mean_ba"]
    null = []
    for _ in range(n):
        null.append(avalia(X, rng.permutation(y), g, folds,
                           bands, n_csp)["mean_ba"])
    null = np.asarray(null)
    return {"observado": obs, "nulo_mediana": round(float(np.median(null)), 4),
            "nulo_p95": round(float(np.percentile(null, 95)), 4),
            "p": round(float((1 + np.sum(null >= obs)) / (1 + n)), 4),
            "n": n}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, g = carrega()
    folds = json.loads(Path("docs/validation/folds_piloto_groupkf.json")
                       .read_text(encoding="utf-8"))["folds"]
    res = {"n_windows": int(len(y)), "n_trials": int(len(np.unique(g))),
           "configs": {}}
    for nome, (bands, n_csp) in CONFIGS.items():
        print("config", nome, flush=True)
        res["configs"][nome] = avalia(X, y, g, folds, bands, n_csp)
    print("permutacao full", flush=True)
    res["permutacao_full"] = permuta(X, y, g, folds, ALL_BANDS, 4, n=100)
    # por sessao, folds proprios
    per = {}
    for nome, p in [("s01", Path("outputs/eeg/piloto_me_mi_s01")),
                    ("s02", Path("outputs/eeg/piloto_me_mi_s02"))]:
        Xs, ys, gs, _ = carrega_own(p, fs=128, l_freq=4.0, h_freq=40.0,
                                    window_sec=2.0, step_sec=0.5,
                                    trial_sec=4.0, fase="MI")
        sg = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        fs_ = []
        for tr, te in sg.split(Xs, ys, groups=gs):
            m = FBCSP(bands=ALL_BANDS, fs=128, n_csp_components=4,
                      random_state=42).fit(Xs[tr], ys[tr])
            pr = m.predict(Xs[te]); a = []; b = []
            for tid in np.unique(gs[te]):
                q = gs[te] == tid
                a.append(int(ys[te][q][0]))
                b.append(int(np.bincount(pr[q]).argmax()))
            fs_.append(balanced_accuracy_score(a, b))
        per[nome] = {"fold_ba": [round(float(x), 4) for x in fs_],
                     "mean_ba": round(float(np.mean(fs_)), 4)}
    res["por_sessao"] = per
    (OUT / "fbcsp_comparacao.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    linhas = ["# FBCSP — auditoria independente", "",
              "| config | BA trial | dp | folds |", "|---|---:|---:|---|"]
    for nome, r in res["configs"].items():
        linhas.append(f"| {nome} | {r['mean_ba']:.4f} | {r['std_ba']:.4f} | {r['fold_ba']} |")
    p = res["permutacao_full"]
    linhas += ["", "## Permutação (config full)", "",
               f"- observado: {p['observado']:.4f}",
               f"- nulo mediana: {p['nulo_mediana']:.4f}",
               f"- nulo p95: {p['nulo_p95']:.4f}",
               f"- p: {p['p']:.4f} ({p['n']} permutações)",
               "", "## Por sessão (folds internos)", ""]
    for nome, r in res["por_sessao"].items():
        linhas.append(f"- {nome}: BA {r['mean_ba']:.4f} (folds {r['fold_ba']})")
    linhas += ["", "## Comparação com auditoria de features simples", "",
               "- features simples: 0,683; p=0,005",
               "- CSP+LDA: 0,583"]
    (OUT / "fbcsp_comparacao.md").write_text("\n".join(linhas) + "\n",
                                             encoding="utf-8")
    print("\n".join(linhas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
