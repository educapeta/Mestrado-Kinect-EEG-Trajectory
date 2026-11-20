#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testa se a janela curta de MI melhora o CSP+LDA.

No nosso paradigma o cue visual (769/770) vem ANTES da ME; a MI comeca no
marcador 780. Portanto "primeiros 2 s apos o cue" incluiria ME e nao responde a
pergunta. Este tool compara:
  - early: janela 0-2 s do bloco de MI (primeira janela de 2 s);
  - full: todas as 5 janelas de 2 s (passo 0,5 s) do bloco de MI;
  - late: ultima janela (2-4 s).
Validação: 5-fold agrupado por trial (janelas do mesmo trial no mesmo fold),
CSP+LDA com agregacao de probabilidades por trial.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mne.decoding import CSP

_SRC = Path(__file__).resolve().parents[1]
for _p in (_SRC, _SRC / "eeg", _SRC / "core", _SRC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eeg.mi_eegnet.data import carrega_own  # noqa: E402


def _finite(z):
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _pipe():
    return Pipeline([
        ("csp", CSP(n_components=2, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(_finite)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def _janelas_por_grupo(g: np.ndarray) -> List[np.ndarray]:
    return [np.where(g == tid)[0] for tid in np.unique(g)]


def seleciona(X, y, g, indices: Tuple[int, ...]):
    """Seleciona posicoes de janela dentro de cada trial (0..4)."""
    grupos = _janelas_por_grupo(g)
    idx = []
    for pos in grupos:
        for k in indices:
            if k < len(pos):
                idx.append(pos[k])
    idx = np.asarray(idx, int)
    return X[idx], y[idx], g[idx]


def avalia_cv(X, y, g) -> Dict:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    yt, yp = [], []
    for tr, te in sgkf.split(X, y, groups=g):
        pipe = _pipe(); pipe.fit(X[tr], y[tr])
        prob = pipe.predict_proba(X[te])
        for tid in np.unique(g[te]):
            m = g[te] == tid
            yt.append(int(y[te][m][0]))
            yp.append(int(np.mean(prob[m], axis=0).argmax()))
    yt = np.asarray(yt); yp = np.asarray(yp)
    from sklearn.metrics import balanced_accuracy_score
    ba = float(balanced_accuracy_score(yt, yp))
    return {"ba": round(ba, 4), "n_trials": int(len(yt)),
            "confusao": confusion_matrix(yt, yp, labels=[0, 1]).tolist()}


def avalia_cross(Xtr, ytr, gtr, Xte, yte, gte) -> Dict:
    pipe = _pipe(); pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)
    yt, yp = [], []
    for tid in np.unique(gte):
        m = gte == tid
        yt.append(int(yte[m][0]))
        yp.append(int(np.mean(prob[m], axis=0).argmax()))
    yt = np.asarray(yt); yp = np.asarray(yp)
    from sklearn.metrics import balanced_accuracy_score
    return {"ba": round(float(balanced_accuracy_score(yt, yp)), 4),
            "n_trials": int(len(yt)),
            "confusao": confusion_matrix(yt, yp, labels=[0, 1]).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alvo", type=Path, nargs="+",
                    default=[Path("outputs/eeg/piloto_me_mi_s01"),
                             Path("outputs/eeg/piloto_me_mi_s02")])
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/janela_curta_mi"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    Xs, ys, gs = [], [], []
    offset = 0
    for pasta in args.alvo:
        X, y, g, _ = carrega_own(pasta, fs=128, fase="MI")
        Xs.append(X); ys.append(y); gs.append(g + offset)
        offset += int(g.max()) + 1
        print(f"{pasta.name}: {X.shape} trials={len(np.unique(g))}")
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    print("alvo:", X.shape, "classes", dict(zip(*np.unique(y, return_counts=True))),
          "trials", len(np.unique(g)))
    if args.smoke:
        keep = np.isin(g, np.unique(g)[:40])
        X, y, g = X[keep], y[keep], g[keep]

    # posicoes: 5 janelas/trial; early=0, late=4, full=0..4
    variantes = {
        "early_0_2s": (0,),
        "mid_1_3s": (2,),
        "late_2_4s": (4,),
        "full_0_4s": (0, 1, 2, 3, 4),
    }
    res = {}
    print("\n=== CSP+LDA, 5-fold agrupado por trial ===")
    for nome, idx in variantes.items():
        Xv, yv, gv = seleciona(X, y, g, idx)
        r = avalia_cv(Xv, yv, gv)
        res[nome] = {**r, "n_janelas": int(len(yv)),
                     "janelas_por_trial": len(idx)}
        print(f"{nome}: BA={r['ba']:.3f} trials={r['n_trials']} "
              f"janelas={len(yv)} confusao={r['confusao']}")

    # cross-session: s01 -> s02 e s02 -> s01 por variante
    cross = {}
    if len(args.alvo) == 2:
        X1, y1, g1, _ = carrega_own(args.alvo[0], fs=128, fase="MI")
        X2, y2, g2, _ = carrega_own(args.alvo[1], fs=128, fase="MI")
        for nome, idx in variantes.items():
            A = seleciona(X1, y1, g1, idx)
            B = seleciona(X2, y2, g2, idx)
            cross[nome] = {
                "s01_para_s02": avalia_cross(*A, *B),
                "s02_para_s01": avalia_cross(*B, *A),
            }
    (args.saida / "resultado.json").write_text(
        json.dumps({"dados": {"n_janelas": int(len(y)),
                              "n_trials": int(len(np.unique(g)))},
                    "resultados": res,
                    "cross_session": cross,
                    "observacao": ("early=0-2s do bloco de MI (marcador 780); "
                                   "cue 769/770 precede a ME e nao e' o inicio "
                                   "da MI.")}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    if cross:
        print("\n=== Cross-session ===")
        for nome, vals in cross.items():
            print(nome, "s01->s02", vals["s01_para_s02"]["ba"],
                  "| s02->s01", vals["s02_para_s01"]["ba"])
    linhas = ["# Janela curta de MI vs janela completa", "",
              "| variante | janelas/trial | BA | trials | confusao |",
              "|---|---:|---:|---:|---|"]
    for nome, r in res.items():
        linhas.append(f"| {nome} | {r['janelas_por_trial']} | {r['ba']:.3f} | "
                      f"{r['n_trials']} | {r['confusao']} |")
    (args.saida / "relatorio.md").write_text("\n".join(linhas) + "\n",
                                             encoding="utf-8")
    print("\nsalvo em", args.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
