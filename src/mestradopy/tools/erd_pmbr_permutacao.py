#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ERD/ERS banda-especifico (mu e beta) e permutacao na janela late.

Testa a hipotese de Post-Movement Beta Rebound (PMBR):
  - se o early 0-2 s tem beta AUMENTADO (ERS) e o late 2-4 s tem beta
    DIMINUIDO (ERD), o PMBR do ME contamina o inicio da MI.
  - permutacao por trial na janela late 2-4 s quantifica se o BA observado
    difere do acaso.

Uso:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\erd_pmbr_permutacao.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mne.decoding import CSP

_SRC = Path(__file__).resolve().parents[1]
for _p in (_SRC, _SRC / "eeg", _SRC / "tools", _SRC / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import feedback_adaptativo as fa  # noqa: E402

FS = 128
BANDS = {"mu": (8.0, 12.0), "beta": (13.0, 30.0)}
NOMES = ["C3", "C4", "Cz"]


def _pipe():
    def finite(z):
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    # funcao de modulo para joblib, mas aqui nao serializamos
    return Pipeline([
        ("csp", CSP(n_components=2, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(finite)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def carrega(alvo: List[Path]):
    trials = []
    for pasta in alvo:
        t, _ = fa.carrega_sessao_eeg(pasta)
        trials.extend(t)
    return trials


def potencia(x: np.ndarray, faixa) -> np.ndarray:
    x = np.asarray(x, np.float64)
    f, p = welch(x, fs=FS, nperseg=min(128, x.shape[-1]), axis=-1)
    m = (f >= faixa[0]) & (f <= faixa[1])
    return p[..., m].mean(axis=-1)


def erd_por_janela(trials) -> Dict:
    """ERD% de janelas de 1 s ao longo da MI, por banda/canal/classe."""
    janelas = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    out = {}
    for banda, faixa in BANDS.items():
        out[banda] = {}
        for (a, b) in janelas:
            chave = f"{a:.0f}-{b:.0f}s"
            vals = {0: [], 1: []}
            for t in trials:
                base = potencia(t["X_base"], faixa)
                seg = t["X_mi"][:, int(a * FS):int(b * FS)]
                if seg.shape[-1] < 32:
                    continue
                p = potencia(seg, faixa)
                vals[int(t["classe"])].append(100.0 * (p / np.maximum(base, 1e-20) - 1.0))
            out[banda][chave] = {
                "esquerda": np.median(np.asarray(vals[0]), axis=0).tolist()
                if vals[0] else [float("nan")] * 3,
                "direita": np.median(np.asarray(vals[1]), axis=0).tolist()
                if vals[1] else [float("nan")] * 3,
            }
    return out


def avalia_late(trials, n_perm=1000, seed=42) -> Dict:
    """CSP+LDA na janela 2-4 s; permutacao de labels por trial."""
    n_win = int(2 * FS)
    segs = []
    for t in trials:
        x = np.asarray(t["X_mi"], np.float64)
        if x.shape[1] < 4 * FS:
            x = np.pad(x, ((0, 0), (0, 4 * FS - x.shape[1])))
        segs.append(x[:, int(2 * FS):int(2 * FS) + n_win])
    X = np.stack(segs)
    y = np.array([int(t["classe"]) for t in trials])
    g = np.arange(len(y))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y, groups=g))

    def ba_de(yy):
        yt, yp = [], []
        for tr, te in folds:
            pipe = _pipe(); pipe.fit(X[tr], yy[tr])
            pred = pipe.predict(X[te])
            yt.extend(yy[te].tolist()); yp.extend(pred.tolist())
        return float(balanced_accuracy_score(yt, yp))

    obs = ba_de(y)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_perm)):
        null.append(ba_de(rng.permutation(y)))
    null = np.asarray(null)
    p = (1.0 + float(np.sum(null >= obs))) / (1.0 + len(null))
    return {"ba_observado": round(obs, 4), "n_trials": int(len(y)),
            "n_perm": int(n_perm), "p_permutacao": round(p, 4),
            "nulo_mediana": round(float(np.median(null)), 4),
            "nulo_p95": round(float(np.percentile(null, 95)), 4),
            "nulo_max": round(float(np.max(null)), 4), "nulo": null.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alvo", type=Path, nargs="+",
                    default=[Path("outputs/eeg/piloto_me_mi_s01"),
                             Path("outputs/eeg/piloto_me_mi_s02")])
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/erd_pmbr"))
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    trials = carrega(args.alvo)
    print("trials:", len(trials), "classes:", {c: sum(1 for t in trials if int(t['classe']) == c) for c in (0, 1)})
    erd = erd_por_janela(trials)
    perm = avalia_late(trials, n_perm=args.n_perm)
    print("permutacao late:", {k: v for k, v in perm.items() if k != "nulo"})
    # figura ERD por janela
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    jan = ["0-1s", "1-2s", "2-3s", "3-4s"]
    for ax, banda in zip(axes, BANDS):
        for cls, cor in (("esquerda", "tab:blue"), ("direita", "tab:red")):
            vals = [erd[banda][k][cls][0] for k in jan]  # C3
            ax.plot(range(len(jan)), vals, "o-", color=cor, label=f"{cls} C3")
        ax.axhline(0, color="gray", ls="--")
        ax.set_ylabel(f"ERD/ERS {banda} C3 (%)")
        ax.grid(alpha=.25); ax.legend()
    axes[-1].set_xticks(range(len(jan))); axes[-1].set_xticklabels(jan)
    axes[0].set_title("ERD/ERS por janela de 1 s — C3 (negativo=ERD, positivo=ERS)")
    fig.tight_layout(); fig.savefig(args.saida / "erd_janelas.png", dpi=160)
    plt.close(fig)
    # figura nulo
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(perm["nulo"], bins=30, color="tab:gray", alpha=.7)
    ax.axvline(perm["ba_observado"], color="tab:red", lw=2,
               label=f"observado {perm['ba_observado']:.3f}")
    ax.set_xlabel("BA sob labels permutados")
    ax.set_ylabel("frequencia")
    ax.set_title(f"Permutacao late 2-4s (p={perm['p_permutacao']:.3f})")
    ax.legend(); fig.tight_layout(); fig.savefig(args.saida / "permutacao_late.png", dpi=160)
    plt.close(fig)
    out = {"n_trials": len(trials), "erd": erd, "permutacao_late": perm,
           "nota": ("early=0-2s; late=2-4s; ERD relativo a 1 s pre-cue; "
                    "permutacao por trial, 5-fold fixo.")}
    (args.saida / "resultado.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    linhas = ["# ERD/ERS e permutacao — teste PMBR", "",
              "## ERD/ERS em C3 por janela (medianas, % vs baseline)", ""]
    for banda in BANDS:
        linhas.append(f"### {banda}")
        for k in jan:
            linhas.append(f"- {k}: L={erd[banda][k]['esquerda']} | "
                          f"R={erd[banda][k]['direita']}")
    linhas += ["", "## Permutacao late 2-4 s", "",
               f"- BA observado: {perm['ba_observado']:.3f}",
               f"- p unilateral: {perm['p_permutacao']:.4f}",
               f"- nulo mediana: {perm['nulo_mediana']:.3f}; "
               f"p95: {perm['nulo_p95']:.3f}",
               "- n trials: {}".format(perm["n_trials"])]
    (args.saida / "relatorio.md").write_text("\n".join(linhas) + "\n",
                                             encoding="utf-8")
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
