#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Espectro medio do piloto ME/MI por mao e canal (validacao do FBCSP).

Usa as MESMAS janelas de 2 s do FBCSP (`carrega_own`, fs=128, 4-40 Hz,
passo 0,5 s) para calcular:
- PSD medio por classe/fase/canal (Welch);
- potencia por banda;
- tamanho de efeito direita-esquerda (Cohen's d) e p por permutacao
  no nivel do trial.

Saidas: espectro_medio.png, bandas_efeito.png, resultados.json, relatorio.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

_SRC = Path(__file__).resolve().parents[2]
for _p in (_SRC, _SRC / "mestradopy"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mestradopy.eeg.mi_eegnet.data import carrega_own  # noqa: E402

FS = 128
CH = ["C3", "C4", "Cz"]
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
         (24, 28), (28, 32), (32, 36), (36, 40)]
CLASSES = {0: "esquerda", 1: "direita"}


def psd_trial(X, y, g):
    """PSD media por trial (Welch em cada janela de 2 s)."""
    n = int(len(np.unique(g)))
    f, P = welch(X, fs=FS, nperseg=min(256, X.shape[-1]),
                 noverlap=min(128, X.shape[-1] // 2), axis=-1,
                 detrend="constant")
    sel = (f >= 4) & (f <= 40)
    f = f[sel]
    Ptr = np.zeros((n, 3, sel.sum()), dtype=np.float64)
    labels = np.zeros(n, dtype=int)
    for k, tid in enumerate(np.unique(g)):
        m = g == tid
        Ptr[k] = P[m].mean(axis=0)[:, sel]
        labels[k] = int(y[m][0])
    return f, Ptr, labels


def load_phase(pastas, fase):
    F, P, Y, S = [], [], [], []
    for si, pasta in enumerate(pastas):
        X, y, g, ty = carrega_own(pasta, fs=FS, l_freq=4.0, h_freq=40.0,
                                  window_sec=2.0, step_sec=0.5, fase=fase)
        f, p, lab = psd_trial(X, y, g)
        F.append(f); P.append(p); Y.append(lab)
        S.append(np.full(len(lab), si))
    return F[0], np.concatenate(P), np.concatenate(Y), np.concatenate(S)


def band_power(P, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    return P[:, :, m].mean(axis=2)


def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def perm_p(a, b, n=2000, seed=42):
    """Permutacao de rotulos no nivel do trial (direita-esquerda)."""
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    z = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(n):
        rng.shuffle(z)
        if abs(z[:na].mean() - z[na:].mean()) >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=Path,
                    default=Path("outputs/eeg/piloto_me_mi_s01"))
    ap.add_argument("--s2", type=Path,
                    default=Path("outputs/eeg/piloto_me_mi_s02"))
    ap.add_argument("--saida", type=Path,
                    default=Path("outputs/eeg/espectro_piloto"))
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    res = {}
    for fase in ("ME", "MI"):
        f, P, y, sess = load_phase([args.s1, args.s2], fase)
        assert len(np.unique(y)) == 2
        print(f"{fase}: trials={len(y)} esquerda={(y==0).sum()} "
              f"direita={(y==1).sum()} janelas/trial={P.shape[0]}", flush=True)
        dados = {}
        for cls in (0, 1):
            dados[CLASSES[cls]] = {
                "psd": P[y == cls].mean(axis=0).tolist(),
                "n_trials": int((y == cls).sum()),
            }
        efeitos = []
        for (lo, hi) in BANDS:
            bp = band_power(P, f, lo, hi)  # (trials, 3)
            for ci, ch in enumerate(CH):
                a = bp[y == 0, ci]; b = bp[y == 1, ci]
                efeitos.append({
                    "band": f"{lo}-{hi}", "channel": ch,
                    "left_mean": float(a.mean()),
                    "right_mean": float(b.mean()),
                    "log2_ratio_right_left": float(np.log2(
                        (b.mean() + 1e-12) / (a.mean() + 1e-12))),
                    "cohen_d": cohen_d(a, b),
                    "p_perm": perm_p(a, b, n=2000),
                })
        # lateralizacao: contralateral (C3 para direita / C4 para esquerda)
        lat = []
        for (lo, hi) in BANDS:
            bp = band_power(P, f, lo, hi)
            for cls, name, contra, ipsi in (
                    (1, "direita", 0, 1), (0, "esquerda", 1, 0)):
                lat.append({
                    "band": f"{lo}-{hi}", "class": name,
                    "contralateral_ch": CH[contra],
                    "ipsilateral_ch": CH[ipsi],
                    "contra_minus_ipsi_log2": float(np.log2(
                        (bp[y == cls, contra].mean() + 1e-12) /
                        (bp[y == cls, ipsi].mean() + 1e-12))),
                })
        res[fase] = {
            "freq": f.tolist(),
            "classes": dados,
            "effects": efeitos,
            "lateralization": lat,
            "n_trials": int(len(y)),
        }
        # plot PSD medio
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for ci, ch in enumerate(CH):
            ax = axes[ci]
            for cls, color in ((0, "tab:blue"), (1, "tab:red")):
                psd = np.array(dados[CLASSES[cls]]["psd"])[ci]
                ax.plot(f, 10 * np.log10(psd + 1e-12), color=color,
                        label=CLASSES[cls], lw=1.8)
            ax.set_title(f"{fase} — {ch}")
            ax.set_xlabel("Hz")
            ax.grid(alpha=0.25)
            ax.axvspan(12, 24, color="gray", alpha=0.12, label="12–24 Hz")
        axes[0].set_ylabel("PSD (dB re 1 µV²/Hz)")
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"PSD médio — {fase} (2 s, Welch, 4–40 Hz)")
        fig.tight_layout()
        fig.savefig(args.saida / f"espectro_{fase}.png", dpi=160)
        plt.close(fig)

    # Figura resumo de efeito por banda (média dos canais)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ai, fase in enumerate(("ME", "MI")):
        ax = axes[ai]
        for bi, (lo, hi) in enumerate(BANDS):
            vals = [e["cohen_d"] for e in res[fase]["effects"]
                    if e["band"] == f"{lo}-{hi}"]
            ax.bar(bi, np.mean(vals), yerr=np.std(vals), capsize=3,
                   color="tab:purple")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels([f"{a}-{b}" for a, b in BANDS], rotation=45,
                           ha="right")
        ax.set_ylabel("Cohen's d (direita − esquerda)")
        ax.set_title(f"{fase}: efeito de classe por banda (média canais)")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.saida / "bandas_efeito.png", dpi=160)
    plt.close(fig)

    (args.saida / "resultados.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    linhas = ["# Espectro médio do piloto — ME e MI", "",
              "Janelas de 2 s, fs=128 Hz, filtro 4–40 Hz, passo 0,5 s "
              "(mesmas janelas do FBCSP).",
              "Efeito = direita − esquerda; d de Cohen; p por permutação "
              "no nível do trial (2000 permutações).", ""]
    for fase in ("ME", "MI"):
        linhas += [f"## {fase}", "",
                   "| banda | canal | d | p | log2(right/left) |",
                   "|---|---:|---:|---:|---:|"]
        for e in res[fase]["effects"]:
            if e["p_perm"] <= 0.05 or abs(e["cohen_d"]) >= 0.5:
                linhas.append(
                    f"| {e['band']} | {e['channel']} | {e['cohen_d']:.3f} | "
                    f"{e['p_perm']:.4f} | {e['log2_ratio_right_left']:.3f} |")
        linhas.append("")
    (args.saida / "relatorio.md").write_text("\n".join(linhas) + "\n",
                                              encoding="utf-8")
    print("Salvo em", args.saida, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
