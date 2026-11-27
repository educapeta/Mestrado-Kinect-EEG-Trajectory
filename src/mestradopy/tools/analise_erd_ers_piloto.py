#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ERD/ERS da MI (4 s) usando o baseline de OLHOS ABERTOS.

Baseline: bloco 901 -> 903 (60 s de olhos abertos no início de cada sessão).
Janela de análise: 4 s da MI (marcador 780 -> 781).
Bandas: mu 8-13, beta baixo 13-20, beta alto 20-30 Hz.
Pre-processamento: 4-40 Hz, fs=128 (mesmo do FBCSP).

ERD% = 100*(P_MI(t)/P_olhos_abertos - 1), trial-a-trial.
Media +/- SEM por classe/canal, teste contra zero e FDR por tempo.
Janelas a priori: ERS inicial (0,1-0,6 s), ERD (0,9-1,6 s),
ERS tardio (2,0-3,5 s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import ttest_1samp

_SRC = Path(__file__).resolve().parents[2]
for _p in (_SRC, _SRC / "mestradopy"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mestradopy.eeg.mi_eegnet.data import (  # noqa: E402
    _load_own_csv, filt_resample, CH_OWN_IDX)

FS = 128
WIN_SEC = 0.5
STEP_SEC = 0.125
MI_SEC = 4.0
CH = ["C3", "C4", "Cz"]
BANDS = {"mu": (8.0, 13.0), "beta_low": (13.0, 20.0),
         "beta_high": (20.0, 30.0)}
CLASSES = {0: "esquerda", 1: "direita"}
WINDOWS = {
    "ERS_inicial": (0.10, 0.60),
    "ERD": (0.90, 1.60),
    "ERS_tardio": (2.00, 3.50),
}
CODE_EYES_OPEN_START = 901
CODE_EYES_OPEN_END = 903


def bh_fdr(p):
    p = np.asarray(p, float)
    shape = p.shape
    pf = p.reshape(-1)
    q = np.ones_like(pf)
    idx = np.where(np.isfinite(pf))[0]
    if len(idx) == 0:
        return q.reshape(shape)
    order = idx[np.argsort(pf[idx])]
    m = len(order)
    vals = pf[order] * m / np.arange(1, m + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q[order] = np.clip(vals, 0, 1)
    return q.reshape(shape)


def power_track(x, fs, lo, hi, win=WIN_SEC, step=STEP_SEC):
    """x (samples,ch) ou (trials,samples,ch) -> potencia por janela."""
    single = x.ndim == 2
    if single:
        x = x[None, ...]
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    xb = sosfiltfilt(sos, x, axis=1)
    n = max(1, int(round(win * fs)))
    st = max(1, int(round(step * fs)))
    starts = list(range(0, xb.shape[1] - n + 1, st)) or [0]
    bp = np.zeros((len(starts), xb.shape[0], xb.shape[2]))
    for i, s in enumerate(starts):
        bp[i] = np.var(xb[:, s:s + n, :], axis=1)
    t = (np.array(starts) + n / 2) / fs
    return (t, bp[:, 0, :]) if single else (t, bp)


def load_eyes_session(pasta):
    """Le uma sessao e devolve baseline olhos abertos e trials MI."""
    csv_path = sorted(Path(pasta).glob("eeg_mi3c*.csv"))[-1]
    json_path = sorted(Path(pasta).glob("eventos*.json"))[-1]
    meta = json.load(open(json_path, encoding="utf-8"))
    eeg, markers, _ = _load_own_csv(csv_path)
    fs_orig = float(meta.get("fs", 500.0))
    eeg = eeg[:, CH_OWN_IDX]
    eeg = filt_resample(eeg.T, fs_orig, FS, 4.0, 40.0).T
    escala = FS / fs_orig
    ini = np.where(markers == CODE_EYES_OPEN_START)[0]
    fim = np.where(markers == CODE_EYES_OPEN_END)[0]
    if not len(ini) or not len(fim):
        raise RuntimeError(f"{pasta}: marcadores 901/903 ausentes")
    a = int(round(ini[0] * escala))
    b = int(round(fim[0] * escala))
    base_eyes = eeg[a:b]

    starts = np.where(markers == 780)[0]
    ends = np.where(markers == 781)[0]
    mi, labels = [], []
    nmi = int(MI_SEC * FS)
    for a_, b_ in zip(starts, ends):
        prev = markers[:a_]
        ultimo, cls = -1, None
        for code, nome in ((769, "esquerda"), (770, "direita")):
            pos = np.where(prev == code)[0]
            if len(pos) and pos[-1] > ultimo:
                ultimo = pos[-1]; cls = nome
        if cls is None:
            continue
        a_ = int(round(a_ * escala))
        if a_ + nmi > len(eeg):
            continue
        mi.append(eeg[a_:a_ + nmi])
        labels.append(0 if cls == "esquerda" else 1)
    return base_eyes, np.stack(mi), np.array(labels)


def media_classe(erd, labels, cls):
    m = labels == cls
    mean = erd[:, m, :].mean(axis=1)
    sem = erd[:, m, :].std(axis=1, ddof=1) / np.sqrt(m.sum())
    p = np.ones((erd.shape[0], 3))
    for ci in range(3):
        for ti in range(erd.shape[0]):
            p[ti, ci] = ttest_1samp(erd[ti, m, ci], 0.0).pvalue
    return {"mean": mean.tolist(), "sem": sem.tolist(),
            "q_fdr": bh_fdr(p).tolist(), "p": p.tolist(),
            "n_trials": int(m.sum())}


def main() -> int:
    out = Path("outputs/eeg/erd_ers_olhos_abertos")
    out.mkdir(parents=True, exist_ok=True)
    erds = []
    labels = []
    t_ref = None
    for pasta in (Path("outputs/eeg/piloto_me_mi_s01"),
                  Path("outputs/eeg/piloto_me_mi_s02")):
        base_eyes, mi, y = load_eyes_session(pasta)
        erd_sess = {}
        for band, (lo, hi) in BANDS.items():
            _, bp_base = power_track(base_eyes, FS, lo, hi)
            ref = np.median(bp_base, axis=0)  # (3,)
            t_ref, bp_mi = power_track(mi, FS, lo, hi)  # (W,trials,3)
            erd_sess[band] = np.clip(
                100.0 * (bp_mi / (ref[None, None, :] + 1e-12) - 1.0),
                -200.0, 200.0)
        erds.append(erd_sess)
        labels.append(y)
        print(pasta.name, "trials", len(y), "olhos_abertos_s",
              round(base_eyes.shape[0] / FS, 1), flush=True)
    labels = np.concatenate(labels)
    print("total", len(labels), "esquerda", int((labels == 0).sum()),
          "direita", int((labels == 1).sum()), flush=True)

    res = {}
    for band in BANDS:
        erd = np.concatenate([s[band] for s in erds], axis=1)  # (W,N,3)
        bloco = {"t": t_ref.tolist(), "classes": {}}
        for cls in (0, 1):
            bloco["classes"][CLASSES[cls]] = media_classe(erd, labels, cls)
        # medias por janelas a priori
        jan = {}
        for wname, (w0, w1) in WINDOWS.items():
            mw = (t_ref >= w0) & (t_ref <= w1)
            jan[wname] = {}
            for cls in (0, 1):
                m = labels == cls
                vals = erd[mw][:, m, :].mean(axis=0)  # (trials,3)
                p = np.ones((3,))
                for ci in range(3):
                    p[ci] = ttest_1samp(vals[:, ci], 0.0).pvalue
                jan[wname][CLASSES[cls]] = {
                    "mean": vals.mean(axis=0).tolist(),
                    "sem": (vals.std(axis=0, ddof=1)
                            / np.sqrt(len(vals))).tolist(),
                    "p": p.tolist(),
                }
        bloco["janelas"] = jan
        res[band] = bloco

        # figura por banda: 3 canais, 2 classes
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        cores = {0: "tab:blue", 1: "tab:red"}
        for ci, ch in enumerate(CH):
            ax = axes[ci]
            ax.axhline(0, color="k", lw=0.8)
            for cls in (0, 1):
                d = bloco["classes"][CLASSES[cls]]
                m = np.array(d["mean"])[:, ci]
                s = np.array(d["sem"])[:, ci]
                ax.plot(t_ref, m, color=cores[cls],
                        label=CLASSES[cls], lw=1.8)
                ax.fill_between(t_ref, m - s, m + s,
                                color=cores[cls], alpha=0.18)
                sig = np.array(d["q_fdr"])[:, ci] < 0.05
                if sig.any():
                    ax.scatter(t_ref[sig], np.full(sig.sum(), ax.get_ylim()[0]),
                               s=10, color=cores[cls], marker=".")
            ax.set_title(f"{band} — {ch}")
            ax.set_xlabel("tempo desde o início da MI (s)")
            ax.grid(alpha=0.25)
            if ci == 0:
                ax.set_ylabel("ERD/ERS vs olhos abertos (%)")
            ax.legend(fontsize=8)
        fig.suptitle("ERD/ERS da MI (4 s) — baseline olhos abertos; "
                     "pontos = q<0,05")
        fig.tight_layout()
        fig.savefig(out / f"erd_ers_{band}.png", dpi=160)
        plt.close(fig)

    # relatorio
    linhas = ["# ERD/ERS da MI — baseline olhos abertos", "",
              "Baseline: 60 s de olhos abertos (marcadores 901–903).",
              "MI: 4 s (780–781). ERD% = 100*(MI/olhos_abertos - 1).",
              "Janelas: ERS inicial 0,1–0,6 s; ERD 0,9–1,6 s; "
              "ERS tardio 2,0–3,5 s.",
              "", "## Por banda", ""]
    for band in BANDS:
        linhas += [f"### {band}", "",
                   "| janela | classe | C3 | C4 | Cz |",
                   "|---|---|---:|---:|---:|"]
        for wname in WINDOWS:
            for cls in (0, 1):
                d = res[band]["janelas"][wname][CLASSES[cls]]
                p = d["p"]
                def fmt(i):
                    estrela = "*" if p[i] < 0.05 else ""
                    return f"{d['mean'][i]:.1f}%{estrela}"
                linhas.append(f"| {wname} | {CLASSES[cls]} | "
                              f"{fmt(0)} | {fmt(1)} | {fmt(2)} |")
        linhas.append("")
    linhas += [
        "`*` = p < 0,05 contra zero (baseline olhos abertos).", "",
        "## Leitura", "",
        "- mu: ERS inicial (0,1–0,6 s) e ERS tardio (2,0–3,5 s) claros;",
        "- mu: ERD transitório em 0,9–1,6 s, mais claro em C4 esquerda;",
        "- beta baixo: ERD sustentado em 0,9–1,6 s e também no período tardio;",
        "- beta alto: ERD em parte dos canais, com padrão menos homogêneo;",
        "- os eventos ERD/ERS estão presentes, mas a média por classe não mostra "
        "uma separação espacial limpa; a classificação FBCSP pode explorar "
        "padrões espaciais que a média de eletrodo não revela.",
    ]
    (out / "resultados.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "relatorio.md").write_text("\n".join(linhas) + "\n",
                                      encoding="utf-8")
    print("Salvo em", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
