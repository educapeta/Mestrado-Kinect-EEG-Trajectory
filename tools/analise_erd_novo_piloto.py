#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ERD/ERS de mu e beta no novo piloto ME/MI.

Para cada trial e fase (ME/MI):
- baseline: 1 s antes do inicio da fase;
- janela: 4 s da fase;
- envelope por Hilbert em mu (8-13) e beta (13-30);
- ERD% = 100*(P - baseline)/baseline;
- media por classe, sessao e canal;
- lateralizacao: contralateral - ipsilateral.

Saidas: ERD_ME_MI.png, erd_medio.png, resultados.json, relatorio.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import uniform_filter1d

mne.set_log_level("ERROR")
FS = 500.0
CH = [14, 15, 16]  # C3, C4, Cz
CH_NAMES = ["C3", "C4", "Cz"]
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
BASE_SEC = 1.0
PHASE_SEC = 4.0
WIN_EVAL = (0.5, 3.5)  # janela pre-definida para media


def load_session(path: Path):
    f = sorted(path.glob("eeg_mi3c*.csv"))[-1]
    j = sorted(path.glob("eventos*.json"))[-1]
    rows = []
    with open(f, encoding="utf-8") as h:
        for r in csv.reader(h):
            if r and not r[0].startswith("#"):
                rows.append(r)
    eeg = np.array([[float(v) for v in r[1:-1]] for r in rows[1:]],
                   dtype=np.float64)[:, CH]
    markers = np.array([int(float(r[-1])) for r in rows[1:]], dtype=int)
    meta = json.load(open(j, encoding="utf-8"))
    return eeg, markers, float(meta.get("fs", 500.0))


def bandpass(x, fs, lo, hi):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def phase_data(eeg, markers, fase):
    starts = np.where(markers == (778 if fase == "ME" else 780))[0]
    ends = np.where(markers == (779 if fase == "ME" else 781))[0]
    fs = FS
    nbase = int(BASE_SEC * fs)
    nphase = int(PHASE_SEC * fs)
    out = {"mu": [], "beta": []}
    labels, trials = [], []
    for a, b in zip(starts, ends):
        prev = markers[:a]
        ultimo = -1; cls = None
        for code, nome in [(769, "esquerda"), (770, "direita")]:
            pos = np.where(prev == code)[0]
            if len(pos) and pos[-1] > ultimo:
                ultimo = pos[-1]; cls = nome
        if cls is None:
            continue
        s = a - nbase
        if s < 0 or s + nbase + nphase > len(eeg):
            continue
        seg = eeg[s:s + nbase + nphase]
        labels.append(0 if cls == "esquerda" else 1)
        trials.append(seg)
        for band, (lo, hi) in BANDS.items():
            # envelope por canal no segmento inteiro
            xb = bandpass(seg, fs, lo, hi)
            env = np.abs(hilbert(xb, axis=0)) ** 2
            env = uniform_filter1d(env, size=max(1, int(0.25 * fs)), axis=0)
            base = env[:nbase].mean(axis=0)
            p = env[nbase:nbase + nphase]
            erd = 100.0 * (p - base) / (base + 1e-12)
            out[band].append(erd)
    for k in out:
        out[k] = np.stack(out[k]) if out[k] else np.empty((0, nphase, 3))
    return np.array(labels), out


def summarize(labels, erd, fs=FS):
    """labels, erd (trials, time, channels) -> media por classe/canal."""
    t = np.arange(erd.shape[1]) / fs
    m = (t >= WIN_EVAL[0]) & (t <= WIN_EVAL[1])
    out = {}
    for cls, name in [(0, "esquerda"), (1, "direita")]:
        msk = labels == cls
        out[name] = {
            "C3": float(erd[msk][:, m, 0].mean()),
            "C4": float(erd[msk][:, m, 1].mean()),
            "Cz": float(erd[msk][:, m, 2].mean()),
        }
        # contralateral: direita->C3, esquerda->C4
        contra_idx = 0 if cls == 1 else 1
        ipsi_idx = 1 if cls == 1 else 0
        out[name]["contralateral"] = float(erd[msk][:, m, contra_idx].mean())
        out[name]["ipsilateral"] = float(erd[msk][:, m, ipsi_idx].mean())
        out[name]["lateralizacao_contra_menos_ipsi"] = (
            out[name]["contralateral"] - out[name]["ipsilateral"])
    return t, out


def media_sessoes(lista):
    """Media de ERD trial-a-trial entre sessoes (mesmo n de trials)."""
    # concatena trials das sessoes
    labels = np.concatenate([x[0] for x in lista])
    erd = np.concatenate([x[1] for x in lista], axis=0)
    return labels, erd


def plot_timecourses(labels, mu, beta, out_dir: Path):
    fs = FS
    n = mu.shape[1]
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for row, (band, erd) in enumerate((("Mu", mu), ("Beta", beta))):
        for col, cls in enumerate((0, 1)):
            ax = axes[row, col]
            m = labels == cls
            name = "esquerda" if cls == 0 else "direita"
            ax.axhline(0, color="k", lw=0.8)
            for ch, cor in ((0, "tab:blue"), (1, "tab:red"),
                            (2, "tab:gray")):
                y = erd[m][:, :, ch].mean(axis=0)
                ax.plot(t, y, color=cor, label=CH_NAMES[ch], lw=1.5)
            ax.set_title(f"{band} — {name}")
            ax.set_xlabel("tempo desde a fase (s)")
            ax.set_ylabel("ERD/ERS (%)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("ERD/ERS médio — ME/MI, duas sessões")
    fig.tight_layout()
    fig.savefig(out_dir / "ERD_ME_MI.png", dpi=160)
    plt.close(fig)


def plot_summary(summary, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(4)
    vals = []
    labels = []
    for fase in ("ME", "MI"):
        for cls in ("esquerda", "direita"):
            d = summary[fase]["mu"][cls]
            vals.append([d["C3"], d["C4"], d["Cz"]])
            labels.append(f"{fase}\n{cls}")
    vals = np.array(vals)
    width = 0.25
    for i, ch in enumerate(CH_NAMES):
        ax.bar(x + (i - 1) * width, vals[:, i], width, label=ch)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ERD médio (%) na janela 0,5–3,5 s")
    ax.set_title("ERD médio por fase/classe/canal")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "erd_medio.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=Path, default=Path("_piloto_me_mi_s01"))
    ap.add_argument("--s2", type=Path, default=Path("_piloto_me_mi_s02"))
    ap.add_argument("--saida", type=Path, default=Path("_erd_novo_piloto"))
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    all_data = {}
    for fase in ("ME", "MI"):
        all_data[fase] = []
        for sname, path in (("s1", args.s1), ("s2", args.s2)):
            eeg, markers, fs = load_session(path)
            labels, out = phase_data(eeg, markers, fase)
            print(f"{fase} {sname}: trials={len(labels)}", flush=True)
            all_data[fase].append((labels, out))
    summary = {}
    for fase in ("ME", "MI"):
        labels, erd_mu, erd_beta = [], [], []
        for labels_s, out in all_data[fase]:
            labels.extend(labels_s.tolist())
            erd_mu.append(out["mu"])
            erd_beta.append(out["beta"])
        labels = np.array(labels)
        mu = np.concatenate(erd_mu, axis=0)
        beta = np.concatenate(erd_beta, axis=0)
        t, sm = summarize(labels, mu)
        _, sb = summarize(labels, beta)
        summary[fase] = {"mu": sm, "beta": sb,
                         "n_trials": int(len(labels)),
                         "n_esquerda": int((labels == 0).sum()),
                         "n_direita": int((labels == 1).sum())}
        plot_timecourses(labels, mu, beta, args.saida)
        # plots por fase (o timecourse atual e' das duas fases juntas;
        # refaz separado abaixo)
    # plots separados por fase
    for fase in ("ME", "MI"):
        labels, erd_mu, erd_beta = [], [], []
        for labels_s, out in all_data[fase]:
            labels.extend(labels_s.tolist())
            erd_mu.append(out["mu"]); erd_beta.append(out["beta"])
        plot_timecourses(np.array(labels), np.concatenate(erd_mu),
                         np.concatenate(erd_beta), args.saida)
        # renomeia para nao sobrescrever a versao final
        src = args.saida / "ERD_ME_MI.png"
        src.replace(args.saida / f"ERD_{fase}.png")
    # ultima figura combinada: usa ME (apenas ilustrativo) - gera de novo
    # combinando ME+MI para o grafico final
    labels_all, mu_all, beta_all = [], [], []
    for fase in ("ME", "MI"):
        for labels_s, out in all_data[fase]:
            labels_all.extend(labels_s.tolist())
            mu_all.append(out["mu"]); beta_all.append(out["beta"])
    plot_timecourses(np.array(labels_all), np.concatenate(mu_all),
                     np.concatenate(beta_all), args.saida)
    plot_summary(summary, args.saida)
    (args.saida / "resultados.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# ERD/ERS — ME/MI (duas sessões)", "",
             "Janela de média: 0,5–3,5 s após início da fase;",
             "baseline: 1 s antes da fase.", ""]
    for fase in ("ME", "MI"):
        for band in ("mu", "beta"):
            lines += [f"## {fase} — {band}", ""]
            for cls in ("esquerda", "direita"):
                d = summary[fase][band][cls]
                lines += [f"### {cls}", "",
                          f"- C3: {d['C3']:.2f}%",
                          f"- C4: {d['C4']:.2f}%",
                          f"- Cz: {d['Cz']:.2f}%",
                          f"- contralateral: {d['contralateral']:.2f}%",
                          f"- ipsilateral: {d['ipsilateral']:.2f}%",
                          f"- lateralização (contra-ipsi): "
                          f"{d['lateralizacao_contra_menos_ipsi']:.2f}%",
                          ""]
    (args.saida / "relatorio.md").write_text("\n".join(lines),
                                              encoding="utf-8")
    print("Salvo em", args.saida)


if __name__ == "__main__":
    main()
