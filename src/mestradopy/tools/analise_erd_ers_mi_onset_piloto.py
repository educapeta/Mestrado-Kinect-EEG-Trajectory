#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ERD/ERS em torno do inicio da MI: janela -1 s a +3 s em relacao a 780.

Baseline: bloco de olhos abertos (901 -> 903) de cada sessao.
Evento: inicio da MI (marcador 780); classe pelo cue 769/770 anterior;
janela [-1 s, +3 s] em relacao ao inicio da MI, fs=128, 4-40 Hz.
A janela cobre o ultimo 1 s da transicao ME->MI e os primeiros 3 s da MI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, spectrogram
from scipy.stats import ttest_1samp

_SRC = Path(__file__).resolve().parents[2]
for _p in (_SRC, _SRC / "mestradopy"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mestradopy.eeg.mi_eegnet.data import (  # noqa: E402
    _load_own_csv, filt_resample, CH_OWN_IDX)

FS = 128
PRE = 1.0
POST = 3.0
WIN_SEC = 0.5
STEP_SEC = 0.125
CH = ["C3", "C4", "Cz"]
BANDS = {"mu": (8.0, 13.0), "beta_low": (13.0, 20.0),
         "beta_high": (20.0, 30.0)}
CLASSES = {0: "esquerda", 1: "direita"}
WINDOWS = {
    "pre_MI": (-1.0, 0.0),
    "MI_0_1": (0.0, 1.0),
    "MI_1_2": (1.0, 2.0),
    "MI_2_3": (2.0, 3.0),
}


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


def power_track(x, lo, hi, win=WIN_SEC, step=STEP_SEC):
    """x (samples,3) ou (trials,samples,3); potencia por janela."""
    single = x.ndim == 2
    if single:
        x = x[None, ...]
    sos = butter(4, [lo / (FS / 2), hi / (FS / 2)],
                 btype="band", output="sos")
    xb = sosfiltfilt(sos, x, axis=1)
    n = max(1, int(round(win * FS)))
    st = max(1, int(round(step * FS)))
    starts = list(range(0, xb.shape[1] - n + 1, st)) or [0]
    bp = np.zeros((len(starts), xb.shape[0], xb.shape[2]))
    for i, s in enumerate(starts):
        bp[i] = np.var(xb[:, s:s + n, :], axis=1)
    t = (np.array(starts) + n / 2) / FS - PRE
    return (t, bp[:, 0, :]) if single else (t, bp)


def tf_spec(x):
    """Espectrograma (freq, canais, tempo), tempo relativo ao início da MI."""
    f, t, S = spectrogram(x, fs=FS, nperseg=64, noverlap=48,
                          axis=0, mode="psd", detrend="constant")
    sel = (f >= 4.0) & (f <= 40.0)
    return f[sel], t - PRE, S[sel]


def load_session(pasta):
    csv_path = sorted(Path(pasta).glob("eeg_mi3c*.csv"))[-1]
    json_path = sorted(Path(pasta).glob("eventos*.json"))[-1]
    meta = json.load(open(json_path, encoding="utf-8"))
    eeg, markers, _ = _load_own_csv(csv_path)
    fs_orig = float(meta.get("fs", 500.0))
    eeg = eeg[:, CH_OWN_IDX]
    eeg = filt_resample(eeg.T, fs_orig, FS, 4.0, 40.0).T
    escala = FS / fs_orig
    ini = np.where(markers == 901)[0]
    fim = np.where(markers == 903)[0]
    if not len(ini) or not len(fim):
        raise RuntimeError(f"{pasta}: sem marcadores 901/903")
    base_eyes = eeg[int(round(ini[0] * escala)):int(round(fim[0] * escala))]

    npre = int(PRE * FS)
    npost = int(POST * FS)
    segs, labels = [], []
    # Ancora: inicio da MI (780). A classe vem do cue 769/770 que o antecede.
    for pos780 in np.where(markers == 780)[0]:
        prev = markers[:pos780]
        ultimo, cls = -1, None
        for code, nome in ((769, "esquerda"), (770, "direita")):
            pos = np.where(prev == code)[0]
            if len(pos) and pos[-1] > ultimo:
                ultimo = pos[-1]
                cls = 0 if nome == "esquerda" else 1
        if cls is None:
            continue
        a = int(round(pos780 * escala))
        s, e = a - npre, a + npost
        if s < 0 or e > len(eeg):
            continue
        segs.append(eeg[s:e])
        labels.append(cls)
    return base_eyes, np.stack(segs), np.array(labels)


def media_classe(erd, labels, cls, t):
    m = labels == cls
    mean = erd[:, m, :].mean(axis=1)
    sem = erd[:, m, :].std(axis=1, ddof=1) / np.sqrt(m.sum())
    p = np.ones((erd.shape[0], 3))
    for ci in range(3):
        for ti in range(erd.shape[0]):
            p[ti, ci] = ttest_1samp(erd[ti, m, ci], 0.0).pvalue
    return {"mean": mean.tolist(), "sem": sem.tolist(),
            "q_fdr": bh_fdr(p).tolist(), "p": p.tolist(),
            "n_trials": int(m.sum()), "t": t.tolist()}


def main() -> int:
    out = Path("outputs/eeg/erd_ers_mi_onset")
    out.mkdir(parents=True, exist_ok=True)
    erds, labels, t_ref = [], [], None
    tf_trials, tf_labels, tf_freq, tf_time = [], [], None, None
    for pasta in (Path("outputs/eeg/piloto_me_mi_s01"),
                  Path("outputs/eeg/piloto_me_mi_s02")):
        base_eyes, segs, y = load_session(pasta)
        tf_freq, _, base_tf = tf_spec(base_eyes)
        base_tf = np.median(base_tf, axis=2)
        t_sess = None
        tf_sess = []
        for seg in segs:
            _, t_mi, S = tf_spec(seg)
            if t_mi is None:
                pass
            if tf_time is None:
                tf_time = t_mi
            erd_tf = 100.0 * (S / (base_tf[:, :, None] + 1e-12) - 1.0)
            tf_sess.append(np.clip(erd_tf, -200.0, 200.0).transpose(0, 2, 1))
        erd_sess = {}
        for band, (lo, hi) in BANDS.items():
            _, bp_base = power_track(base_eyes, lo, hi)
            ref = np.median(bp_base, axis=0)
            t_ref, bp = power_track(segs, lo, hi)
            erd = 100.0 * (bp / (ref[None, None, :] + 1e-12) - 1.0)
            erd_sess[band] = np.clip(erd, -200.0, 200.0)
        erds.append(erd_sess)
        labels.append(y)
        tf_trials.append(np.stack(tf_sess))
        tf_labels.append(y)
        print(pasta.name, "trials", len(y), "janela", t_ref[0], t_ref[-1],
              flush=True)
    labels = np.concatenate(labels)
    tf_all = np.concatenate(tf_trials)
    tf_y = np.concatenate(tf_labels)
    tf_mean = {cls: tf_all[tf_y == cls].mean(axis=0) for cls in (0, 1)}
    np.savez_compressed(out / "tempo_frequencia_mi_onset.npz",
                        freq=tf_freq, time=tf_time,
                        tf_esquerda=tf_mean[0], tf_direita=tf_mean[1],
                        tf_todos=tf_all.mean(axis=0), labels=tf_y)

    res = {}
    for band in BANDS:
        erd = np.concatenate([s[band] for s in erds], axis=1)
        bloco = {"t": t_ref.tolist(), "classes": {}}
        for cls in (0, 1):
            bloco["classes"][CLASSES[cls]] = media_classe(
                erd, labels, cls, t_ref)
        jan = {}
        for wname, (w0, w1) in WINDOWS.items():
            mw = (t_ref >= w0) & (t_ref <= w1)
            jan[wname] = {}
            for cls in (0, 1):
                m = labels == cls
                vals = erd[mw][:, m, :].mean(axis=0)
                p = np.ones(3)
                for ci in range(3):
                    p[ci] = ttest_1samp(vals[:, ci], 0.0).pvalue
                jan[wname][CLASSES[cls]] = {
                    "mean": vals.mean(axis=0).tolist(),
                    "sem": (vals.std(axis=0, ddof=1)
                            / np.sqrt(len(vals))).tolist(),
                    "p": p.tolist()}
        bloco["janelas"] = jan
        res[band] = bloco

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        cores = {0: "tab:blue", 1: "tab:red"}
        for ci, ch in enumerate(CH):
            ax = axes[ci]
            ax.axvline(0, color="k", lw=1.2)
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
            ax.set_xlabel("tempo em relação ao início da MI (s)")
            ax.set_xlim(-1.0, 3.0)
            ax.grid(alpha=0.25)
            if ci == 0:
                ax.set_ylabel("ERD/ERS vs olhos abertos (%)")
            ax.legend(fontsize=8)
        fig.suptitle("ERD/ERS em torno do início da MI (-1 s a +3 s) — baseline olhos abertos")
        fig.tight_layout()
        fig.savefig(out / f"erd_ers_mi_onset_{band}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ci, ch in enumerate(CH):
        for cj, cls in enumerate((0, 1)):
            ax = axes[ci, cj]
            Z = tf_mean[cls][:, :, ci].T
            im = ax.imshow(np.clip(Z, -100, 100), origin="lower",
                           aspect="auto", extent=[-1.0, 3.0,
                                                  tf_freq[0], tf_freq[-1]],
                           cmap="RdBu_r", vmin=-100, vmax=100)
            for fl in (8, 13, 20, 30):
                ax.axhline(fl, color="k", lw=0.4, alpha=0.35)
            ax.axvline(0, color="k", lw=1.0)
            ax.set_title(f"{ch} — {CLASSES[cls]}")
            if cj == 0:
                ax.set_ylabel("Hz")
            if ci == 2:
                ax.set_xlabel("tempo em relação ao início da MI (s)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="ERD/ERS vs olhos abertos (%)")
    fig.suptitle("Tempo-frequência em torno do início da MI por canal e classe")
    fig.savefig(out / "tempo_frequencia_mi_onset_por_canal.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    linhas = ["# ERD/ERS em torno do início da MI — janela −1 s a +3 s", "",
              "Baseline: olhos abertos (901–903). Âncora: início da MI (780); classe pelo cue anterior (769/770).",
              "A janela cobre os 1 s finais da transição ME→MI e os primeiros "
              "3 s da MI.", "",
              "| banda | janela | classe | C3 | C4 | Cz |",
              "|---|---|---|---:|---:|---:|"]
    for band in BANDS:
        for wname in WINDOWS:
            for cls in (0, 1):
                d = res[band]["janelas"][wname][CLASSES[cls]]
                p = d["p"]
                linhas.append(
                    f"| {band} | {wname} | {CLASSES[cls]} | "
                    f"{d['mean'][0]:.1f}%{'*' if p[0]<0.05 else ''} | "
                    f"{d['mean'][1]:.1f}%{'*' if p[1]<0.05 else ''} | "
                    f"{d['mean'][2]:.1f}%{'*' if p[2]<0.05 else ''} |")
    linhas += ["", "`*` = p < 0,05 contra zero."]
    (out / "resultados.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "relatorio.md").write_text("\n".join(linhas) + "\n",
                                      encoding="utf-8")
    print("Salvo em", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
