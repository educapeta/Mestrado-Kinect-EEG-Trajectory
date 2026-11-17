#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSP+LDA nas duas sessões do novo piloto (ME e MI).

Avaliações:
- within-session: 5-fold agrupado por trial;
- cross-session: treina em uma sessão e testa na outra (e vice-versa);
- ME e MI separadamente;
- soft voting por trial.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mne.set_log_level("ERROR")
FS = 128.0
WIN = 256
STEP = 64
CH = [14, 15, 16]  # C3, C4, Cz


def _finite(Z):
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def load_session(path: Path):
    f = sorted(path.glob("eeg_mi3c*.csv"))[-1]
    j = sorted(path.glob("eventos*.json"))[-1]
    rows = []
    with open(f, encoding="utf-8") as h:
        for r in csv.reader(h):
            if r and not r[0].startswith("#"):
                rows.append(r)
    eeg = np.array([[float(v) for v in r[1:-1]] for r in rows[1:]],
                   dtype=np.float64)
    markers = np.array([int(float(r[-1])) for r in rows[1:]], dtype=int)
    meta = json.load(open(j, encoding="utf-8"))
    fs = float(meta.get("fs", 500.0))
    eeg = eeg[:, CH]
    eeg = mne.filter.filter_data(eeg.T, fs, 8.0, 32.0, verbose=False).T
    eeg = _resample(eeg, fs)
    return eeg, markers, fs, meta


def _resample(eeg, fs_in):
    if abs(fs_in - FS) < 1e-9:
        return eeg
    from scipy.signal import resample_poly
    up = int(round(FS)); down = int(round(fs_in))
    return resample_poly(eeg, up, down, axis=0)


def extract_phase(eeg, markers, fs_orig, phase):
    ini, fim = (778, 779) if phase == "ME" else (780, 781)
    starts = np.where(markers == ini)[0]
    ends = np.where(markers == fim)[0]
    xs, ys, gs = [], [], []
    n_win = int(2.0 * FS)
    step = int(0.5 * FS)
    for tid, (a, b) in enumerate(zip(starts, ends)):
        s = int(round(a * FS / fs_orig)); e = int(round(b * FS / fs_orig))
        seg = eeg[s:e].T
        if seg.shape[1] < int(4 * FS):
            seg = np.pad(seg, ((0, 0), (0, int(4 * FS) - seg.shape[1])))
        seg = seg[:, :int(4 * FS)]
        # ultimo cue antes do trial
        prev = markers[:a]
        ultimo = -1; cls = None
        for code, nome, lab in [(769, "esquerda", 0), (770, "direita", 1)]:
            pos = np.where(prev == code)[0]
            if len(pos) and pos[-1] > ultimo:
                ultimo = pos[-1]; cls = lab
        if cls is None:
            continue
        for w in range(0, seg.shape[1] - n_win + 1, step):
            xs.append(seg[:, w:w + n_win])
            ys.append(cls); gs.append(tid)
    return np.stack(xs).astype(np.float32), np.array(ys), np.array(gs)


def make_csp():
    return Pipeline([
        ("csp", CSP(n_components=2, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(_finite)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr",
                                           shrinkage="auto")),
    ])


def soft_vote(p, g):
    uniq = np.unique(g)
    pm = np.stack([p[g == x].mean(axis=0) for x in uniq])
    return uniq, pm, pm.argmax(1)


def metrics(y, pred, proba=None):
    y = np.asarray(y); pred = np.asarray(pred)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "confusion": cm.tolist(),
    }
    if proba is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, np.asarray(proba)[:, 1]))
    return out


def eval_within(X, y, g):
    clf = make_csp()
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    yt, yp, pp = [], [], []
    for tr, te in skf.split(X, y, groups=g):
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        _, pm, pred = soft_vote(p, g[te])
        ytrue = np.array([y[te][g[te] == x][0] for x in np.unique(g[te])])
        yt.extend(ytrue); yp.extend(pred); pp.extend(pm.tolist())
    return metrics(yt, yp, np.array(pp))


def eval_cross(Xtr, ytr, gtr, Xte, yte, gte):
    clf = make_csp()
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)
    _, _, pred = soft_vote(p, gte)
    ytrue = np.array([yte[gte == x][0] for x in np.unique(gte)])
    return metrics(ytrue, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=Path, default=Path("outputs/eeg/piloto_me_mi_s01"))
    ap.add_argument("--s2", type=Path, default=Path("outputs/eeg/piloto_me_mi_s02"))
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/csp_lda_novo_piloto"))
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, path in (("s1", args.s1), ("s2", args.s2)):
        eeg, markers, fs, meta = load_session(path)
        data[name] = {"eeg": eeg, "markers": markers, "fs": fs}
        print(name, eeg.shape, "fs", fs, flush=True)

    res = {}
    for phase in ("ME", "MI"):
        print(f"\n=== {phase} ===", flush=True)
        ph = {}
        for sname, d in data.items():
            X, y, g = extract_phase(d["eeg"], d["markers"], d["fs"], phase)
            ph[sname] = {"X": X, "y": y, "g": g}
            print(f"{sname}: {X.shape} trials={len(np.unique(g))}", flush=True)
            r = eval_within(X, y, g)
            print(f"{sname} within: BA={r['balanced_accuracy']:.3f} "
                  f"acc={r['accuracy']:.3f}", flush=True)
            ph[sname]["within"] = r
        # cross-session
        s1, s2 = ph["s1"], ph["s2"]
        ph["cross_1_2"] = eval_cross(s1["X"], s1["y"], s1["g"],
                                     s2["X"], s2["y"], s2["g"])
        ph["cross_2_1"] = eval_cross(s2["X"], s2["y"], s2["g"],
                                     s1["X"], s1["y"], s1["g"])
        print(f"cross s1->s2: BA={ph['cross_1_2']['balanced_accuracy']:.3f}",
              flush=True)
        print(f"cross s2->s1: BA={ph['cross_2_1']['balanced_accuracy']:.3f}",
              flush=True)
        res[phase] = ph

    # salva apenas metricas, sem arrays de janelas
    out = {}
    for phase, ph in res.items():
        out[phase] = {}
        for key in ("s1", "s2"):
            out[phase][key] = {"within": ph[key]["within"]}
        for key in ("cross_1_2", "cross_2_1"):
            out[phase][key] = ph[key]
    (args.saida / "resultados.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# CSP+LDA — novo piloto ME/MI", ""]
    for phase, ph in res.items():
        lines += [f"## {phase}", ""]
        for key in ("s1", "s2"):
            r = ph[key]["within"]
            lines += [f"### {key} within-session", "",
                      f"- BA: {r['balanced_accuracy']:.3f}",
                      f"- acc: {r['accuracy']:.3f}",
                      f"- confusão: {r['confusion']}", ""]
        for key in ("cross_1_2", "cross_2_1"):
            r = ph[key]
            lines += [f"### {key} cross-session", "",
                      f"- BA: {r['balanced_accuracy']:.3f}",
                      f"- acc: {r['accuracy']:.3f}",
                      f"- confusão: {r['confusion']}", ""]
    (args.saida / "relatorio.md").write_text("\n".join(lines),
                                              encoding="utf-8")
    print("Salvo em", args.saida)


if __name__ == "__main__":
    main()
