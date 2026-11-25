#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Controles FBCSP: baseline 2 s, ME/MI e contribuiÃ§Ã£o por banda."""
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
from eeg.mi_eegnet.data import _load_own_csv, filt_resample, CH_OWN_IDX  # noqa: E402
from eeg.fbcsp.fbcsp import FBCSP  # noqa: E402

FS = 128
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
         (24, 28), (28, 32), (32, 36), (36, 40)]
OUT = Path("outputs/eeg/auditoria_piloto")


def load_trials():
    trials = []
    for p in [Path("outputs/eeg/piloto_me_mi_s01"),
              Path("outputs/eeg/piloto_me_mi_s02")]:
        t, _ = fa.carrega_sessao_eeg(p)
        trials.extend(t)
    return trials


def baseline_2s() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, gs = [], [], []
    off = 0
    for p in [Path("outputs/eeg/piloto_me_mi_s01"),
              Path("outputs/eeg/piloto_me_mi_s02")]:
        csv = sorted(p.glob("eeg_mi3c*.csv"))[-1]
        eeg, markers, _ = _load_own_csv(csv)
        eeg = eeg[:, CH_OWN_IDX]
        eeg = filt_resample(eeg.T, 500.0, FS, 4.0, 40.0)
        for code, lab in ((769, 0), (770, 1)):
            for orig in np.where(markers == code)[0]:
                start = int(round(orig * FS / 500.0)) - 2 * FS
                end = start + 2 * FS
                if start >= 0 and end <= eeg.shape[1]:
                    Xs.append(eeg[:, start:end]); ys.append(lab)
                    gs.append(len(ys) - 1 + off)
        off += 60
    return np.stack(Xs), np.asarray(ys), np.asarray(gs)


def avalia(X, y, g, folds, bands=BANDS, n_csp=4):
    bas = []
    for f in folds:
        tr = np.isin(g, f["train_trials"]); te = np.isin(g, f["test_trials"])
        if not tr.any() or not te.any():
            continue
        m = FBCSP(bands=bands, fs=FS, n_csp_components=n_csp,
                  random_state=42).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        a, b = [], []
        for tid in np.unique(g[te]):
            q = g[te] == tid
            a.append(int(y[te][q][0]))
            b.append(int(np.bincount(pred[q]).argmax()))
        bas.append(balanced_accuracy_score(a, b))
    return {"fold_ba": [round(float(x), 4) for x in bas],
            "mean_ba": round(float(np.mean(bas)), 4),
            "std_ba": round(float(np.std(bas, ddof=1)), 4)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    folds = json.loads(Path("docs/validation/folds_piloto_groupkf.json")
                       .read_text(encoding="utf-8"))["folds"]
    trials = load_trials()
    res = {}
    # baseline 2 s
    Xb, yb, gb = baseline_2s()
    res["baseline_2s"] = avalia(Xb, yb, gb, folds)
    print("baseline 2s", res["baseline_2s"], flush=True)
    # ME/MI via feedback_adaptativo
    for fase, key in (("ME", "X_me"), ("MI", "X_mi")):
        Xs, ys, gs = [], [], []
        for i, t in enumerate(trials):
            X = np.asarray(t[key], float)
            n = 2 * FS
            if X.shape[1] < n:
                X = np.pad(X, ((0, 0), (0, n - X.shape[1])))
            X = X[:, :4 * FS]
            for st in range(0, X.shape[1] - n + 1, int(0.5 * FS)):
                Xs.append(X[:, st:st + n]); ys.append(int(t["classe"])); gs.append(i)
        X = np.stack(Xs); y = np.asarray(ys); g = np.asarray(gs)
        res[fase.lower()] = avalia(X, y, g, folds)
        print(fase, res[fase.lower()], flush=True)
    # contribuicao por banda (MI)
    Xs, ys, gs = [], [], []
    off = 0
    for p in [Path("outputs/eeg/piloto_me_mi_s01"),
              Path("outputs/eeg/piloto_me_mi_s02")]:
        X, y, g, _ = __import__("eeg.mi_eegnet.data", fromlist=["carrega_own"]).carrega_own(
            p, fs=FS, l_freq=4.0, h_freq=40.0, window_sec=2.0,
            step_sec=0.5, trial_sec=4.0, fase="MI")
        Xs.append(X); ys.append(y); gs.append(g + off); off += int(g.max()) + 1
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    bands_out = {}
    for b in BANDS:
        r = avalia(X, y, g, folds, bands=[b], n_csp=4)
        bands_out[f"{b[0]}-{b[1]}"] = r
        print("band", b, r["mean_ba"], flush=True)
    res["bandas_mi"] = bands_out
    # padroes espaciais do full
    m = FBCSP(bands=BANDS, fs=FS, n_csp_components=4, random_state=42).fit(X, y)
    patterns = {}
    for (b, csp) in zip(BANDS, m.csps_):
        pat = np.asarray(csp.patterns_)  # (2*n_comp, n_channels)
        patterns[f"{b[0]}-{b[1]}"] = {
            ch: [round(float(v), 4) for v in np.abs(pat[:, i]).tolist()]
            for i, ch in enumerate(["C3", "C4", "Cz"])}
    res["padroes_abs"] = patterns
    (OUT / "fbcsp_controles.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    linhas = ["# FBCSP â€” baseline 2 s, ME/MI e bandas", "",
              "| fase | BA trial | dp | folds |", "|---|---:|---:|---|"]
    for k in ("baseline_2s", "me", "mi"):
        r = res[k]
        linhas.append(f"| {k} | {r['mean_ba']:.4f} | {r['std_ba']:.4f} | {r['fold_ba']} |")
    linhas += ["", "## ContribuiÃ§Ã£o por banda (MI)", "",
               "| banda | BA trial |", "|---|---:|"]
    for k, r in bands_out.items():
        linhas.append(f"| {k} Hz | {r['mean_ba']:.4f} |")
    (OUT / "fbcsp_controles.md").write_text("\n".join(linhas) + "\n",
                                             encoding="utf-8")
    print("\n".join(linhas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

