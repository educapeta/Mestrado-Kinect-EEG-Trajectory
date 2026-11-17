#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validação 1 e 2: ME positivo + labels embaralhados (CPU).

- Dados: piloto `outputs/eeg/piloto3c_memi` (ME e MI, 4 s), canais C3/C4/Cz.
- Janelas: 2 s, passo 0,5 s (5 por trial).
- Features: log bandpower (alpha/beta) por canal.
- Classificador: LDA com shrinkage.
- Validação: StratifiedGroupKFold por trial (5 folds); todas as janelas do
  mesmo trial ficam no mesmo fold.
- Teste 1: ME esquerda × direita acima do acaso.
- Teste 2: labels embaralhados no nível do trial; p-valor por permutação.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import mne
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import treino_baseline_3classes as tb

mne.set_log_level("ERROR")
FS = 500.0
WIN_SEC = 2.0
STEP_SEC = 0.5
N_WIN = int(WIN_SEC * FS)
STEP = int(STEP_SEC * FS)
BANDS = ((4, 8), (8, 13), (13, 30))


def carrega_fase(pasta: Path, fase: str):
    f = sorted(glob.glob(str(pasta / "eeg_mi3c*.csv")))[-1]
    j = sorted(glob.glob(str(pasta / "eventos*.json")))[-1]
    eeg, markers, ts = tb.load_eeg_csv(Path(f))
    meta = json.load(open(j, encoding="utf-8"))
    eeg = eeg[:, [14, 15, 16]]  # C3, C4, Cz
    fs = float(meta.get("fs", FS))
    eeg = mne.filter.filter_data(eeg.T, fs, 4.0, 40.0, verbose=False).T
    ini, fim = (778, 779) if fase == "ME" else (780, 781)
    ep, labels, tri, blocks = tb.extract_epochs_from_markers(
        eeg, markers, ts, fs, mi_start_code=ini, mi_end_code=fim,
        tmin=0.0, tmax=4.0, cue_codes=tb.CODE_TO_CLASS)
    return np.stack(ep), np.array(labels), np.array(tri), np.array(blocks)


def janelas(trial):
    out = []
    for s in range(0, trial.shape[0] - N_WIN + 1, STEP):
        out.append(trial[s:s + N_WIN, :])
    return np.stack(out)


def features(win):
    """win (n, t, ch) -> (n, ch*bands)."""
    fr, psd = signal.welch(win, fs=FS, nperseg=256, axis=1)
    fs_list = []
    for lo, hi in BANDS:
        m = (fr >= lo) & (fr <= hi)
        fs_list.append(np.log(psd[:, m, :].mean(axis=1) + 1e-12))
    F = np.concatenate(fs_list, axis=1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def prepara(fase):
    X4, y_str, tri, blocks = carrega_fase(Path("outputs/eeg/piloto3c_memi"), fase)
    mapa = {"esquerda": 0, "direita": 1, "repouso": 2}
    y = np.array([mapa[s] for s in y_str])
    # A validação 1/2 é binária: somente mão esquerda e direita.
    mask = np.isin(y, [0, 1])
    X4, y = X4[mask], y[mask]
    Xw, yw, gw = [], [], []
    for i, t in enumerate(X4):
        w = janelas(t)
        Xw.append(w)
        yw.extend([int(y[i])] * len(w))
        gw.extend([i] * len(w))
    Xw = np.concatenate(Xw, axis=0)
    F = features(Xw)
    return F, np.array(yw), np.array(gw), y


def cv_trial(F, yw, gw):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    yt, yp = [], []
    n_trials = len(np.unique(gw))
    for tr, te in sgkf.split(F, yw, groups=gw):
        clf = make_pipeline(StandardScaler(),
                            LinearDiscriminantAnalysis(solver="lsqr",
                                                       shrinkage="auto"))
        clf.fit(F[tr], yw[tr])
        p = clf.predict_proba(F[te])
        # agrega as janelas por trial
        for g in np.unique(gw[te]):
            m = gw[te] == g
            yt.append(int(yw[te][m][0]))
            yp.append(int(p[m].mean(axis=0).argmax()))
    yt = np.array(yt); yp = np.array(yp)
    return (float(np.mean(yt == yp)),
            float(balanced_accuracy_score(yt, yp)),
            confusion_matrix(yt, yp, labels=[0, 1]),
            yt, yp)


def permuta(F, yw, gw, n_perm=200, seed=123):
    rng = np.random.default_rng(seed)
    # labels por trial
    trials = np.unique(gw)
    y_trial = np.array([yw[gw == g][0] for g in trials])
    bas = []
    for _ in range(n_perm):
        yperm = y_trial[rng.permutation(len(y_trial))]
        yw_perm = np.array([yperm[np.where(trials == g)[0][0]] for g in gw])
        _, ba, _, _, _ = cv_trial(F, yw_perm, gw)
        bas.append(ba)
    return np.array(bas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/validacao_me_shuffle"))
    ap.add_argument("--permutacoes", type=int, default=200)
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    resultados = {}
    for fase in ("ME", "MI"):
        F, yw, gw, y_trial = prepara(fase)
        acc, ba, cm, yt, yp = cv_trial(F, yw, gw)
        print(f"\n[{fase}] trials={len(yt)} acc={acc:.3f} BA={ba:.3f}")
        print(cm)
        shuf = permuta(F, yw, gw, n_perm=args.permutacoes)
        p = (1 + np.sum(shuf >= ba)) / (1 + len(shuf))
        print(f"[{fase}] shuffle BA média={shuf.mean():.3f} "
              f"p95={np.percentile(shuf, 95):.3f} p={p:.4f}")
        resultados[fase] = {
            "n_trials": int(len(yt)),
            "acc": acc, "ba": ba,
            "confusao": cm.tolist(),
            "shuffle_mean": float(shuf.mean()) if len(shuf) else None,
            "shuffle_p95": float(np.percentile(shuf, 95)) if len(shuf) else None,
            "p_perm": float(p) if np.isfinite(p) else None,
        }

    # ME e MI usam só left/right (20+20). Para 3 classes o script poderia ser
    # estendido, mas o alvo aqui é validar o hardware nas duas mãos.
    lines = ["# Validação ME positivo + labels embaralhados", "",
             "Janelas de 2 s (passo 0,5 s), 5 folds agrupados por trial.", ""]
    for fase, r in resultados.items():
        lines += [f"## {fase}", "",
                  f"- trials: {r['n_trials']}",
                  f"- acc: {r['acc']:.3f}",
                  f"- balanced accuracy: {r['ba']:.3f}",
                  f"- confusão: `{r['confusao']}`"]
        if r["p_perm"] is not None:
            lines += [f"- shuffle BA média: {r['shuffle_mean']:.3f}",
                      f"- shuffle p95: {r['shuffle_p95']:.3f}",
                      f"- p-valor: {r['p_perm']:.4f}"]
        lines.append("")
    (args.saida / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")
    (args.saida / "resultados.json").write_text(
        json.dumps(resultados, indent=2), encoding="utf-8")
    print("\nRelatório:", args.saida / "relatorio.md")


if __name__ == "__main__":
    main()
