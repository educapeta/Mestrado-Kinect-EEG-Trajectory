#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transferência BCI IV 2a -> piloto, com o pipeline alinhado.

- BCI IV 2a: apenas sessões T rotuladas, canais C3/C4/Cz, 100 Hz, janela MI 4 s.
- Piloto: mesmos 3 canais, 100 Hz, janela MI 4 s.
- Normalização por época e por canal (z-score) nos dois.
- Modelo treinado apenas no BCI; o piloto é usado como validação externa.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import mne
from scipy import signal

import sys
sys.path.insert(0, str(Path(__file__).parent))
import treino_baseline_3classes as tb

mne.set_log_level("ERROR")
FS_ALVO = 100.0
PICKS = ["EEG-C3", "EEG-C4", "EEG-Cz"]
CODE_BCI = {769: "esquerda", 770: "direita", 771: "pes", 772: "lingua"}


def _zscore_epocas(x: np.ndarray) -> np.ndarray:
    """(n, ch, t) -> z-score por época e canal."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True) + 1e-9
    return (x - mu) / sd


def _bandpower(x: np.ndarray, fs: float = FS_ALVO) -> np.ndarray:
    fr, psd = signal.welch(x, fs=fs, nperseg=min(256, x.shape[-1]), axis=-1)
    feats = []
    for lo, hi in ((8, 13), (13, 30)):
        m = (fr >= lo) & (fr <= hi)
        feats.append(np.log(psd[:, :, m].mean(axis=2) + 1e-12))
    return np.concatenate(feats, axis=1)


def carrega_bci(pasta: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for f in sorted(pasta.glob("*T.gdf")):
        raw = mne.io.read_raw_gdf(str(f), preload=True, verbose=False)
        raw.filter(0.5, 40.0, verbose=False)
        raw.resample(FS_ALVO, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        code_to_class = {}
        for code_orig, cls in CODE_BCI.items():
            chave = str(code_orig)
            if chave in event_id:
                code_to_class[int(event_id[chave])] = cls
        if not code_to_class:
            print(f"[bci] {f.name}: sem cues rotulados; ignorado")
            continue
        ev = events[np.isin(events[:, 2], list(code_to_class))]
        # cue em 2 s; MI até 6 s => 4 s depois do cue
        ep = mne.Epochs(raw, ev, event_id={cls: c for c, cls in code_to_class.items()},
                        tmin=0.0, tmax=4.0, baseline=None, preload=True,
                        picks=PICKS, verbose=False)
        dados = _zscore_epocas(ep.get_data())
        labels = np.array([code_to_class[int(c)] for c in ep.events[:, 2]])
        xs.append(dados.astype(np.float64))
        ys.append(labels)
        print(f"[bci] {f.name}: {dados.shape[0]} épocas, {dados.shape[-1]} amostras")
    return np.concatenate(xs), np.concatenate(ys)


def carrega_piloto(pasta: Path) -> Tuple[np.ndarray, np.ndarray]:
    f = sorted(glob.glob(str(pasta / "eeg_mi3c*.csv")))[-1]
    j = sorted(glob.glob(str(pasta / "eventos*.json")))[-1]
    eeg, markers, ts = tb.load_eeg_csv(Path(f))
    meta = json.load(open(j, encoding="utf-8"))
    # 15=C3, 16=C4, 17=Cz (índices 0-based 14,15,16)
    eeg = eeg[:, [14, 15, 16]]
    fs = float(meta.get("fs", 500.0))
    # extrai a 500 Hz (marcadores alinhados) e reamostra cada época para 100 Hz
    ep, labels, _, _ = tb.extract_epochs_from_markers(
        eeg, markers, ts, fs, mi_start_code=780, mi_end_code=781,
        tmin=0.0, tmax=4.0, cue_codes=tb.CODE_TO_CLASS)
    ep100 = []
    for e in ep:
        e = np.asarray(e, np.float64)
        if fs != FS_ALVO:
            e = signal.resample(e, int(round(e.shape[0] * FS_ALVO / fs)), axis=0)
        ep100.append(e)
    X = _zscore_epocas(np.stack(ep100).transpose(0, 2, 1))
    return X, np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bci", type=Path, default=Path("BCICIV_2a_gdf"))
    ap.add_argument("--piloto", type=Path, default=Path("_piloto3c_memi"))
    ap.add_argument("--saida", type=Path, default=Path("_bci2a_transfer_100hz"))
    ap.add_argument("--avaliar-bci", action="store_true")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    Xb, yb = carrega_bci(args.bci)
    print("BCI:", Xb.shape, "classes:", dict(zip(*np.unique(yb, return_counts=True))))
    Fb = _bandpower(Xb)
    lda = tb.lda_train(Fb, yb, reg=1e-2)

    # avaliação rápida no BCI: últimos 3 sujeitos (A07T..A09T) como holdout
    if args.avaliar_bci:
        n_arq = 9
        corte = int(len(yb) * 6 / n_arq)
        lda_h = tb.lda_train(Fb[:corte], yb[:corte], reg=1e-2)
        pred_h = tb.lda_predict(Fb[corte:], lda_h)
        print("[bci] holdout BA:", round(tb.balanced_accuracy(yb[corte:], pred_h), 3))

    Xp, yp = carrega_piloto(args.piloto)
    Fp = _bandpower(Xp)
    print("Piloto:", Xp.shape, "labels:", dict(zip(*np.unique(yp, return_counts=True))))

    pred = tb.lda_predict(Fp, lda)
    proba = tb.lda_predict_proba(Fp, lda)
    print("Predições 4 classes no piloto:", dict(zip(*np.unique(pred, return_counts=True))))
    print("Concordância com labels do piloto:", round(float(np.mean(pred == yp)), 3))

    # binário esquerda x direita (treinado só no BCI)
    mask_b = np.isin(yb, ["esquerda", "direita"])
    lda_b = tb.lda_train(Fb[mask_b], yb[mask_b], reg=1e-2)
    mask_p = np.isin(yp, ["esquerda", "direita"])
    pred_b = tb.lda_predict(Fp[mask_p], lda_b)
    proba_b = tb.lda_predict_proba(Fp[mask_p], lda_b)
    print("Binário no piloto:", dict(zip(*np.unique(pred_b, return_counts=True))),
          "concordância:", round(float(np.mean(pred_b == yp[mask_p])), 3))

    np.savez(args.saida / "modelo_bci2a_100hz.npz",
             classes=np.array(lda["classes"]), w=lda["W_lda"],
             mean=lda["mean_global"],
             projected_means=np.array([lda["projected_means"][c]
                                       for c in lda["classes"]]),
             priors=np.array([lda["class_priors"][c] for c in lda["classes"]]))
    np.savez(args.saida / "predicoes_piloto_100hz.npz", y_true=yp,
             y_pred=pred, proba=proba, classes=np.array(lda["classes"]))
    linhas = ["# Transferência BCI IV 2a -> piloto (100 Hz, 4 s)", "",
              f"BCI: {Xb.shape} | classes {dict(zip(*np.unique(yb, return_counts=True)))}",
              f"Piloto: {Xp.shape}", "",
              "## 4 classes", "",
              f"Distribuição predita: {dict(zip(*np.unique(pred, return_counts=True)))}",
              f"Concordância com labels do piloto: {np.mean(pred == yp):.3f}", "",
              "## Binário L/R", "",
              f"Distribuição predita: {dict(zip(*np.unique(pred_b, return_counts=True)))}",
              f"Concordância: {np.mean(pred_b == yp[mask_p]):.3f}"]
    (args.saida / "relatorio_transferencia_100hz.md").write_text(
        "\n".join(linhas), encoding="utf-8")
    print("Relatório:", args.saida / "relatorio_transferencia_100hz.md")


if __name__ == "__main__":
    main()
