#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transferência BCI IV 2a -> piloto C3/C4/Cz.

Treina um LDA sobre features de bandpower (alpha/beta) usando APENAS C3, C4 e
Cz do BCI IV 2a (9 sujeitos x 2 sessões, 4 classes: esquerda, direita, pés,
língua), congela o modelo e aplica nas épocas de MI do piloto.

IMPORTANTE: não é um modelo pré-treinado publicado; é um baseline treinado
neste script. A transferência entre datasets/sujeitos é apenas exploratória.
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

CLASSES_BCI = ["esquerda", "direita", "pes", "lingua"]
CODE_BCI = {769: "esquerda", 770: "direita", 771: "pes", 772: "lingua"}
PICKS = ["EEG-C3", "EEG-C4", "EEG-Cz"]


def filtro_banda(x: np.ndarray, fs: float,
                 lo: float = 8.0, hi: float = 30.0) -> np.ndarray:
    return mne.filter.filter_data(x, fs, lo, hi, verbose=False)


def features_bandpower(x: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """x: (n_epochs, n_channels, n_samples) -> (n_epochs, 6)."""
    fr, psd = signal.welch(x, fs=fs, nperseg=min(256, x.shape[-1]),
                           axis=-1)
    feats = []
    for lo, hi in ((8, 13), (13, 30)):
        m = (fr >= lo) & (fr <= hi)
        feats.append(np.log(psd[:, :, m].mean(axis=2) + 1e-12))
    return np.concatenate(feats, axis=1)


def carrega_bci(pasta: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for f in sorted(pasta.glob("*.gdf")):
        raw = mne.io.read_raw_gdf(str(f), preload=True, verbose=False)
        raw.filter(8.0, 30.0, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        # O MNE devolve códigos internos (7,8,9,10); as anotações originais
        # 769..772 estão nas chaves de event_id.
        code_to_class = {}
        for code_orig, cls in CODE_BCI.items():
            chave = str(code_orig)
            if chave in event_id:
                code_to_class[int(event_id[chave])] = cls
        if not code_to_class:
            # Sessões E do BCI IV 2a têm apenas o cue 783 (desconhecido):
            # são avaliação, sem rótulo de classe. Não entram no treino.
            if f.name.endswith("E.gdf"):
                print(f"[bci] {f.name}: sessão de avaliação sem labels; ignorada")
            else:
                print(f"[aviso] sem cues rotulados em {f.name} (event_id={event_id})")
            continue
        ev = events[np.isin(events[:, 2], list(code_to_class))]
        if len(ev) == 0:
            print(f"[aviso] sem cues em {f.name}")
            continue
        # 1 a 4 s após o cue (janela motora clássica)
        ep = mne.Epochs(raw, ev, event_id={cls: code for code, cls in
                                          code_to_class.items()},
                        tmin=1.0, tmax=4.0, baseline=None, preload=True,
                        picks=PICKS, verbose=False)
        dados = ep.get_data()  # (n, 3, n_samples)
        labels = np.array([code_to_class[int(c)] for c in ep.events[:, 2]])
        xs.append(dados.astype(np.float64))
        ys.append(labels)
        print(f"[bci] {f.name}: {dados.shape[0]} épocas")
    return np.concatenate(xs, axis=0), np.concatenate(ys)


def carrega_piloto(pasta: Path) -> Tuple[np.ndarray, np.ndarray]:
    f = sorted(glob.glob(str(pasta / "eeg_mi3c*.csv")))[-1]
    j = sorted(glob.glob(str(pasta / "eventos*.json")))[-1]
    eeg, markers, ts = tb.load_eeg_csv(Path(f))
    meta = json.load(open(j, encoding="utf-8"))
    # canais físicos: 15=C3, 16=C4, 17=Cz (1-based)
    eeg = eeg[:, [14, 15, 16]]
    fs = float(meta.get("fs", 500.0))
    eeg = filtro_banda(eeg, fs)
    # Extrai a 500 Hz (marcadores e EEG no mesmo relógio) e só depois
    # decima cada época para 250 Hz, igual ao BCI IV 2a.
    ep, labels, _, _ = tb.extract_epochs_from_markers(
        eeg, markers, ts, fs, mi_start_code=780,
        mi_end_code=781, tmin=0.5, tmax=3.5, cue_codes=tb.CODE_TO_CLASS)
    ep250 = []
    for e in ep:
        e = np.asarray(e, np.float64)
        if fs > 250:
            e = e[::2, :]
        ep250.append(e)
    # extract_epochs devolve (n_samples, n_channels); o modelo espera
    # (n_epochs, n_channels, n_samples).
    return np.stack(ep250).transpose(0, 2, 1), np.array(labels)


def padroniza_treino(X, X_val=None):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xn = (X - mu) / sd
    if X_val is None:
        return Xn, mu, sd
    return Xn, (X_val - mu) / sd, mu, sd


def avalia_bci():
    """Split por sujeito: treina em A01..A06, testa em A07..A09."""
    X, y = carrega_bci(Path("BCICIV_2a_gdf"))
    # arquivos vêm em ordem alfabética; ~288 trials por arquivo
    n_arq = 18
    corte = int(len(y) * 12 // 18)
    Xtr, ytr = X[:corte], y[:corte]
    Xte, yte = X[corte:], y[corte:]
    Ftr = features_bandpower(Xtr)
    Fte = features_bandpower(Xte)
    Ftr, Fte, mu, sd = padroniza_treino(Ftr, Fte)
    lda = tb.lda_train(Ftr, ytr, reg=1e-2)
    pred = tb.lda_predict(Fte, lda)
    ba = tb.balanced_accuracy(yte, pred)
    print(f"[bci] split sujeito treino={len(ytr)} teste={len(yte)} "
          f"balanced_accuracy={ba:.3f}")
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bci", type=Path, default=Path("BCICIV_2a_gdf"))
    ap.add_argument("--piloto", type=Path, default=Path("outputs/eeg/piloto3c_memi"))
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/bci2a_transfer"))
    ap.add_argument("--avaliar-bci", action="store_true")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    print("Carregando BCI IV 2a...")
    Xb, yb = carrega_bci(args.bci)
    print("BCI:", Xb.shape, "classes:", dict(zip(*np.unique(yb, return_counts=True))))
    if args.avaliar_bci:
        # avaliação rápida por sujeito (12 arquivos treino / 6 teste)
        corte = int(len(yb) * 12 // 18)
        Ftr = features_bandpower(Xb[:corte]); Fte = features_bandpower(Xb[corte:])
        Ftr, Fte, mu, sd = padroniza_treino(Ftr, Fte)
        lda = tb.lda_train(Ftr, yb[:corte], reg=1e-2)
        pred = tb.lda_predict(Fte, lda)
        print("BCI holdout BA:", round(tb.balanced_accuracy(yb[corte:], pred), 3))

    # treina no BCI inteiro e congela
    Fb = features_bandpower(Xb)
    Fb, mu, sd = padroniza_treino(Fb)
    lda = tb.lda_train(Fb, yb, reg=1e-2)
    np.savez(args.saida / "modelo_bci2a_c3c4cz.npz",
             mu=mu, sd=sd, classes=np.array(lda["classes"]),
             w=lda["W_lda"], mean=lda["mean_global"],
             projected_means=np.array([lda["projected_means"][c]
                                       for c in lda["classes"]]),
             priors=np.array([lda["class_priors"][c] for c in lda["classes"]]))
    print("Modelo BCI salvo.")

    # piloto
    Xp, yp = carrega_piloto(args.piloto)
    Fp = features_bandpower(Xp)
    Fp = (Fp - mu) / sd
    pred = tb.lda_predict(Fp, lda)
    proba = tb.lda_predict_proba(Fp, lda)
    print("\nPiloto:", Xp.shape, "labels:", dict(zip(*np.unique(yp, return_counts=True))))
    print("Predições BCI:", dict(zip(*np.unique(pred, return_counts=True))))
    print("Concordância com label do piloto:",
          round(float(np.mean(pred == yp)), 3))
    np.savez(args.saida / "predicoes_piloto.npz",
             y_true=yp, y_pred=pred, proba=proba,
             classes=np.array(lda["classes"]))
    # Modelo binário esquerda x direita (mais próximo do alvo do piloto)
    mask_bin = np.isin(yb, ["esquerda", "direita"])
    lda_bin = tb.lda_train(Fb[mask_bin], yb[mask_bin], reg=1e-2)
    mask_p = np.isin(yp, ["esquerda", "direita"])
    pred_bin = tb.lda_predict(Fp[mask_p], lda_bin)
    proba_bin = tb.lda_predict_proba(Fp[mask_p], lda_bin)
    print("\nBinário BCI -> piloto:")
    print("  piloto L/R:", dict(zip(*np.unique(pred_bin, return_counts=True))))
    print("  concordância L/R:", round(float(np.mean(pred_bin == yp[mask_p])), 3))
    np.savez(args.saida / "modelo_bci2a_lr.npz",
             mu=mu, sd=sd, classes=np.array(lda_bin["classes"]),
             w=lda_bin["W_lda"], mean=lda_bin["mean_global"],
             projected_means=np.array([lda_bin["projected_means"][c]
                                       for c in lda_bin["classes"]]),
             priors=np.array([lda_bin["class_priors"][c]
                              for c in lda_bin["classes"]]))

    # relatório simples
    linhas = ["# Transferência BCI IV 2a -> piloto", "",
              f"Épocas BCI: {Xb.shape[0]} | C3,C4,Cz | 4 classes", 
              f"Épocas piloto: {Xp.shape[0]} | 3 s", "",
              "## Predições no piloto (4 classes)", ""]
    for i in range(len(yp)):
        linhas.append(f"- ep {i+1}: label={yp[i]} | pred={pred[i]} | "
                      f"proba={dict(zip(lda['classes'], np.round(proba[i],3)))}")
    linhas += ["", "## Predições binárias esquerda x direita", ""]
    for i, (yt, yp_) in enumerate(zip(yp[mask_p], pred_bin)):
        linhas.append(f"- ep {i+1}: label={yt} | pred={yp_}")
    linhas += ["", f"Concordância binária: {np.mean(pred_bin == yp[mask_p]):.3f}"]
    (args.saida / "relatorio_transferencia.md").write_text(
        "\n".join(linhas), encoding="utf-8")
    print("Relatório:", args.saida / "relatorio_transferencia.md")


if __name__ == "__main__":
    main()
