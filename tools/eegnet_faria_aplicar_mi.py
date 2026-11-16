#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Treina EEGNet 3c/4cls no BCI IV 2a e aplica nos 4 s de MI do piloto.

- Modelo: EEGNet do Faria (1.412 parâmetros para 3ch/4cls).
- Treino: BCIC-IV 2a sessões T, C3/C4/Cz, 128 Hz, 4 classes.
- Aplicação: MI do piloto, 4 s, janelas deslizantes de 2 s, agregação por
  média de probabilidades.
- O piloto tem 3 classes (esquerda/direita/repouso), o BCI tem 4
  (esquerda/direita/pés/língua). Portanto a acurácia só é calculável para
  esquerda/direita; repouso é reportado como distribuição.
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
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import eegnet_faria_benchmark as ef

mne.set_log_level("ERROR")
FS = 128.0
N_TIMES = 256
BCI_CLASSES = ["esquerda", "direita", "pes", "lingua"]
PILOTO_CLASSES = ["esquerda", "direita", "repouso"]


def treina_final(X, y, device, epochs=500):
    """Treina nos dados BCI inteiros e devolve o modelo."""
    torch.manual_seed(42)
    model = ef.EEGNet(X.shape[1], X.shape[2], 4, kernel_length=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx].to(device))
            loss.backward(); opt.step()
        if ep == 0 or (ep + 1) % 100 == 0:
            print(f"  epoca {ep + 1:03d}/{epochs}", flush=True)
    model.eval()
    return model


def carrega_mi_piloto(pasta: Path):
    """4 s de MI do piloto, 128 Hz, padronizacao exponencial."""
    sys.path.insert(0, str(Path(__file__).parent))
    import treino_baseline_3classes as tb
    f = sorted(glob.glob(str(pasta / "eeg_mi3c*.csv")))[-1]
    j = sorted(glob.glob(str(pasta / "eventos*.json")))[-1]
    eeg, markers, ts = tb.load_eeg_csv(Path(f))
    meta = json.load(open(j, encoding="utf-8"))
    eeg = eeg[:, [14, 15, 16]]  # 15=C3, 16=C4, 17=Cz
    fs_orig = float(meta.get("fs", 500.0))
    # filtro e reamostragem contínuos, como no BCI
    eeg = mne.filter.filter_data(eeg.T, fs_orig, 4.0, 40.0,
                                 verbose=False).T
    eeg128 = signal.resample_poly(eeg, up=128, down=int(fs_orig),
                                  axis=0)
    eeg128 = ef.ema_standardize(eeg128.T).T
    # marcadores: 780 = início MI, 781 = fim; labels do cue anterior
    starts = np.where(markers == 780)[0]
    ends = np.where(markers == 781)[0]
    cue_map = {v: k for k, v in tb.CODE_TO_CLASS.items()}
    X, y, blocks = [], [], []
    for a, b in zip(starts, ends):
        s128 = int(round(a * 128 / fs_orig))
        e128 = int(round(b * 128 / fs_orig))
        if e128 - s128 < N_TIMES:
            continue
        # label do cue mais recente antes do início do MI
        prev = markers[:a]
        ultimo = -1; ultima_cls = "repouso"
        for code, nome in tb.CODE_TO_CLASS.items():
            pos = np.where(prev == code)[0]
            if len(pos) and pos[-1] > ultimo:
                ultimo = pos[-1]; ultima_cls = nome
        y.append(PILOTO_CLASSES.index(ultima_cls))
        seg = eeg128[s128:e128].T  # (3, ~512)
        # garante exatamente 4 s a 128 Hz
        alvo = int(4 * 128)
        if seg.shape[1] > alvo:
            seg = seg[:, :alvo]
        elif seg.shape[1] < alvo:
            seg = np.pad(seg, ((0, 0), (0, alvo - seg.shape[1])))
        X.append(seg)
        # bloco: 30 trials por bloco
        blocks.append(0 if len(X) <= 30 else 1)
    return np.stack(X).astype(np.float32), np.array(y), np.array(blocks)


def janelas_2s(trial):
    """(3,512) -> (n_windows,3,256), janela 2 s com passo 0.5 s."""
    n = trial.shape[1]
    out = []
    for s in range(0, n - N_TIMES + 1, 64):
        out.append(trial[:, s:s + N_TIMES])
    return np.stack(out).astype(np.float32)


def prediz_trial(model, trial, device):
    """Agrega janelas de 2 s por média de probabilidades."""
    wins = janelas_2s(trial)
    with torch.no_grad():
        p = model(torch.from_numpy(wins).to(device)).softmax(1).cpu().numpy()
    return p.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-2a", type=Path,
                    default=Path("Datasets/BCICIV_2a_gdf"))
    ap.add_argument("--piloto", type=Path, default=Path("_piloto3c_memi"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--saida", type=Path, default=Path("_eegnet_faria_3ch4cls"))
    ap.add_argument("--carregar-modelo", type=Path, default=None,
                    help="Se informado, pula o treino e carrega este .pt")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                  else "cpu")
    else:
        device = args.device
    print("device:", device, flush=True)

    if args.carregar_modelo:
        model = ef.EEGNet(3, N_TIMES, 4, kernel_length=32).to(device)
        model.load_state_dict(torch.load(args.carregar_modelo, map_location=device))
        model.eval()
        print("Modelo carregado:", args.carregar_modelo, flush=True)
    else:
        X, y, subj, _ = ef.carrega_2a(args.dataset_2a, 3, 4)
        print("BCI:", X.shape, flush=True)
        model = treina_final(X, y, device, args.epochs)
        torch.save(model.state_dict(), args.saida / "eegnet_2a_3ch_4cls.pt")
        print("Modelo salvo.", flush=True)

    Xp, yp, blocos = carrega_mi_piloto(args.piloto)
    print("Piloto MI:", Xp.shape, "labels:", np.unique(yp, return_counts=True),
          "blocos:", np.unique(blocos), flush=True)

    # Salva também os dados de janela deslizante usados na inferência.
    janelas = np.stack([janelas_2s(t) for t in Xp])  # (trials, 5, 3, 256)
    np.savez_compressed(args.saida / "janelas_mi_2s.npz",
                        X=janelas, y=yp, blocos=blocos,
                        window_sec=2.0, stride_sec=0.5)
    print("Janelas salvas:", janelas.shape, flush=True)

    probs = np.stack([prediz_trial(model, t, device) for t in Xp])
    pred = probs.argmax(1)
    pred_nomes = np.array(BCI_CLASSES)[pred]

    print("\nPredições por trial:")
    for i, (yt, ypred) in enumerate(zip(yp, pred_nomes)):
        print(f"  {i+1:02d}: label={PILOTO_CLASSES[yt]:>8s} "
              f"| pred={ypred:>8s} | probs="
              f"{dict(zip(BCI_CLASSES, np.round(probs[i], 3)))}")

    mask_lr = np.isin(yp, [0, 1])
    # acurácia 4-class nas classes que existem no piloto (left/right)
    acc4_lr = np.mean(pred_nomes[mask_lr] == np.array(BCI_CLASSES)[yp[mask_lr]])
    # acurácia binária restrita: argmax entre as probabilidades left/right
    p_lr = probs[mask_lr][:, :2]
    pred_lr = p_lr.argmax(1)
    acc_lr = np.mean(pred_lr == yp[mask_lr])
    print(f"\nLeft/right: acc 4-class={acc4_lr:.3f} | "
          f"acc binária restrita={acc_lr:.3f} | n={mask_lr.sum()}")
    print("Repouso: predições:", dict(zip(*np.unique(pred_nomes[~mask_lr],
                                                    return_counts=True))))

    np.savez(args.saida / "predicoes_piloto.npz",
             probs=probs, y_true=yp, classes=np.array(BCI_CLASSES))
    (args.saida / "resultado_aplicacao_mi.md").write_text(
        "# EEGNet BCI IV 2a 3ch/4cls aplicado no MI do piloto\n\n"
        f"- modelo: EEGNet Faria, 1.412 parâmetros, {args.epochs} epochs\n"
        f"- janelas: 2 s, passo 0,5 s, agregadas por média de probabilidades\n"
        f"- left/right: acc 4-class={acc4_lr:.3f}, binária={acc_lr:.3f}, "
        f"n={int(mask_lr.sum())}\n"
        f"- repouso: {dict(zip(*np.unique(pred_nomes[~mask_lr], return_counts=True)))}\n",
        encoding="utf-8")
    print("Resultado:", args.saida / "resultado_aplicacao_mi.md", flush=True)


if __name__ == "__main__":
    main()
