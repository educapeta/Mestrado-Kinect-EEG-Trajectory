#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BCIC-IV 2b: EEGNet 3ch/2cls (1.154 parametros) e teste binário no piloto.

- Treina no BCIC-IV 2b, sessões T, C3/Cz/C4, 128 Hz, janela 2 s.
- Salva o modelo final.
- Gera as janelas de 2 s das épocas de MI do piloto e testa SOMENTE as
  épocas de mão esquerda e direita (descarta repouso).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import eegnet_faria_benchmark as ef
from eegnet_faria_aplicar_mi import carrega_mi_piloto, janelas_2s

N_TIMES = 256
CLASSES = ["esquerda", "direita"]


def treina_final_2b(X, y, device, epochs=500, batch_size=128,
                    progress_every=10):
    torch.manual_seed(42)
    model = ef.EEGNet(3, N_TIMES, 2, kernel_length=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).long().to(device)
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
        if ep == 0 or (ep + 1) % progress_every == 0 or ep == epochs - 1:
            print(f"  epoca {ep + 1:03d}/{epochs}", flush=True)
    model.eval()
    return model


def prediz_trial(model, trial, device):
    wins = janelas_2s(trial)
    with torch.no_grad():
        p = model(torch.from_numpy(wins).to(device)).softmax(1).cpu().numpy()
    return p.mean(axis=0), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-2b", type=Path,
                    default=Path("Datasets/BCICIV_2b_gdf"))
    ap.add_argument("--piloto", type=Path, default=Path("_piloto3c_memi"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Protocolo Faria = 32; 128 acelera o teste")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--carregar-modelo", type=Path, default=None)
    ap.add_argument("--saida", type=Path,
                    default=Path("_eegnet_faria_2b_3ch2cls"))
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
        model = ef.EEGNet(3, N_TIMES, 2, kernel_length=32).to(device)
        model.load_state_dict(torch.load(args.carregar_modelo,
                                         map_location=device))
        model.eval()
        print("Modelo carregado:", args.carregar_modelo, flush=True)
    else:
        cache = args.saida / "cache_bci_2b.npz"
        if cache.exists():
            d = np.load(cache)
            X, y = d["X"], d["y"]
            print("Cache 2b:", X.shape, flush=True)
        else:
            X, y, subj, run = ef.carrega_2b(args.dataset_2b)
            np.savez_compressed(cache, X=X, y=y)
            print("Cache 2b salvo:", X.shape, flush=True)
        model = treina_final_2b(X, y, device, args.epochs,
                                args.batch_size, args.progress_every)
        torch.save(model.state_dict(), args.saida / "eegnet_2b_3ch_2cls.pt")
        print("Modelo salvo.", flush=True)

    Xp, yp, blocos = carrega_mi_piloto(args.piloto)
    # somente mão esquerda (0) e direita (1)
    mask = np.isin(yp, [0, 1])
    Xp, yp = Xp[mask], yp[mask]
    print("Piloto left/right:", Xp.shape, "labels:",
          dict(zip(*np.unique(yp, return_counts=True))), flush=True)

    probs = []
    janelas = []
    for t in Xp:
        pr, w = prediz_trial(model, t, device)
        probs.append(pr); janelas.append(w)
    probs = np.stack(probs)
    janelas = np.stack(janelas)  # (trials, 5, 2)
    pred = probs.argmax(1)
    acc = float(np.mean(pred == yp))
    rec = [float(np.mean(pred[yp == c] == c)) for c in (0, 1)]
    print(f"\nBINÁRIO NO PILOTO: acc={acc:.3f} | "
          f"BA={np.mean(rec):.3f} | n={len(yp)}", flush=True)
    print("Confusão (linha=verdade, coluna=pred):")
    cm = np.zeros((2, 2), dtype=int)
    for a, b in zip(yp, pred):
        cm[a, b] += 1
    print(cm, flush=True)
    print("Predições:", dict(zip(*np.unique(pred, return_counts=True))))

    np.savez_compressed(args.saida / "janelas_mi_left_right.npz",
                        X=janelas, probs=probs, y=yp,
                        trial_windows=janelas)
    (args.saida / "resultado_2b_teste_mi.md").write_text(
        "# EEGNet BCIC-IV 2b 3ch/2cls → MI do piloto (left/right)\n\n"
        f"- modelo: 1.154 parâmetros, {args.epochs} epochs, "
        f"batch={args.batch_size}\n"
        f"- trials de teste: {len(yp)} (esquerda/direita)\n"
        f"- acurácia: {acc:.3f}\n"
        f"- balanced accuracy: {np.mean(rec):.3f}\n"
        f"- confusão:\n```\n{cm}\n```\n",
        encoding="utf-8")
    print("Relatório:", args.saida / "resultado_2b_teste_mi.md", flush=True)


if __name__ == "__main__":
    main()
