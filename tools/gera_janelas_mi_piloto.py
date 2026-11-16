#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera e salva as janelas deslizantes de 2 s do MI do piloto.

Entrada: CSV/JSON de `_piloto3c_memi` (4 s de MI).
Saída: NPZ com X (trials, 5 janelas, 3 canais, 256 amostras), y, blocos.
O modelo EEGNet do Faria espera 2 s; as 5 janelas de 2 s (passo 0,5 s)
são agregadas por média de probabilidades na aplicação.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eegnet_faria_aplicar_mi import carrega_mi_piloto, janelas_2s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piloto", type=Path, default=Path("_piloto3c_memi"))
    ap.add_argument("--saida", type=Path, default=Path("_janelas_mi_piloto"))
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    X, y, blocos = carrega_mi_piloto(args.piloto)
    W = np.stack([janelas_2s(t) for t in X])  # (n_trials, 5, 3, 256)
    np.savez_compressed(args.saida / "janelas_mi_2s.npz",
                        X=W, y=y, blocos=blocos,
                        window_sec=2.0, stride_sec=0.5,
                        fs=128, n_canais=3)
    meta = {
        "n_trials": int(W.shape[0]),
        "n_janelas_por_trial": int(W.shape[1]),
        "shape_janela": list(W.shape[2:]),
        "fs": 128,
        "window_sec": 2.0,
        "stride_sec": 0.5,
        "canais": ["C3", "C4", "Cz"],
        "labels": ["esquerda", "direita", "repouso"],
        "y": y.tolist(),
        "blocos": blocos.tolist(),
    }
    (args.saida / "janelas_mi_2s.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("X:", W.shape, "y:", np.unique(y, return_counts=True))
    print("Salvo em", args.saida)


if __name__ == "__main__":
    main()
