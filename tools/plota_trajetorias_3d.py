#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera gráfico 3D da trajetória média dos 21 landmarks do Gate B.

Entrada: CSV longo gerado por
    tools/validacao_groundtruth_deslocamento.py --salvar-landmarks

O gráfico é separado por mão (direita/esquerda). Para cada repetição:
  - usa apenas o método `kinect`;
  - mantém frames com lateralidade correta;
  - alinha a trajetória pelo início do estado `movimento`;
  - reamostra para N pontos e faz a média entre repetições.

As coordenadas são as salvas no CSV (espaço de câmera do Kinect, em metros).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def carrega(csv_path: Path, metodo: str = "kinect"):
    dados = defaultdict(list)  # (lado, rep, landmark) -> [(t, estado, p)]
    with open(csv_path, newline="", encoding="utf-8") as f:
        import csv
        for row in csv.DictReader(f):
            if row["metodo"] != metodo:
                continue
            lado = row["lado"]
            # o CSV longo ja' e' do lado alvo; nao tem coluna lateralidade
            try:
                p = np.array([float(row["x_m"]), float(row["y_m"]),
                              float(row["z_m"])], float)
                t = float(row["timestamp_s"])
            except ValueError:
                continue
            if not np.isfinite(p).all():
                continue
            dados[(lado, int(row["repeticoes"]),
                   int(row["landmark"]))].append((t, row["estado"], p))
    for k in dados:
        dados[k].sort(key=lambda x: x[0])
    return dados


def reamostra_trajetoria(amostras, n_pontos=80):
    """Reamostra o trecho de movimento (com pequena margem) para n_pontos."""
    estados = [a[1] for a in amostras]
    try:
        i_mov = estados.index("movimento")
    except ValueError:
        return None
    # margem antes do movimento para pegar o hold inicial estabilizado
    ini = max(0, i_mov - 6)
    # vai até o começo do hold final + 3 frames
    try:
        i_fim = estados.index("hold_fim")
        fim = min(len(amostras), i_fim + 4)
    except ValueError:
        fim = len(amostras)
    trecho = amostras[ini:fim]
    if len(trecho) < 4:
        return None
    ts = np.array([a[0] for a in trecho])
    ps = np.stack([a[2] for a in trecho])
    # tempo normalizado pelo trecho; interpola cada eixo
    u = ts - ts[0]
    if u[-1] <= 0:
        return None
    u = u / u[-1]
    grid = np.linspace(0.0, 1.0, n_pontos)
    out = np.empty((n_pontos, 3), float)
    for eixo in range(3):
        out[:, eixo] = np.interp(grid, u, ps[:, eixo])
    return out


def medias_por_lado(dados, lado, n_pontos=80):
    por_lm = defaultdict(list)
    for (l, rep, lm), amostras in dados.items():
        if l != lado:
            continue
        traj = reamostra_trajetoria(amostras, n_pontos)
        if traj is not None:
            por_lm[lm].append(traj)
    medias = {}
    for lm, trajs in por_lm.items():
        arr = np.stack(trajs)
        medias[lm] = np.median(arr, axis=0)  # mediana robusta entre reps
    return medias


def plota(csv_path: Path, saida: Path, n_pontos=80):
    dados = carrega(csv_path)
    saida.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    cores = cm.viridis(np.linspace(0, 1, 21))
    eixos = []
    for painel, lado in enumerate(("direita", "esquerda"), start=1):
        medias = medias_por_lado(dados, lado, n_pontos)
        ax = fig.add_subplot(1, 2, painel, projection="3d")
        eixos.append(ax)
        for lm in range(21):
            if lm not in medias:
                continue
            tr = medias[lm]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                    color=cores[lm], lw=1.8, alpha=0.9)
            ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2],
                       color=cores[lm], s=18, marker="o")
            ax.scatter(tr[-1, 0], tr[-1, 1], tr[-1, 2],
                       color=cores[lm], s=24, marker="s")
        # centróide médio dos 21 landmarks (linha preta grossa)
        if medias:
            comum = np.stack([medias[lm] for lm in medias])
            cent = np.median(comum, axis=0)
            ax.plot(cent[:, 0], cent[:, 1], cent[:, 2],
                    color="black", lw=3.0, ls="--")
        ax.set_title(f"Trajetória média — mão {lado}", fontsize=13, pad=16)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=24, azim=-62)
        ax.grid(True, alpha=0.35)
        # mantém proporção visual dos eixos
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    # barra de cores compartilhada para os 21 landmarks
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=Normalize(vmin=0, vmax=20))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=eixos, shrink=0.72, pad=0.02,
                        ticks=[0, 5, 9, 12, 16, 20])
    cbar.set_label("landmark")
    # legenda única para o centróide
    fig.legend([Line2D([0], [0], color="black", lw=3, ls="--")],
               ["centróide dos 21 landmarks"], loc="lower center",
               ncol=1, framealpha=0.9)
    fig.suptitle("Kinect — trajetória média dos 21 landmarks "
                 "(mediana entre repetições; espaço da câmera, metros)",
                 fontsize=14)
    fig.savefig(saida, dpi=160)
    plt.close(fig)
    print(f"[OK] Figura salva: {saida}")


def main():
    ap = argparse.ArgumentParser(description="Gráfico 3D da trajetória média dos landmarks")
    ap.add_argument("--csv", type=Path, required=True,
                    help="landmarks_*.csv gerado com --salvar-landmarks")
    ap.add_argument("--saida", type=Path,
                    default=Path("analise_trajetoria/trajetorias_3d_media.png"))
    ap.add_argument("--pontos", type=int, default=80)
    args = ap.parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV não encontrado: {args.csv}")
    plota(args.csv, args.saida, args.pontos)


if __name__ == "__main__":
    main()
