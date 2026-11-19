#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualiza os 21 landmarks 3D processados dos vídeos das auxiliares.

Entrada: pasta `processed/` com `landmarks_3d.npz` (handmvnet_video_process.py).
Saídas: PNG estático (trajetórias + pose) e GIF animado do esqueleto 3D.

Uso:
    python src/mestradopy/tools/visualiza_landmarks3d.py \
        --session F:\handmvnet_videos\teste_sync3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
for p in (_SRC, _SRC / "core"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from kinect_imu_groundtruth import HAND_CONNECTIONS  # noqa: E402


def carrega(session: Path, npz_path=None):
    npz = Path(npz_path) if npz_path else (
        Path(session) / "processed" / "landmarks_3d.npz")
    if not npz.is_file():
        raise SystemExit(f"nao encontrei {npz}; rode handmvnet_video_process.py")
    d = np.load(npz, allow_pickle=True)
    return d, npz


def _xyz_plot(j):
    """Kinect: x direita, y para baixo, z para frente -> x, z, -y."""
    return np.array([j[0], j[2], -j[1]], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--npz", type=Path, default=None,
                    help="NPZ processado alternativo (ex.: processed_1mao)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    out = args.saida or (Path(args.session) / "processed" / "visualizacao_3d")
    out.mkdir(parents=True, exist_ok=True)
    d, npz = carrega(args.session, args.npz)
    xyz = np.asarray(d["joints_xyz"], dtype=float)
    valid = np.asarray(d["valid"], dtype=bool)
    hands = np.asarray(d["hand"]).astype(str)
    n = len(xyz)
    if args.max_frames:
        n = min(n, args.max_frames)
    # frames em que pelo menos um joint e' valido
    keep = np.array([valid[i].any() for i in range(n)])
    idxs = np.where(keep)[0]
    if len(idxs) == 0:
        raise SystemExit("nenhum frame com joint valido")
    # limites com margem
    pts = xyz[idxs][valid[idxs]]
    P = np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])
    lo = np.nanpercentile(P, 2, axis=0); hi = np.nanpercentile(P, 98, axis=0)
    pad = np.maximum(hi - lo, 0.05) * 0.25
    lo -= pad; hi += pad
    # ---------- figura estatica: trajetorias de todos os joints ----------
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    colors = plt.cm.turbo(np.linspace(0, 1, 21))
    for j in range(21):
        q = xyz[idxs, j]
        ok = valid[idxs, j] & np.isfinite(q).all(axis=1)
        if not ok.any():
            continue
        Q = np.array([_xyz_plot(v) for v in q[ok]])
        ax.plot(Q[:, 0], Q[:, 1], Q[:, 1], color=colors[j], lw=0.7, alpha=0.55)
        ax.scatter(Q[-1, 0], Q[-1, 1], Q[-1, 2], color=colors[j], s=8)
        ax2.plot(d["time_sync_s"][idxs][ok] - d["time_sync_s"][idxs][0],
                 Q[:, 1], color=colors[j], lw=0.7, alpha=0.55)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z profundidade (m)")
    ax.set_zlabel("Y para cima (m)")
    ax.set_title("Trajetórias 3D dos 21 landmarks")
    ax2.set_xlabel("tempo (s)"); ax2.set_ylabel("Z (m)")
    ax2.set_title("Profundidade Z de cada landmark")
    ax2.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "landmarks3d_trajetorias.png", dpi=160)
    plt.close(fig)

    # ---------- animacao ----------
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z profundidade (m)")
    ax.set_zlabel("Y para cima (m)")
    ax.set_title("landmarks 3D — frame 0")
    lines = []
    dots = []
    for _ in range(21):
        ln, = ax.plot([], [], [], "o-", color="tab:blue", lw=1.2, ms=2)
        lines.append(ln)
    text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(k):
        i = idxs[k]
        j = xyz[i]
        for joint in range(21):
            if not np.isfinite(j[joint]).all():
                lines[joint].set_data([], []); lines[joint].set_3d_properties([])
                continue
            q = _xyz_plot(j[joint])
            # ponto atual
            lines[joint].set_data([q[0]], [q[1]])
            lines[joint].set_3d_properties([q[2]])
        # esqueleto: usar os pontos atuais diretamente
        # (cada linha e' um joint; desenhamos conexoes como linhas extras)
        return lines + [text]

    # Para desenhar conexoes, recriamos as linhas como segmentos
    segs = [ax.plot([], [], [], color="tab:blue", lw=1.2)[0]
            for _ in HAND_CONNECTIONS]
    pts = [ax.plot([], [], [], "o", color="tab:red", ms=3)[0] for _ in range(21)]

    def update2(k):
        i = idxs[k]
        j = xyz[i]
        for p_, joint in zip(pts, range(21)):
            if np.isfinite(j[joint]).all():
                q = _xyz_plot(j[joint]); p_.set_data([q[0]], [q[1]]); p_.set_3d_properties([q[2]])
            else:
                p_.set_data([], []); p_.set_3d_properties([])
        for s_, (a, b) in zip(segs, HAND_CONNECTIONS):
            if np.isfinite(j[a]).all() and np.isfinite(j[b]).all():
                qa = _xyz_plot(j[a]); qb = _xyz_plot(j[b])
                s_.set_data([qa[0], qb[0]], [qa[1], qb[1]])
                s_.set_3d_properties([qa[2], qb[2]])
            else:
                s_.set_data([], []); s_.set_3d_properties([])
        text.set_text(f"frame {k+1}/{len(idxs)}  t={d['time_sync_s'][i]-d['time_sync_s'][idxs[0]]:.2f}s  {hands[i]}")
        ax.set_title(f"landmarks 3D — frame {k+1}/{len(idxs)}")
        return segs + pts + [text]

    anim = FuncAnimation(fig, update2, frames=len(idxs), interval=100, blit=False)
    gif = out / "landmarks3d.gif"
    anim.save(gif, writer=PillowWriter(fps=10))
    plt.close(fig)
    (out / "README.txt").write_text(
        f"Fonte: {npz}\nframes: {len(idxs)}\n"
        f"imagem: {(out/'landmarks3d_trajetorias.png')}\n"
        f"gif: {gif}\n", encoding="utf-8")
    print("salvo:", out)
    print("PNG:", out / "landmarks3d_trajetorias.png")
    print("GIF:", gif)


if __name__ == "__main__":
    main()
