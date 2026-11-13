#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualizador ao vivo do g.Nautilus (janela móvel, sem salvar dados).

Objetivo: inspecionar rapidamente a qualidade dos canais e detectar problemas de
ground/referência (canal flat, ruído comum, 50/60 Hz, saturação, offset).

- Topo: canais selecionados (default C3, Cz, C4), em janela móvel.
- Baixo: os 32 canais empilhados para ver quais estão ativos/ruidosos.
- Não grava CSV; só visualiza.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import gpype as gp
from gpype.backend.core.io_node import IONode
from gpype.common.constants import Constants

PORT_IN = Constants.Defaults.PORT_IN

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
    "P8", "PO7", "PO3", "PO4", "PO8", "O1", "Oz", "O2",
]


class LiveSink(IONode):
    """Sink que recebe blocos de EEG e publica no buffer do visualizador."""

    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.n_channels = None

    def setup(self, data, port_context_in):
        ctx = port_context_in[PORT_IN]
        self.n_channels = int(ctx[Constants.Keys.CHANNEL_COUNT])
        return super().setup(data, port_context_in)

    def step(self, data):
        block = data.get(PORT_IN)
        if block is not None and len(block):
            self.callback(np.asarray(block, dtype=np.float64))
        return None


def _orienta(block, n_channels):
    arr = np.asarray(block, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.shape[0] == n_channels and arr.shape[1] != n_channels:
        return arr.T
    return arr


def main():
    ap = argparse.ArgumentParser(description="Visualizador ao vivo do g.Nautilus")
    ap.add_argument("--canais", default="C3,Cz,C4",
                    help="Canais grandes no topo (nomes ou índices 1-based)")
    ap.add_argument("--janela", type=float, default=8.0,
                    help="Janela móvel em segundos")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--sem-uv", action="store_true",
                    help="Mostra counts brutos (default: escala aproximada µV)")
    args = ap.parse_args()

    import pyqtgraph as pg
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    # buffer circular
    fs = float(args.fs)
    max_samples = int(args.janela * fs) + int(2 * fs)
    buffer = np.full((max_samples, len(CHANNEL_NAMES)), np.nan)
    write_pos = 0
    filled = 0

    def on_block(block):
        nonlocal write_pos, filled
        arr = _orienta(block, len(CHANNEL_NAMES))
        if arr.shape[1] != len(CHANNEL_NAMES):
            # fonte pode ter outro número de canais; adapta
            nomes = [f"Ch{i+1:02d}" for i in range(arr.shape[1])]
        n = arr.shape[0]
        if n > max_samples:
            arr = arr[-max_samples:]
            n = max_samples
        primeiro = min(n, max_samples - write_pos)
        buffer[write_pos:write_pos + primeiro, :arr.shape[1]] = arr[:primeiro]
        if primeiro < n:
            buffer[:n - primeiro, :arr.shape[1]] = arr[primeiro:]
        write_pos = (write_pos + n) % max_samples
        filled = min(max_samples, filled + n)

    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    win = pg.GraphicsLayoutWidget(show=True,
                                 title="g.Nautilus — sinal ao vivo (sem gravar)")
    win.resize(1400, 900)

    plot_top = win.addPlot(row=0, col=0, title="Canais selecionados")
    plot_top.showGrid(x=True, y=True, alpha=0.25)
    plot_top.setLabel("left", "amplitude", units="uV")
    plot_top.setLabel("bottom", "tempo", units="s")

    plot_all = win.addPlot(row=1, col=0, title="32 canais (offset para visualização)")
    plot_all.showGrid(x=True, y=True, alpha=0.25)
    plot_all.setLabel("left", "canal")
    plot_all.setLabel("bottom", "tempo", units="s")

    # seleciona canais do topo
    desejados = [s.strip() for s in args.canais.split(",") if s.strip()]
    indices_topo = []
    for nome in desejados:
        if nome in CHANNEL_NAMES:
            indices_topo.append(CHANNEL_NAMES.index(nome))
        else:
            try:
                indices_topo.append(int(nome) - 1)
            except ValueError:
                pass
    if not indices_topo:
        indices_topo = [12, 13, 14]
    cores = [(255, 80, 80), (80, 255, 80), (80, 160, 255),
             (255, 220, 80), (220, 80, 255), (80, 255, 220)]
    curvas_topo = []
    for k, idx in enumerate(indices_topo):
        cor = cores[k % len(cores)]
        curvas_topo.append(plot_top.plot(
            pen=pg.mkPen(cor, width=2), name=CHANNEL_NAMES[idx]))
    legenda = plot_top.addLegend()
    for idx in indices_topo:
        legenda.addItem(pg.PlotDataItem(pen=pg.mkPen(
            cores[indices_topo.index(idx) % len(cores)], width=2)),
            CHANNEL_NAMES[idx])

    curvas_all = []
    for i in range(len(CHANNEL_NAMES)):
        c = pg.intColor(i, hues=32, values=1, maxValue=255, minValue=80)
        curvas_all.append(plot_all.plot(pen=pg.mkPen(c, width=1)))

    # eixo X comum: [-janela, 0]
    for p in (plot_top, plot_all):
        p.setXRange(-args.janela, 0, padding=0)

    estado = {"ultimo": 0.0}

    def atualiza():
        if filled < 10:
            return
        # ordena o buffer circular
        if filled < max_samples:
            dados = buffer[:filled]
        else:
            dados = np.vstack((buffer[write_pos:], buffer[:write_pos]))
        dados = dados[-int(args.janela * fs):]
        n = len(dados)
        t = np.arange(n) / fs
        t = t - t[-1]
        # escala aproximada counts -> uV (só para visualização)
        shown = dados if args.sem_uv else dados * 0.000134
        for k, idx in enumerate(indices_topo):
            y = shown[:, idx]
            # remove NaN para o plot
            y = np.nan_to_num(y, nan=0.0)
            curvas_topo[k].setData(t, y)
        # 32 canais empilhados; normaliza cada um para visualização
        for i in range(len(CHANNEL_NAMES)):
            y = np.nan_to_num(shown[:, i], nan=0.0)
            scale = np.nanmax(np.abs(y)) if np.isfinite(y).any() else 1.0
            if scale <= 0:
                scale = 1.0
            curvas_all[i].setData(t, y / scale - i * 1.5)
        agora = time.strftime("%H:%M:%S")
        plot_top.setTitle(f"Canais {', '.join(CHANNEL_NAMES[i] for i in indices_topo)} "
                          f"— janela {args.janela:.1f}s — {agora}")

    timer = QTimer()
    timer.timeout.connect(atualiza)
    timer.start(40)

    # pipeline
    fonte = gp.GNautilus(sampling_rate=fs, channel_count=len(CHANNEL_NAMES),
                         frame_size=1)
    sink = LiveSink(on_block)
    pipeline = gp.Pipeline()
    pipeline.connect(fonte, sink)
    pipeline.start()

    def fecha():
        try:
            pipeline.stop()
        except Exception:
            pass
        app.quit()

    app.aboutToQuit.connect(fecha)
    win.show()
    print("[visualizador] janela aberta; feche a janela para parar.")
    try:
        app.exec()
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
