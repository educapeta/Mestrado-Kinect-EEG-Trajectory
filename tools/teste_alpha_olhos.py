#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teste rápido de alpha: olhos fechados x abertos no g.Nautilus.

Sequência:
  5 s preparação -> 20 s olhos fechados -> 5 s transição -> 20 s olhos abertos.

Grava 32 canais + Marker em CSV e um JSON de eventos. Não é paradigma de MI;
serve para confirmar que o EEG está captando sinal e que alpha aparece.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import gpype as gp
from gpype.backend.core.io_node import IONode
from gpype.backend.filters.bandpass import Bandpass
from gpype.backend.filters.bandstop import Bandstop
from gpype.backend.sinks.csv_writer import CsvWriter
from gpype.backend.sources.g_nautilus import GNautilus
from gpype.common.constants import Constants

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT
CHANNEL_COUNT = 32
FS = 500.0

CODE_PREP = 900
CODE_CLOSED_START = 910
CODE_CLOSED_END = 911
CODE_OPEN_START = 912
CODE_OPEN_END = 913
CODE_END = 32765

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
except ImportError:
    QApplication = None


class MarkerMuxer(IONode):
    """Adiciona a coluna Marker ao EEG (mesmo princípio do paradigma)."""

    def __init__(self, marker_queue, hold_samples=1, **kwargs):
        super().__init__(**kwargs)
        self._queue = marker_queue
        self._hold = max(1, int(hold_samples))
        self._marker = 0
        self._remaining = 0
        self._n_in = None

    def setup(self, data, port_context_in):
        ctx = port_context_in[PORT_IN]
        self._n_in = int(ctx[Constants.Keys.CHANNEL_COUNT])
        out = super().setup(data, port_context_in)
        for context in out.values():
            context[Constants.Keys.CHANNEL_COUNT] = self._n_in + 1
        return out

    def step(self, data):
        block = data.get(PORT_IN)
        if block is None or len(block) == 0:
            return None
        n = block.shape[0]
        try:
            while True:
                ev = self._queue.get_nowait()
                self._marker = int(ev["code"])
                self._remaining = self._hold
        except queue.Empty:
            pass
        col = np.zeros((n, 1), dtype=block.dtype)
        for i in range(n):
            if self._remaining > 0:
                col[i, 0] = self._marker
                self._remaining -= 1
                if self._remaining == 0:
                    self._marker = 0
        return {PORT_OUT: np.column_stack((block, col))}


class AlphaTest:
    def __init__(self, pasta: Path, sujeito: str, sessao: str,
                 dur: float = 20.0, noshow: bool = False):
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.sujeito = sujeito
        self.sessao = sessao
        self.dur = float(dur)
        self.noshow = noshow
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.pasta / f"alpha_{sujeito}_s{sessao}_{stamp}.csv"
        self.json_path = self.pasta / f"alpha_{sujeito}_s{sessao}_{stamp}_eventos.json"
        self.marker_queue = queue.Queue()
        self.events = []
        self.state = {"texto": "PREPARANDO..."}
        self.app = None
        self.label = None
        self.pipeline = None
        self.source = None
        self.muxer = None
        self.writer = None
        self._running = False

    def _emit(self, code, nome):
        self.marker_queue.put({"code": int(code), "nome": nome})
        self.events.append({"code": int(code), "nome": nome,
                            "t_mono": time.monotonic(),
                            "t_epoch": time.time()})

    def _mostra(self, texto):
        self.state["texto"] = texto
        print(f"[alpha] {texto}", flush=True)

    def build(self):
        self.source = GNautilus(sampling_rate=FS, channel_count=CHANNEL_COUNT,
                                frame_size=1)
        bp = Bandpass(f_lo=0.5, f_hi=40.0, order=4, phase="causal")
        notch = Bandstop(f_lo=58.0, f_hi=62.0, order=4, phase="causal")
        self.muxer = MarkerMuxer(self.marker_queue, hold_samples=1)
        self.writer = CsvWriter(file_name=str(self.csv_path))
        self.pipeline = gp.Pipeline()
        self.pipeline.connect(self.source, bp)
        self.pipeline.connect(bp, notch)
        self.pipeline.connect(notch, self.muxer)
        self.pipeline.connect(self.muxer, self.writer)

    def _ui(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("Teste alpha — olhos fechados/abertos")
        self.window.resize(900, 500)
        self.label = QLabel("PREPARANDO...")
        self.label.setStyleSheet(
            "font-size: 54px; font-weight: bold; color: white; background: #101820;")
        self.label.setAlignment(Qt.AlignCenter)
        self.window.setCentralWidget(self.label)
        self.window.showFullScreen()
        self.timer = QTimer()
        self.timer.timeout.connect(
            lambda: self.label.setText(self.state["texto"]))
        self.timer.start(100)

    def _run(self):
        self._running = True
        try:
            time.sleep(3.0)
            self._emit(CODE_PREP, "preparacao")
            self._mostra("PREPARANDO\n\n3 s")
            time.sleep(3.0)
            self._emit(CODE_CLOSED_START, "olhos_fechados_inicio")
            self._mostra("OLHOS FECHADOS\n\nRelaxe a mandíbula")
            time.sleep(self.dur)
            self._emit(CODE_CLOSED_END, "olhos_fechados_fim")
            self._mostra("OLHOS ABERTOS\n\nem 3 s")
            time.sleep(3.0)
            self._emit(CODE_OPEN_START, "olhos_abertos_inicio")
            self._mostra("OLHOS ABERTOS\n\nOlhe para um ponto fixo")
            time.sleep(self.dur)
            self._emit(CODE_OPEN_END, "olhos_abertos_fim")
            self._mostra("FIM\n\nObrigado")
            time.sleep(1.5)
        except Exception as exc:
            print(f"[alpha] ERRO: {type(exc).__name__}: {exc}")
        finally:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self._emit(CODE_END, "fim")
            meta = {
                "subject_id": self.sujeito,
                "session_id": self.sessao,
                "date": datetime.now().isoformat(),
                "protocol": "alpha_eyes_closed_open",
                "fs": FS,
                "channels": CHANNEL_COUNT,
                "channel_names": [
                    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
                    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
                    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
                    "P8", "PO7", "PO3", "PO4", "PO8", "O1", "Oz", "O2"],
                "duration_eyes_s": self.dur,
                "events": self.events,
                "closed_codes": [CODE_CLOSED_START, CODE_CLOSED_END],
                "open_codes": [CODE_OPEN_START, CODE_OPEN_END],
            }
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"[alpha] CSV: {self.csv_path}")
            print(f"[alpha] JSON: {self.json_path}")
            if self.app is not None:
                self.app.quit()

    def run(self):
        if not self.noshow and QApplication is not None:
            self._ui()
        self.build()
        self.pipeline.start()
        print(f"[alpha] gravando {self.csv_path}", flush=True)
        th = threading.Thread(target=self._run, daemon=True)
        th.start()
        if self.app is not None and not self.noshow:
            self.app.exec()
        else:
            th.join(timeout=300)
        try:
            self.pipeline.stop()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Teste alpha olhos fechados/abertos")
    ap.add_argument("--pasta-saida", type=Path, default=Path("_alpha_test"))
    ap.add_argument("--sujeito", default="S01")
    ap.add_argument("--sessao", default="01")
    ap.add_argument("--duracao-olhos", type=float, default=20.0)
    ap.add_argument("--sem-janela", action="store_true")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    t = AlphaTest(args.pasta_saida, args.sujeito, args.sessao,
                  args.duracao_olhos, args.sem_janela)
    t.run()


if __name__ == "__main__":
    main()
