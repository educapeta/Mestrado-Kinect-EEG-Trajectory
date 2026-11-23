#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aquisição de ruído EEG CRU (sem notch/bandpass) em condições controladas.

Condições padrão:
  A: monitor ligado, posição normal
  B: monitor desligado, posição normal
  C: monitor ligado, afastado
  D: monitor desligado, afastado

Grava um único CSV contínuo com coluna Marker (código 1..4 no início de cada
condição). O analisador `analisa_ruido_eeg.py` segmenta por esses marcadores.

Uso (hardware):
  .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\aquisicao_ruido_eeg.py `
      --saida outputs\\eeg\\ruido_bruto --duracao 60

Teste sem hardware:
  ... --test-generator --duracao 3 --saida outputs\\eeg\\_teste_ruido
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import gpype as gp
from gpype.backend.sinks.csv_writer import CsvWriter
from gpype.backend.sources.g_nautilus import GNautilus
from gpype.backend.sources.generator import Generator
from gpype.backend.core.io_node import IONode
from gpype.common.constants import Constants

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT
FS = 500.0
N_CH = 32

PERFIS = {
    "desktop": [
        ("A", 1, "monitor LIGADO, posicao normal"),
        ("B", 2, "monitor DESLIGADO, posicao normal"),
        ("C", 3, "monitor LIGADO, afastado ~1 m"),
        ("D", 4, "monitor DESLIGADO, afastado ~1 m"),
        ("E", 5, "monitor LIGADO, MUITO perto do desktop (~0,3 m)"),
    ],
    "laptop": [
        ("A", 1, "laptop na BATERIA, longe do desktop"),
        ("B", 2, "laptop na BATERIA, perto do desktop"),
        ("C", 3, "laptop na TOMADA, longe do desktop"),
        ("D", 4, "laptop na TOMADA, perto do desktop"),
        ("E", 5, "laptop na BATERIA, perto do desktop e monitor/tela LIGADA"),
    ],
}
CONDICOES = PERFIS["desktop"]


class MarkerMuxer(IONode):
    """Adiciona uma coluna Marker sincrona ao EEG (hold de 1 amostra)."""

    def __init__(self, marker_queue, **kwargs):
        super().__init__(**kwargs)
        self._queue = marker_queue
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
        try:
            while True:
                ev = self._queue.get_nowait()
                self._marker = int(ev["code"])
                self._remaining = 1
        except queue.Empty:
            pass
        col = np.zeros((block.shape[0], 1), dtype=block.dtype)
        for i in range(block.shape[0]):
            if self._remaining > 0:
                col[i, 0] = self._marker
                self._remaining -= 1
                if self._remaining == 0:
                    self._marker = 0
        return {PORT_OUT: np.column_stack((block, col))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/ruido_bruto"))
    ap.add_argument("--duracao", type=float, default=60.0,
                    help="segundos por condição")
    ap.add_argument("--perfil", choices=["desktop", "laptop"], default="desktop",
                    help="conjunto de condições A-E")
    ap.add_argument("--condicoes", default="A,B,C,D,E",
                    help="quais condições executar")
    ap.add_argument("--test-generator", action="store_true",
                    help="usa fonte sintética, sem g.Nautilus")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.saida / f"ruido_bruto_{stamp}.csv"
    json_path = args.saida / f"ruido_bruto_{stamp}.json"

    pipeline = gp.Pipeline()
    if args.test_generator:
        source = Generator(sampling_rate=FS, channel_count=N_CH, frame_size=1)
    else:
        source = GNautilus(sampling_rate=FS, channel_count=N_CH, frame_size=1)
    q = queue.Queue()
    mux = MarkerMuxer(q)
    writer = CsvWriter(file_name=str(csv_path))
    pipeline.connect(source, mux)
    pipeline.connect(mux, writer)

    desejadas = [c.strip().upper() for c in args.condicoes.split(",") if c.strip()]
    perfil = PERFIS[args.perfil]
    selecionadas = [c for c in perfil if c[0] in desejadas]
    if not selecionadas:
        print("nenhuma condição válida")
        return 2
    eventos = []
    print("\n=== AQUISIÇÃO DE RUÍDO EEG CRU ===")
    print("Sem notch e sem bandpass; NÃO altere cabos durante a gravação.")
    print(f"Duração por condição: {args.duracao:.0f}s")
    print("Para cada condição: prepare a posição, aperte ENTER e fique imóvel.\n")

    pipeline.start()
    time.sleep(1.0)
    t0 = time.monotonic()
    try:
        for codigo, code, descricao in selecionadas:
            print(f"[{codigo}] {descricao}")
            input("    Prepare a posição e pressione ENTER para começar... ")
            print("    estabilizando 3s...")
            time.sleep(3.0)
            marca = time.monotonic()
            q.put({"code": code})
            eventos.append({"condicao": codigo, "code": code,
                            "descricao": descricao,
                            "inicio_s": round(marca - t0, 3)})
            for s in range(int(args.duracao), 0, -1):
                if s % 10 == 0 or s <= 3:
                    print(f"    gravando... {s}s", flush=True)
                time.sleep(1.0)
            print("    fim.\n")
    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] finalizando e salvando o que foi gravado...")
    finally:
        time.sleep(1.0)
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
    dados = {"csv": str(csv_path), "duracao_alvo_s": args.duracao,
             "test_generator": bool(args.test_generator),
             "eventos": eventos,
             "nota": ("EEG cru, sem filtros; coluna Marker indica o início de "
                      "cada condição.")}
    json_path.write_text(json.dumps(dados, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print("CSV:", csv_path)
    print("JSON:", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
