"""Teste da PARADA das threads de bancada do paradigma (bug do `self._stop`).

Contexto (bancada, 28/09/2026): `MotionRecorder`, `LinkWatchdog` e
`ImpedanceMonitor` herdam de `threading.Thread` e guardavam o pedido de parada
em `self._stop` -- que e' o nome de um metodo INTERNO do `threading.Thread`
(chamado pelo `join()`/`is_alive()` quando a thread termina). O atributo
sombreava o metodo e a parada morria com

    TypeError: 'Event' object is not callable

dentro do `stop()` das tres -- exatamente no caminho do "cancelar trial /
abortar sessao". Foi o que derrubou a suite `test_paradigm_protocol`.

Este teste TRAVA isso de duas formas: (1) estrutural -- nenhuma subclasse de
Thread do modulo pode definir os nomes internos; (2) comportamental -- start e
stop de verdade nas tres threads, sem hardware nenhum:

  1. nomes internos de threading.Thread nao sao sombreados pelas subclasses;
  2. MotionRecorder grava cabecalho + linhas e para limpo;
  3. LinkWatchdog marca a perda e a volta do link e para limpo;
  4. ImpedanceMonitor para limpo (com e sem aparelho);
  5. stop() e' idempotente (duas chamadas seguidas nao levantam nada).

    .venv\\Scripts\\python.exe test_parada_threads.py
"""
import os
import queue
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

import eeg_motor_paradigm as p

# =============================================================================
# 1) ESTRUTURAL: os nomes internos do threading.Thread tem de ficar livres.
# =============================================================================
# `_stop` e' metodo interno (chamado pelo join); `_started` e' um Event interno
# do proprio Thread. Sobrescrever qualquer um dos dois quebra o join de um jeito
# dificil de ler (era o `TypeError: 'Event' object is not callable`).
INTERNOS = ("_stop", "_started", "_tstate_lock", "_bootstrap_inner")

subclasses = []
for nome in dir(p):
    objeto = getattr(p, nome)
    if isinstance(objeto, type) and issubclass(objeto, threading.Thread) \
            and objeto is not threading.Thread:
        subclasses.append((nome, objeto))
nomes = sorted(nome for nome, _ in subclasses)
assert {"MotionRecorder", "LinkWatchdog", "ImpedanceMonitor"} <= set(nomes), \
    f"as threads de bancada sumiram do modulo? achei {nomes}"
for nome, classe in subclasses:
    for interno in INTERNOS:
        assert interno not in vars(classe), (
            f"{nome}.{interno} sombreia o interno do threading.Thread: "
            "o join()/is_alive() deixa de funcionar")

# =============================================================================
# 2) MotionRecorder: grava o CSV e PARA limpo (join sem excecao)
# =============================================================================
pasta = tempfile.mkdtemp(prefix="paradigma_threads_")
try:
    caminho = os.path.join(pasta, "movimento.csv")
    estado = p.SharedState()
    gravador = p.MotionRecorder(estado, caminho, hz=50.0, fs=100.0)
    gravador.start()
    limite = 300
    while gravador.rows < 3 and limite:
        time.sleep(0.01)
        limite -= 1
    gravador.stop()                       # era AQUI que estourava o TypeError
    assert not gravador.is_alive(), "stop() tem de deixar a thread morta"
    assert gravador.error is None, gravador.error
    gravador.stop()                       # 5) idempotente
    with open(caminho, encoding="utf-8") as handle:
        linhas = [linha for linha in handle.read().splitlines() if linha]
    assert linhas[0].startswith("t_mono_s,t_epoch_s,"), linhas[0][:60]
    assert len(linhas) >= 4, f"esperava cabecalho + 3 amostras, veio {len(linhas)}"
    colunas = linhas[1].split(",")
    assert len(colunas) == 2 + p.N_MOTION_COLS, len(colunas)
    # =========================================================================
    # 3) LinkWatchdog: detecta a QUEDA e a VOLTA do link, e para limpo
    # =========================================================================
    class MuxerFake:
        """Muxer minimo: o vigia so' olha `samples_written`."""

        def __init__(self):
            self.samples_written = 0

    muxer = MuxerFake()
    fila = queue.Queue()
    estado = p.SharedState()
    vigia = p.LinkWatchdog(muxer, estado, fila, timeout=0.05, period=0.01)

    vistos = []

    def espera(codigo, limite=500):
        """Drena a fila para `vistos` e espera o `codigo` aparecer (com prazo)."""
        while codigo not in vistos and limite:
            while True:
                try:
                    vistos.append(fila.get_nowait()["code"])
                except queue.Empty:
                    break
            time.sleep(0.01)
            limite -= 1
        return codigo in vistos

    vigia.start()
    assert espera(p.CODE_LINK_LOST), "o vigia nao marcou a perda de link"
    assert vigia.perdas == 1, vigia.perdas
    assert estado.link_ok == 0, estado.link_ok

    muxer.samples_written += 1             # o EEG voltou a entregar amostras
    assert espera(p.CODE_LINK_OK), "o vigia nao marcou a volta do link"
    assert estado.link_ok == 1, estado.link_ok

    vigia.stop()                           # join de thread ja' encerrada
    assert not vigia.is_alive()
    vigia.stop()                           # 5) idempotente

    # =========================================================================
    # 4) ImpedanceMonitor: sem aparelho a thread SAI sozinha e o stop e' limpo
    # =========================================================================
    estado = p.SharedState()
    monitor = p.ImpedanceMonitor(None, estado, period=0.01)
    monitor.start()
    limite = 300
    while monitor.is_alive() and limite:
        time.sleep(0.01)
        limite -= 1
    assert not monitor.is_alive(), "sem aparelho o monitor tem de sair sozinho"
    assert "indisponivel" in estado.impedance_text, estado.impedance_text
    monitor.stop()
    assert not monitor.is_alive()
    monitor.stop()                         # 5) idempotente

    # 4b) com aparelho falso: mede, publica a tabela e para limpo
    class AparelhoFake:
        """Uma medida por chamada: 2 canais bons, 1 acima do limite, 1 nao medido."""

        def __init__(self):
            self.leituras = 0

        def get_impedance(self, _primeira):
            self.leituras += 1
            return np.array([1.0, 2.0, 99.0, -10.0])

    estado = p.SharedState()
    aparelho = AparelhoFake()
    monitor = p.ImpedanceMonitor(aparelho, estado, limite_kohm=50.0, period=0.01)
    monitor.start()
    limite = 300
    while aparelho.leituras < 2 and limite:
        time.sleep(0.01)
        limite -= 1
    monitor.stop()
    assert not monitor.is_alive()
    assert aparelho.leituras >= 2, aparelho.leituras
    assert estado.impedance_values is not None, "a medida nao foi publicada"
    assert "99.0" in estado.impedance_text, estado.impedance_text

finally:
    shutil.rmtree(pasta, ignore_errors=True)

print("PARADA_THREADS_OK: 5/5 grupos (nomes internos do Thread livres, "
      "MotionRecorder, LinkWatchdog, ImpedanceMonitor e stop idempotente)")
sys.exit(0)
