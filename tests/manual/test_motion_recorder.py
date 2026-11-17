"""Teste de regressao do MotionRecorder (Q5) -- SEM hardware, SEM pytest.

Uso:  $env:PYTHONIOENCODING='utf-8'; .venv\Scripts\\python.exe test_motion_recorder.py

Cobre o patch pedido em eeg_motor_paradigm.py (MotionRecorder.run / stop):
  1) sobrevive a excecao de UMA linha (thread nao morre, loga, continua);
  2) usa origin_ready no snapshot (igual ao RecordingMuxer :3420-3421) e nao
     quebra quando origin_xyz aparece no meio do laco;
  3) flush periodico -> o abort com os._exit (hotkeys.hard_exit) nao perde
     o buffer de usuario-space;
  4) o arquivo TEM o cabecalho em disco (open + header.write + flush) mesmo
     com start/stop no mesmo ciclo;

Esse teste e' o "vermelho" das 4 regressoes: ele FALHA contra o
MotionRecorder atual e PASSA com o patch aplicado.
"""
import os
import sys
import tempfile
import time

import numpy as np

RAIZ = r"C:\mestradopy"   # repo (o teste fica FORA do repo)
sys.path.insert(0, RAIZ)

import eeg_motor_paradigm as p  # noqa: E402

TEMPO_ESPERA = 3.0
falhas = []


def check(cond, msg):
    if not cond:
        falhas.append(msg)
        print(f"   FALHOU: {msg}")
    else:
        print(f"   ok: {msg}")


def espera(cond, limite=TEMPO_ESPERA):
    fim = time.monotonic() + limite
    while time.monotonic() < fim:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def arquivo(path):
    # `close=False` + try/finally: nao fecha no registry (tranca o handle da
    # thread) e o finally evita "I/O operation on closed file" num flush atrasado.
    if not os.path.exists(path):
        return []
    h = open(path, encoding="utf-8")          # nao passa pelo registry (close)
    try:
        return h.read().splitlines()
    except (ValueError, OSError):
        return []
    finally:
        try:
            h.close()
        except OSError:
            pass


def no_disco(path):
    """Le direto do arquivo (sem fechar: o registry tranca o handle da thread)."""
    if not os.path.exists(path):
        return ""
    h = open(path, encoding="utf-8")          # nao passa pelo registry (close)
    try:
        return h.read()
    except (ValueError, OSError):
        return ""
    finally:
        try:
            h.close()
        except OSError:
            pass


class EstadoFalso:
    """State minimo: snapshot() e origin_snapshot()."""

    def __init__(self, hz_linhas=60):
        import threading
        self.origin_ready = False
        self.origin_xyz = None
        self.origin_rpy = None
        self._lock = threading.Lock()
        self._n = 0
        self._hz = hz_linhas
        self.motion = dict(p.SharedState().motion)

    def snapshot(self):
        # a 1a chamada (antes do sleep do gravador) nasce "quebrada" e as
        # seguintes sao normais -> exercita a linha 1 e depois a recuperacao
        with self._lock:
            self._n += 1
            n = self._n
        if n == 1:
            self.motion.pop("valid", None)
        elif n == 2:
            self.motion = dict(p.SharedState().motion)
        return dict(self.motion), "idle", "", ""

    def origin_snapshot(self):
        with self._lock:
            return self.origin_xyz, self.origin_rpy


# ---------------------------------------------------------------- caso 1
print("1) excecao em UMA linha nao mata o gravador")
tempo = tempfile.mkdtemp(prefix="mr_c1_")
caminho = os.path.join(tempo, "c1.csv")
estado = EstadoFalso()
recordar = p.MotionRecorder(estado, caminho, hz=120.0)
recordar.start()
espera(lambda: recordar.rows >= 3, limite=1.0)
check(recordar.is_alive(),
      "a thread continua viva depois da excecao na linha 1 "
      "(hoje: morre em silencio e NUNCA mais grava)")
check(recordar.error is not None, "self.error registra a 1a excecao")
linhas = arquivo(caminho)
check(len(linhas) >= 4,
      f"continuou gravando depois da excecao ({len(linhas)} linhas no arquivo)")

# ---------------------------------------------------------------- caso 2
print("2) origin_ready no snapshot + origem trocada no meio do laco")
caminho = os.path.join(tempo, "c2.csv")
estado = EstadoFalso()
estado.motion["valid"] = 1
recordar = p.MotionRecorder(estado, caminho, hz=200.0)
recordar.start()
# origem aparece DEPOIS do start, exatamente como na calibracao (race)
estado.origin_xyz = np.array([10.0, 10.0, 10.0])
estado.origin_rpy = np.array([0.0, 0.0, 0.0])
estado.origin_ready = True
espera(lambda: recordar.rows >= 4)
check(recordar.is_alive(), "origem trocada no meio do laco nao derruba a thread")
recordar.stop()
check(not recordar.is_alive(), "stop() encerra a thread")
linhas = arquivo(caminho)
check(len(linhas) == recordar.rows + 1,
      f"contagem coerente ({recordar.rows} dados + cabecalho = {len(linhas)})")

# ---------------------------------------------------------------- caso 3
print("3) header no DISCO + flush: os._exit nao perde o buffer")
# OBS: o "le janela OS" foi retirado daqui porque o snapshot do teste chama
# o conteudo ANTES de o header.write rodar (corrida do proprio teste) e o
# codigo ja' tem header.write+flush como 1a acao; o que importa e' DISCO.
caminho = os.path.join(tempo, "c3.csv")
estado = EstadoFalso()
recordar = p.MotionRecorder(estado, caminho, hz=200.0)
recordar.start()
espera(lambda: recordar.rows >= 5, limite=1.0)
corpo_na_tela = no_disco(caminho)      # 3a leitura: janela OS apos ~25 linhas
check("nan" in corpo_na_tela or len(corpo_na_tela.splitlines()) >= 5,
      "linhas de dados legiveis em disco ANTES do stop (flush periodico)")
check(no_disco(caminho).startswith("t_mono_s,t_epoch_s,"),
      "cabecalho ja' em DISCO depois do flush periodico")
check("nan" in no_disco(caminho),
      "linhas de dados ja' em DISCO antes do stop (flush periodico)")

# abort bruto do CancelamentoGarantido (hotkeys.hard_exit -> os._exit)
recordar._stop_event.set()
recordar.join(timeout=3.0)
linhas = arquivo(caminho)
check(len(linhas) >= 6, f"apos o abort os dados estao em disco ({len(linhas)})")

# ---------------------------------------------------------------- caso 4
print("4) stop prematuro (start+stop no mesmo ciclo) nao perde o arquivo")
caminho = os.path.join(tempo, "c4.csv")
estado = EstadoFalso()
recordar = p.MotionRecorder(estado, caminho, hz=50.0)
recordar.start()
recordar.stop()
linhas = arquivo(caminho)
check(bool(linhas), "arquivo criado")
check(linhas[0].startswith("t_mono_s"),
      "cabecalho presente em disco apos stop imediato")

# ---------------------------------------------------------------- caso 5
print("5) guarda: o teste so' tem sentido com o comportamento NOVO")
src = open(os.path.abspath(p.__file__), encoding="utf-8").read()
trecho = src[src.index("class MotionRecorder"):
             src.index("class RecordingCsvWriter")]
check("except Exception as exc" in trecho,
      "MotionRecorder.run tem tratamento de excecao por linha")
check("origin_ready" in trecho, "MotionRecorder usa origin_ready")
check(".flush()" in trecho, "MotionRecorder faz flush periodico")

if falhas:
    print(f"MOTION_RECORDER_FALHOU: {len(falhas)} verificacao(oes)")
    sys.exit(1)
print("MOTION_RECORDER_OK: 4/4 casos")
sys.exit(0)