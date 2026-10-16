"""Probe ISOLADO da impedancia do g.Nautilus (sem o paradigma, sem gpype).

Mede o erro REAL que o `get_impedance` levanta neste aparelho, nos DOIS
estados que importam:

  A) APARELHO ABERTO E PARADO (logo apos SetConfiguration, sem stream):
     e' o estado do `_GNautilusCore` ANTES do pipeline.start();
  B) APARELHO TRANSMITINDO (a thread de dados rodando): e' o estado em
     que o `ImpedanceMonitor` do paradigma realmente mede (a tela de
     preparacao so' aparece DEPOIS do pipeline.start()).

O paradigma engolia a excecao e mostrava "sem aparelho direto nem API do
g.Pype" (item 59); este probe mostra o tipo e a mensagem EXATOS, para
distinguir CRC transitrio do link sem fio de erro de estado.

FEQUE O PARADIGMA ANTES (o GDS abre o aparelho em modo EXCLUSIVO):
    python tools/probe_impedancia.py
"""
import sys
import threading
import time
import traceback

import numpy as np


def _titulo(texto):
    print(f"\n=== {texto} " + "=" * max(0, 60 - len(texto)), flush=True)


def main():
    import gtec_gds as gds
    from gtec_gds.lib.gtec_gds_wrapper import ConnectedDevices
    from gtec_gds.lib.constants import Constants as C

    _titulo("Aparelhos conectados (ConnectedDevices)")
    conectados = [(n, t, u) for n, t, u in ConnectedDevices()]
    for nome, tipo, uso in conectados:
        marca = "g.Nautilus" if tipo == C.DEVICE_TYPE_GNAUTILUS else f"tipo={tipo}"
        print(f"  {nome}  {marca}  em_uso={uso}")
    if not any(t == C.DEVICE_TYPE_GNAUTILUS for _, t, _ in conectados):
        print("NENHUM g.Nautilus visivel -- ligue o aparelho e rode de novo.")
        return 1

    _titulo("A) Abrir o aparelho (parado, sem stream)")
    try:
        aparelho = gds.GNautilus(sampling_rate=500, channel_count=32,
                                 frame_size=1)
    except Exception as exc:
        print(f"FALHOU ao abrir: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    print(f"aberto: serial={aparelho.serial_number} "
          f"canais={aparelho.channel_count} fs={aparelho._sampling_rate}")

    _titulo("A1) get_impedance(first=True)  [parado]")
    try:
        z = aparelho.get_impedance(True)
        z = np.asarray(z, dtype=float)
        print(f"OK: {len(z)} canais | min={np.nanmin(z):.1f} "
              f"max={np.nanmax(z):.1f} | primeiros: {z[:6]}")
    except Exception as exc:
        print(f"FALHOU: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    _titulo("A2) get_impedance(first=False)  [parado, 2a leitura]")
    try:
        z = aparelho.get_impedance(False)
        z = np.asarray(z, dtype=float)
        print(f"OK: min={np.nanmin(z):.1f} max={np.nanmax(z):.1f}")
    except Exception as exc:
        print(f"FALHOU: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    _titulo("B) Ligar o stream (como o paradigma faz) e medir de novo")
    frames = []
    erro_stream = []

    def _cb(dado):
        frames.append(np.asarray(dado).copy())

    try:
        aparelho.set_data_callback(_cb)
        aparelho.start()
        time.sleep(1.0)
        print(f"stream: {len(frames)} frames em 1 s")
        for primeira in (True, False):
            try:
                z = aparelho.get_impedance(primeira)
                z = np.asarray(z, dtype=float)
                print(f"get_impedance(first={primeira}) TRANSMITINDO: OK "
                      f"min={np.nanmin(z):.1f} max={np.nanmax(z):.1f}")
            except Exception as exc:
                print(f"get_impedance(first={primeira}) TRANSMITINDO: "
                      f"FALHOU: {type(exc).__name__}: {exc}")
                traceback.print_exc()
    except Exception as exc:
        print(f"stream FALHOU: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        try:
            aparelho.stop()
        except Exception:
            pass
        if getattr(aparelho, "stream_error", None) is not None:
            print(f"stream_error: {aparelho.stream_error}")

    _titulo("Fechar")
    try:
        aparelho.close()
        print("fechado.")
    except Exception as exc:
        print(f"fecho falhou: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
