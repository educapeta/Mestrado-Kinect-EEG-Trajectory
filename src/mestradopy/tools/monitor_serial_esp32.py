"""Monitor da serial do ESP32: diz se a PLACA esta' viva (item 86).

POR QUE EXISTE (bancada 19/10/2026): o upload falhava e a COM4 do CP210x
estava 100% MUDA -- nem o boot do ESP32 aparecia depois de pulsar o reset.
Sem saber se a placa esta' ligada/transmitindo, o erro do esptool
("No serial data received") e' ambiguo: pode ser placa sem energia, cabo de
dados ruim, ou a COM4 ser de OUTRO dispositivo (um dongle CP210x qualquer).

Uso (a placa plugada no USB):
    .venv\\Scripts\\python.exe tools\\monitor_serial_esp32.py [COMx] [--segundos 40]

DURANTE a execucao:
  1) aperte o EN (reset) 1-2 vezes;
  2) aperte o BOOT e o EN juntos (BOOT segurado) e solte;
  3) desligue e religue o USB da placa (se for facil).

O que o resultado significa:
  - APARECERAM bytes   -> a placa esta' viva e a serial funciona: o problema do
                          upload e' o modo boot/download (use o GRAVAR_ESP32.bat).
  - ZERO bytes         -> a placa NAO esta' transmitindo: confira energia
                          (LED/chave), cabo de DADOS e se a COMx e' mesmo desta
                          placa (desplugue a placa: o dispositivo COMx tem de
                          desaparecer do Gerenciador de Dispositivos).
"""
from __future__ import annotations

import argparse
import sys
import time


def le_por(porta, segundos, baud):
    import serial

    recebido = bytearray()
    try:
        with serial.Serial(porta, baud, timeout=0.2) as p:
            p.reset_input_buffer()
            t0 = time.time()
            while time.time() - t0 < segundos:
                pedaco = p.read(p.in_waiting or 1)
                if pedaco:
                    recebido += pedaco
    except Exception as exc:            # noqa: BLE001 - caminho de hardware
        print(f"  {baud} baud: nao consegui ler ({type(exc).__name__}: {exc})")
        return bytearray()
    texto = recebido.decode("utf-8", errors="replace")
    print(f"  {baud} baud: {len(recebido)} bytes em {segundos:g} s")
    if recebido:
        for linha in texto.splitlines()[:12]:
            print("    | " + linha[:120])
    return recebido


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("porta", nargs="?", default="COM4")
    parser.add_argument("--segundos", type=float, default=20.0,
                        help="tempo de escuta em 115200 (padrao 20 s)")
    args = parser.parse_args(argv)

    print("=" * 74)
    print(f"  MONITOR DA SERIAL DO ESP32 -- {args.porta}")
    print("  AGORA: aperte EN (reset), depois BOOT+EN, e (se facil) religue o USB.")
    print("=" * 74, flush=True)

    total = bytearray()
    total += le_por(args.porta, args.segundos, 115200)
    if not total:
        # O ESP32 imprime o boot do ROM a 74880 baud (se a app ainda nao subiu).
        print("  nada a 115200; tentando 74880 (boot do ROM)...", flush=True)
        total += le_por(args.porta, min(args.segundos, 10.0), 74880)

    print()
    if total:
        print("  RESULTADO: a PLACA ESTA' VIVA e a serial funciona.")
        print("  Proximo passo: GRAVAR_ESP32.bat COMx (segure o BOOT e tecle ENTER).")
        return 0
    print("  RESULTADO: ZERO bytes -- a placa NAO esta' transmitindo.")
    print("  Confira, nesta ordem:")
    print("   1) LED de energia da placa aceso? Existe chave ON/OFF ou BAT/USB? Ligue.")
    print("   2) Cabo e' de DADOS (nao so' carga)? Troque o cabo.")
    print("   3) A placa esta' plugada no USB dela? Desplugue: o CP210x (COMx) SOME")
    print("      do Gerenciador de Dispositivos? Se NAO sumir, esse COMx e' de OUTRO")
    print("      dispositivo (dongle) e a placa nao esta' conectada por USB.")
    print("   4) Teste em outra porta USB/outro PC; se continuar mudo, e' energia")
    print("      ou defeito da placa/cabo.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
