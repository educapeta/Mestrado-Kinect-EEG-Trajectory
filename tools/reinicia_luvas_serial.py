"""Reinicia as luvas pela SERIAL e espera a calibracao de bias do giroscopio.

POR QUE EXISTE (medido em 21/10/2026, item 99): o firmware calibra o bias do
giroscopio UMA UNICA VEZ no boot (`calibrateGyroscopeAt` no setup) e assume que
a luva esta' PARADA. Se o ESP32 religa enquanto a mao/luva se mexe, o "bias"
vira a MEDIA do movimento -> o giroscopio fica com offset constante e o yaw
deriva: medido naquela bancada -12,5 dps no eixo Y da luva DIREITA = +434
graus/min de yaw (7 voltas em 10 min); a esquerda menor (0,1-0,3 dps = 6
graus/min). Roll/pitch ficam perfeitos (gravidade corrige) -- so' o yaw deriva.

Este script evita depender do botao: da' o pulso de RESET pela serial
(RTS -> EN; e' o pulso que funciona no CP210x desta placa -- o pulso DTR/RTS
do esptool derrubava a porta, item 84), escuta o boot a 115200 e mostra:

  * `rst:... (POWERON_RESET)` = prova de que reiniciou;
  * `[imu] calibrando giroscopio MPU 0x68/0x69 ...` = calibracao comecou;
  * `[imu] bias MPU 0x68/0x69: (x, y, z)` = bias medido AGORA (deve ser pequeno;
    compare com a deriva: yaw ~ bias projetado na vertical).

Depois do reset, MANTENHA AS LUVAS PARADAS ate' o script terminar (a calibracao
roda no setup, nos primeiros ~3 s). Para conferir o resultado no UDP:
`python tools\\deriva_estatica_imu.py` (grava 10 min e analisa) ou o tool.

Uso:
    .venv\\Scripts\\python.exe tools\\reinicia_luvas_serial.py [COM4] [--segundos 8]
"""
from __future__ import annotations

import argparse
import re
import sys
import time


def _reset_por_serial(porta_nome, segundos=8.0):
    """Pulso de RESET (RTS) e devolve (linhas, reiniciou, biases)."""
    import serial

    linhas = []
    reiniciou = False
    biases = []
    reiniciou_re = re.compile(r"rst:|bias MPU|calibrando giroscopio",
                              re.IGNORECASE)
    with serial.Serial(porta_nome, 115200, timeout=0.2) as porta:
        porta.reset_input_buffer()
        # Pulso de RESET: RTS = EN (ativo baixo). DTR fica alto para nao forcar
        # o GPIO0 (nao queremos download mode, so' reboot).
        porta.dtr = False
        porta.rts = True
        time.sleep(0.15)
        porta.rts = False
        time.sleep(0.05)
        porta.reset_input_buffer()
        t0 = time.monotonic()
        pendente = b""
        while time.monotonic() - t0 < segundos:
            pendente += porta.read(porta.in_waiting or 1)
            while b"\n" in pendente:
                bruta, pendente = pendente.split(b"\n", 1)
                texto = bruta.decode("utf-8", "replace").strip()
                if not texto:
                    continue
                # As linhas de dados do firmware começam com "1," ou "2," --
                # nao interessam aqui (so' o boot/calibracao).
                if texto[:2] in ("1,", "2,"):
                    continue
                linhas.append(texto)
                if reiniciou_re.search(texto):
                    reiniciou = True
                if "bias MPU" in texto:
                    biases.append(texto)
    return linhas, reiniciou, biases


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("porta", nargs="?", default="COM4")
    parser.add_argument("--segundos", type=float, default=8.0,
                        help="tempo de escuta do boot (padrao 8 s)")
    args = parser.parse_args(argv)

    print("=" * 74)
    print("  REINICIAR LUVAS PELA SERIAL (calibracao de bias do giroscopio)")
    print("  MANTENHA AS LUVAS PARADAS NA MESA ATE' O FIM DESTE SCRIPT")
    print("=" * 74, flush=True)
    try:
        linhas, reiniciou, biases = _reset_por_serial(args.porta, args.segundos)
    except Exception as exc:                      # noqa: BLE001 - caminho de HW
        print(f"  FALHOU ao abrir/pulsar {args.porta}: {type(exc).__name__}: {exc}")
        print("  Verifique se COMx esta' livre (feche monitor serial) e tente de novo.")
        return 2

    for texto in linhas:
        print("  | " + texto[:160])
    print()
    if not reiniciou:
        print("  AVISO: nao vi 'rst:' nem mensagem de calibracao no boot.")
        print("  O pulso RTS pode nao ter resetado: use EN/RST na mao ou")
        print("  desligue/religue o USB (segure o EN enquanto religa).")
        return 1
    if biases:
        print("  OK: calibracao de bias executada:")
        for texto in biases:
            print("    " + texto)
        print("  Confira a deriva depois: gyro parado deve ficar ~0 no eixo;")
        print("  se nao ficar, repita com as luvas 100% paradas.")
        return 0
    print("  AVISO: reiniciou, mas nao vi '[imu] bias MPU ...' no tempo escutado.")
    print("  A calibracao pode ter rodado antes da escuta: aumente --segundos")
    print("  (ex.: 12) e rode de novo, mantendo a luva parada.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
