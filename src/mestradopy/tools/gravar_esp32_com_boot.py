"""Grava o firmware do ESP32 com ASSISTENCIA DE BOOT (item 85).

POR QUE EXISTE (medido em 19/10/2026, item 84): o auto-reset desta placa nao
entra em download mode, e o pulso DTR/RTS que o esptool faz para resetar
derruba o bridge CP210x no meio da configuracao da porta:
    PermissionError(13, 'Um dispositivo conectado ao sistema nao esta funcionando')
A saida foi tirar o reset do esptool (`board_upload.before_reset = no_reset`,
`upload_speed = 115200` no platformio.ini) e entra' em download mode NA MAO.
Acertar o tempo de DOIS botoes (BOOT + EN) e' chato; aqui o RESET e' dado pelo
proprio PC pela serial (RTS = EN), entao o usuario so' precisa SEGURAR o BOOT:

    1) segure o BOOT
    2) ENTER aqui -> o script pulsa o EN (RTS) e a placa sobe em download mode
    3) o upload roda com --before=no_reset e grava

Se o pulso nao funcionar nesta montagem (EN nao ligado no RTS), o plano B e'
segurar o BOOT ENQUANTO pluga o USB (a placa sobe em download mode) e rodar
com --sem-pulso.

Uso:
    .venv\\Scripts\\python.exe tools\\gravar_esp32_com_boot.py [COMx]
    .venv\\Scripts\\python.exe tools\\gravar_esp32_com_boot.py COM4 --sem-pulso
    (ou o atalho na raiz: GRAVAR_ESP32.bat COMx)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PROJETO = RAIZ / "firmware" / "esp32_duas_imus"


def encontra_pio():
    """pio.exe: extensao do VS Code, .venv do projeto, PATH (mesma ordem do gravar.bat)."""
    candidatos = [
        Path(os.environ.get("USERPROFILE", "")) / ".platformio" / "penv" / "Scripts" / "pio.exe",
        RAIZ / ".venv" / "Scripts" / "pio.exe",
    ]
    for candidato in candidatos:
        if candidato.exists():
            return str(candidato)
    return "pio"


def pulso_reset(porta):
    """Pulso de RESET pela serial: RTS = EN (DTR fica alto para nao forcar o GPIO0)."""
    import serial  # pyserial (mesmo que o PlatformIO usa)

    with serial.Serial(porta, 115200, timeout=0.2) as porta_serial:
        porta_serial.dtr = False     # GPIO0 nao forcado (o BOOT e' apertado na mao)
        porta_serial.rts = True      # EN baixo = placa em reset
        time.sleep(0.15)
        porta_serial.rts = False     # solta o reset: com o BOOT apertado, sobe em download
        time.sleep(0.10)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("porta", nargs="?", default="COM4",
                        help="porta serial do ESP32 (padrao COM4)")
    parser.add_argument("--env", default="esp32_duas_imus",
                        help="ambiente do PlatformIO (padrao esp32_duas_imus)")
    parser.add_argument("--sem-pulso", action="store_true",
                        help=("NAO pulsa o reset: use quando a placa JA' esta' em "
                              "download mode (BOOT segurado enquanto plugou o USB)"))
    args = parser.parse_args(argv)

    print("=" * 74)
    print("  GRAVAR ESP32 - assistencia de BOOT (item 85)")
    print("  Como funciona: voce SEGURA o BOOT; este script da' o pulso de RESET")
    print("  (EN, pela serial) e chama o upload com --before=no_reset.")
    print("  Pode SOLTAR o BOOT assim que aparecer 'Connecting...'.")
    print("=" * 74)
    if args.sem_pulso:
        print("  --sem-pulso: assumindo que a placa JA' esta' em download mode")
        print("  (segure o BOOT e PLUGUE o USB; solte depois de ~2 s).")
    else:
        try:
            resposta = input("  [1] SEGURE o BOOT e tecle ENTER para pulsar o reset "
                             "(ou 'n' para cancelar): ").strip().lower()
        except EOFError:
            resposta = ""
        if resposta == "n":
            print("  cancelado")
            return 1
        try:
            pulso_reset(args.porta)
            print("  [2] reset pulsado; a placa deve estar em DOWNLOAD MODE")
        except Exception as exc:                      # noqa: BLE001 - caminho de HW
            print(f"  [2] FALHOU o pulso pela serial ({type(exc).__name__}: {exc})")
            print("      Plano B: segure o BOOT e PLUGUE o USB (download mode),")
            print("      depois rode de novo com --sem-pulso.")
            return 2
        time.sleep(0.2)

    pio = encontra_pio()
    comando = [pio, "run", "-d", str(PROJETO), "-e", args.env, "-t", "upload",
               "--upload-port", args.porta]
    print("  [3] gravando:", " ".join(comando), flush=True)
    codigo = subprocess.call(comando, cwd=str(RAIZ))
    if codigo == 0:
        print("\n  SUCESSO. Desligue e religue o USB da placa para o firmware novo subir.")
    else:
        print("\n  FALHOU. Se apareceu 'No serial data received', a placa nao entrou em")
        print("  download mode: repita (segure BOOT -> ENTER) ou use o plano B")
        print("  (BOOT apertado enquanto PLUGA o USB e rode com --sem-pulso).")
        print("  Se apareceu 'Could not open port'/ocupada: feche monitor serial/")
        print("  PlatformIO Monitor/Arduino IDE e tente de novo.")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
