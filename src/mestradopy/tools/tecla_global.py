"""Injeta UMA tecla GLOBAL no Windows (`keybd_event`) -- bancada do cancelamento.

Por que existe
--------------
O vigia do paradigma (`hotkeys.GlobalHotkeyWatcher`) le o estado das teclas no
Windows INTEIRO pelo `GetAsyncKeyState`, e nao pelo foco do Qt -- foi o que
resolveu a bancada de 25/09/2026 (a janela do OpenCV roubava o foco e
ESPACO/ESC "morriam"). Isso deixou uma pergunta pratica em aberto: "como testar
o cancelamento sem ficar apertando o teclado no meio da sessao?" -- e' o que
esta ferramenta responde.

`keybd_event` (user32) pressiona e solta a tecla NO SISTEMA, e o
`GetAsyncKeyState` enxerga isso como tecla de verdade (borda solto ->
pressionado). Foi assim que a bancada de 28/09/2026 provou, com o paradigma
rodando de verdade (`--source generator --blocos 1 --trials-por-bloco 4`):
    C    -> "trial 3/4 CANCELADO pelo experimentador" + marcador 796 no CSV
    ESC  -> "sessao ABORTADA" + resumo dos arquivos + o processo saiu < 7 s

Uso
---
    .venv\\Scripts\\python.exe tools\\tecla_global.py c        # cancela a trial
    .venv\\Scripts\\python.exe tools\\tecla_global.py esc      # aborta a sessao
    .venv\\Scripts\\python.exe tools\\tecla_global.py space    # pula a espera
    .venv\\Scripts\\python.exe tools\\tecla_global.py esc --espera 20   # ESC daqui
                                                          # a 20 s (deixe a sessao
                                                          # chegar na fase)

    --ms N      quanto tempo a tecla fica pressionada (padrao 120 ms; o vigia
                varre o teclado a cada 20 ms, entao 120 ms tem folga)
    --espera S  espera S segundos antes de injetar

CUIDADO: a tecla e' global de VERDADE
------------------------------------
Ela entra na fila de teclado do sistema, entao vai tambem para a JANELA EM FOCO
(e nao so' para o vigia do paradigma): com um editor em primeiro plano, o ESC/
ESPACO vao para o editor. Deixe o paradigma em primeiro plano. Nao ha' como ser
visto pelo `GetAsyncKeyState` sem passar pela fila do sistema -- e' a mesma
limitacao que faz o recurso funcionar.

Nao ha' dependencia nova: `ctypes` e' da stdlib, igual ao `hotkeys.py`.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotkeys import VK_C, VK_ESCAPE, VK_SPACE  # noqa: E402

#: Codigos VK_* vindos do proprio `hotkeys.py`: nada de numero magico
#: duplicado aqui (se o paradigma trocar uma tecla, esta ferramenta segue).
TECLAS = {"esc": VK_ESCAPE, "space": VK_SPACE, "c": VK_C}

KEYEVENTF_KEYUP = 0x0002

#: Tempo (ms) com a tecla pressionada. O vigia varre o teclado a cada 20 ms:
#: 120 ms garante que a borda seja vista ate' com a bancada consumindo CPU.
MS_PADRAO = 120


def injetar(vk, ms=MS_PADRAO):
    """Pressiona e solta a tecla `vk`; o vigia do paradigma ve UMA borda."""
    if sys.platform != "win32":
        raise RuntimeError("keybd_event e' da API do Windows "
                           "(o paradigma inteiro e' Windows-only)")
    user32 = ctypes.windll.user32
    user32.keybd_event(int(vk), 0, 0, 0)
    time.sleep(max(0.01, float(ms) / 1000.0))
    user32.keybd_event(int(vk), 0, KEYEVENTF_KEYUP, 0)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tecla", choices=sorted(TECLAS),
                        help="tecla a injetar: c (cancela a trial), "
                             "esc (aborta a sessao), space (pula a espera)")
    parser.add_argument("--ms", type=float, default=MS_PADRAO,
                        help=f"tempo com a tecla pressionada "
                             f"(padrao {MS_PADRAO:g} ms)")
    parser.add_argument("--espera", type=float, default=0.0,
                        help="espera (s) antes de injetar a tecla")
    args = parser.parse_args()

    if args.espera > 0:
        print(f"[tecla] esperando {args.espera:g} s para injetar "
              f"'{args.tecla}' -- deixe a janela do paradigma em primeiro plano",
              flush=True)
        time.sleep(args.espera)

    codigo = TECLAS[args.tecla]
    injetar(codigo, args.ms)
    time.sleep(0.05)                 # deixa a fila do sistema drenar
    print(f"[tecla] injetada: {args.tecla} (vk={codigo:#04x}, "
          f"{args.ms:g} ms)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
