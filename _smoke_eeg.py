"""Teste de fumaca do Programa 1 com o protocolo final acelerado (~25 s).

Bloco unico com 2 trials, tempos curtos, sem preparacao e sem baseline.
Rode dentro do terminal do VS Code (o g.Pype exige IDE suportada).
"""
import sys

sys.argv = [
    "eeg_motor_paradigm.py",
    "--source", "generator",
    "--blocos", "1",
    "--trials-por-bloco", "1",
    "--contagem", "0",
    "--home-sec", "0.2",
    "--me-sec", "0.3",
    "--video-sec", "0.2",
    "--mi-sec", "0.2",
    "--pausa-bloco-sec", "1",
    "--olhos-abertos", "0",
    "--olhos-fechados", "0",
    "--repouso-ativo", "0",
    "--hold-sec", "0.1",
    "--sem-preparacao",
    # O questionario do participante e' interativo (input()): desligado no
    # smoke test para nao travar a automacao.
    "--sem-questionario",
    "--csv", "_smoke_eeg.csv",
    "--seed", "7",
]

import eeg_motor_paradigm as p

# Calibracao de origem curta (le direto da constante do modulo).
p.ORIGIN_CAL_SEC = 0.5
# Trecho do video de priming curto (frame_buffer ainda vazio sem Kinect).
p.SLOWMO_END_LAG_SEC = 0.05
p.SLOWMO_SPAN_SEC = 0.2

p.main()

