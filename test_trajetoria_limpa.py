"""Regressao: a trajetoria desenhada no clipe nao pode ter zigue-zague.

Queixa do trial real de 21/10/2026: no video de priming (frames do Kinect com o
esqueleto) o desenho da trajetoria da mao ficou "todo contorcido/zigzagueante".

Causa medida na sessao `gravacao_MEMI_20261021_145700`: o historico da
trajetoria (`TrackingThread._traj_history`) recebe a posicao de QUALQUER fonte
(depth do Kinect, esqueleto do SDK, triangulacao) e de QUALQUER lado. Na ME2 o
`KT_src` alternou 2<->3 (depth <-> esqueleto) com saltos de 74-90 cm entre dois
quadros e o desenho ligava os dois pontos com uma RETA -- na sessao inteira
houve 47 saltos >10 cm (27 >25 cm, max 135 cm).

Correcao: `separa_trajetoria` quebra o caminho em TRECHOS fisicamente
plausiveis (passo > TRAJ_MAX_PASSO_M ou gap > TRAJ_MAX_GAP_S) e o clipe desenha
um `polylines` por trecho -- nunca uma reta entre pontos de fontes/lados
diferentes.

Este teste roda as funcoes puras (sem Qt, sem Kinect, sem video real) e trava a
estrutura criada no codigo.
"""
from __future__ import annotations

import inspect

import numpy as np

import eeg_motor_paradigm as p

assert p.TRAJ_MAX_PASSO_M > 0, p.TRAJ_MAX_PASSO_M
assert p.TRAJ_MAX_GAP_S > 0, p.TRAJ_MAX_GAP_S


def reta(t0=1000.0, n=10, passo=0.01, z=0.0):
    """Trajetoria reta (1 cm por quadro a 30 Hz) em (t, (x, y, z))."""
    return [(t0 + i * 0.033, np.array([i * passo, 0.0, z]))
            for i in range(n)]


# 1) Trajetoria continua = UM trecho, todos os pontos preservados
trechos = p.separa_trajetoria(reta())
assert len(trechos) == 1, trechos
assert len(trechos[0]) == 10, len(trechos[0])

# 2) Salto de FONTE/LADO (ex.: depth -> esqueleto SDK, 50 cm num quadro) =
#    DOIS trechos; o desenho NAO liga um no outro
t = 1000.0
pontos = []
for i in range(6):
    pontos.append((t, np.array([0.01 * i, 0.0, 0.0])))
    t += 0.033
for i in range(6):
    pontos.append((t, np.array([0.5 + 0.01 * i, 0.0, 0.0])))
    t += 0.033
trechos = p.separa_trajetoria(pontos)
assert len(trechos) == 2, [len(t) for t in trechos]
assert len(trechos[0]) == 6 and len(trechos[1]) == 6, \
    [len(t) for t in trechos]

# 3) Gap de tempo (mao perdida por > TRAJ_MAX_GAP_S) tambem quebra o trecho
t = 1000.0
pontos = []
for i in range(5):
    pontos.append((t, np.array([0.01 * i, 0.0, 0.0])))
    t += 0.033
t += p.TRAJ_MAX_GAP_S + 0.5           # buraco: mao perdida
for i in range(5):
    pontos.append((t, np.array([0.2 + 0.01 * i, 0.0, 0.0])))
    t += 0.033
trechos = p.separa_trajetoria(pontos)
assert len(trechos) == 2, [len(t) for t in trechos]

# 4) Linhas invalidas (NaN/None) sao IGNORADAS e nao quebram a trajetoria
pontos = reta()
meio = len(pontos) // 2
pontos.insert(meio, (pontos[meio][0], np.array([np.nan, 0.0, 0.0])))
pontos.insert(meio, None)
pontos.insert(meio, (None, np.array([0.0, 0.0, 0.0])))
trechos = p.separa_trajetoria(pontos)
assert len(trechos) == 1, trechos
assert len(trechos[0]) == 10, len(trechos[0])

# 5) Limite exato: passo == TRAJ_MAX_PASSO_M NAO quebra; um pouco acima quebra
limite = [(1000.0, np.array([0.0, 0.0, 0.0])),
          (1000.033, np.array([p.TRAJ_MAX_PASSO_M, 0.0, 0.0]))]
assert len(p.separa_trajetoria(limite)) == 1
acima = [(1000.0, np.array([0.0, 0.0, 0.0])),
         (1000.033, np.array([p.TRAJ_MAX_PASSO_M + 0.01, 0.0, 0.0]))]
assert len(p.separa_trajetoria(acima)) == 0, \
    "um trecho de 2 pontos separados por salto NAO pode ser desenhado"

# 6) Caso REAL da ME2 (21/10): alternancia depth(2) <-> esqueleto(3) a cada
#    quadro, 0.8 m de distancia -> nenhum trecho desenhavel (era isto que virava
#    o zigue-zague no video)
t = 1000.0
alternado = []
for i in range(20):
    x = 0.0 if i % 2 == 0 else 0.8
    alternado.append((t, np.array([x, 0.0, 0.0])))
    t += 0.033
assert p.separa_trajetoria(alternado) == [], \
    "a alternancia de fontes ainda desenha um zigue-zague"

# 7) Trecho com menos de 2 pontos nao e' desenhado
assert p.separa_trajetoria([]) == []
assert p.separa_trajetoria([(1000.0, np.zeros(3))]) == []
assert p.separa_trajetoria([None, (1000.0, np.zeros(3))]) == []

# 8) O CLIPE usa a separacao (um polylines por trecho) -- nao voltar ao array
#    unico que ligava os saltos
fonte_pixels = inspect.getsource(p.TrackingThread._trajectory_pixels)
assert "separa_trajetoria" in fonte_pixels, fonte_pixels
assert "return pixels or None" in fonte_pixels, fonte_pixels
fonte_clip = inspect.getsource(p.TrackingThread._publish_slowmo)
assert "for trecho in trechos" in fonte_clip, fonte_clip
assert "polylines" in fonte_clip, fonte_clip
assert "_trajectory_pixels(t0, t1)" in fonte_clip, fonte_clip
# a docstring/constantes explicam o motivo (o proximo dev nao "otimiza" de volta)
assert "zigue-zague" in p.separa_trajetoria.__doc__, \
    p.separa_trajetoria.__doc__[:200]

print("TRAJETORIA_LIMPA_OK",
      "reta = 1 trecho | salto de fonte/lado = 2 trechos (sem reta entre eles) "
      "| gap de tempo quebra | NaN/None ignorados | limite de passo exato | "
      "alternancia depth<->SDK nao desenha zigue-zague | clipe desenha 1 "
      "polylines por trecho")
