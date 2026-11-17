"""Regressao: medidor de exposicao/saturacao das webcams (item 104).

Contexto: luz extra na bancada estourou a Trust (e encostou no Kinect) e a
calibracao J/V piorou. `tools/ajusta_exposicao_cameras.py` mede por exposicao:
brilho medio, p99, % de pixels saturados e % escuros; escolhe a melhor (sem
estouro, brilho perto do alvo) e imprime a linha de comando do tool.

Este teste cobre a PARTE PURA (sem hardware):
  1. `_metricas_quadro`: quadro preto -> 0% estourado/escuro total; quadro
     estourado -> % estourado alto; meio-termo -> media esperada;
  2. `_escolhe_melhor`: prefere a exposicao sem estouro com brilho perto do
     alvo; se TODAS estouram, devolve a de MENOR estouro;
  3. integracao: o tool de exposicao existe, aceita --exposicoes e --kinect, e
     a linha recomendada sai no formato "--exposicao \"0:N,1:N\"".
"""
from __future__ import annotations

import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "tools"))
import ajusta_exposicao_cameras as expo                 # noqa: E402

FALHAS = []


def check(nome, condicao, detalhe=""):
    if condicao:
        print(f"   ok: {nome}")
    else:
        FALHAS.append(nome)
        print(f"   FALHOU: {nome} {detalhe}")


print("1) metricas de um quadro")
preto = np.zeros((40, 40, 3), dtype=np.uint8)
media, p99, estourados, escuros = expo._metricas_quadro(preto)
check("quadro preto: media 0 e 100% escuro",
      media == 0.0 and escuros == 100.0 and estourados == 0.0,
      (media, p99, estourados, escuros))
branco = np.full((40, 40, 3), 255, dtype=np.uint8)
media, p99, estourados, escuros = expo._metricas_quadro(branco)
check("quadro branco: 100% estourado", estourados == 100.0 and media == 255.0,
      (media, p99, estourados, escuros))
meio = np.full((40, 40, 3), 120, dtype=np.uint8)
media, p99, estourados, escuros = expo._metricas_quadro(meio)
check("quadro medio: sem estouro/escuro",
      media == 120.0 and estourados == 0.0 and escuros == 0.0,
      (media, estourados, escuros))
misto = np.zeros((40, 40, 3), dtype=np.uint8)
misto[:10, :, :] = 255          # 25% estourado
media, p99, estourados, escuros = expo._metricas_quadro(misto)
check("quadro 25% estourado: fracao ~25%",
      abs(estourados - 25.0) < 1e-9, estourados)

print("2) escolha da melhor exposicao")
resultados = {
    -8.0: {"media": 40.0, "p99": 60.0, "estourados": 0.0, "escuros": 60.0},
    -6.0: {"media": 110.0, "p99": 200.0, "estourados": 0.0, "escuros": 5.0},
    -4.0: {"media": 180.0, "p99": 255.0, "estourados": 20.0, "escuros": 0.0},
}
melhor = expo._escolhe_melhor(resultados, alvo_media=115.0)
check("escolhe a exposicao sem estouro com brilho perto do alvo",
      melhor == -6.0, melhor)
todas_estouradas = {
    -4.0: {"media": 200.0, "p99": 255.0, "estourados": 60.0, "escuros": 0.0},
    -3.0: {"media": 210.0, "p99": 255.0, "estourados": 70.0, "escuros": 0.0},
}
melhor = expo._escolhe_melhor(todas_estouradas)
check("se todas estouram, devolve a de MENOR estouro",
      melhor == -4.0, melhor)
resultados_claros = {
    -6.0: {"media": 110.0, "p99": 200.0, "estourados": 0.0, "escuros": 5.0},
    -4.0: {"media": 160.0, "p99": 240.0, "estourados": 1.0, "escuros": 2.0},
}
check("respeita alvo_media (alvo alto prefere a exposicao mais clara)",
      expo._escolhe_melhor(resultados_claros, alvo_media=170.0) == -4.0)

print("3) integracao")
raiz_tool = os.path.join(RAIZ, "tools", "ajusta_exposicao_cameras.py")
check("ferramenta existe", os.path.exists(raiz_tool))
fonte = open(raiz_tool, encoding="utf-8").read()
check("aceita --exposicoes e --kinect",
      '"--exposicoes"' in fonte and '"--kinect"' in fonte)
check("mede o Kinect RGB e avisa quando estoura",
      "_mede_kinect" in fonte and "ESTOURADO" in fonte)
check("imprime a linha de comando recomendada",
      '--exposicao "' in fonte)

if FALHAS:
    print()
    print(f"EXPOSICAO_CAMERAS_FALHOU: {len(FALHAS)} verificacao(oes): "
          + " | ".join(FALHAS))
    raise SystemExit(1)
print()
print("EXPOSICAO_CAMERAS_OK: metricas (media/p99/%estourado/%escuro) | escolha "
      "da melhor exposicao sem estouro e perto do alvo | fallback pela menor "
      "saturacao | ferramenta com --exposicoes/--kinect e linha de comando")
