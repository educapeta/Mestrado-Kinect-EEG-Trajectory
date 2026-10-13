"""Cue da trial: a seta NAO pode invadir o objeto, e a bola de tenis tem de
ter costura (nao so' um circulo).

Sintomas reais (bancada Eduardo-PC, 30/09/2026, reportados pelo usuario):
  1. em telas quasi-quadradas a CAUDA da seta encostava/invadia o icone
     (medido: 900x900 = -9 px, 700x900 = -7 px; em 1920x1050 havia 372 px de
     folga, por isso so' aparecia na janela normal) -- "a seta sobreposta no
     icone da garrafa/caneta/bola";
  2. a bola era "so' um circulo" e parecia uma LAMPADA: o moveTo+arcTo antigo
     deixava uma RETA atravessada e dois arcos 10% MAIORES que o circulo (para
     fora); a costura agora e' a do Twemoji (U+1F3BE), duas cubicas por costura.

Aqui e' aritimetica pura (sem Qt, sem render): `geometria_cue` e
`costura_bola` sao exatamente o que `_paint_arrow`/`_paint_object` e
`_draw_ball` usam para desenhar -- se alguem voltar o calculo antigo, a
suite derruba.
"""
import math

import eeg_motor_paradigm as p

TELAS = [(900, 900), (700, 900), (1920, 1050), (1200, 800), (600, 1000),
         (1000, 600), (800, 450), (640, 480), (3840, 2160)]
MAOS = ("esquerda", "direita")

# =============================================================================
# 1) folga: em TODA tela, nas DUAS maos, a seta fica separada do objeto
# =============================================================================
for width, height in TELAS:
    for mao in MAOS:
        geo = p.geometria_cue(width, height, mao)
        x0, x1 = geo["seta_x"]
        o0, o1 = geo["objeto_x"]
        folga = (o0 - x1) if mao == "esquerda" else (x0 - o1)
        assert folga >= geo["folga"] - 1e-6, (width, height, mao, folga, geo)
        assert folga > 0, (width, height, mao, "seta SOBREPOE o objeto")
        # ponta encostada na borda certa, tudo dentro do canvas, comprimento
        # > 0 (a seta continua existindo em qualquer proporcao)
        if mao == "esquerda":
            assert abs(x0 - width * p.MARGEM_SETA_FRAC) < 1e-6, (width, x0)
        else:
            assert abs(x1 - (width - width * p.MARGEM_SETA_FRAC)) < 1e-6, \
                (width, x1)
        assert 0 <= x0 < x1 <= width, (width, height, mao, x0, x1)
        assert geo["comprimento"] > 0, (width, height, mao)

# =============================================================================
# 2) objeto: caixa quadrada centralizada, dentro do canvas, na linha do cue
# =============================================================================
for width, height in TELAS:
    geo = p.geometria_cue(width, height)          # sem mao: so' o objeto
    o0, o1 = geo["objeto_x"]
    assert abs((o0 + o1) / 2 - width / 2) < 1e-6, (width, "fora de centro")
    assert 0 <= o0 < o1 <= width, (width, height, o0, o1)
    assert abs(geo["lado"] - min(width, height) * 0.30) < 1e-6, geo["lado"]
    assert abs(geo["cy"] - height * 0.40) < 1e-6, geo["cy"]
    # a folga pedida e' 5% do menor lado (e' ela que a seta respeita)
    assert abs(geo["folga"] - min(width, height) * 0.05) < 1e-6

# =============================================================================
# 3) regressao EXATA do caso reportado: 900x900 tinha -9 px de sobreposicao
# =============================================================================
geo = p.geometria_cue(900, 900, "direita")
sobra = geo["seta_x"][0] - geo["objeto_x"][1]
assert sobra >= 45 - 1e-6, f"900x900 voltou a sobrepor: {sobra:.1f} px"
geo = p.geometria_cue(700, 900, "esquerda")
sobra = geo["objeto_x"][0] - geo["seta_x"][1]
assert sobra >= 35 - 1e-6, f"700x900 voltou a sobrepor: {sobra:.1f} px"

# =============================================================================
# 4) costuras da bola de tenis (Twemoji U+1F3BE): os QUATRO pes SOBRE o circulo,
#    todo o tracado POR DENTRO, mergulho de 0,39 R no meio (nem colada na borda
#    = "circulo com um risco", nem passando pelo centro = "X") e as duas
#    costuras espelhadas esquerda/direita
# =============================================================================
LADO_CAIXA = min(900, 900) * p.CAIXA_OBJETO_FRAC    # caixa do objeto (px)
CENTRO = 450.0
RAIO = LADO_CAIXA * 0.46                            # exatamente o de _draw_ball
pontos = p.costura_bola(CENTRO, CENTRO, RAIO)
# 14 pontos: 2 costuras x (1 inicio + 2 cubicas x 3 pontos)
assert len(pontos) == 2 * (1 + 2 * 3), len(pontos)


def _dist(pt):
    return math.hypot(pt[0] - CENTRO, pt[1] - CENTRO)


# os pés = inicio/fim de cada costura (x=30.374 e x=28.94 no viewBox r=18)
PES = (pontos[0], pontos[6], pontos[7], pontos[13])
for pe in PES:
    assert abs(_dist(pe) - RAIO) <= 0.01 * RAIO, (_dist(pe), pe)

# NENHUM ponto do tracado sai do circulo (o bug antigo tinha arcos 10% maiores)
for pt in pontos:
    assert _dist(pt) <= RAIO + 1e-6, (pt, _dist(pt))

# e a costura MERGULHA ate' ~0,39 R: o pico para dentro nao pode ser raso
# (viraria "circulo com um risco") nem profundo (cruzaria o centro, um "X")
INTERIORES = [pt for pt in pontos if pt not in PES]
mergulho = min(_dist(pt) for pt in INTERIORES)
assert mergulho <= 0.45 * RAIO, ("costura colada no circulo", mergulho / RAIO)
assert mergulho >= 0.30 * RAIO, ("costura invadindo o centro", mergulho / RAIO)

# sao DUAS costuras (uma em cada metade), e o espelho em x fecha ponto a ponto
assert (pontos[0][0] - CENTRO) * (pontos[7][0] - CENTRO) < 0, pontos[:1] + pontos[7:8]
for (x, y) in pontos:
    espelho = (2 * CENTRO - x, y)
    assert any(abs(espelho[0] - qx) < 1e-6 and abs(espelho[1] - qy) < 1e-6
               for (qx, qy) in pontos), (x, y)

# a bola inteira (circulo + costura) cabe dentro da caixa do objeto
assert RAIO <= LADO_CAIXA * 0.5, (RAIO, LADO_CAIXA)

print("test_cue_estimulos_OK (folga seta x objeto em 9 telas x 2 maos | "
      "regressao -9 px travada | costura Twemoji da bola: 4 pes no circulo, "
      "mergulho 0,30-0,45 R, 2 costuras espelhadas)")