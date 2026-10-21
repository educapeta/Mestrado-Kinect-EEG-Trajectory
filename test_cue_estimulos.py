"""Cue da trial: o SIMBOLO DA MAO nao pode invadir o objeto, a bola de tenis
tem de ter costura (nao so' um circulo) e os slides nao podem falar em SETA.

Sintomas reais (bancada Eduardo-PC, 30/09/2026, reportados pelo usuario):
  1. em telas quasi-quadradas a CAUDA da seta encostava/invadia o icone
     (medido: 900x900 = -9 px, 700x900 = -7 px; em 1920x1050 havia 372 px de
     folga, por isso so' aparecia na janela normal) -- "a seta sobreposta no
     icone da garrafa/caneta/bola";
  2. a bola era "so' um circulo" e parecia uma LAMPADA: o moveTo+arcTo antigo
     deixava uma RETA atravessada e dois arcos 10% MAIORES que o circulo (para
     fora); a costura agora e' a do Twemoji (U+1F3BE), duas cubicas por costura.

E a queixa do trial real de 21/10/2026, que trocou a seta pelo simbolo da mao:
  "a seta para a esquerda (mao esquerda) deixa entender que a bola esta a
  esquerda, mas muitas vezes o objeto esta no lado direito da mesa" -- a seta
  dizia o lado da MAO e era lida como o lado do OBJETO. Agora o cue mostra o
  desenho de uma mao (esquerda != direita, dedao do lado de dentro) ao lado do
  objeto, com rotulo curto, e os slides explicam isso.

Aqui e' aritimetica pura (sem Qt, sem render): `geometria_cue`,
`geometria_mao`, `geometria_rotulo_mao` e `costura_bola` sao exatamente o que
`_paint_hand`/`_paint_object`/`_draw_ball` usam para desenhar -- se alguem
voltar o calculo antigo (ou a seta), a suite derruba.
"""
import inspect
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

# =============================================================================
# 5) SIMBOLO DA MAO (21/10/2026): a seta foi trocada por um desenho de mao do
#    lado pedido -- ela dizia o lado da MAO, mas o participante lia como o lado
#    do OBJETO ("a seta para a esquerda deixa entender que a bola esta a
#    esquerda, mas muitas vezes o objeto esta no lado direito da mesa").
#    Aqui: em TODA tela, nas DUAS maos, o desenho (caixa E tinta) respeita a
#    MESMA folga da seta, nao invade a margem da borda e fica do lado certo.
# =============================================================================
for width, height in TELAS:
    objeto = p.geometria_cue(width, height)
    margem = width * p.MARGEM_SETA_FRAC
    for mao in MAOS:
        geo = p.geometria_mao(width, height, mao)
        assert geo is not None, (width, height, mao)
        x0, y0, x1, y1 = geo["caixa"]
        # caixa do simbolo: quadrada, sem sobrepor o objeto, dentro do canvas
        assert abs((x1 - x0) - (y1 - y0)) < 1e-6, ("nao e' quadrada", geo)
        assert abs(geo["lado"] - (x1 - x0)) < 1e-6, geo["lado"]
        folga = ((objeto["objeto_x"][0] - x1) if mao == "esquerda"
                 else (x0 - objeto["objeto_x"][1]))
        assert abs(folga - objeto["folga"]) < 1e-6, \
            (width, height, mao, "folga errada", folga, objeto["folga"])
        assert folga > 0, (width, height, mao, "simbolo SOBREPOE o objeto")
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, \
            (width, height, mao, geo["caixa"])
        # o lado do simbolo acompanha o lado da mao (esquerda a esquerda)
        if mao == "esquerda":
            assert x0 >= margem - 1e-6, (width, "invade a margem", x0)
            assert x1 <= objeto["objeto_x"][0], (width, "passou do objeto", x1)
        else:
            assert x1 <= width - margem + 1e-6, (width, "invade a margem", x1)
            assert x0 >= objeto["objeto_x"][1], (width, "passou do objeto", x0)
        # mesma linha do cue do objeto
        assert abs(geo["centro"][1] - objeto["cy"]) < 1e-6, geo["centro"]
        # o desenho NAO e' degenerado: em nenhuma das 9 telas o corredor
        # apertou o simbolo (o limite MARGEM/lado so' e' seguro, nao aperta)
        assert geo["lado"] >= objeto["lado"] * p.CAIXA_MAO_FRAC - 1e-6, \
            (width, height, mao, "corredor apertou o simbolo", geo["lado"])
        assert geo["lado"] >= 60, (width, height, mao, geo["lado"])
        # a TINTA (o que o participante ve) tambem esta dentro da caixa e
        # longe do objeto: a folga e' medida pelo desenho, nao so' pela caixa
        t0, ti0, t1, ti1 = geo["tinta"]
        assert x0 - 1e-6 <= t0 < t1 <= x1 + 1e-6, (width, mao, geo["tinta"])
        assert y0 - 1e-6 <= ti0 < ti1 <= y1 + 1e-6, (width, mao, geo["tinta"])
        if mao == "esquerda":
            assert t1 <= objeto["objeto_x"][0] - objeto["folga"] + 1e-6, \
                (width, height, "tinta na folga do objeto", t1)
            assert t0 >= margem, (width, "tinta na margem", t0)
        else:
            assert t0 >= objeto["objeto_x"][1] + objeto["folga"] - 1e-6, \
                (width, height, "tinta na folga do objeto", t0)
            assert t1 <= width - margem, (width, "tinta na margem", t1)
        # todos os pontos do tracado (ancoras E controles) dentro da caixa:
        # e' o que garante que o desenho nao "vaza" para cima do objeto
        assert geo["pontos"], "simbolo sem tracado"
        for (px, py) in geo["pontos"]:
            assert x0 - 1e-6 <= px <= x1 + 1e-6, (width, mao, px, geo["caixa"])
            assert y0 - 1e-6 <= py <= y1 + 1e-6, (width, mao, py, geo["caixa"])

# =============================================================================
# 6) LADO ESQUERDO != DIREITO: o desenho e' o ESPELHO em x (dedao do lado de
#    dentro) -- nao e' o mesmo icone deslocado nem, muito menos, uma seta.
#    Anatomia travada: na mao DIREITA o dedao fica a ESQUERDA do centro (e' o
#    que a propria mao direita mostra para quem olha de frente).
# =============================================================================
for width, height in TELAS:
    esquerda = p.geometria_mao(width, height, "esquerda")
    direita = p.geometria_mao(width, height, "direita")
    x0_e, y0_e = esquerda["caixa"][0], esquerda["caixa"][1]
    x0_d, y0_d = direita["caixa"][0], direita["caixa"][1]
    lado = direita["lado"]
    assert abs(lado - esquerda["lado"]) < 1e-9, (width, "lados diferentes")
    # 1) na MESMA caixa, a mao esquerda e' o espelho exato da direita
    local_e = sorted((round(px - x0_e, 9), round(py - y0_e, 9))
                     for (px, py) in esquerda["pontos"])
    espelho_d = sorted((round(lado - (px - x0_d), 9), round(py - y0_d, 9))
                       for (px, py) in direita["pontos"])
    assert len(local_e) == len(espelho_d), (width, len(local_e))
    for (ex, ey), (dx, dy) in zip(local_e, espelho_d):
        assert abs(ex - dx) < 1e-6 and abs(ey - dy) < 1e-6, \
            (width, height, "nao e' espelho", (ex, ey), (dx, dy))
    # 2) o desenho NAO e' simetrico (senao o espelho nao mudaria nada e o
    #    lado esquerdo seria igual ao direito na tela)
    simetrico = {(round(lado - ex, 3), round(ey, 3)) for (ex, ey) in local_e}
    direto = {(round(ex, 3), round(ey, 3)) for (ex, ey) in local_e}
    assert direto != simetrico, (width, "desenho simetrico: lados iguais")
    # 3) anatomia: na mao DIREITA o dedao fica a ESQUERDA do centro (e' o que a
    #    propria mao direita mostra para quem olha de frente) e vice-versa
    meio_e = (esquerda["caixa"][0] + esquerda["caixa"][2]) / 2.0
    meio_d = (direita["caixa"][0] + direita["caixa"][2]) / 2.0
    dedo_e = sum(px for (px, _py) in esquerda["dedao"]) / 2.0
    dedo_d = sum(px for (px, _py) in direita["dedao"]) / 2.0
    assert dedo_d < meio_d, ("mao direita sem o dedao a esquerda", dedo_d, meio_d)
    assert dedo_e > meio_e, ("mao esquerda sem o dedao a direita", dedo_e, meio_e)
    # 4) quatro dedos apontando para CIMA + um dedao saindo de lado
    assert len(direita["dedos"]) == 4, len(direita["dedos"])
    for (base, ponta) in direita["dedos"]:
        assert ponta[1] < base[1], ("dedo apontando para baixo", base, ponta)
    (bx, by), (tx, ty) = direita["dedao"]
    assert ty < by, ("dedao apontando para baixo", (bx, by), (tx, ty))
    assert tx < bx, ("dedao para dentro da mao", (bx, by), (tx, ty))
    assert abs(tx - bx) > 0.25 * abs(ty - by), \
        ("dedao quase vertical: nao da' para ver de que lado ele esta", tx, bx)
    # 5) mao nao-lateral nao desenha nada (o cue sem mao continua valido)
    for invalido in (None, "cima", ""):
        assert p.geometria_mao(900, 900, invalido) is None, invalido

# =============================================================================
# 7) ROTULO curto ("MÃO ESQUERDA"): cabe INTEIRO no corredor livre entre a mao
#    e o objeto (nao encosta no objeto), fica na tela e e' legivel -- abaixo do
#    minimo o proprio `geometria_rotulo_mao` devolve None em vez de desenhar
#    um texto ilegivel ou sobreposto.
# =============================================================================
for width, height in TELAS:
    objeto = p.geometria_cue(width, height)
    margem = width * p.MARGEM_SETA_FRAC
    for mao in MAOS:
        rotulo = p.geometria_rotulo_mao(width, height, mao)
        assert rotulo is not None, (width, height, mao, "sem rotulo")
        geo = p.geometria_mao(width, height, mao)
        assert rotulo["texto"] in ("MÃO ESQUERDA", "MÃO DIREITA"), \
            rotulo["texto"]
        x0, y0, x1, y1 = rotulo["caixa"]
        # legivel: nunca abaixo do minimo que o modulo aceita desenhar
        assert rotulo["tamanho"] >= p.TAMANHO_MINIMO_ROTULO, rotulo
        assert rotulo["linhas"] >= 1, rotulo
        # dentro do corredor da mao: nao sobrepoe o objeto nem sai da tela
        assert 0 <= x0 < x1 <= width, (width, mao, rotulo["caixa"])
        if mao == "esquerda":
            assert x0 >= margem - 1e-6, (width, mao, "sai na margem", x0)
            assert x1 <= objeto["objeto_x"][0] - objeto["folga"] + 1e-6, \
                (width, mao, "rotulo no objeto", x1)
        else:
            assert x0 >= objeto["objeto_x"][1] + objeto["folga"] - 1e-6, \
                (width, mao, "rotulo no objeto", x0)
            assert x1 <= width - margem + 1e-6, (width, mao, "sai na margem")
        # abaixo do simbolo (nao por cima do desenho) e dentro da tela
        assert y0 >= geo["caixa"][3] - 1e-6, (mao, "rotulo em cima da mao")
        assert y1 <= height, (width, height, mao, "rotulo fora da tela", y1)
        # a caixa e' do tamanho do texto: a estimativa (o mesmo
        # LARGURA_CARACTERE_EM que o codigo usa) tem de caber nela...
        em = (max(len(parte) for parte in rotulo["texto"].split())
              if rotulo["linhas"] > 1 else len(rotulo["texto"]))
        largura_estimada = em * p.LARGURA_CARACTERE_EM * rotulo["tamanho"]
        assert largura_estimada <= (x1 - x0) + 1e-6, \
            (width, mao, "estimativa nao cabe", largura_estimada, x1 - x0)
        # ...e o Arial Bold REAL medido no Qt cabe com folga (0,9175 em/char
        # medido a 100 pt: a estimativa com 0,95 e' conservadora)
        assert p.LARGURA_CARACTERE_EM > 0.9175, p.LARGURA_CARACTERE_EM
        assert (x1 - x0) >= 0.98 * 0.9175 * em * rotulo["tamanho"], \
            (width, mao, "rotulo apertado", x1 - x0)
        # altura = uma linha por linha do texto (o desenho nao corta embaixo)
        assert (y1 - y0) >= p.LINHA_ROTULO_EM * rotulo["tamanho"] - 1e-6, \
            (width, mao, y1 - y0)

# =============================================================================
# 8) A PINTURA usa a geometria pura e NAO tem mais seta (senao o teste acima
#    mede uma coisa e a tela mostra outra)
# =============================================================================
assert not hasattr(p._StimulusCanvas, "_paint_arrow"), "a seta voltou"
fonte_mao = inspect.getsource(p._StimulusCanvas._paint_hand)
assert "geometria_mao(" in fonte_mao, "pintura da mao sem geometria_mao"
assert "cubicTo" in fonte_mao and "drawLine" in fonte_mao, fonte_mao
assert "geometria_rotulo_mao(" in inspect.getsource(
    p._StimulusCanvas._paint_rotulo_mao), "rotulo sem geometria_rotulo_mao"
assert "Qt.TextWordWrap" in inspect.getsource(
    p._StimulusCanvas._paint_rotulo_mao), "rotulo sem quebra de palavra"
fonte_cue = inspect.getsource(p._StimulusCanvas.paintEvent)
assert "_paint_hand(" in fonte_cue and "_paint_arrow(" not in fonte_cue, \
    "paintEvent ainda chama a seta"

# =============================================================================
# 9) Os SLIDES de instrucao falam do simbolo da mao (nao da seta) e continuam
#    com as palavras-chave do protocolo (test_paradigm_melhorias depende delas)
# =============================================================================
slides = p.slides_instrucao()
assert len(slides) == 4, len(slides)
texto_slides = " ".join(slides).upper()
assert "SETA" not in texto_slides, "slide ainda fala da SETA"
for chave in ("SIMBOLO", "MAO", "ESQUERDA", "DIREITA", "GARRAFA", "BOLA",
              "CANETA", "EXECUTE", "IMAGINE", "HOME", "ESPACO"):
    assert chave in texto_slides, f"slide sem a palavra {chave}"
assert "MAO ESQUERDA" in texto_slides and "MAO DIREITA" in texto_slides

print("test_cue_estimulos_OK (folga seta x objeto em 9 telas x 2 maos | "
      "regressao -9 px travada | costura Twemoji da bola: 4 pes no circulo, "
      "mergulho 0,30-0,45 R, 2 costuras espelhadas | SIMBOLO DA MAO: caixa + "
      "tinta fora da folga do objeto e da margem em 9 telas x 2 maos, "
      "esquerda == espelho de direita com o dedao do lado de dentro, rotulo "
      "'MÃO ESQUERDA'/'MÃO DIREITA' dentro do corredor e legivel, pintura sem "
      "seta, slides sem SETA)")