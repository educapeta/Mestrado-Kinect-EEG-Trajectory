"""Teste da deteccao de tabuleiro em CASCATA (sem hardware).

Contexto medido na bancada: a MESMA configuracao de deteccao acertou numa rodada
e falhou na outra (640 px e 960 px ja' fizeram os dois), com o tabuleiro na
mesma faixa de tamanho. Ou seja, o resultado sai por sorte e o detector esta' na
fronteira por causa da imagem (luz/ruido), nao por causa da escala. O codigo
tenta varias larguras em cascata (DETECTION_WIDTHS), guardando a que funcionou
para tentar primeiro.

Este teste garante o que passaria despercebido e corromperia a calibracao:

1. a cascata acha o tabuleiro do usuario (o TAMANHO vem de
   sc.CHECKERBOARD_SIZE -- hoje 6x4 quadrados = 5x3 cantos, o mesmo gravado nas
   calibracoes aux1/aux2; o teste antes fixava 11x7 e quebrava quando o
   tabuleiro da bancada mudou);
2. as coordenadas VOLTAM para a escala do FRAME ORIGINAL -- se ficassem na
   escala reduzida, a calibracao sairia silenciosamente errada;
3. a largura que funcionou e' memorizada e tentada primeiro;
4. prepare_gray respeita a largura alvo e devolve a escala de volta coerente;
5. largura alvo 0 = resolucao cheia (escape hatch);
6. refina_cantos reduz o erro do canto que veio de uma deteccao REDUZIDA (o
   risco real: achar a 427 px e multiplicar por 3 deixa ~1,5 px de erro, perto
   do MAX_STEREO_RMS_PX de 2 px);
7. abre_kinect reclama ALTO quando o Kinect nao entrega frame -- antes ele
   ficava mudo, porque o loop fazia `continue` antes do cv2.imshow e nenhuma
   janela aparecia ("rodei e nao apareceu nada");
8. abre_kinect devolve o runtime quando o Kinect funciona;
9. abre_kinect traduz o erro do pykinect2 numa instrucao acionavel.

A tecla T da ferramenta (diagnostico do tabuleiro) tambem e' testada aqui, sem
hardware, com tracker/calibracao falsos:

10. centro e bordas do resumo saem da GEOMETRIA do tabuleiro (o resumo usava
    `errors[24]`, indice da placa ANTIGA de 11x7, e estourava IndexError com a
    de 5x3);
11. a ordem canonica dos cantos e' a MESMA da calibracao
    (`stereo_calibration.canonical_order`), conferida em tabuleiros aleatorios
    com perspectiva e simetria sorteadas -- sao duas copias da mesma regra;
12. o pareamento dos cantos do diagnostico e' o de MENOR erro entre as 4
    simetrias da grade (a ordem que o detector devolve nao diz qual canto fisico
    e' o primeiro, e projetar as listas cruas compara cantos que nao se
    correspondem: centenas de px com a calibracao perfeita);
13. erro grande vira aviso acionavel ("RECALIBRE nesta maquina"), porque o RMS
    do arquivo de calibracao so' mede a coerencia INTERNA dele -- medido
    28/09/2026: a calibracao do repositorio deu 491 px nesta bancada e uma
    homografia nova do mesmo par fechou em 0,3 px;
14. a HOMOGRAFIA NOVA (ajustada com os MESMOS cantos detectados) separa as duas
    causas, que pedem consertos OPOSTOS: deteccao boa + erro grande = RECALIBRE;
    deteccao ruim (tabuleiro que mudou entre as duas capturas) = MELHORE A
    CAPTURA -- sem a segunda mensagem o dia de bancada comecaria recalibrando por
    causa de uma foto borrada;
15. npz que diz ter sido gravado em OUTRA camera e' denunciado antes de qualquer
    numero: o indice do OpenCV e' a posicao da camera no USB e troca sozinho.
"""
import contextlib
import io
import os
import sys

import cv2
import numpy as np

import stereo_calibration as sc
import kinect_imu_groundtruth as gt      # geometria do tabuleiro (centro/bordas)

RAIZ = os.path.dirname(os.path.abspath(__file__))
# O diagnostico da tecla T vive na ferramenta (tools/), mas e' chamavel sem
# hardware: importar NAO abre Kinect nem camera (o `main()` so' roda em
# `__main__`). Aqui e' so' para o import achar o arquivo.
sys.path.insert(0, os.path.join(RAIZ, "tools"))
import kinect_groundtruth_tool as tool  # noqa: E402

LADO_PX = 44          # reproduz os ~46 px/quadrado medidos na bancada


def renderiza(quadrados_x, quadrados_y, lado_px, margem_px=60):
    """Tabuleiro sintetico: quadrados alternados com margem branca."""
    largura = quadrados_x * lado_px + 2 * margem_px
    altura = quadrados_y * lado_px + 2 * margem_px
    img = np.full((altura, largura), 255, np.uint8)
    for j in range(quadrados_y):
        for i in range(quadrados_x):
            if (i + j) % 2 == 0:
                y0 = margem_px + j * lado_px
                x0 = margem_px + i * lado_px
                img[y0:y0 + lado_px, x0:x0 + lado_px] = 0
    return img


def em_canvas(img):
    """Centra o tabuleiro num canvas 1280x720 (o tamanho real da auxiliar)."""
    canvas = np.full((720, 1280), 255, np.uint8)
    h, w = img.shape
    y0, x0 = (720 - h) // 2, (1280 - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def verdade(margem_px, lado_px, y0, x0):
    """Posicoes EXATAS dos cantos internos do tabuleiro sintetico.

    A QUANTIDADE vem de `sc.CHECKERBOARD_SIZE` (nao de um numero fixo): o
    tabuleiro da bancada ja' mudou uma vez e o teste nao pode ficar validando o
    antigo.

    O primeiro canto interno fica a UM quadrado da borda (por isso `i + 1`):
    e' o que separa "cantos internos" de "quadrados da borda".
    O -0.5 final acerta a convencao do OpenCV, que devolve o CENTRO do pixel:
    o canto de um quadrado e' o canto do pixel, e o centro dele esta' meio
    pixel antes. Sem isso sobra um offset constante de (-0.5, -0.5) px que nao
    e' erro de deteccao e estraga a leitura do numero.
    """
    xs = [margem_px + (i + 1) * lado_px for i in range(sc.CHECKERBOARD_SIZE[0])]
    ys = [margem_px + (j + 1) * lado_px for j in range(sc.CHECKERBOARD_SIZE[1])]
    return np.array([[x0 + x - 0.5, y0 + y - 0.5]
                     for y in ys for x in xs], np.float64)


def erro_medio(estimados, exatos):
    """Erro medio (px) entre o canto estimado e o exato, SEM depender da ordem.

    `findChessboardCorners*` devolve os cantos na ordem canonica DO PADRAO
    (linha a linha), mas pode comecar por qualquer um dos cantos do tabuleiro:
    no tabuleiro sintetico o padrao veio rotacionado 180 graus (o primeiro
    canto e' o inferior-direito). O padrao e' um retangulo, entao so' as rotacoes
    de 180 em 180 preservam a forma -- nao ha' o que testar em 90 graus.

    Ordenar por (y, x) NAO serve e foi um erro cometido aqui: dentro da mesma
    linha o `y` varia ~0.03 px e esse desempate EMBARALHA as colunas -- num
    tabuleiro sintetico perfeito isso aparecia como "erro" de 161 px.
    """
    forma = (sc.CHECKERBOARD_SIZE[1], sc.CHECKERBOARD_SIZE[0], 2)
    a = np.asarray(estimados, np.float64).reshape(forma)
    b = np.asarray(exatos, np.float64).reshape(forma)
    return min(float(np.linalg.norm(np.rot90(a, k) - b, axis=2).mean())
               for k in (0, 2))


casos = 0
QUADRADOS_X = sc.CHECKERBOARD_SIZE[0] + 1        # quadrados = cantos + 1
QUADRADOS_Y = sc.CHECKERBOARD_SIZE[1] + 1
bgr = em_canvas(renderiza(QUADRADOS_X, QUADRADOS_Y, LADO_PX))
sc.DETECTION_WIDTHS = (427, 960, 640)   # o default atual do codigo
sc._ultima_largura_ok = None

# 1) a cascata acha o tabuleiro do usuario (cantos = sc.CHECKERBOARD_SIZE)
cantos = sc.find_corners(bgr)
assert cantos is not None, "a cascata nao achou o tabuleiro"
assert cantos.shape == (sc.CHECKERBOARD_SIZE[0] * sc.CHECKERBOARD_SIZE[1], 2), \
    cantos.shape
casos += 1

# 2) coordenadas na escala do FRAME ORIGINAL, nao da imagem reduzida
assert cantos[:, 0].max() > 640 and cantos[:, 1].max() > 360, (
    f"coordenadas ficaram na escala reduzida: {cantos.max(axis=0)}")
assert cantos[:, 0].min() > 0 and cantos[:, 1].min() > 0
casos += 1

# 3) a largura que funcionou e' tentada PRIMEIRO (custo minimo no proximo frame)
assert sc._ultima_largura_ok in sc.DETECTION_WIDTHS, sc._ultima_largura_ok
preferida = sc._ultima_largura_ok
cantos2 = sc.find_corners(bgr)
assert cantos2 is not None and sc._ultima_largura_ok == preferida
assert np.allclose(cantos, cantos2), "deteccoes seguidas divergiram"
casos += 1

# 4) prepare_gray: largura alvo respeitada + escala de volta coerente
for alvo in (427, 640, 960):
    gray, escala = sc.prepare_gray(bgr, alvo)
    assert gray.shape[1] == alvo, (alvo, gray.shape)
    assert abs(escala - 1280.0 / alvo) < 1e-6, (alvo, escala)
casos += 1

# 5) largura alvo 0 = resolucao cheia (escape hatch)
gray, escala = sc.prepare_gray(bgr, 0)
assert gray.shape[1] == 1280 and escala == 1.0
casos += 1

# 6) refina_cantos: refinar na escala CHEIA reduz o erro do canto que veio de
#    uma deteccao REDUZIDA. Era o risco real de calibrar com a cascata: achar a
#    427 px e multiplicar por 3 deixa ~1,5 px de erro, perto do limite de 2 px.
#    O tabuleiro sintetico tem posicao conhecida, entao o erro e' medivel.
margem = 60
exatos = verdade(margem, LADO_PX,
                 (720 - (QUADRADOS_Y * LADO_PX + 2 * margem)) // 2,
                 (1280 - (QUADRADOS_X * LADO_PX + 2 * margem)) // 2)
sc.DETECTION_WIDTHS = (427,)          # forca o pior caso (fator 3x)
sc._ultima_largura_ok = None
sc.DETECTION_REFINO_CHEIO = False
bruto = sc.find_corners(bgr)
sc.DETECTION_REFINO_CHEIO = True
sc._ultima_largura_ok = None
refinado = sc.find_corners(bgr)
assert bruto is not None and refinado is not None, "cascata nao achou a 427 px"
erro_bruto = erro_medio(bruto, exatos)
erro_refinado = erro_medio(refinado, exatos)
#: Medido: sem refino o erro cresce com a reducao (0.67 px na escala cheia,
#: 1.27 px a 640, 2.20 px a 427 -- este ULTIMO acima do MAX_STEREO_RMS_PX de
#: 2 px). Com refino as tres larguras ficam iguais, em 0.707 px, que e' so' a
#: convencao de meio pixel do OpenCV: o erro real de deteccao e' ~0.
assert erro_refinado < 0.5, (
    f"refino na escala cheia nao ficou sub-pixel: {erro_refinado:.3f} px")
assert erro_refinado < erro_bruto, (
    f"o refinamento nao melhorou o canto: {erro_bruto:.3f} -> "
    f"{erro_refinado:.3f} px")
casos += 1

# 6b) refina_cantos com o tabuleiro NA BORDA. MEDIDO em 30/09/2026 e aberto
#     desde entao: com um canto a menos de meia janela da borda o
#     `cv2.cornerSubPix` do OpenCV 5 LEVANTA e a excecao encerrava a sessao de
#     captura INTEIRA, sem mensagem final. E' o quadro que o arranjo Kinect-only
#     PEDE (levar o tabuleiro as bordas e aos cantos e' o que ensina a DISTORCAO
#     do Kinect), entao os cantos SEM folga ficam com o valor do detector e os
#     outros continuam sendo refinados -- a amostra entra com um canto mais
#     ruidoso, em vez de a sessao morrer.
estreita = bgr[:, :640]
levantou = False
try:
    cv2.cornerSubPix(
        cv2.cvtColor(estreita, cv2.COLOR_BGR2GRAY),
        np.array([[[700.0, 300.0]]], np.float32), (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
except cv2.error:
    levantou = True
assert levantou, ("o cv2 deste venv aceitou canto FORA do quadro: a guarda de "
                  "borda de refina_cantos virou decorativa -- reveja a medicao")
misto = exatos.copy()
misto[0] = [4.0, exatos[0][1]]           # canto forjado na borda esquerda
junto = sc.refina_cantos(bgr, misto)
separado = sc.refina_cantos(bgr, exatos[1:])
assert junto.shape == misto.shape, (junto.shape, misto.shape)
assert np.allclose(junto[0], misto[0]), (junto[0], misto[0])
assert np.allclose(junto[1:], separado), "a borda mudou o refino dos outros"
casos += 1
#    tem de falhar ALTO e claro. Antes o programa ficava mudo para sempre,
#    porque o loop fazia `continue` antes do cv2.imshow.
class KinectMudo:
    def has_new_color_frame(self):
        return False

    def close(self):
        pass


class KinectAusente:
    def __init__(self, *args, **kwargs):
        raise OSError("Kinect not found")


original = sc.PyKinectRuntime
sc.PyKinectRuntime = lambda *a, **k: KinectMudo()
try:
    sc.abre_kinect(espera_s=0.2)
except RuntimeError as exc:
    assert "NAO entregou frame" in str(exc), exc
else:
    raise AssertionError("abre_kinect aceitou um Kinect que nunca entrega frame")
casos += 1

# 8) Kinect que funciona -> devolve o runtime (e NAO fecha a camera)
class KinectOk:
    def __init__(self):
        self.consultas = 0

    def has_new_color_frame(self):
        self.consultas += 1
        return self.consultas >= 2     # so' entrega a partir da 2a consulta

    def close(self):
        raise AssertionError("abre_kinect fechou um Kinect que funcionava")


sc.PyKinectRuntime = lambda *a, **k: KinectOk()
runtime = sc.abre_kinect(espera_s=1.0)
assert runtime.consultas >= 2, runtime.consultas
casos += 1

# 9) erro do pykinect2 (Kinect ausente) -> mensagem acionavel, nao traceback cru
sc.PyKinectRuntime = KinectAusente
try:
    sc.abre_kinect(espera_s=0.2)
except RuntimeError as exc:
    assert "USB 3.0" in str(exc), exc
else:
    raise AssertionError("abre_kinect nao reclamou do Kinect ausente")
finally:
    sc.PyKinectRuntime = original
casos += 1

# 10) DISPLAY_SCALE e' reducao SO' da JANELA: o helper tem de reduzir pelo
# fator e nada mais (a deteccao acima ja' provou que ela roda na resolucao
# cheia, ANTES desta reducao -- e' a ordem importa: reduzir antes do detector
# seria cometer o erro que DETECTION_WIDTHS documenta).
assert 0.0 < sc.DISPLAY_SCALE <= 1.0, sc.DISPLAY_SCALE
assert sc.reduzir_janela(None) is None
pequeno = sc.reduzir_janela(np.zeros((1080, 1920, 3), np.uint8))
assert pequeno.shape[:2] == (int(1080 * sc.DISPLAY_SCALE),
                             int(1920 * sc.DISPLAY_SCALE)), pequeno.shape
assert pequeno.shape[2] == 3, pequeno.shape
casos += 1

# 11) o resumo do diagnostico (tecla T) indexa centro e bordas pela GEOMETRIA
#     do tabuleiro, nao por indices fixos. O bug medido em 28/09/2026: com a
#     placa de entao (5x3 = 15 cantos) o resumo fazia `errors[24]` e morria com
#     IndexError DEPOIS de acertar o tabuleiro -- 24 era o "centro" da placa
#     ANTIGA, de 11x7 = 77 cantos. Este bloco trava a regra para qualquer placa.
exatos = verdade(60, LADO_PX, 0, 0)
indice_centro, bordas, raios = gt.grupos_centro_borda(exatos)
assert raios.size == sc.CHECKERBOARD_SIZE[0] * sc.CHECKERBOARD_SIZE[1], raios.size
assert raios[indice_centro] == raios.min(), indice_centro
# "bordas" = os cantos mais distantes do centro pela DEFINICAO (>= 75% do raio
# maximo): na placa 5x3 davam 6 dos 15 (as 4 quinas e os 2 extremos do meio das
# laterais). O teste trava a DEFINICAO, nao um numero que muda com a placa:
# 4 <= bordas < metade dos cantos.
assert bool(np.all(raios[bordas] >= 0.75 * raios.max())), raios
assert 4 <= int(bordas.sum()) <= len(exatos) // 2, int(bordas.sum())
assert indice_centro not in np.flatnonzero(bordas)   # centro nao e' borda
# O indice FIXO do bug antigo (`errors[24]`) tem de continuar sendo um erro nesta
# placa. Duas formas, ambas erradas do mesmo jeito: se 24 estiver FORA da grade,
# o resumo morre com IndexError (foi o caso da placa de 5x3 = 15 cantos); se
# estiver DENTRO, ele aponta outro canto que nao o centro -- o resumo acerta por
# acidente e mostra o numero errado. Trocar a placa nao pode deixar este bloco
# mudo (foi o que aconteceu em 01/10/2026 com a placa de 7x5 = 35 cantos).
INDICE_FIXO_DO_BUG = 24
if INDICE_FIXO_DO_BUG >= len(exatos):
    print("  indice fixo do bug (24) fica FORA da grade atual (%d cantos) -> "
          "IndexError, como em 28/09/2026" % len(exatos))
else:
    assert INDICE_FIXO_DO_BUG != int(indice_centro), (
        "o indice fixo do bug (24) E' o centro da placa atual (%d cantos): "
        "escolha outro exemplo para o resumo" % len(exatos))
    print("  indice fixo do bug (24) esta dentro da grade atual (%d cantos) e "
          "NAO e' o centro: erro silencioso, tambem coberto pela regra "
          "geometrica" % len(exatos))
for total in (len(exatos), 21, 77):                  # nenhuma placa estoura
    erros = np.zeros(total)
    _, mascara, _ = gt.grupos_centro_borda(np.random.default_rng(3).random(
        (total, 2)) * 100)
    assert erros[mascara].size >= 1, total
casos += 1

# 12) ordem_canonica_do_tabuleiro: e' a MESMA regra que a calibracao usa para
#     casar os dois tabuleiros (`stereo_calibration.canonical_order`), e as duas
#     implementacoes tem de concordar -- sao duas copias da mesma regra (a
#     biblioteca nao pode importar `stereo_calibration`: ele importa a
#     biblioteca) e divergir aqui da' centenas de px de erro com a calibracao
#     perfeita.
grade = exatos.reshape(sc.CHECKERBOARD_SIZE[1], sc.CHECKERBOARD_SIZE[0], 2)
for nome, indices in sc.symmetry_indices().items():
    permutado = grade.reshape(-1, 2)[indices]
    canonico, achado = gt.ordem_canonica_do_tabuleiro(permutado)
    assert canonico.shape == permutado.shape, (nome, canonico.shape)
    assert sorted(map(tuple, canonico)) == sorted(map(tuple, permutado)), nome
    nome_sc, pontos_sc = sc.canonical_order(permutado)
    assert nome_sc == achado, (nome, nome_sc, achado)
    assert np.allclose(pontos_sc, canonico), (nome, nome_sc)
    casos += 1

# 12b) tabuleiros ALEATORIOS (quadrado de tamanho e perspectiva variados, com
#      simetria aleatoria aplicada): as duas implementacoes tem de dar a MESMA
#      ordem -- e' o que trava a duplicacao da regra acima.
rng = np.random.default_rng(5)
for _ in range(40):
    lado = float(rng.uniform(18.0, 60.0))
    base = np.array([[320.0 + lado * i + rng.uniform(-1, 1),
                      240.0 + lado * j + rng.uniform(-1, 1)]
                     for j in range(sc.CHECKERBOARD_SIZE[1])
                     for i in range(sc.CHECKERBOARD_SIZE[0])])
    # perspectiva leve + simetria aleatoria (o detector tambem parte de um canto
    # qualquer quando o tabuleiro esta' torto na imagem)
    H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [640, 0], [640, 480], [0, 480]]),
        np.float32([[rng.uniform(0, 40), rng.uniform(0, 40)],
                    [640 + rng.uniform(-30, 30), rng.uniform(0, 30)],
                    [640 + rng.uniform(-30, 30), 480 + rng.uniform(-30, 0)],
                    [rng.uniform(0, 40), 480 + rng.uniform(-30, 30)]]))
    torto = cv2.perspectiveTransform(
        base.astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
    nome_sim = list(sc.symmetry_indices())[int(rng.integers(0, 4))]
    entrada = torto.reshape(-1, 2)[sc.symmetry_indices()[nome_sim]]
    pontos_gt, _nome_gt = gt.ordem_canonica_do_tabuleiro(entrada)
    nome_sc, pontos_sc = sc.canonical_order(entrada)
    assert nome_sc == _nome_gt, (nome_sc, _nome_gt)
    assert np.allclose(pontos_sc, pontos_gt), np.abs(pontos_sc - pontos_gt).max()
casos += 1

# 12c) tamanho de tabuleiro diferente e' erro claro (calibracao de outra placa)
try:
    gt.ordem_canonica_do_tabuleiro(exatos[:6])
except ValueError as exc:
    assert "placa" in str(exc), exc
else:
    raise AssertionError("tamanho de tabuleiro diferente passou calado")
casos += 1

# 13) CameraAlignment.add_sample guarda as DUAS listas na ORDEM CANONICA (a
#     homografia legada vai para o disco: ordem crua de um lado e canonica do
#     outro = calibracao errada). A regra da calibracao pode escolher uma
#     simetria diferente da identidade (medido: para um tabuleiro deitado ela
#     prefere 'hflip'), entao o teste compara com a ordem canonica, nao com a
#     ordem crua -- e exige que a canonicalizacao tenha mesmo acontecido.
class AlinhamentoLegado(gt.CameraAlignment):
    def __init__(self, mapa):
        self.auxiliary_points = []
        self.kinect_points = []
        self.mapa = mapa

    def find_board(self, frame):
        return np.asarray(self.mapa[frame], np.float64)


cru_aux = grade.reshape(-1, 2)[::-1]
esperado_kin, _ = gt.ordem_canonica_do_tabuleiro(exatos)
esperado_aux, _ = gt.ordem_canonica_do_tabuleiro(cru_aux)
assert not np.allclose(esperado_kin, exatos), \
    "escolha um tabuleiro em que a ordem canonica NAO seja a identidade"
falso = AlinhamentoLegado({"aux": cru_aux, "kin": exatos})
assert falso.add_sample("aux", "kin") is True
assert np.allclose(falso.kinect_points[0], esperado_kin)
assert np.allclose(falso.auxiliary_points[0], esperado_aux)
casos += 1

# 14) diagnostico da tecla T de ponta a ponta, SEM hardware: a funcao do tool
#     imprime o resumo e devolve os numeros. Este bloco reproduz a falha medida
#     em 28/09/2026 -- o resumo indexava `errors[24]` (o "centro" da placa ANTIGA
#     de 11x7 = 77 cantos) e a tecla T morria com IndexError DEPOIS de acertar o
#     tabuleiro e medir a profundidade, com a placa atual de 5x3 = 15 cantos.
class TrackerFalso:
    """Depth do Kinect o suficiente para o diagnostico (sem hardware)."""

    def __init__(self, profundidade_mm=1200.0, faltando=()):
        self.last_depth = np.full((424, 512), 1200, np.uint16)
        self.depth_frame_index = 40
        self.profundidade_mm = profundidade_mm
        self.faltando = set(faltando)

    def depth_stats(self):
        return (192320, 0, 7999, 0.89)

    def _color_to_depth_pixel(self, x, y):
        return (int(x) // 4, int(y) // 4)

    def _median_depth(self, dx, dy, radius=None):
        return None if (dx, dy) in self.faltando else self.profundidade_mm


class AlinhamentoFalso:
    """Calibracao stereo falsa, fisicamente coerente.

    `ordem[j]` diz qual canto do Kinect corresponde ao j-esimo canto que a
    AUXILIAR devolveu -- e' o que a projecao precisa saber (na vida real quem
    resolve isso e' a geometria da calibracao; aqui, o teste).
    """

    stereo_usable = True

    def __init__(self, kinect_cantos, aux_cantos, ordem, ruido=0.4,
                 quadro_aux=None):
        self.kinect = np.asarray(kinect_cantos, np.float64).reshape(-1, 2)
        self.aux = np.asarray(aux_cantos, np.float64).reshape(-1, 2)
        self.ordem = np.asarray(ordem, int)
        self.ruido = ruido
        self.quadro_aux = quadro_aux

    def find_board(self, frame):
        # O quadro da auxiliar chega como IMAGEM (e' o que a ferramenta passa):
        # identificado por identidade, nao por texto.
        return self.aux if frame is self.quadro_aux else self.kinect

    def auxiliary_to_kinect_stereo(self, pontos, profundidades):
        pontos = np.asarray(pontos, np.float64).reshape(-1, 2)
        indices = [int(np.argmin(np.linalg.norm(self.aux - ponto, axis=1)))
                   for ponto in pontos]
        rng = np.random.default_rng(11)
        return self.kinect[self.ordem[indices]] + rng.normal(
            0.0, self.ruido, (len(indices), 2))


colunas, linhas = sc.CHECKERBOARD_SIZE
grade_kinect = np.array([[820.0 + 40 * i, 500.0 + 40 * j]
                         for j in range(linhas) for i in range(colunas)])
recuada = np.rot90(grade_kinect.reshape(linhas, colunas, 2), 2).reshape(-1, 2)
COR = np.zeros((4, 4, 4), np.uint8)
COR_LIMPA = np.zeros((4, 4, 3), np.uint8)
QUADRO_AUX = np.ones((4, 4, 3), np.uint8)      # a "imagem" que vem da auxiliar

saida = io.StringIO()
with contextlib.redirect_stdout(saida):
    resumo = tool.diagnostico_tabuleiro(
        COR, COR_LIMPA, QUADRO_AUX, TrackerFalso(),
        AlinhamentoFalso(grade_kinect, recuada, np.arange(len(recuada))[::-1],
                         quadro_aux=QUADRO_AUX),
        aux_ok=True, aux_index=0)
texto = saida.getvalue()
assert resumo is not None, texto
assert resumo["cantos"] == len(grade_kinect) == colunas * linhas, resumo
assert resumo["cantos_com_profundidade"] == resumo["cantos"], resumo
# A auxiliar devolveu os cantos na meia volta: o diagnostico testa as 4
# simetrias da grade e fica com a de menor erro (a correspondencia fisica).
assert resumo["pareamento_cantos"] == "rot180", resumo
assert resumo["media_px"] < 2.0, resumo          # pareamento certo = erro baixo
assert resumo["max_px"] < 5.0, resumo
assert resumo["bordas_cantos"] >= 4, resumo
assert resumo["pareamento_px"]["id"] > 100.0, resumo   # pareamento errado: longe
assert resumo["homografia_nova_px"] < 1.0, resumo
assert resumo["correspondencia_ok"] is True, resumo
assert "pareamento dos cantos" in texto, texto
assert "centro" in texto and "bordas" in texto, texto
# A homografia NOVA (ajustada na hora, com os mesmos cantos) prova que a
# deteccao/correspondencia esta' boa: o erro grande que aparecer depois e' da
# CALIBRACAO do arquivo, nao do detector.
assert "deteccao/correspondencia OK" in texto, texto
assert "RECALIBRE" not in texto, texto           # erro baixo: nada de alarme
casos += 1

# 15) quando as duas cameras concordam na ordem, nada de aviso e o erro segue
#     baixo (a canonicalizacao nao pode "consertar" o que ja' estava certo).
saida = io.StringIO()
with contextlib.redirect_stdout(saida):
    resumo = tool.diagnostico_tabuleiro(
        COR, COR_LIMPA, QUADRO_AUX, TrackerFalso(),
        AlinhamentoFalso(grade_kinect, grade_kinect,
                         np.arange(len(grade_kinect)), quadro_aux=QUADRO_AUX),
        aux_ok=True, aux_index=0)
assert resumo is not None
assert resumo["pareamento_cantos"] == "id", resumo
assert resumo["media_px"] < 2.0 and resumo["max_px"] < 5.0, resumo
assert "pareamento dos cantos id" in saida.getvalue()
casos += 1

# 15b) calibracao de OUTRA montagem: erro grande tem de virar aviso acionavel
#      (o RMS do arquivo nao detecta isso -- medido 491 px nesta bancada com a
#      calibracao feita na maquina antiga).
class AlinhamentoDeslocado(AlinhamentoFalso):
    def auxiliary_to_kinect_stereo(self, pontos, profundidades):
        return super().auxiliary_to_kinect_stereo(pontos, profundidades) \
            + np.array([-485.0, 77.0])


saida = io.StringIO()
with contextlib.redirect_stdout(saida):
    resumo = tool.diagnostico_tabuleiro(
        COR, COR_LIMPA, QUADRO_AUX, TrackerFalso(),
        AlinhamentoDeslocado(grade_kinect, grade_kinect,
                             np.arange(len(grade_kinect)),
                             quadro_aux=QUADRO_AUX),
        aux_ok=True, aux_index=0)
texto = saida.getvalue()
assert resumo is not None and resumo["mediana_px"] > 400, resumo
assert abs(resumo["dx_px"] + 485.0) < 1.0, resumo
# correspondencia BOA + erro grande = calibracao de outra bancada (o conserto e'
# recalibrar, nao mexer na captura)
assert resumo["correspondencia_ok"] is True, resumo
assert resumo["homografia_nova_px"] < 1.0, resumo
assert "RECALIBRE" in texto and "stereo_calibration.py" in texto, texto
assert "diagnostico_cameras.py" in texto, texto      # ...e onde conferir o mapa
casos += 1

# 17) correspondencia RUIM: a deteccao da AUXILIAR nao descreve o MESMO tabuleiro
#     do Kinect (borrao/reflexo, ou o tabuleiro mudou entre as duas capturas --
#     elas nao sao sincronizadas). Aqui o conserto e' OUTRO: mandar recalibrar
#     seria trabalho perdido, porque o erro grande reapareceria no arquivo novo.
#     A perturbacao vai na DETECCAO (nao na projecao): a homografia nova e' uma
#     medida da deteccao/correspondencia, independente do stereo carregado.
class TabuleiroQueMudou(AlinhamentoFalso):
    def find_board(self, frame):
        if frame is not self.quadro_aux:
            return self.kinect
        i = np.arange(len(self.aux), dtype=np.float64)
        return self.aux + np.stack([50.0 + 30.0 * np.sin(1.7 * i),
                                    20.0 + 30.0 * np.cos(2.3 * i)], axis=1)


saida = io.StringIO()
with contextlib.redirect_stdout(saida):
    resumo = tool.diagnostico_tabuleiro(
        COR, COR_LIMPA, QUADRO_AUX, TrackerFalso(),
        TabuleiroQueMudou(grade_kinect, grade_kinect,
                          np.arange(len(grade_kinect)),
                          quadro_aux=QUADRO_AUX),
        aux_ok=True, aux_index=0)
texto = saida.getvalue()
assert resumo is not None, texto
assert resumo["mediana_px"] > tool.LIMITE_PX_CALIBRACAO_SUSPEITA, resumo
assert resumo["correspondencia_ok"] is False, resumo
assert resumo["homografia_nova_px"] > tool.LIMITE_PX_CORRESPONDENCIA, resumo
assert "CORRESPONDENCIA esta' ruim" in texto, texto
assert "antes de recalibrar" in texto, texto
assert "RECALIBRE nesta maquina" not in texto, texto
casos += 1

# 18) NPZ DE OUTRA CAMERA: quem carrega diz de quem o arquivo e'. So' os npz
#     gravados a partir de 29/09/2026 tem esse campo -- e ele e' a unica defesa
#     contra o indice do OpenCV trocar de camera (posicao no USB) e o programa
#     injetar uma geometria de outro par sem sintoma.
class NomesFalsos:
    MAPA = {0: "SIGMA-W780M", 1: "Trust USB Camera"}

    @staticmethod
    def nome_da_camera(indice):
        return NomesFalsos.MAPA.get(int(indice))


class AlinhamentoOutraCamera(AlinhamentoFalso):
    def camera_name_conflict(self, aux_index=None):
        return "Trust USB Camera"


nomes_originais = gt.camera_names
gt.camera_names = NomesFalsos
try:
    saida = io.StringIO()
    with contextlib.redirect_stdout(saida):
        resumo = tool.diagnostico_tabuleiro(
            COR, COR_LIMPA, QUADRO_AUX, TrackerFalso(),
            AlinhamentoOutraCamera(grade_kinect, grade_kinect,
                                   np.arange(len(grade_kinect)),
                                   quadro_aux=QUADRO_AUX),
            aux_ok=True, aux_index=0)
finally:
    gt.camera_names = nomes_originais
texto = saida.getvalue()
assert resumo is not None, texto
assert "GRAVADO PARA OUTRA CAMERA" in texto, texto
assert "Trust USB Camera" in texto and "SIGMA-W780M" in texto, texto
assert "python stereo_calibration.py" in texto, texto
casos += 1

# 16) a mensagem diz QUAL camera nao viu (a antiga dizia "nao visto nas duas
#     cameras" quando UMA falhava -- foi o que mandou procurar problema no
#     tabuleiro com o tabuleiro na mesa) e o resumo nao sai.
class SemTabuleiro(AlinhamentoFalso):
    def find_board(self, frame):
        return None if frame is self.quadro_aux else self.kinect


saida = io.StringIO()
with contextlib.redirect_stdout(saida):
    resumo = tool.diagnostico_tabuleiro(
        COR, COR_LIMPA, QUADRO_AUX, TrackerFalso(),
        SemTabuleiro(grade_kinect, recuada, np.arange(len(recuada)),
                     quadro_aux=QUADRO_AUX),
        aux_ok=True, aux_index=1)
texto = saida.getvalue()
assert resumo is None, resumo
assert "Kinect=OK" in texto and "auxiliar (indice 1)=nao" in texto, texto
casos += 1

print(f"DETECCAO_CASCATA_OK: {casos}/24 (cascata acha "
      f"{sc.CHECKERBOARD_SIZE[0]}x{sc.CHECKERBOARD_SIZE[1]}, coordenadas "
      f"reescalonadas para o frame original, largura preferida memorizada, "
      f"refino na escala cheia reduz o erro {erro_bruto:.2f}->"
      f"{erro_refinado:.2f} px, abre_kinect falha alto e claro, janela "
      f"reduzida por DISPLAY_SCALE={sc.DISPLAY_SCALE:g} sem mexer na deteccao, "
      f"centro/bordas do resumo pela GEOMETRIA (nao pelos indices da placa "
      f"antiga), ordem canonica do tabuleiro identica a da CALIBRACAO em "
      f"tabuleiros aleatorios, pareamento dos cantos pelo MENOR erro, "
      f"correspondencia x calibracao separadas pela homografia NOVA -- com "
      f"conserto distinto para cada uma -- e npz de outra camera denunciado)")
sys.exit(0)
