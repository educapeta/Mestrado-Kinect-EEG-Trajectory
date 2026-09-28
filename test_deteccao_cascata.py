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
"""
import sys

import cv2
import numpy as np

import stereo_calibration as sc

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

# 7) abre_kinect: Kinect que abre mas NUNCA entrega frame (o caso da bancada)
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

print(f"DETECCAO_CASCATA_OK: {casos}/10 (cascata acha "
      f"{sc.CHECKERBOARD_SIZE[0]}x{sc.CHECKERBOARD_SIZE[1]}, coordenadas "
      f"reescalonadas para o frame original, largura preferida memorizada, "
      f"refino na escala cheia reduz o erro {erro_bruto:.2f}->"
      f"{erro_refinado:.2f} px, abre_kinect falha alto e claro, janela "
      f"reduzida por DISPLAY_SCALE={sc.DISPLAY_SCALE:g} sem mexer na deteccao)")
sys.exit(0)
