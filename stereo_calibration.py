"""Calibracao stereo Kinect RGB + camera auxiliar.

Tabuleiro (placa atual, impressa em 23/09/2026): 6 x 4 QUADRADOS de 57 mm ->
5 x 3 CANTOS INTERNOS. Os valores ficam nas constantes do arquivo
(CHECKERBOARD_COLS/ROWS e SQUARE_SIZE_M) e precisam casar com
kinect_imu_groundtruth.py: o npz grava placa e quadrado, e um npz de placa
diferente e' RECUSADO com o motivo por extenso (CameraAlignment.stereo_status).
Trocar de placa = mudar as constantes nos DOIS arquivos (ou passar
--checkerboard CxR --quadrado MM aqui) e CALIBRAR DE NOVO.

Regra geral do cv2: ele conta CANTOS INTERNOS = (quadrados - 1) por eixo; se a
placa tiver quadrados cortados nas bordas, o detector ENXERGA os cantos deles
(contar as colunas/linhas cortadas tambem).
As capturas sao automaticas (2 s com o tabuleiro estavel nas duas
cameras); ESC encerra sem salvar.

O Kinect RGB e' OBRIGATORIO: o par estereo e' (Kinect RGB, webcam auxiliar), e
nao ha' como calibrar so' com a auxiliar. Se o Kinect nao entregar frame, o
programa agora falha ALTO e com instrucao. Antes ele ficava MUDO: o loop fazia
`continue` antes do `cv2.imshow` e nenhuma janela aparecia -- o sintoma era
"rodei e nao apareceu nada", sem pista nenhuma de que faltava a camera.

Por que esta versao existe (o RMS stereo ficava em 30-70 px com RMS
individual de ~0.5-1.0 px):

1. ORDEM DOS CANTOS. findChessboardCorners/SB devolve os N cantos em
   ordem canonica de IMAGEM (o canto mais proximo do canto superior
   esquerdo da imagem primeiro). Como cada camera ve o tabuleiro de um
   angulo diferente, o "primeiro canto" pode corresponder a cantos
   fisicos diferentes nas duas cameras, e a correspondencia muda de
   pose para pose (quando o tabuleiro gira no plano da imagem). Uma
   troca de 180 graus e uma rotacao rigida do tabuleiro: ela e
   absorvida pela pose individual de cada view e por isso o RMS
   individual continua excelente, mas a pose RELATIVA entre as cameras
   fica inconsistente e o RMS stereo explode. Aqui cada captura e
   reordenada de forma canonica e o solver testa as 4 simetrias da
   grade por pose: corrige as poses trocadas e descarta as poses ruins
   (diagnostico por residuo stereo por pose).
2. INTRINSECAS CONGELADAS. CALIB_FIX_INTRINSIC congela fx/fy/cx/cy/
   distorcao obtidos separadamente. Se as poses tiverem pouca variacao
   de distancia/inclinacao (tabuleiro sempre de frente e na mesma
   distancia), essas intrinsecas ficam imprecisas e o erro so aparece
   no stereo. O solver final tenta tambem CALIB_USE_INTRINSIC_GUESS
   (refinamento conjunto de intrinsecas + extrinsecas) e salva o
   resultado com menor RMS.
3. QUALIDADE DAS AMOSTRAS. Capturas com tabuleiro pequeno na imagem
   (foco/escala ruins) ou com erro planar do par alto (movimento,
   deteccao ruim) sao rejeitadas antes de entrar na solucao.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime

# Cameras auxiliares que formam o par estereo com o Kinect (indices do OpenCV).
# ATENCAO: o indice do OpenCV NAO e' estavel -- e' a POSICAO da camera na
# enumeracao do DirectShow, e ela muda quando uma camera entra/sai do USB.
# MEDIDO nesta maquina (23/09/2026), com as DUAS externas ligadas:
#   0 = webcam externa virada para a bancada/mesa (vista ampla)
#   1 = webcam DO LAPTOP (Integrated Camera) -- sem baseline com o Kinect
#   2 = webcam externa montada por cima da bancada (ve' as duas maos)
# No MESMO dia, quando as externas cairam fora do USB, o indice 0 passou a ser a
# webcam do LAPTOP: o par abaixo e' so' um PONTO DE PARTIDA. Quem manda e' o NOME
# (camera_names.py) -- confira com tools/diagnostico_cameras.py --salvar, que
# imprime o nome do Windows de cada indice.
# Foi por assumir 1="usb" que a calibracao de `aux1` foi feita na webcam do
# laptop: o par util hoje e' (2, 0), as DUAS webcams externas.
# Precisa casar com AUX_CAMERA_INDICES de kinect_imu_groundtruth.py.
AUX_CAMERA_INDICES_PADRAO = (2, 0)
AUX_CAMERA_NAMES = {
    0: "externa-ampla",
    1: "laptop",
    2: "externa-bancada",
}
# Indice em uso agora (trocado por calibrate_one a cada camera).
AUX_CAMERA_INDEX = AUX_CAMERA_INDICES_PADRAO[0]
# Must match the resolution used by kinect_imu_groundtruth.py.
AUX_REQUEST_SIZE = (1280, 720)  # (width, height)
#: Backends tentados ao abrir a webcam auxiliar, NA ORDEM. O DSHOW costuma
#: abrir em ~1 s (contra os 20-40 s do MSMF) e enfileira MENOS frames -- as
#: duas coisas aparecem como "video lento/atrasado" quando erradas.
AUX_BACKENDS = ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"))
#: Frames descartados logo apos abrir: o driver entrega imagem instavel/escura
#: enquanto a exposicao automatica estabiliza.
AUX_DESCARTE = 5
# The raw auxiliary stream arrives mirrored (virtual camera / phone app);
# flipping restores the physical, unmirrored coordinates. Keep the same
# value as AUXILIARY_MIRROR_HORIZONTAL in kinect_imu_groundtruth.py.
AUXILIARY_MIRROR_HORIZONTAL = True
AUXILIARY_DISPLAY_MIRROR = False
#: Fator de reducao das JANELAS de video (0.5 = metade). Manter IGUAL a
#: WINDOW_SCALE de kinect_imu_groundtruth.py: com o Kinect em 1920x1080 e as
#: duas webcams em 1280x720, so' cabe tudo na tela se cada janela ocupar ~1/4
#: dela -- no tamanho real o experimentador nao consegue ver as duas auxiliares
#: ao mesmo tempo. A reducao e' SO' de EXIBICAO: a deteccao de cantos, a
#: captura de amostras e a calibracao seguem na resolucao CHEIA (reduzir antes
#: do detector seria cometer o erro que DETECTION_WIDTHS documenta).
DISPLAY_SCALE = 0.5
#: Quanto esperar o PRIMEIRO frame do Kinect RGB ao abrir (s). O Kinect leva
#: alguns segundos para comecar a transmitir; se nao vier nada nesse tempo, o
#: programa para com mensagem clara em vez de ficar mudo para sempre.
ESPERA_KINECT_S = 20.0
# Tabuleiro: PLACA ATUAL (impressa 23/09/2026) = 6 x 4 QUADRADOS de 57 mm ->
# 5 x 3 CANTOS internos (a regra do cv2 e' sempre quadrados - 1 por lado).
# Precisa casar com CHECKERBOARD_SIZE/SQUARE_SIZE_M em
# kinect_imu_groundtruth.py: o stereo salvo guarda estes dois valores e a
# biblioteca RECUSA (stereo_ready=False, com o motivo dito em
# CameraAlignment.stereo_status) qualquer npz de placa/quadrado diferente.
# Historico: 4x3/52 mm (subgrade, geometria 2x grande) -> 12x8 quadrados/26 mm
# (11x7 cantos, a placa antiga) -> 6x4/57 mm (a atual).
# Sobrescrevivel na linha de comando com --checkerboard CxR e --quadrado MM.
CHECKERBOARD_COLS = 5
CHECKERBOARD_ROWS = 3
CHECKERBOARD_SIZE = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)
SQUARE_SIZE_M = 0.057
SAMPLES_REQUIRED = 15
SOLVE_EVERY_EXTRA = 3          # depois das 15, resolve a cada +3 amostras
MAX_STEREO_RMS_PX = 2.0
STABLE_CAPTURE_SECONDS = 2.0
MIN_BOARD_SPAN_FRACTION = 0.08  # tabuleiro >= 8% da diagonal da imagem (duas cams)
MAX_PAIR_ERROR_PX = 3.0         # erro planar homografia aux -> kinect
MIN_POSE_SEPARATION = 0.30      # amostra nova a >= 30% do span de TODAS as outras
MAX_POSE_RMS_PX = 4.0           # residuo stereo por pose (limite duro)
OUTLIER_MEDIAN_FACTOR = 2.5     # pior pose vs mediana
MIN_ACTIVE_POSES = 8
RANSAC_SEED_TRIALS = 40         # mini-solves para achar um subconjunto limpo
RANSAC_SEED_SIZE = 5
ASSIGNMENT_ROUNDS = 3
MAX_TOTAL_SAMPLES = 60
DETECTION_INTERVAL_S = 0.1     # detecta cantos a ~10 Hz e reusa entre frames
FALLBACK_EVERY = 1             # classico em TODO ciclo: com N>1 a deteccao
                               # OSCILA (SB nao acha o frame do Kinect; sem
                               # fallback ele falha, a estabilidade de 2 s
                               # zera e NENHUMA amostra e' coletada -- medido
                               # ao vivo com batimento de diagnostico)
# --- Robustez e custo do detector ------------------------------------------
# Por que existem (dois sintomas classicos na bancada):
#   (a) VIDEO TRAVADO: findChessboardCornersSB custa ~O(px^2) e com
#       CALIB_CB_EXHAUSTIVE|CALIB_CB_ACCURACY em 1920x1080 ele sozinho passa de
#       100 ms/frame -- a interface engasga e o detector reusa cantos velhos.
#   (b) TABULEIRO "PISCANDO" com ele visivel: impressao de baixo contraste,
#       papel brilhante ou luz desigual deixam o detector na fronteira, e ele
#       oscila entre achar e nao achar.
#: CLAHE no cinza antes de detectar (0 desliga). E' o que mais ajuda em (b).
DETECTION_CLAHE_CLIP = 2.0
DETECTION_CLAHE_GRID = 8
#: Larguras-ALVO da deteccao, tentadas EM CASCATA (a primeira que achar vence).
#: 0 = resolucao cheia.
#: MEDIDO na bancada, com TODAS as configs rodando na MESMA imagem:
#:   rodada 1 (tabuleiro a 36% da largura, quadrado de 45 px em 1280):
#:     ACERTA 640 e 427 px | FALHA 960 px e a resolucao cheia;
#:   rodada 2 (tabuleiro a 31%, quadrado de 39 px em 1280):
#:     ACERTA 960 e 427 px | FALHA 640 px e a resolucao cheia.
#: As duas rodadas juntas NAO obedecem a regra nenhuma de tamanho: 640 acerta
#: numa e falha na outra, e 960 faz o mesmo. Ou seja, a hipotese anterior
#: ("quadrado grande demais atrapalha o SB") esta' REFUTADA -- o detector esta'
#: na FRONTEIRA e o resultado sai por sorte. Quem decide e' a qualidade da
#: imagem (ruido de ISO e borrao de movimento com pouca luz), nao a largura.
#: Consequencia pratica: NAO tente consertar "piscando" a largura; melhore a luz.
#: A cascata cobre as tres larguras que ja' acertaram alguma vez, com a MAIS
#: confiavel primeiro (427 acertou nas DUAS rodadas), e a memoizacao
#: (_ultima_largura_ok) evita pagar a cascata inteira a cada frame.
#: Custo medido: 427 px = 13 ms | 640 px = 59 ms | 960 px = 70 ms | cheia = 108 ms.
DETECTION_WIDTHS = (427, 960, 640)
#: Largura que ACHOU o tabuleiro na deteccao anterior: tentada PRIMEIRO, porque
#: entre frames o tabuleiro muda pouco. Sem isso a cascata inteira seria paga a
#: cada frame, justamente quando ela e' desnecessaria.
_ultima_largura_ok = None
#: Flags do SB. EXHAUSTIVE procura com mais afinco (tabuleiro pequeno/inclinado)
#: e e' barato comparado ao ACCURACY, que faz refinamento extra e era o que
#: mais pesava no video travado.
DETECTION_SB_FLAGS = cv2.CALIB_CB_EXHAUSTIVE
#: Refinar os cantos com cornerSubPix na resolucao CHEIA (nao na reduzida).
#: Por que: a cascata pode achar o tabuleiro a 427 px e, ao voltar para o frame
#: original, o fator de 3x transforma o erro de meio pixel da deteccao em 1,5 px
#: -- perto do MAX_STEREO_RMS_PX de 2 px. Refinando no frame original o erro
#: volta a ser sub-pixel. Efeito colateral bom: TODAS as amostras passam a ter
#: a mesma precisao, mesmo as que vieram de larguras diferentes (antes a amostra
#: detectada a 427 px entrava com 3x mais ruido de pixel que a de 960 px, e a
#: calibracao misturava as duas).
DETECTION_REFINO_CHEIO = True
# k1, k2, k3, p1, p2 livres (k3 congelado degradava a precisao nas bordas
# da imagem, onde a distorcao do Kinect e maior).
CALIB_FLAGS = 0
CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
STEREO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
OUTPUT_FILE = Path(__file__).with_name("stereo_calibration.npz")
# Regenerado junto com a calibracao stereo para manter os dois arquivos
# consistentes (a homografia e derivada dos pontos do tabuleiro da
# propria solucao; o refinamento afim recomeca da identidade).
CAMERA_ALIGNMENT_FILE = Path(__file__).with_name("camera_alignment.npz")
CAMERA_ALIGNMENT_VERSION = 7


def output_file(aux_index):
    """Arquivo stereo por camera auxiliar (legado = indice 0)."""
    if int(aux_index) == 0:
        return OUTPUT_FILE
    return Path(__file__).with_name(
        f"stereo_calibration_aux{int(aux_index)}.npz")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--largura-deteccao", type=int, nargs="+",
                        default=list(DETECTION_WIDTHS), metavar="N",
                        help="Largura(s)-ALVO da deteccao, tentadas EM CASCATA "
                             "(0 = resolucao cheia). Medido com o tabuleiro a "
                             "31-36%% da imagem: 427 px acertou nas DUAS "
                             "rodadas, e 640 e 960 alternaram entre acertar e "
                             "falhar -- o que decide e' a qualidade da imagem "
                             "(luz/ruido), nao a largura. Custo: 427 px = 13 "
                             "ms, 960 px = 70 ms, resolucao cheia = 108 ms.")
    parser.add_argument("--clahe", type=float, default=DETECTION_CLAHE_CLIP,
                        help="Forca do CLAHE no cinza antes de detectar. 0 "
                             "desliga; 2-4 ajuda muito com impressao fraca, "
                             "papel brilhante ou luz desigual.")
    parser.add_argument("--rapido", action="store_true",
                        help="Tira o CALIB_CB_EXHAUSTIVE do detector SB: mais "
                             "rapido, porem menos robusto com o tabuleiro "
                             "pequeno ou muito inclinado.")
    parser.add_argument("--checkerboard", type=str, default=None, metavar="CxR",
                        help="Padrao do tabuleiro como 'CANTOSxCANTOS' (5x3 = "
                             "6x4 quadrados). Padrao: usa as constantes do "
                             "arquivo (%dx%d)." % (CHECKERBOARD_COLS,
                                                   CHECKERBOARD_ROWS))
    parser.add_argument("--quadrado", type=float, default=None, metavar="MM",
                        help="Tamanho do quadrado do tabuleiro em mm. "
                             "Padrao: usa a constante do arquivo "
                             "(%g mm)." % (SQUARE_SIZE_M * 1000))
    parser.add_argument("--aux-index", type=int, nargs="+",
                        default=list(AUX_CAMERA_INDICES_PADRAO), metavar="N",
                        help="Indice(s) da(s) webcam(s) auxiliar(es) a "
                             "calibrar, na ordem. Aceita varias de uma vez "
                             "(ex.: --aux-index 1 2). Salva em "
                             "stereo_calibration_aux{N}.npz (0 usa o arquivo "
                             "legado stereo_calibration.npz).")
    return parser.parse_args()


def prepare_gray(frame, largura_alvo):
    """Cinza pronto para o detector: reduzido para `largura_alvo` + CLAHE.

    Devolve (gray, escala_de_volta). Os cantos achados na imagem reduzida
    precisam ser multiplicados por `escala_de_volta` para voltar ao tamanho do
    frame original. `largura_alvo` 0 = nao reduz.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    escala = 1.0
    if largura_alvo and gray.shape[1] > largura_alvo:
        fator = largura_alvo / float(gray.shape[1])
        gray = cv2.resize(gray, None, fx=fator, fy=fator,
                          interpolation=cv2.INTER_AREA)
        escala = 1.0 / fator
    if DETECTION_CLAHE_CLIP > 0:
        clahe = cv2.createCLAHE(
            clipLimit=DETECTION_CLAHE_CLIP,
            tileGridSize=(DETECTION_CLAHE_GRID, DETECTION_CLAHE_GRID))
        gray = clahe.apply(gray)
    return gray, escala


def refina_cantos(frame, cantos):
    """Refina cantos (JA' no sistema de coordenadas do frame) com cornerSubPix.

    Ver DETECTION_REFINO_CHEIO para o porque. O refinamento roda no cinza
    ORIGINAL, sem CLAHE: o CLAHE realca contraste mas tambem amplifica o ruido,
    e e' o ruido que desloca o minimo sub-pixel. A janela 11x11 (raio de 5 px)
    e' segura: o rescalonamento erra no maximo ~1,5 px e a estrutura vizinha
    mais proxima (a borda de um quadrado) esta' a ~1/2 lado de quadrado de
    distancia -- bem fora da janela.
    """
    if not DETECTION_REFINO_CHEIO or len(cantos) == 0:
        return cantos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    refinados = cv2.cornerSubPix(
        gray,
        cantos.reshape(-1, 1, 2).astype(np.float32),
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return refinados.reshape(-1, 2).astype(np.float64)


def find_corners(frame, allow_fallback=True):
    """Detecta todos os cantos internos (float64, ordem de deteccao).

    Tenta as larguras de DETECTION_WIDTHS EM CASCATA e devolve a primeira que
    achar; a que funcionou por ultimo e' tentada primeiro. A cascata existe
    porque o SB e' instavel na FRONTEIRA: MEDIDO, a mesma configuracao acertou
    numa rodada e falhou na outra (640 px e 960 px ja' fizeram os dois, com o
    tabuleiro na mesma faixa de tamanho -- ver DETECTION_WIDTHS). Entao vale
    cobrir varias escalas em vez de apostar numa.

    Os cantos achados na imagem reduzida sao reescalonados para o frame
    ORIGINAL e refinados la' (refina_cantos). Sem isso o erro de deteccao
    cresce com a reducao: medido 0.67 px na escala cheia, 1.27 px a 640 px e
    2.20 px a 427 px -- este ultimo ja' acima do MAX_STEREO_RMS_PX de 2 px.

    O detector classico entra apenas como fallback eventual (allow_fallback)
    porque e' caro em imagens grandes.
    """
    global _ultima_largura_ok
    larguras = list(DETECTION_WIDTHS)
    if _ultima_largura_ok in larguras:
        larguras.remove(_ultima_largura_ok)
        larguras.insert(0, _ultima_largura_ok)
    for largura in larguras:
        gray, escala = prepare_gray(frame, largura)
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray,
                CHECKERBOARD_SIZE,
                DETECTION_SB_FLAGS,
            )
            if found:
                _ultima_largura_ok = largura
                cantos = corners.reshape(-1, 2).astype(np.float64) * escala
                return refina_cantos(frame, cantos)
    if not allow_fallback:
        return None
    #: O classico entra EM CASCATA tambem, da MENOR para a MAIOR largura (0 =
    #: resolucao cheia, sempre por ultimo): adaptive threshold local lida melhor
    #: com quadrados pequenos e com a impressao de baixo contraste, e ha' caso
    #: medido (Kinect 1080p, tabuleiro impresso) em que o SB rejeita o frame
    #: INTEIRO em todas as escalas -- EXH e rapido, com e sem CLAHE -- enquanto
    #: o classico acha a 960 px (e falha a 640/427 px). Fazer o classico tentar
    #: so' a menor largura perdia exatamente esses casos.
    ordem = sorted(w for w in larguras if w)
    if any(w == 0 for w in larguras):
        ordem.append(0)
    for largura in ordem:
        gray, escala = prepare_gray(frame, largura)
        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            break
    if not found:
        return None
    #: O classico ja' devolve cantos refinados, mas na escala REDUZIDA: refinar
    #: de novo no frame original deixa a precisao igual a do caminho SB.
    cantos = corners.reshape(-1, 2).astype(np.float64) * escala
    return refina_cantos(frame, cantos)


def board_object_points():
    board = np.zeros((CHECKERBOARD_COLS * CHECKERBOARD_ROWS, 1, 3), np.float64)
    grid = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    board[:, 0, :2] = grid
    return board * SQUARE_SIZE_M


def symmetry_indices():
    """Mapeia cada simetria da grade para uma permutacao de indices.

    Chaves: 'id' (identidade), 'rot180' (grade girada 180 graus),
    'hflip' (espelhada ao longo do eixo de ROWS cantos) e 'vflip'
    (espelhada ao longo do eixo de COLS cantos). O layout dos indices e
    row-major, com COLS cantos por linha.
    """
    cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
    rr, cc = np.mgrid[0:rows, 0:cols]
    return {
        "id": np.arange(rows * cols),
        "rot180": (rows * cols - 1) - np.arange(rows * cols),
        "hflip": (rr * cols + (cols - 1 - cc)).reshape(-1),
        "vflip": ((rows - 1 - rr) * cols + cc).reshape(-1),
    }


def canonical_order(corners):
    """Reordena a lista de cantos detectados em um caminho canonico.

    Regra: o primeiro canto e o canto externo mais proximo do canto
    superior esquerdo da imagem; a primeira linha segue o eixo mais
    horizontal a partir do inicio; as linhas avancam para baixo.
    Aplicada as duas cameras antes de calibrar para que os dois
    conjuntos de pontos usem a mesma convencao.
    """
    cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
    points = np.asarray(corners, np.float64).reshape(cols * rows, 2)
    best_name, best_points, best_penalty = "id", points, None
    for name, indices in symmetry_indices().items():
        candidate = points[indices]
        start = candidate[0]
        row_vec = candidate[cols - 1] - start
        col_vec = candidate[cols] - start
        others = np.array([candidate[cols - 1], candidate[cols], candidate[-1]])
        penalty = (
            max(0.0, float(others.sum(axis=1).min()) - float(start.sum()))
            + max(0.0, abs(row_vec[1]) - abs(row_vec[0]))
            + max(0.0, -col_vec[1])
        )
        if best_penalty is None or penalty < best_penalty - 1e-9:
            best_name, best_points, best_penalty = name, candidate, penalty
    return best_name, best_points


def board_span(corners):
    """Maior distancia entre cantos externos do tabuleiro (px)."""
    cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
    points = np.asarray(corners, np.float64).reshape(-1, 2)
    outer = np.array([points[0], points[cols - 1], points[(rows - 1) * cols], points[-1]])
    diffs = outer[:, None, :] - outer[None, :, :]
    return float(np.max(np.linalg.norm(diffs, axis=2)))


def board_tilt_ratio(corners):
    """Razao entre as diagonais do quadrilatero externo.

    ~1.0 significa tabuleiro de frente; valores maiores indicam
    inclinacao (bom para diversidade de poses).
    """
    cols, rows = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
    points = np.asarray(corners, np.float64).reshape(-1, 2)
    d1 = np.linalg.norm(points[0] - points[-1])
    d2 = np.linalg.norm(points[cols - 1] - points[(rows - 1) * cols])
    return float(max(d1, d2) / max(1e-9, min(d1, d2)))


def pair_planar_error(kinect_corners, auxiliary_corners):
    """Erro medio da homografia planar aux -> kinect.

    Detecta capturas ruins (movimento entre frames, detecco ruim), mas
    NAO detecta troca de ordem dos cantos (a homografia absorve as
    simetrias da grade); por isso a correcao de simetria e feita depois
    pelo residuo stereo 3D em solve_with_repair.
    """
    aux = np.asarray(auxiliary_corners, np.float64).reshape(-1, 1, 2)
    kin = np.asarray(kinect_corners, np.float64).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(aux, kin, cv2.RANSAC, 3.0)
    if homography is None:
        return float("inf")
    projected = cv2.perspectiveTransform(aux, homography).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(kin.reshape(-1, 2) - projected, axis=1)))


def calibrate_single(object_points, image_points, size):
    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        size,
        None,
        None,
        flags=CALIB_FLAGS,
        criteria=CALIB_CRITERIA,
    )
    return rms, matrix, dist, rvecs, tvecs


def _unpack_stereo(output, kinect_matrix, kinect_dist, auxiliary_matrix, auxiliary_dist):
    """Lida com os diferentes tamanhos de tupla de cv2.stereoCalibrate.

    Os bindings Python do OpenCV 4.x/5.x retornam 9 valores:
    (rms, K1, D1, K2, D2, R, T, E, F). Bindings mais antigos retornam
    (rms, R, T, E, F). Com CALIB_USE_INTRINSIC_GUESS os K/D devolvidos
    sao os REFINADOS e devem substituir os de calibrateCamera.
    """
    if len(output) >= 9:
        rms = output[0]
        kinect_matrix, kinect_dist = output[1], output[2]
        auxiliary_matrix, auxiliary_dist = output[3], output[4]
        rotation, translation = output[5], output[6]
    else:
        rms, rotation, translation = output[0], output[1], output[2]
    return (
        rms,
        kinect_matrix,
        kinect_dist,
        auxiliary_matrix,
        auxiliary_dist,
        rotation,
        translation,
    )


def _as_float32_points(points_list):
    """cv2.calibrateCamera/stereoCalibrate exigem Point3f/Point2f."""
    return [np.asarray(points, np.float32) for points in points_list]


def run_stereo(
    object_points,
    kinect_points,
    auxiliary_points,
    kinect_size,
    auxiliary_size,
    refine_intrinsics,
):
    """Roda uma solucao completa (individual + stereo) e devolve um dict."""
    object_points = _as_float32_points(object_points)
    kinect_points = _as_float32_points(kinect_points)
    auxiliary_points = _as_float32_points(auxiliary_points)
    (
        kinect_rms,
        kinect_matrix,
        kinect_dist,
        kinect_rvecs,
        kinect_tvecs,
    ) = calibrate_single(object_points, kinect_points, kinect_size)
    auxiliary_rms, auxiliary_matrix, auxiliary_dist, _, _ = calibrate_single(
        object_points, auxiliary_points, auxiliary_size
    )
    if refine_intrinsics:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | CALIB_FLAGS
    else:
        flags = cv2.CALIB_FIX_INTRINSIC
    output = cv2.stereoCalibrate(
        object_points,
        kinect_points,
        auxiliary_points,
        kinect_matrix,
        kinect_dist,
        auxiliary_matrix,
        auxiliary_dist,
        kinect_size,
        flags=flags,
        criteria=STEREO_CRITERIA,
    )
    (
        stereo_rms,
        kinect_matrix,
        kinect_dist,
        auxiliary_matrix,
        auxiliary_dist,
        rotation,
        translation,
    ) = _unpack_stereo(
        output, kinect_matrix, kinect_dist, auxiliary_matrix, auxiliary_dist
    )
    return {
        "stereo_rms": float(stereo_rms),
        "kinect_rms": float(kinect_rms),
        "auxiliary_rms": float(auxiliary_rms),
        "kinect_matrix": kinect_matrix,
        "kinect_dist": kinect_dist,
        "auxiliary_matrix": auxiliary_matrix,
        "auxiliary_dist": auxiliary_dist,
        "rotation": np.asarray(rotation, np.float64),
        "translation": np.asarray(translation, np.float64),
        "kinect_size": tuple(kinect_size),
        "auxiliary_size": tuple(auxiliary_size),
        "refine_intrinsics": bool(refine_intrinsics),
        "kinect_rvecs": kinect_rvecs,
        "kinect_tvecs": kinect_tvecs,
    }


def _project_camera_points(points_camera, matrix, dist):
    projected, _ = cv2.projectPoints(
        np.asarray(points_camera, np.float64).reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        matrix,
        dist,
    )
    return projected.reshape(-1, 2)


def _single_pose_error(result, object_point, kinect_point, auxiliary_point, rvec, tvec):
    """RMS stereo ( Kinect + auxiliar, px) de uma unica pose."""
    rot_i, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    tvec_i = np.asarray(tvec, np.float64).reshape(1, 3)
    points_kinect = object_point.reshape(-1, 3) @ rot_i.T + tvec_i
    points_aux = points_kinect @ np.asarray(result["rotation"], np.float64).T + np.asarray(
        result["translation"], np.float64
    ).reshape(1, 3)
    kinect_error = np.linalg.norm(
        _project_camera_points(points_kinect, result["kinect_matrix"], result["kinect_dist"])
        - np.asarray(kinect_point, np.float64).reshape(-1, 2),
        axis=1,
    )
    auxiliary_error = np.linalg.norm(
        _project_camera_points(
            points_aux, result["auxiliary_matrix"], result["auxiliary_dist"]
        )
        - np.asarray(auxiliary_point, np.float64).reshape(-1, 2),
        axis=1,
    )
    combined = np.concatenate([kinect_error, auxiliary_error])
    return float(np.sqrt(np.mean(combined ** 2)))


def refit_view_poses(object_points, kinect_points, kinect_matrix, kinect_dist, kinect_size):
    """Poses por view do Kinect consistentes com intrinsecas fixas."""
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_FOCAL_LENGTH
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_TANGENT_DIST
    )
    _, _, _, rvecs, tvecs = cv2.calibrateCamera(
        _as_float32_points(object_points),
        _as_float32_points(kinect_points),
        kinect_size,
        np.asarray(kinect_matrix, np.float64).copy(),
        np.asarray(kinect_dist, np.float64).copy(),
        flags=flags,
        criteria=CALIB_CRITERIA,
    )
    return rvecs, tvecs


def pose_rms(result, object_points, kinect_points, auxiliary_points, rvecs=None, tvecs=None):
    """Residuo stereo por pose (px), combinando as duas cameras."""
    if rvecs is None or tvecs is None:
        rvecs, tvecs = refit_view_poses(
            object_points,
            kinect_points,
            result["kinect_matrix"],
            result["kinect_dist"],
            result["kinect_size"],
        )
    return np.array(
        [
            _single_pose_error(
                result,
                object_points[i],
                kinect_points[i],
                auxiliary_points[i],
                rvecs[i],
                tvecs[i],
            )
            for i in range(len(object_points))
        ],
        dtype=np.float64,
    )


def solve_best(object_points, kinect_points, auxiliary_points, kinect_size, auxiliary_size):
    """Resolve com intrinsecas congeladas e com refinadas; guarda a melhor."""
    candidates = []
    for refine in (False, True):
        try:
            candidates.append(
                run_stereo(
                    object_points,
                    kinect_points,
                    auxiliary_points,
                    kinect_size,
                    auxiliary_size,
                    refine_intrinsics=refine,
                )
            )
        except cv2.error as exc:
            print(f"  modo {'refinado' if refine else 'congelado'} falhou: {exc}")
    if not candidates:
        raise RuntimeError("stereoCalibrate falhou nos dois modos")
    return min(candidates, key=lambda item: item["stereo_rms"])


def solve_with_repair(
    object_points,
    kinect_points,
    auxiliary_points,
    kinect_size,
    auxiliary_size,
    verbose=True,
):
    """Resolve a calibracao stereo reparando trocas de ordem dos cantos.

    Estrategia:
    1. Semente RANSAC: resolve mini-subconjuntos aleatorios de poses
       (simetrias 'id'); um subconjunto sem poses trocadas da RMS ~1 px,
       enquanto qualquer subconjunto contaminado da RMS dezenas de px.
       Com menos de 50% das poses trocadas, algumas dezenas de tentativas
       acham um subconjunto limpo com probabilidade altissima.
    2. Atribuicao por pose: para cada pose, a pose do Kinect e obtida por
       solvePnP (independente do R/T global) e as 4 simetrias da grade sao
       testadas contra o modelo da semente; a simetria correta da erro
       ~1 px e as erradas dezenas de px. Re-resolve e repete.
    3. Poses que continuam ruins mesmo com a melhor simetria sao
       descartadas. Termina com solve final testando os dois modos de
       intrinsecas.
    """
    symmetries = symmetry_indices()
    total = len(object_points)
    name_map = {i: "id" for i in range(total)}
    active = list(range(total))

    def subset(indices):
        obj = [object_points[i] for i in indices]
        kin = [kinect_points[i] for i in indices]
        aux = []
        for i in indices:
            points = np.asarray(auxiliary_points[i], np.float64).reshape(-1, 2)
            aux.append(points[symmetries[name_map[i]]].reshape(-1, 1, 2))
        return obj, kin, aux

    def assign_symmetries(model, indices):
        assignments, errors = {}, {}
        for i in indices:
            found, rvec_i, tvec_i = cv2.solvePnP(
                np.asarray(object_points[i], np.float32),
                np.asarray(kinect_points[i], np.float32),
                model["kinect_matrix"],
                model["kinect_dist"],
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not found:
                assignments[i], errors[i] = "id", float("inf")
                continue
            best_name, best_error = "id", float("inf")
            points = np.asarray(auxiliary_points[i], np.float64).reshape(-1, 2)
            for name, indices_perm in symmetries.items():
                trial = points[indices_perm].reshape(-1, 1, 2)
                error = _single_pose_error(
                    model,
                    object_points[i],
                    kinect_points[i],
                    trial,
                    rvec_i,
                    tvec_i,
                )
                if error < best_error:
                    best_name, best_error = name, error
            assignments[i], errors[i] = best_name, best_error
        return assignments, errors

    # ---- etapa 1: semente RANSAC ----
    seed_size = min(RANSAC_SEED_SIZE, total)
    seed_result, seed_subset, seed_rms = None, None, float("inf")
    if total >= 6:
        rng = np.random.default_rng(12345)
        if verbose:
            print(f"  procurando subconjunto limpo ({RANSAC_SEED_TRIALS} tentativas)...")
        for trial_index in range(RANSAC_SEED_TRIALS):
            if verbose and trial_index and trial_index % 10 == 0:
                print(f"  ... {trial_index}/{RANSAC_SEED_TRIALS}")
            trial = sorted(rng.choice(total, size=seed_size, replace=False).tolist())
            try:
                trial_result = run_stereo(
                    *subset(trial), kinect_size, auxiliary_size, refine_intrinsics=False
                )
            except cv2.error:
                continue
            if trial_result["stereo_rms"] < seed_rms:
                seed_result, seed_subset, seed_rms = trial_result, trial, trial_result["stereo_rms"]
        if verbose and seed_result is not None:
            print(
                f"  semente RANSAC: poses {[i + 1 for i in seed_subset]} "
                f"(RMS {seed_rms:.2f} px)"
            )
    if seed_result is None:
        obj, kin, aux = subset(active)
        seed_result = run_stereo(obj, kin, aux, kinect_size, auxiliary_size, refine_intrinsics=False)

    # ---- etapa 2: atribuicao de simetrias por pose + re-solve ----
    model = seed_result
    for _ in range(ASSIGNMENT_ROUNDS):
        assignments, assignment_errors = assign_symmetries(model, active)
        changed = [
            f"pose {i + 1}:'{assignments[i]}'"
            for i in active
            if assignments[i] != name_map[i]
        ]
        if verbose and changed:
            print("  ordem dos cantos corrigida: " + ", ".join(changed))
        for i in active:
            name_map[i] = assignments[i]
        obj, kin, aux = subset(active)
        model = run_stereo(obj, kin, aux, kinect_size, auxiliary_size, refine_intrinsics=False)

    # ---- etapa 3: descarte de poses ruins ----
    obj, kin, aux = subset(active)
    errors = pose_rms(model, obj, kin, aux, model["kinect_rvecs"], model["kinect_tvecs"])
    limit = max(MAX_POSE_RMS_PX, OUTLIER_MEDIAN_FACTOR * float(np.median(errors)))
    order = np.argsort(errors)[::-1]
    for position in order:
        if errors[position] <= limit or len(active) <= MIN_ACTIVE_POSES:
            break
        dropped = active[position]
        if verbose:
            print(
                f"  pose {dropped + 1} descartada (erro {errors[position]:.2f} px "
                "mesmo apos reordenar os cantos)"
            )
        active.remove(dropped)
        del name_map[dropped]
        obj, kin, aux = subset(active)
        model = run_stereo(obj, kin, aux, kinect_size, auxiliary_size, refine_intrinsics=False)
        errors = pose_rms(model, obj, kin, aux, model["kinect_rvecs"], model["kinect_tvecs"])

    # ---- solve final ----
    obj, kin, aux = subset(active)
    final = solve_best(obj, kin, aux, kinect_size, auxiliary_size)
    errors = pose_rms(final, obj, kin, aux)
    return final, name_map, active, errors, obj, kin, aux


def print_solve_report(final, errors, active, spans, tilt_ratios):
    mode = "intrinsecas refinadas" if final["refine_intrinsics"] else "intrinsecas congeladas"
    print(
        f"RMS stereo={final['stereo_rms']:.4f} px ({mode}) | "
        f"RMS Kinect={final['kinect_rms']:.4f} px | RMS auxiliar={final['auxiliary_rms']:.4f} px"
    )
    for position, index in enumerate(active):
        print(f"  pose {index + 1}: {errors[position]:.2f} px")
    if spans and tilt_ratios:
        span_ratio = max(spans) / max(1e-9, min(spans))
        tilted = sum(1 for value in tilt_ratios if value > 1.2)
        print(
            f"Diversidade: span do tabuleiro max/min = {span_ratio:.2f} | "
            f"poses inclinadas (diag ratio > 1.2): {tilted}/{len(tilt_ratios)}"
        )
        if span_ratio < 1.6:
            print(
                "  AVISO: pouca variacao de distancia do tabuleiro "
                "(aproxime e afaste o tabuleiro entre capturas)."
            )
        if tilted < 5:
            print(
                "  AVISO: poucas poses inclinadas (inclua o tabuleiro 20-45 graus "
                "para os lados em varias capturas)."
            )
    if final["auxiliary_rms"] > 3.0:
        print(
            "  AVISO: RMS individual da auxiliar alto; verifique "
            "AUXILIARY_MIRROR_HORIZONTAL (a imagem pode estar espelhada)."
        )


def print_capture_guidance():
    print("COMO CAPTURAR (o RMS stereo depende muito disso):")
    print("- LUZ e' o fator numero 1. Com pouca luz a webcam sobe o tempo de")
    print("  exposicao e o ganho: a imagem sai BORRADA (movimento da mao) e")
    print("  GRANULADA, e o detector passa a achar/nao achar por sorte -- medido")
    print("  na bancada, a MESMA configuracao acertou numa rodada e falhou na")
    print("  outra. Acenda uma luz DIFUSA apontada para o tabuleiro (lampada")
    print("  com uma folha de papel na frente). Luz de teto nao resolve: ela")
    print("  cria sombra da propria mao e estoura no reflexo.")
    print("- Prenda o tabuleiro numa base rigida (caixa, encosto de cadeira).")
    print("  Na mao, a pouca luz + tremor = borrao e a deteccao fica instavel.")
    print("- Segure o tabuleiro de frente para as cameras, sempre com a parte")
    print("  de cima para cima (nao gire o tabuleiro no plano da imagem).")
    print("- Varie a distancia (0.8 a 2.5 m) e a posicao na imagem; todas as")
    print("  regioes da imagem devem aparecer em pelo menos uma captura.")
    print("- Incline o tabuleiro 20-45 graus para os lados/cima/baixo em")
    print("  varias capturas; sem inclinacao o foco fica impreciso.")
    print("- Inclua capturas com o tabuleiro nas bordas e cantos da imagem")
    print("  das duas cameras (o modelo de distorcao precisa delas).")
    print("- Tabuleiro PARADO na captura (as cameras nao sao sincronizadas),")
    print("  imagem nitida e sem reflexo.")


def save_calibration(final, active, errors, obj, kin, aux):
    output = output_file(AUX_CAMERA_INDEX)
    np.savez(
        output,
        kinect_matrix=final["kinect_matrix"],
        kinect_dist=final["kinect_dist"],
        auxiliary_matrix=final["auxiliary_matrix"],
        auxiliary_dist=final["auxiliary_dist"],
        rotation=final["rotation"],
        translation=final["translation"],
        rms=final["stereo_rms"],
        kinect_rms=final["kinect_rms"],
        auxiliary_rms=final["auxiliary_rms"],
        square_size_m=SQUARE_SIZE_M,
        checkerboard_size=CHECKERBOARD_SIZE,
        kinect_size=np.asarray(final["kinect_size"]),
        auxiliary_size=np.asarray(final["auxiliary_size"]),
        per_pose_rms=errors,
        used_poses=np.asarray(active),
        refined_intrinsics=1 if final["refine_intrinsics"] else 0,
    )
    print(
        f"Melhor calibracao salva em {output}; "
        f"RMS stereo={final['stereo_rms']:.4f} px"
    )
    save_alignment(obj, kin, aux)


def save_alignment(obj, kin, aux):
    """Regenera o camera_alignment.npz a partir da calibracao stereo.

    Mantem camera_alignment.npz e stereo_calibration.npz sincronizados:
    a homografia e ajustada (RANSAC) aos pontos do tabuleiro usados na
    propria solucao stereo e o refinamento afim recomeca da identidade.
    """
    aux_pixels = np.concatenate(
        [np.asarray(points, np.float64).reshape(-1, 2) for points in aux]
    )
    kinect_pixels = np.concatenate(
        [np.asarray(points, np.float64).reshape(-1, 2) for points in kin]
    )
    homography, _ = cv2.findHomography(aux_pixels, kinect_pixels, cv2.RANSAC, 3.0)
    if homography is None:
        print("AVISO: nao foi possivel derivar a homografia do alinhamento")
        return
    np.savez(
        CAMERA_ALIGNMENT_FILE,
        homography=homography,
        square_size_m=SQUARE_SIZE_M,
        version=CAMERA_ALIGNMENT_VERSION,
        refinement_affine=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
    )
    print(f"Alinhamento (homografia) regenerado em {CAMERA_ALIGNMENT_FILE}")


def reduzir_janela(frame):
    """Reduz o frame para o TAMANHO DA JANELA (DISPLAY_SCALE).

    Chamado no ultimo passo, so' no que vai para o `cv2.imshow`: e' reducao de
    EXIBICAO. Tudo o que decide algo (deteccao de cantos, amostras, solver)
    roda antes, na resolucao CHEIA.
    """
    if frame is None or DISPLAY_SCALE == 1.0:
        return frame
    return cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                      interpolation=cv2.INTER_AREA)


def abre_auxiliar(indice):
    """Abre a webcam auxiliar pelo backend de MENOR ATRASO, com fallback.

    O problema que isto resolve: por padrao os backends do OpenCV ENFILEIRAM
    varios frames e o `read()` devolve o mais ANTIGO. O video aparece "lento"
    (mostra o passado) e o detector roda em cima de imagem defasada -- o
    tabuleiro ja' se moveu quando o resultado sai. `CAP_PROP_BUFFERSIZE = 1`
    corta a fila; onde o backend ignora a propriedade, o descarte inicial ja'
    ajuda. Se o DSHOW nao entregar frame, cai para o MSMF.
    """
    for backend, nome in AUX_BACKENDS:
        captura = cv2.VideoCapture(indice, backend)
        if not captura.isOpened():
            captura.release()
            print(f"  auxiliar {indice}: backend {nome} nao abriu")
            continue
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, AUX_REQUEST_SIZE[0])
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, AUX_REQUEST_SIZE[1])
        aceitou_buffer = captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(AUX_DESCARTE):
            captura.read()
        ok, quadro = captura.read()
        if ok and quadro is not None:
            print(f"  auxiliar {indice}: backend {nome} OK "
                  f"({quadro.shape[1]}x{quadro.shape[0]}, buffer=1: "
                  f"{'sim' if aceitou_buffer else 'backend ignorou'})")
            return captura
        captura.release()
        print(f"  auxiliar {indice}: backend {nome} abriu mas nao entregou frame")
    raise RuntimeError(
        f"Nao consegui abrir a webcam auxiliar {indice}. Ela esta' conectada "
        "e livre (nenhum outro programa usando)?")


def abre_kinect(espera_s=ESPERA_KINECT_S):
    """Abre o Kinect RGB e ESPERA o primeiro frame antes de seguir.

    O Kinect e' OBRIGATORIO: o par estereo e' (Kinect RGB, auxiliar), nao ha'
    como calibrar so' com a auxiliar. Sem esta espera o sintoma e' cruel, e foi
    o que aconteceu na bancada: se o Kinect nao entrega frame, o loop principal
    faz `continue` ANTES do `cv2.imshow` e NENHUMA JANELA APARECE. O programa
    fica mudo, com cara de travado, sem dizer que o problema e' a camera.
    Aqui ele falha ALTO e diz exatamente o que conferir.
    """
    print("  Kinect: abrindo o RGB (pode levar alguns segundos)...")
    try:
        runtime = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    except Exception as exc:
        raise RuntimeError(
            "Nao consegui abrir o Kinect v2. Confira: cabo USB 3.0 (porta "
            "azul) DIRETO na maquina (sem hub/extensor), SDK do Kinect v2 "
            "instalado, e nenhum outro programa (Kinect Studio, ferramenta de "
            f"ground truth) usando a camera. Erro: {exc}") from exc
    inicio = time.time()
    while time.time() - inicio < espera_s:
        if runtime.has_new_color_frame():
            print(f"  Kinect: primeiro frame em {time.time() - inicio:.1f} s")
            return runtime
        time.sleep(0.005)
    runtime.close()
    raise RuntimeError(
        f"O Kinect abriu mas NAO entregou frame em {espera_s:.0f} s. Ele e' "
        "obrigatorio para a calibracao stereo (o par e' Kinect RGB + webcam "
        "auxiliar). Confira o cabo USB 3.0 direto na maquina e se outro "
        "programa nao esta' usando a camera.")


def calibrate_one(aux_index):
    """Calibra o par (Kinect RGB, webcam auxiliar `aux_index`).

    Devolve 0 em caso de sucesso. Cada camera auxiliar tem a SUA propria sessao
    de captura e o seu proprio arquivo stereo_calibration_aux{N}.npz.
    """
    global AUX_CAMERA_INDEX
    AUX_CAMERA_INDEX = int(aux_index)
    # Nome REAL (Windows) da camera deste indice: e' o que denuncia o erro caro
    # -- calibrar "a auxiliar" na webcam do LAPTOP (mesmo ponto de vista do
    # Kinect, sem baseline). O nome vem de camera_names.py (enumeracao
    # DirectShow, a mesma ordem do CAP_DSHOW).
    nome_real = None
    try:
        import camera_names as nomes_mod
    except ImportError:                   # pragma: no cover - sem comtypes
        nomes_mod = None
    if nomes_mod is not None:
        nome_real = nomes_mod.nome_da_camera(AUX_CAMERA_INDEX)
    print("=" * 72)
    print(f"Calibrando a webcam auxiliar de indice {AUX_CAMERA_INDEX} "
          f"({nome_real or AUX_CAMERA_NAMES.get(AUX_CAMERA_INDEX, '?')}) -> "
          f"{output_file(AUX_CAMERA_INDEX)}")
    if nomes_mod is not None and nomes_mod.e_do_laptop(nome_real):
        print(nomes_mod.aviso_laptop(
            AUX_CAMERA_INDEX, "webcam do par estereo com o Kinect"))
    print_capture_guidance()
    kinect = abre_kinect()
    auxiliary = abre_auxiliar(AUX_CAMERA_INDEX)
    aux_width = int(auxiliary.get(cv2.CAP_PROP_FRAME_WIDTH))
    aux_height = int(auxiliary.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kinect_width = kinect.color_frame_desc.Width
    kinect_height = kinect.color_frame_desc.Height
    kinect_size = (kinect_width, kinect_height)
    auxiliary_size = (aux_width, aux_height)
    print(
        f"Resolucao Kinect RGB: {kinect_width}x{kinect_height} | "
        f"auxiliar: {aux_width}x{aux_height}"
    )
    board = board_object_points()
    object_points = []
    kinect_points = []
    auxiliary_points = []
    kinect_spans = []
    tilt_ratios = []
    kinect_diagonal = float(np.hypot(kinect_width, kinect_height))
    auxiliary_diagonal = float(np.hypot(aux_width, aux_height))
    stable_since = None
    best_rms = float("inf")
    last_solved = 0
    last_detect = 0.0
    detection_count = 0
    cached_kinect_corners = None
    cached_auxiliary_corners = None
    #: O SB falhou na deteccao anterior? Se sim, o classico entra na proxima --
    #: e' exatamente o caso (contraste/ruido) em que ele mais ajuda.
    sb_falhou_antes = False
    #: Ultimo aviso de "camera sem frame" impresso (evita repetir a cada frame).
    aviso_anterior = ""
    #: Batimento de deteccao (ver bloco "sem par completo" no loop).
    ultimo_aviso_detecao = 0.0
    try:
        while True:
            kinect_frame = None
            if kinect.has_new_color_frame():
                data = kinect.get_last_color_frame()
                kinect_frame = data.reshape((kinect_height, kinect_width, 4))
                kinect_frame = cv2.cvtColor(kinect_frame, cv2.COLOR_BGRA2BGR)
            auxiliary_ok, auxiliary_frame = auxiliary.read()
            if AUXILIARY_MIRROR_HORIZONTAL and auxiliary_ok:
                auxiliary_frame = cv2.flip(auxiliary_frame, 1)
            if kinect_frame is None or not auxiliary_ok:
                # AVISO VISIVEL: mostrar NADA quando uma camera cai deixa o
                # programa com cara de travado -- foi exatamente o sintoma
                # "rodei e nao apareceu nada". A janela diz QUAL camera caiu;
                # o texto no terminal sai so' quando a lista muda.
                faltando = []
                if kinect_frame is None:
                    faltando.append("Kinect RGB")
                if not auxiliary_ok:
                    faltando.append(f"auxiliar {AUX_CAMERA_INDEX}")
                texto = ", ".join(faltando)
                # Aviso montado JA' no tamanho da JANELA (DISPLAY_SCALE): aqui
                # nao ha' imagem de camera para reduzir, entao as 4 linhas sao
                # desenhadas direto no tamanho final (reduzir o quadro pronto
                # deixaria as letras com ~5 px).
                aviso = np.full(
                    (int(200 * DISPLAY_SCALE), int(780 * DISPLAY_SCALE), 3),
                    40, np.uint8)
                cv2.putText(aviso, "SEM IMAGEM", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(aviso, f"sem frame de: {texto}", (12, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(aviso, "confira o cabo USB (3.0 direto, sem hub) e",
                            (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1)
                cv2.putText(aviso, "se outro programa nao usa a camera | ESC sai",
                            (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1)
                cv2.imshow("Stereo - Kinect RGB", aviso)
                cv2.imshow("Stereo - camera auxiliar", aviso)
                if texto != aviso_anterior:
                    print(f"  SEM FRAME de: {texto}. A janela mostra o aviso; "
                          "confira o cabo/USB e se outro programa nao esta' "
                          "usando a camera.")
                    aviso_anterior = texto
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            #: Recuperou: zera para que uma NOVA queda volte a avisar.
            aviso_anterior = ""
            now = cv2.getTickCount() / cv2.getTickFrequency()
            # Detecta a ~10 Hz e reusa os cantos entre frames: SB+fallback
            # em 1080p+720p por frame deixaria a interface inviavel.
            if now - last_detect >= DETECTION_INTERVAL_S:
                # O classico entra em TODO ciclo (FALLBACK_EVERY = 1): com o SB
                # falhando em frames como os do Kinect, um gate "de vez em
                # quando" faz a deteccao oscilar acha/falha e a estabilidade
                # de 2 s nunca completa. Ver FALLBACK_EVERY.
                allow_fallback = (sb_falhou_antes
                                  or detection_count % FALLBACK_EVERY == 0)
                kinect_corners = find_corners(kinect_frame, allow_fallback=allow_fallback)
                auxiliary_corners = find_corners(auxiliary_frame, allow_fallback=allow_fallback)
                sb_falhou_antes = (kinect_corners is None
                                   or auxiliary_corners is None)
                cached_kinect_corners = kinect_corners
                cached_auxiliary_corners = auxiliary_corners
                last_detect = now
                detection_count += 1
            else:
                kinect_corners = cached_kinect_corners
                auxiliary_corners = cached_auxiliary_corners
            kinect_found = kinect_corners is not None
            auxiliary_found = auxiliary_corners is not None
            # --- JANELAS (DISPLAY_SCALE) ------------------------------------
            # A deteccao acima rodou na resolucao CHEIA; daqui para baixo e' o
            # que o cv2.imshow mostra. Os cantos desenhados sao escalados pelo
            # MESMO fator (senao sairiam fora do tabuleiro) e os TEXTOS entram
            # depois da reducao: desenhados antes, encolheriam junto com a
            # imagem e ficariam ilegiveis na janela da metade.
            kinect_display = reduzir_janela(kinect_frame)
            auxiliary_display = reduzir_janela(auxiliary_frame)
            if kinect_found:
                # ORDEM CANONICA no desenho: o detector devolve os cantos na
                # ordem da IMAGEM e as linhas de ligacao saem cruzadas/loucas
                # (parece que ele achou a coisa errada). `canonical_order` e' a
                # MESMA reordenacao usada nas amostras -> o operador ve' a grade
                # seguindo o tabuleiro.
                _, canonico = canonical_order(kinect_corners)
                cv2.drawChessboardCorners(
                    kinect_display,
                    CHECKERBOARD_SIZE,
                    (canonico * DISPLAY_SCALE).astype(np.float32).reshape(-1, 1, 2),
                    True,
                )
            if auxiliary_found:
                _, canonico = canonical_order(auxiliary_corners)
                cv2.drawChessboardCorners(
                    auxiliary_display,
                    CHECKERBOARD_SIZE,
                    (canonico * DISPLAY_SCALE).astype(np.float32).reshape(-1, 1, 2),
                    True,
                )
            if AUXILIARY_DISPLAY_MIRROR:
                auxiliary_display = cv2.flip(auxiliary_display, 1)
            if kinect_found and auxiliary_found and stable_since is not None:
                stable_text = f"estavel {max(0.0, now - stable_since):.1f}/2.0 s"
            else:
                stable_text = "aguardando ambos"
            # Mostra o TAMANHO do tabuleiro ao vivo: abaixo de
            # MIN_BOARD_SPAN_FRACTION a amostra e' rejeitada, e sem esse numero
            # o sintoma no video e' so' "piscar" sem o usuario saber o motivo.
            kinect_pct = (board_span(kinect_corners) / kinect_diagonal
                          if kinect_found else 0.0)
            auxiliary_pct = (board_span(auxiliary_corners) / auxiliary_diagonal
                             if auxiliary_found else 0.0)
            status = (
                f"Kinect: {'OK' if kinect_found else 'falhou'} "
                f"({kinect_pct:.0%}) | "
                f"Aux: {'OK' if auxiliary_found else 'falhou'} "
                f"({auxiliary_pct:.0%}) | "
                f"{len(object_points)}/{SAMPLES_REQUIRED} | {stable_text}"
            )
            for display in (kinect_display, auxiliary_display):
                cv2.putText(
                    display,
                    status,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if kinect_found and auxiliary_found else (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"tabuleiro >= {MIN_BOARD_SPAN_FRACTION:.0%} da imagem | "
                    "mantenha 2 s parado | ESC encerra",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            cv2.imshow("Stereo - Kinect RGB", kinect_display)
            cv2.imshow("Stereo - camera auxiliar", auxiliary_display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if not (kinect_found and auxiliary_found):
                stable_since = None
                #: Batimento de diagnostico: silencio total quando a deteccao
                #: falha e' cruel -- parece travado. A cada 5 s sem par completo,
                #: imprime O QUE cada camera mostra (achou? brilho medio?) e
                #: salva os frames em _hb_*.png para analise externa.
                if now - ultimo_aviso_detecao >= 5.0:
                    mk = (f"brilho {kinect_frame.mean():.0f}"
                          if kinect_frame is not None else "sem frame")
                    ma = (f"brilho {auxiliary_frame.mean():.0f}"
                          if auxiliary_ok else "sem frame")
                    print(
                        f"  deteccao sem par completo "
                        f"(Kinect {'OK' if kinect_found else 'nao'} {mk} | "
                        f"aux {AUX_CAMERA_INDEX} "
                        f"{'OK' if auxiliary_found else 'nao'} {ma}); "
                        "frames salvos em _hb_*.png", flush=True)
                    ultimo_aviso_detecao = now
                    if kinect_frame is not None:
                        cv2.imwrite("_hb_kinect.png", kinect_frame)
                    if auxiliary_ok:
                        cv2.imwrite("_hb_aux.png", auxiliary_frame)
                continue
            if stable_since is None:
                stable_since = now
            if now - stable_since < STABLE_CAPTURE_SECONDS:
                continue
            stable_since = now
            kinect_span = board_span(kinect_corners)
            auxiliary_span = board_span(auxiliary_corners)
            if (
                kinect_span < MIN_BOARD_SPAN_FRACTION * kinect_diagonal
                or auxiliary_span < MIN_BOARD_SPAN_FRACTION * auxiliary_diagonal
            ):
                print("Tabuleiro pequeno na imagem; aproxime o tabuleiro das cameras.")
                continue
            pair_error = pair_planar_error(kinect_corners, auxiliary_corners)
            if pair_error > MAX_PAIR_ERROR_PX:
                print(
                    f"Par rejeitado (erro planar {pair_error:.2f} px > "
                    f"{MAX_PAIR_ERROR_PX:.1f}); mantenha o tabuleiro PARADO e repita."
                )
                continue
            _, kinect_canonical = canonical_order(kinect_corners)
            _, auxiliary_canonical = canonical_order(auxiliary_corners)
            #: Guarda de variedade: amostra quase IGUAL a uma ja' coletada nao
            #: acrescenta geometria -- 15 amostras de UMA pose = extrinseca
            #: degenerada com RMS enganosamente bom. Exige que o centro do
            #: tabuleiro (na imagem do Kinect) esteja afastado de TODOS os
            #: centros ja' coletados; o tabuleiro parado deixa de virar lixo.
            if kinect_points:
                centros = np.array([
                    p.reshape(-1, 2).mean(axis=0) for p in kinect_points
                ])
                centro_novo = kinect_canonical.reshape(-1, 2).mean(axis=0)
                separacao = float(
                    np.linalg.norm(centros - centro_novo, axis=1).min()
                )
                if separacao < MIN_POSE_SEPARATION * kinect_span:
                    print(
                        f"Pose quase igual a uma ja' coletada "
                        f"({separacao:.0f} px); MEXA no tabuleiro -- mude "
                        "distancia, inclinacao ou posicao antes da proxima."
                    )
                    continue
            object_points.append(board.copy())
            kinect_points.append(kinect_canonical.reshape(-1, 1, 2))
            auxiliary_points.append(auxiliary_canonical.reshape(-1, 1, 2))
            kinect_spans.append(kinect_span)
            tilt_ratios.append(board_tilt_ratio(kinect_corners))
            print(
                f"Amostra aceita: {len(object_points)}/{SAMPLES_REQUIRED} "
                f"(par {pair_error:.2f} px, span {kinect_span / kinect_diagonal:.0%} da imagem)"
            )


            total = len(object_points)
            if total < SAMPLES_REQUIRED:
                continue
            if total != SAMPLES_REQUIRED and total - last_solved < SOLVE_EVERY_EXTRA:
                continue
            last_solved = total
            print(f"Resolvendo com {total} poses...")
            final, name_map, active, errors, final_obj, final_kin, final_aux = solve_with_repair(
                object_points, kinect_points, auxiliary_points, kinect_size, auxiliary_size
            )
            print_solve_report(final, errors, active, kinect_spans, tilt_ratios)
            if final["stereo_rms"] < best_rms:
                best_rms = final["stereo_rms"]
                save_calibration(final, active, errors, final_obj, final_kin, final_aux)
            if final["stereo_rms"] <= MAX_STEREO_RMS_PX:
                print(f"Calibracao aceita; RMS stereo={final['stereo_rms']:.4f} px")
                break
            if total >= MAX_TOTAL_SAMPLES:
                print(
                    f"Limite de {MAX_TOTAL_SAMPLES} amostras atingido com RMS "
                    f"{final['stereo_rms']:.2f} px; a melhor foi salva. Revise a "
                    "variedade das poses (distancia, inclinacao, posicao) e recomece."
                )
                break
            print(
                f"Calibracao ainda acima do limite ({MAX_STEREO_RMS_PX:.1f} px); "
                "coletando mais amostras (varie distancia e inclinacao!)"
            )
    finally:
        auxiliary.release()
        kinect.close()
        cv2.destroyAllWindows()
    return 0


def main():
    """Calibra cada camera auxiliar pedida, uma sessao de captura de cada vez."""
    global DETECTION_WIDTHS, DETECTION_CLAHE_CLIP, DETECTION_SB_FLAGS
    args = parse_args()
    global CHECKERBOARD_COLS, CHECKERBOARD_ROWS, CHECKERBOARD_SIZE, SQUARE_SIZE_M
    if args.checkerboard:
        texto = args.checkerboard.lower().replace(" ", "")
        for separador in ("x", ",", "*"):
            if separador in texto:
                cols_txt, rows_txt = texto.split(separador, 1)
                break
        else:
            print(f"ERRO: --checkerboard '{args.checkerboard}' nao esta' no "
                  "formato COLSxROWS (ex.: 5x3).")
            return 2
        try:
            cols, rows = int(cols_txt), int(rows_txt)
        except ValueError:
            print(f"ERRO: --checkerboard '{args.checkerboard}' nao esta' no "
                  "formato COLSxROWS (ex.: 5x3).")
            return 2
        if cols < 3 or rows < 3:
            print(f"ERRO: --checkerboard {cols}x{rows} invalido (minimo 3x3).")
            return 2
        CHECKERBOARD_COLS, CHECKERBOARD_ROWS = cols, rows
        CHECKERBOARD_SIZE = (cols, rows)
    if args.quadrado:
        if args.quadrado <= 0:
            print(f"ERRO: --quadrado {args.quadrado} tem de ser > 0.")
            return 2
        SQUARE_SIZE_M = args.quadrado / 1000.0
    print(f"Tabuleiro: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS} cantos "
          f"internos, quadrado de {SQUARE_SIZE_M * 1000:g} mm")
    DETECTION_WIDTHS = tuple(int(v) for v in args.largura_deteccao)
    DETECTION_CLAHE_CLIP = float(args.clahe)
    DETECTION_SB_FLAGS = 0 if args.rapido else cv2.CALIB_CB_EXHAUSTIVE
    print(f"Detector: larguras {DETECTION_WIDTHS} (0 = cheia) | "
          f"CLAHE {DETECTION_CLAHE_CLIP:g} | flags {DETECTION_SB_FLAGS}")
    #: dict.fromkeys preserva a ordem e remove indices repetidos.
    indices = list(dict.fromkeys(int(i) for i in args.aux_index))
    if len(indices) > 1:
        print(f"Vao ser calibradas {len(indices)} auxiliares, UMA DE CADA VEZ: "
              f"{indices}")
        print("Cada sessao tem o seu proprio minimo de 15 amostras; ESC "
              "encerra a sessao atual e passa para a proxima.")
    for posicao, aux_index in enumerate(indices, start=1):
        if len(indices) > 1:
            print(f"\n##### AUXILIAR {posicao}/{len(indices)} "
                  f"(indice {aux_index}) #####")
        try:
            if calibrate_one(aux_index) != 0:
                return 1
        except RuntimeError as exc:
            #: Falha de camera sai como INSTRUCAO, nao como traceback cru: e' o
            #: caso "esqueci o Kinect conectado", e um traceback do pykinect2
            #: nao diz nada de util.
            print(f"\nERRO: {exc}")
            return 2
    return 0


if __name__ == "__main__":
    #: sys.exit propaga o codigo: sem isso um erro de camera terminava com
    #: codigo 0 e os .bat/scripts nao tinham como perceber a falha.
    sys.exit(main())
