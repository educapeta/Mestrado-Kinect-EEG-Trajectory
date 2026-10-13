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
2. INTRINSECAS. CALIB_FIX_INTRINSIC congela fx/fy/cx/cy/distorcao obtidos
   separadamente. Se as poses tiverem pouca variacao de distancia/inclinacao
   (tabuleiro sempre de frente e na mesma distancia), essas intrinsecas ficam
   imprecisas. Pior: o tabuleiro de 6x4 x 57 mm NAO preenche o quadro do Kinect
   na distancia de trabalho (a camera de cor do Kinect foca longe), entao o
   modelo do Kinect (fx/fy/cx/cy + k1/k2/k3) estimado por sessao e' fragil.
   MEDIDO em 30/09/2026: tres sessoes da MESMA bancada produziram tres modelos do
   Kinect diferentes, que projetam o mesmo ponto 3D de 24 a 50 px de diferenca
   (ate' 176 px na borda do quadro) -- erro sistematico do overlay que o RMS NAO
   denuncia, porque o RMS so' mede a coerencia INTERNA de cada npz.
   Por isso existe o MODELO CONGELADO do Kinect (`kinect_intrinsics.npz`),
   gravado com --salvar-intrinsecas-kinect (tabuleiro GRANDE preenchendo o
   quadro) e reusado por --intrinsecas-kinect ou automaticamente quando o
   arquivo existe. Com ele o Kinect NUNCA e' re-estimado: o stereo resolve so'
   R/T, que e' o que realmente muda de sessao para sessao. Sem modelo congelado o
   solver ainda tenta CALIB_USE_INTRINSIC_GUESS (refinamento conjunto) e salva o
   de menor RMS -- comportamento historico, mantido como ultimo recurso.

   O arranjo que congela POR ESTE CAMINHO tem um limite fisico: quem manda na
   distancia do tabuleiro e' a AUXILIAR (ela fecha o quadro primeiro), e o
   tabuleiro grande a ~1 m ocupa so' 11-13% da diagonal do Kinect -- abaixo do
   portao de 35%. Foi assim que nasceu o paliativo de 30/09/2026 (medoide de 4
   modelos de sessao, com vies desconhecido de ate' ~46 px). Para congelar DE
   VERDADE use o arranjo Kinect-only (`freeze_kinect_direto.py`): so' o Kinect, o
   tabuleiro chega a 0,3-0,45 m, o span passa dos 35% e o modelo sai de uma
   camera so'.
   MEDIDO em 01/10/2026: trocar a placa deixou os dois npz da MESMA bancada
   68,8 px um do outro porque CADA UM re-estimou o Kinect -- o congelado de
   5x3/57 mm tinha sido recusado POR PLACA. Hoje a placa nao recusa nada (vira
   aviso: as intrinsecas sao da CAMERA); quem decide sao poses + span + RMS, os
   MESMOS portoes na gravacao e na leitura (`portoes_do_modelo_kinect`).
3. QUALIDADE DAS AMOSTRAS. Capturas com tabuleiro pequeno na imagem
   (foco/escala ruins) ou com erro planar do par alto (movimento,
   deteccao ruim) sao rejeitadas antes de entrar na solucao.

MODELO CONGELADO DO KINECT (o fluxo recomendado, UMA vez por bancada):

    # 1) tabuleiro GRANDE (>= ~35% da diagonal do quadro), Kinect + a auxiliar
    #    enxergando: grava o modelo do Kinect e ja' calibra este par
    python stereo_calibration.py --salvar-intrinsecas-kinect
    # 2) sessoes de rotina (uma por auxiliar): reusam o modelo congelado
    python stereo_calibration.py
    # 3) confira SEM hardware o que ficou:
    python tools\\diagnostico_calibracao.py

Sem o modelo congelado, cada sessao re-estima o Kinect (comportamento antigo) e
os npz ficam com modelos diferentes entre si -- o overlay herda dezenas de px de
erro sistematico que nenhum RMS denuncia (medido em 30/09/2026; ver o item 2).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

import kinect_compat      # pykinect2 64 bits: ANTES do import do pykinect2

kinect_compat.aplicar()

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
#: Quantas vezes REABRIR o runtime do Kinect quando ele abre e nao entrega frame.
#: MEDIDO na bancada de 07/10/2026: depois de usar o Kinect Studio o sensor ficou
#: enumerado (VID_045E&PID_02C4, driver KinectSensor, `ConfigFlags=0`) e o
#: `PyKinectRuntime` ABRIA em 0,0 s reportando 1920x1080 -- mas NENHUM frame saiu
#: em 20 s, em color, depth, body e nos tres juntos. Fechar e reabrir o runtime
#: custa um segundo e ja' recuperou sensores nesse estado; esperar para sempre
#: num runtime morto, nao.
TENTATIVAS_KINECT = 3
#: Programas que tomam o sensor para si. Entram na mensagem de erro porque "fechei
#: o Kinect Studio" e' o que o operador lembra -- e o watch/gravador fica vivo.
APLICATIVOS_QUE_SEGURAM_O_KINECT = (
    "Kinect Studio", "Kinect Configuration Verifier", "KinectService",
    "Kinect Monitor", "gravador do Kinect Studio", "any application using the "
    "sensor")
#: Etiqueta de arquivo FORA DO AR: `carrega_intrinsecas_kinect` so' le o nome
#: exato (`KINECT_INTRINSICS_FILE`), entao este sufixo aposenta um npz sem
#: apagar nada e sem que ele volte a ser carregado por engano.
SUFIXO_ARQUIVADO = ".arquivado"
# Tabuleiro: PLACA ATUAL (impressa 23/09/2026) = 6 x 4 QUADRADOS de 57 mm ->
# 5 x 3 CANTOS internos (a regra do cv2 e' sempre quadrados - 1 por lado).
# Precisa casar com CHECKERBOARD_SIZE/SQUARE_SIZE_M em
# kinect_imu_groundtruth.py: o stereo salvo guarda estes dois valores e a
# biblioteca RECUSA (stereo_ready=False, com o motivo dito em
# CameraAlignment.stereo_status) qualquer npz de placa/quadrado diferente.
# Historico: 4x3/52 mm (subgrade, geometria 2x grande) -> 12x8 quadrados/26 mm
# (11x7 cantos, a placa antiga) -> 6x4/57 mm (a atual).
# Sobrescrevivel na linha de comando com --checkerboard CxR e --quadrado MM.
CHECKERBOARD_COLS = 7
CHECKERBOARD_ROWS = 5
CHECKERBOARD_SIZE = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)
SQUARE_SIZE_M = 0.045
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
# --- MODELO CONGELADO DO KINECT (ver item 2 do docstring) -------------------
# O Kinect RGB e' UMA camera: o modelo dela vale para todas as sessoes. Este
# arquivo e' gravado UMA vez (--salvar-intrinsecas-kinect, com um tabuleiro que
# preencha o quadro) e reusado por TODAS as calibracoes de par
# (--intrinsecas-kinect, ou automaticamente quando ele existe ao lado do codigo).
# Com o modelo congelado, o Kinect NAO e' re-estimado em sessao nenhuma -- era
# assim que cada sessao inventava um k1/k2/k3 diferente e o overlay saia dezenas
# de px fora (medido em 30/09/2026; ver tools\diagnostico_calibracao.py).
KINECT_INTRINSICS_FILE = Path(__file__).with_name("kinect_intrinsics.npz")
KINECT_INTRINSICS_VERSION = 1
#: Minimo de poses para GRAVAR o modelo congelado (menos que isso o
#: calibrateCamera tem parametros demais para os dados que tem).
KINECT_INTRINSICS_MIN_POSES = 12
#: Fracao MINIMA da diagonal do quadro do Kinect que o tabuleiro tem de ocupar
#: (span MEDIANO das poses). 0.35 = o tabuleiro grande, feito para isto: com o
#: tabuleiro pequeno (8-20% da diagonal) o modelo NAO e' gravavel -- e' esse o
#: motivo do modelo ser congelado em vez de re-estimado.
#: E' ESTA a grandeza que decide se um modelo congelado serve -- nao a placa em
#: que ele foi gravado (ver `carrega_intrinsecas_kinect`). Os tres portoes sao os
#: mesmos ao GRAVAR e ao CARREGAR (`portoes_do_modelo_kinect`): se divergissem,
#: existiria o arquivo que o programa escreve e depois recusa.
KINECT_INTRINSICS_MIN_SPAN = 0.35
#: Fracao MAXIMA da diagonal aceita na CAPTURA (arranjo Kinect-only). Acima disto
#: o tabuleiro esta' a menos de ~0,28 m: o RGB do Kinect foca longe, o quadro sai
#: borrado e o tabuleiro gigante domina o ajuste de fx (MEDIDO em 30/09/2026:
#: 77% da diagonal = tabuleiro a ~0,16 m). E' portao de CAPTURA, nunca do modelo:
#: a amostra e' so' descartada, com o numero medido.
KINECT_INTRINSICS_MAX_SPAN = 0.60
#: RMS individual maximo do Kinect aceito ao gravar o modelo congelado (o RMS de
#: uma camera so', com o tabuleiro grande, tem de ser bem melhor que o do par).
KINECT_INTRINSICS_MAX_RMS = 0.75
#: Grade de avaliacao de `compara_modelos_kinect` (m, referencial do Kinect):
#: a regiao onde a mao anda na bancada. Espelhada em
#: tools\diagnostico_calibracao.py (JANELA_X/JANELA_Y/PASSOS_X/PASSOS_Y) e
#: travada por test_diagnostico_calibracao.py -- os dois tem de dar o mesmo px.
KINECT_EVAL_JANELA_X = (-0.35, 0.35)
KINECT_EVAL_JANELA_Y = (-0.30, 0.30)
KINECT_EVAL_PASSOS_X = 9
KINECT_EVAL_PASSOS_Y = 7
KINECT_EVAL_PROFUNDIDADE_M = 0.8


def output_file(aux_index):
    """Arquivo stereo por camera auxiliar (legado = indice 0)."""
    if int(aux_index) == 0:
        return OUTPUT_FILE
    return Path(__file__).with_name(
        f"stereo_calibration_aux{int(aux_index)}.npz")


def indices_padrao_auxiliares():
    """Indices das webcams EXTERNAS desta maquina (default de `--aux-index`).

    A constante AUX_CAMERA_INDICES_PADRAO = (2, 0) estava certa na maquina de
    23/09/2026 e esta' ERRADA nesta (28/09/2026): aqui existem 0 = SIGMA-W780M
    e 1 = Trust USB Camera, e NENHUM indice 2 -- com o default antigo a
    calibracao tentava o indice 2 primeiro e o operador nao chegava na camera
    boa. A regra e' a MESMA do resto do projeto (fonte unica:
    `kinect_imu_groundtruth.indices_auxiliares_automaticos`): todas as cameras
    MENOS a webcam do laptop, pelo NOME do Windows, as que ja' tem calibracao
    primeiro (mesma ordem do `tools/kinect_groundtruth_tool.py`, entao calibrar
    na ordem mantem camera e calibracao pareadas).
    """
    try:                                # import tardio: so' quando ninguem fixou
        import kinect_imu_groundtruth as gt
        automaticos = gt.indices_auxiliares_automaticos()
        if automaticos:
            return [int(indice) for indice in automaticos]
    except Exception:                   # noqa: BLE001 - enumeracao e' diagnostico
        pass
    try:
        # `camera_names` e' leve (ctypes/comtypes) e aqui se importa direto: o
        # `nomes_mod` do modulo so' existe DEPOIS (import tardio na hora de
        # abrir a camera).
        import camera_names
        externas = camera_names.indices_fora_do_laptop()
        if externas:
            return [int(indice) for indice in externas]
    except Exception:                   # noqa: BLE001 - idem
        pass
    return list(AUX_CAMERA_INDICES_PADRAO)


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
                        default=None, metavar="N",
                        help="Indice(s) da(s) webcam(s) auxiliar(es) a "
                             "calibrar, na ordem. PADRAO = AUTOMATICO: as "
                             "webcams EXTERNAS desta maquina pelo NOME do "
                             "Windows (todas menos a do laptop; as que ja' tem "
                             "calibracao primeiro). Fixe para forcar (ex.: "
                             "--aux-index 1 0). Salva em "
                             "stereo_calibration_aux{N}.npz (0 usa o arquivo "
                             "legado stereo_calibration.npz).")
    parser.add_argument("--intrinsecas-kinect", type=str, default=None,
                        metavar="ARQUIVO",
                        help="Modelo CONGELADO do Kinect a usar (padrao: "
                             "kinect_intrinsics.npz ao lado do codigo, quando ele "
                             "existe). Com o modelo congelado o Kinect NAO e' "
                             "re-estimado: o stereo resolve so' R/T -- e' isso "
                             "que impede o overlay de sair dezenas de px fora "
                             "(ver o item 2 do docstring).")
    parser.add_argument("--sem-intrinsecas-kinect", action="store_true",
                        help="IGNORA o modelo congelado existente (comportamento "
                             "antigo: cada sessao re-estima o Kinect). Serve para "
                             "comparar A/B na mesma bancada.")
    parser.add_argument("--salvar-intrinsecas-kinect", action="store_true",
                        help=("Ao resolver, grava o modelo congelado do Kinect a "
                              "partir das poses desta sessao -- exige tabuleiro "
                              "GRANDE preenchendo o quadro (>= {span:.0f}%% da "
                              "diagonal, {poses} poses e RMS <= {rms} px), senao "
                              "recusa com o motivo medido. Rode UMA vez por "
                              "bancada, com o tabuleiro grande, antes das sessoes "
                              "de rotina. PREFIRA freeze_kinect_direto.py (arranjo "
                              "Kinect-only): aqui quem manda na distancia do "
                              "tabuleiro e' a auxiliar, que fecha o quadro primeiro "
                              "e deixa o span em 11-13%% -- abaixo do portao.")
                              .format(span=KINECT_INTRINSICS_MIN_SPAN * 100,
                                      poses=KINECT_INTRINSICS_MIN_POSES,
                                      rms=KINECT_INTRINSICS_MAX_RMS))
    return parser.parse_args()


def aplica_tabuleiro(checkerboard=None, quadrado_mm=None):
    """Aplica --checkerboard/--quadrado nas globais. Devolve (ok, mensagem).

    Existe para o arranjo Kinect-only (`freeze_kinect_direto.py`) aceitar os
    MESMOS overrides sem copiar a analise do texto: uma copia aqui seria a
    duplicata que ja' custou uma sessao inteira (MEDIDO em 01/10/2026: os dois
    modulos declarando placas diferentes, os npz recusados na leitura e o Kinect
    de volta ao re-estimado por sessao).
    """
    global CHECKERBOARD_COLS, CHECKERBOARD_ROWS, CHECKERBOARD_SIZE, SQUARE_SIZE_M
    if checkerboard:
        texto = checkerboard.lower().replace(" ", "")
        for separador in ("x", ",", "*"):
            if separador in texto:
                cols_txt, rows_txt = texto.split(separador, 1)
                break
        else:
            return False, (f"--checkerboard '{checkerboard}' nao esta' no "
                           "formato COLSxROWS (ex.: 5x3)")
        try:
            cols, rows = int(cols_txt), int(rows_txt)
        except ValueError:
            return False, (f"--checkerboard '{checkerboard}' nao esta' no "
                           "formato COLSxROWS (ex.: 5x3)")
        if cols < 3 or rows < 3:
            return False, f"--checkerboard {cols}x{rows} invalido (minimo 3x3)"
        CHECKERBOARD_COLS, CHECKERBOARD_ROWS = cols, rows
        CHECKERBOARD_SIZE = (cols, rows)
    if quadrado_mm:
        if quadrado_mm <= 0:
            return False, f"--quadrado {quadrado_mm} tem de ser > 0"
        SQUARE_SIZE_M = quadrado_mm / 1000.0
    return True, None


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
    pontos = np.asarray(cantos, np.float64).reshape(-1, 2).copy()
    altura, largura = frame.shape[:2]
    #: GUARDA DE BORDA (MEDIDA em 30/09/2026, aberta desde entao). Com um canto a
    #: MENOS de meia janela da borda o `cv2.cornerSubPix` do OpenCV 5 LEVANTA
    #: ("initial corner is outside the image") e a excecao sem tratamento encerra
    #: a sessao de captura inteira, sem mensagem final -- reproduzido ao vivo com
    #: o tabuleiro colado na camera (span 53-77%) e num caso isolado (canto em
    #: x=700 de uma imagem de 640 px). E' exatamente o quadro que o arranjo
    #: Kinect-only precisa: `freeze_kinect_direto.py` pede ao operador para levar
    #: o tabuleiro as BORDAS e aos CANTOS do quadro, porque e' isso que ensina a
    #: DISTORCAO do Kinect. Entao: os cantos COM folga sao refinados (o ganho de
    #: precisao continua) e os que nao tem ficam com o valor do detector -- que ja'
    #: entrega sub-pixel (test_deteccao_cascata.py mede a mesma precisao nas tres
    #: larguras da cascata). O pior caso deixa de ser "a sessao morreu" e passa a
    #: ser "esta amostra entrou com ~1,5 px a mais de ruido num canto", que a
    #: limpeza de outliers do solver ja' trata.
    margem = 6.0                     # meia janela (11x11 -> raio 5) + 1 px
    com_folga = ((pontos[:, 0] >= margem) & (pontos[:, 0] <= largura - 1 - margem)
                 & (pontos[:, 1] >= margem)
                 & (pontos[:, 1] <= altura - 1 - margem))
    if not com_folga.any():
        return pontos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        refinados = cv2.cornerSubPix(
            gray,
            pontos[com_folga].reshape(-1, 1, 2).astype(np.float32),
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    except cv2.error:
        #: Rede de seguranca: se o cv2 reclamar mesmo com a folga conferida, a
        #: amostra vale menos que a sessao.
        return pontos
    pontos[com_folga] = refinados.reshape(-1, 2).astype(np.float64)
    return pontos


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


def grade_de_avaliacao(profundidade=KINECT_EVAL_PROFUNDIDADE_M):
    """(N,3) pontos da regiao de trabalho, no referencial do Kinect (m).

    Serve para COMPARAR dois modelos do Kinect em pixel -- a unica medida que
    responde "este npz vai projetar no lugar certo nesta bancada?". O RMS nao
    responde: ele mede so' a coerencia interna do proprio arquivo.
    """
    xs = np.linspace(KINECT_EVAL_JANELA_X[0], KINECT_EVAL_JANELA_X[1],
                     KINECT_EVAL_PASSOS_X)
    ys = np.linspace(KINECT_EVAL_JANELA_Y[0], KINECT_EVAL_JANELA_Y[1],
                     KINECT_EVAL_PASSOS_Y)
    z = float(profundidade)
    return np.array([[x, y, z] for y in ys for x in xs], np.float64)


def compara_modelos_kinect(matriz_a, dist_a, matriz_b, dist_b):
    """(mediana, max) px entre dois modelos do Kinect RGB, na regiao de trabalho.

    Espelha `espalhamento_px` de tools\\diagnostico_calibracao.py (mesma grade e
    mesma profundidade -- travado por test_diagnostico_calibracao.py). Aquela
    ferramenta mede isso entre ARQUIVOS; aqui a mesma medida existe para o
    programa poder avisar em tempo de calibracao, sem depender de tools\\.
    """
    pontos = grade_de_avaliacao()
    px_a = _project_camera_points(pontos, np.asarray(matriz_a, np.float64),
                                  np.asarray(dist_a, np.float64).reshape(-1, 1))
    px_b = _project_camera_points(pontos, np.asarray(matriz_b, np.float64),
                                  np.asarray(dist_b, np.float64).reshape(-1, 1))
    diferencas = np.linalg.norm(px_a - px_b, axis=1)
    return float(np.median(diferencas)), float(diferencas.max())


def carrega_intrinsecas_kinect(caminho=None):
    """(matriz, distorcao, info) do modelo CONGELADO do Kinect, ou (None, None, info).

    Nunca levanta: arquivo ausente, npz sem os campos ou modelo fraco viram
    (None, None, {"motivo": ...}) e o programa segue no caminho antigo. O modelo
    congelado e' melhoria, nunca pre-requisito para calibrar.

    O QUE DECIDE (e o que NAO decide mais):

    * a PLACA em que o modelo foi gravado NAO decide nada -- ela vira AVISO em
      `info["aviso_placa"]`. As intrinsecas sao propriedade da CAMERA: trocar o
      tabuleiro impresso (foi o que aconteceu em 01/10/2026, 5x3/57 mm ->
      7x5/45 mm) nao muda fx/fy/cx/cy/distorcao de camera nenhuma. Recusar por
      placa transformava uma troca de placa em "toda sessao volta a RE-ESTIMAR o
      Kinect" -- exatamente o bug que o congelamento existe para matar, MEDIDO
      nos dois npz de 01/10/2026 (68,8 px de divergencia entre dois arquivos da
      MESMA bancada).
    * quem decide sao os MESMOS portoes da gravacao (`portoes_do_modelo_kinect`):
      numero de poses, cobertura do quadro (span) e RMS. E' o SPAN -- nao a placa
      -- que diz se um modelo do Kinect presta: com o tabuleiro pequeno no quadro
      o calibrateCamera recebe geometria de menos e devolve um modelo diferente a
      cada sessao.
    * um arquivo SEM `poses`/`span` e' recusado: ele foi gravado por um caminho
      que nao media a cobertura, entao nao ha' como saber se o modelo presta (e'
      o caso do paliativo de 30/09/2026).
    """
    caminho = Path(caminho) if caminho is not None else KINECT_INTRINSICS_FILE
    if not caminho.exists():
        return None, None, {"motivo": f"{caminho.name} nao existe"}
    try:
        dados = np.load(caminho, allow_pickle=True)
    except (OSError, ValueError) as exc:
        return None, None, {"motivo": f"nao consegui ler {caminho.name}: {exc}"}
    if "kinect_matrix" not in dados.files or "kinect_dist" not in dados.files:
        return None, None, {"motivo": f"{caminho.name} sem kinect_matrix/dist"}

    def escalar(nome):
        if nome not in dados.files:
            return None
        valor = np.asarray(dados[nome]).reshape(-1)
        return float(valor[0]) if valor.size else None

    def texto(nome):
        return (str(np.asarray(dados[nome]).reshape(-1)[0])
                if nome in dados.files else "")

    poses = escalar("poses")
    info = {
        "arquivo": str(caminho),
        "rms": escalar("rms"),
        "poses": int(poses) if poses is not None else None,
        "span": escalar("span"),
        "congelado_em": texto("frozen_at"),
        #: Resolucao de origem do modelo: fx/cx so' valem NAQUELA resolucao.
        "kinect_size": (tuple(int(v) for v in
                              np.asarray(dados["kinect_size"]).reshape(-1))
                        if "kinect_size" in dados.files else None),
    }
    # --- OS MESMOS portoes da gravacao (nunca uma regra paralela) ------------
    ok, motivo = aceita_modelo_kinect(info["poses"], info["span"], info["rms"])
    if not ok:
        return None, None, {"motivo": f"{caminho.name}: {motivo}"}
    # --- PLACA: AVISO (ver a docstring: as intrinsecas sao da CAMERA) --------
    avisos = []
    if "checkerboard_size" in dados.files:
        tabuleiro = tuple(int(v) for v in
                          np.asarray(dados["checkerboard_size"]).reshape(-1))
        if tabuleiro != tuple(CHECKERBOARD_SIZE):
            avisos.append(
                "gravado com tabuleiro %dx%d != %dx%d da placa atual (as "
                "intrinsecas sao da CAMERA e nao mudam com a placa: segue "
                "valendo)" % (tabuleiro + tuple(CHECKERBOARD_SIZE)))
    if "square_size_m" in dados.files:
        quadrado = float(np.asarray(dados["square_size_m"]).reshape(-1)[0])
        if abs(quadrado - SQUARE_SIZE_M) > 1e-6:
            avisos.append(
                "gravado com quadrado de %g mm != %g mm da placa atual (idem: "
                "segue valendo)" % (quadrado * 1000.0, SQUARE_SIZE_M * 1000.0))
    info["aviso_placa"] = " | ".join(avisos)
    return (np.asarray(dados["kinect_matrix"], np.float64).reshape(3, 3),
            np.asarray(dados["kinect_dist"], np.float64).reshape(-1), info)


def salva_intrinsecas_kinect(matriz, distorcao, kinect_size, rms, poses, span,
                            caminho=None):
    """Grava o modelo congelado do Kinect (ver docstring, item 2).

    `caminho` = arquivo de saida (padrao `KINECT_INTRINSICS_FILE`). Existe para o
    arranjo Kinect-only (`freeze_kinect_direto.py --saida`) poder gravar num
    arquivo de teste sem tocar no modelo bom da bancada.

    `poses` e `span` NAO sao decoracao: sao os portoes que a LEITURA refaz
    (`portoes_do_modelo_kinect`). Um arquivo sem eles e' recusado na leitura, por
    mais bonito que o RMS seja.
    """
    destino = Path(caminho) if caminho is not None else KINECT_INTRINSICS_FILE
    np.savez(
        destino,
        version=KINECT_INTRINSICS_VERSION,
        kinect_matrix=np.asarray(matriz, np.float64).reshape(3, 3),
        kinect_dist=np.asarray(distorcao, np.float64).reshape(-1),
        kinect_size=np.asarray([int(v) for v in kinect_size]),
        rms=float(rms),
        poses=int(poses),
        span=float(span),
        checkerboard_size=np.asarray(CHECKERBOARD_SIZE),
        square_size_m=float(SQUARE_SIZE_M),
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"Modelo do Kinect CONGELADO em {destino.name}: "
          f"RMS {rms:.3f} px | {poses} poses | tabuleiro a {span:.0%} da diagonal. "
          "A partir de agora TODA calibracao de par reusa este modelo.")


def span_do_tabuleiro(pontos_kinect, kinect_size):
    """Fracao MEDIANA da diagonal do quadro do Kinect que o tabuleiro ocupou.

    E' a medida que os portoes do modelo congelado usam (ver
    `portoes_do_modelo_kinect`): com o tabuleiro pequeno no quadro o
    calibrateCamera tem parametros demais para a geometria que recebeu, e o
    modelo do Kinect sai diferente a cada sessao -- foi MEDIDO em 30/09/2026
    (24 a 50 px na projecao do mesmo ponto 3D, ate' 176 px na borda).
    """
    spans = [board_span(np.asarray(pontos, np.float64).reshape(-1, 2))
             for pontos in pontos_kinect]
    diagonal = float(np.hypot(*kinect_size))
    return float(np.median(spans)) / diagonal if spans else 0.0


def portoes_do_modelo_kinect(poses, span, rms=None):
    """(ok, motivo) dos TRES portoes do modelo congelado -- funcao PURA.

    Usada DE PROPOSITO nos DOIS lados: ao GRAVAR
    (`talvez_salvar_intrinsecas_kinect` e `freeze_kinect_direto.py`) e ao
    CARREGAR (`carrega_intrinsecas_kinect`). Se os dois lados usassem regras
    diferentes existiria o arquivo que o proprio programa escreve e depois
    recusa -- e o Kinect voltaria a ser re-estimado por sessao, que e' o bug que
    o congelamento existe para matar.

    `rms=None` = ainda NAO medido, e o portao do RMS e' pulado: o caminho de
    gravacao mede o RMS DEPOIS de saber que ha' poses e cobertura, porque o
    calibrateCamera e' caro e nao vale gastar com o tabuleiro pequeno (que seria
    recusado logo em seguida).
    """
    if poses < KINECT_INTRINSICS_MIN_POSES:
        return False, (
            f"{poses} poses (minimo {KINECT_INTRINSICS_MIN_POSES}) -- junte "
            "mais poses")
    if span < KINECT_INTRINSICS_MIN_SPAN:
        return False, (
            f"o tabuleiro ocupou {span:.0%} da diagonal do quadro (minimo "
            f"{KINECT_INTRINSICS_MIN_SPAN:.0%}) -- use um tabuleiro GRANDE e "
            "chegue mais perto do Kinect: o modelo dele e' fragil por falta de "
            "cobertura, e foi esse o motivo de ele ser congelado")
    if rms is not None and rms > KINECT_INTRINSICS_MAX_RMS:
        return False, (
            f"RMS {rms:.3f} px acima de {KINECT_INTRINSICS_MAX_RMS} px -- "
            "melhore luz/nitidez (o RMS de uma camera so', com o tabuleiro "
            "grande, tem de ser bem baixo)")
    return True, None


def aceita_modelo_kinect(poses, span, rms=None):
    """(ok, motivo) de um modelo congelado -- a decisao COMPLETA, num lugar so'.

    Junta as duas coisas que decidem: os campos obrigatorios (`poses` e `span`,
    sem os quais nao ha' como saber a cobertura do quadro) e os tres portoes.
    Existe para quem LE' o arquivo POR FORA (`tools/diagnostico_calibracao.py`)
    poder dar o MESMO veredito que a calibracao, em vez de reimplementar a regra.
    Foi uma regra paralela -- a recusa por placa -- que deixou o diagnostico
    dizer "modelo congelado do Kinect: SIM" enquanto a calibracao re-estimava o
    Kinect em silencio (01/10/2026).
    """
    faltando = [campo for campo, valor in (("poses", poses), ("span", span))
                if valor is None]
    if faltando:
        return False, ("nao registra %s: gravado por um caminho que nao mede a "
                       "cobertura do quadro, entao nao ha' como saber se o "
                       "modelo presta -- refaca com freeze_kinect_direto.py "
                       "(arranjo Kinect-only) ou com stereo_calibration.py "
                       "--salvar-intrinsecas-kinect" % " nem ".join(faltando))
    return portoes_do_modelo_kinect(poses, span, rms)


def resolve_modelo_kinect(obj, kin, kinect_size):
    """(rms, matriz, distorcao, poses, span) do modelo do Kinect destas poses.

    Unico lugar que sabe COMO o modelo congelado e' calculado, para o arranjo
    Kinect-only (`freeze_kinect_direto.py`) e o caminho do par nao divergirem.
    Nao decide nada: quem decide sao os portoes (`portoes_do_modelo_kinect`).
    """
    poses = len(obj)
    span = span_do_tabuleiro(kin, kinect_size)
    rms, matriz, distorcao, _, _ = calibrate_single(
        _as_float32_points(obj), _as_float32_points(kin), kinect_size)
    return rms, matriz, distorcao, poses, span


def aceita_pose_kinect(cantos, pontos_ja, kinect_size,
                       min_span=MIN_BOARD_SPAN_FRACTION,
                       max_span=KINECT_INTRINSICS_MAX_SPAN,
                       min_separacao=MIN_POSE_SEPARATION):
    """(ok, motivo, fracao) -- portao de CAPTURA das poses do Kinect so'.

    Tres regras, as MESMAS da captura do par onde elas existem
    (MIN_BOARD_SPAN_FRACTION e MIN_POSE_SEPARATION), na forma de funcao pura para
    poder ser provada sem hardware:

    * SPAN MINIMO -- o tabuleiro tem de ocupar ao menos `min_span` da diagonal do
      quadro. A meta de verdade e' KINECT_INTRINSICS_MIN_SPAN (35%), mas aqui a
      porta e' mais frouxa: uma pose distante ainda ajuda na DISTORCAO, e recusar
      a captura seria perder trabalho. Quem reprova o conjunto no fim e' o portao
      do modelo.
    * SPAN MAXIMO -- `max_span` (ver KINECT_INTRINSICS_MAX_SPAN): tabuleiro
      colado na camera sai de foco e domina fx.
    * SEPARACAO -- o centro do tabuleiro tem de estar a >= `min_separacao` do
      span de distancia de TODAS as poses ja' coletadas. Sem isto, 12 amostras de
      UMA pose nao cobrem geometria nenhuma: o calibrateCamera devolve um RMS
      otimo com intrinsecas erradas -- o pior caso possivel para algo que vai ser
      CONGELADO e passar a valer para todas as sessoes seguintes.
    """
    span = board_span(cantos)
    diagonal = float(np.hypot(*kinect_size))
    fracao = span / diagonal if diagonal else 0.0
    if fracao < min_span:
        return False, ("tabuleiro a %.0f%% da diagonal do quadro (minimo %.0f%%)"
                       " -- chegue mais perto do Kinect"
                       % (fracao * 100.0, min_span * 100.0)), fracao
    if max_span and fracao > max_span:
        return False, ("tabuleiro a %.0f%% da diagonal do quadro (maximo %.0f%%)"
                       " -- AFASTE do Kinect: mais perto que isto o RGB fica "
                       "fora de foco e o tabuleiro domina o ajuste de fx"
                       % (fracao * 100.0, max_span * 100.0)), fracao
    if pontos_ja:
        centros = np.array([np.asarray(pontos, np.float64).reshape(-1, 2)
                            .mean(axis=0) for pontos in pontos_ja])
        centro = np.asarray(cantos, np.float64).reshape(-1, 2).mean(axis=0)
        separacao = float(np.linalg.norm(centros - centro, axis=1).min())
        if separacao < min_separacao * span:
            return False, ("pose quase igual a uma ja' coletada (%.0f px) -- "
                           "MEXA no tabuleiro: mude distancia, inclinacao ou "
                           "posicao" % separacao), fracao
    return True, None, fracao


def talvez_salvar_intrinsecas_kinect(obj, kin, kinect_size):
    """Grava o modelo congelado do Kinect se a SESSAO tiver dados para isso.

    Tres portas de qualidade, cada uma negando com o numero medido:
    poses minimas, cobertura do quadro (span MEDIANO do tabuleiro) e RMS
    individual. Congelar um modelo ruim seria PIOR que nao congelar: ele passa a
    valer, calado, para TODAS as sessoes seguintes.

    As tres portas moram em `portoes_do_modelo_kinect` porque a LEITURA usa
    exatamente as mesmas (ver `carrega_intrinsecas_kinect`).
    """
    poses = len(obj)
    span = span_do_tabuleiro(kin, kinect_size)
    ok, motivo = portoes_do_modelo_kinect(poses, span)
    if not ok:
        print(f"Intrinsecas do Kinect NAO gravadas: {motivo}.")
        return False
    rms, matriz, distorcao, _, _ = calibrate_single(
        _as_float32_points(obj), _as_float32_points(kin), kinect_size)
    ok, motivo = portoes_do_modelo_kinect(poses, span, rms)
    if not ok:
        print(f"Intrinsecas do Kinect NAO gravadas: {motivo}.")
        return False
    salva_intrinsecas_kinect(matriz, distorcao, kinect_size, rms, poses, span)
    return True


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
    ref_kinect=None,
):
    """Roda uma solucao completa (individual + stereo) e devolve um dict.

    `ref_kinect` = (matriz, distorcao) do MODELO CONGELADO do Kinect (ver
    `carrega_intrinsecas_kinect`). Quando ele vem, o Kinect NAO e' estimado nem
    refinado nesta sessao: usa-se o modelo congelado e o stereo resolve so' R/T
    (`refine_intrinsics` e' ignorado). O modelo estimado na sessao ainda e'
    calculado, mas so' para MEDIR o desvio (`kinect_deviation_px`): e' o numero
    que diz quanto aquele modelo teria custado em px no overlay.
    """
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
    desvio_px = None
    congelado = ref_kinect is not None
    if congelado:
        referencia = tuple(np.asarray(valor, np.float64) for valor in ref_kinect)
        if len(referencia) == 2:
            desvio_px = compara_modelos_kinect(
                kinect_matrix, kinect_dist, referencia[0], referencia[1])
            kinect_matrix = referencia[0].reshape(3, 3).copy()
            kinect_dist = referencia[1].reshape(-1).copy()
            # Com o modelo congelado nao ha' o que refinar no Kinect, e o
            # candidato "refinado" so' sabe vencer no RMS (que nao mede o
            # problema) -- ver o item 2 do docstring.
            refine_intrinsics = False
        else:
            congelado = False
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
        "kinect_frozen": bool(congelado),
        "kinect_deviation_px": desvio_px,
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


def solve_best(object_points, kinect_points, auxiliary_points, kinect_size,
               auxiliary_size, ref_kinect=None):
    """Resolve com intrinsecas congeladas e com refinadas; guarda a melhor.

    Com `ref_kinect` (modelo do Kinect congelado) ha' UMA candidata so': o
    refinamento conjunto das intrinsecas deixa de existir, porque ele re-estima
    justamente o modelo que NAO pode mudar (o Kinect e' a mesma camera) e tem
    vantagem falsa no RMS.
    """
    if ref_kinect is not None:
        return run_stereo(
            object_points,
            kinect_points,
            auxiliary_points,
            kinect_size,
            auxiliary_size,
            refine_intrinsics=False,
            ref_kinect=ref_kinect,
        )
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
                    ref_kinect=None,
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
    ref_kinect=None,
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
       descartadas. Termina com o solve final, que com `ref_kinect`
       (modelo CONGELADO do Kinect) resolve so' R/T.
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
                    *subset(trial), kinect_size, auxiliary_size,
                    refine_intrinsics=False, ref_kinect=ref_kinect,
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
        seed_result = run_stereo(obj, kin, aux, kinect_size, auxiliary_size,
                                 refine_intrinsics=False, ref_kinect=ref_kinect)

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
        model = run_stereo(obj, kin, aux, kinect_size, auxiliary_size,
                           refine_intrinsics=False, ref_kinect=ref_kinect)

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
        model = run_stereo(obj, kin, aux, kinect_size, auxiliary_size,
                           refine_intrinsics=False, ref_kinect=ref_kinect)
        errors = pose_rms(model, obj, kin, aux, model["kinect_rvecs"], model["kinect_tvecs"])

    # ---- solve final ----
    obj, kin, aux = subset(active)
    final = solve_best(obj, kin, aux, kinect_size, auxiliary_size,
                       ref_kinect=ref_kinect)
    errors = pose_rms(final, obj, kin, aux)
    return final, name_map, active, errors, obj, kin, aux


def print_solve_report(final, errors, active, spans, tilt_ratios):
    mode = "intrinsecas refinadas" if final["refine_intrinsics"] else "intrinsecas congeladas"
    print(
        f"RMS stereo={final['stereo_rms']:.4f} px ({mode}) | "
        f"RMS Kinect={final['kinect_rms']:.4f} px | RMS auxiliar={final['auxiliary_rms']:.4f} px"
    )
    if final.get("kinect_frozen"):
        desvio = final.get("kinect_deviation_px")
        texto = ("--" if desvio is None
                 else "mediana %.1f px / max %.1f px" % desvio)
        print("  Kinect CONGELADO (kinect_intrinsics.npz): o modelo que ESTA "
              "sessao estimaria diverge %s dele" % texto)
        if desvio is not None and desvio[0] > 10.0:
            print("  ... divergencia grande = o tabuleiro desta sessao NAO "
                  "sustenta um modelo do Kinect (pouca cobertura do quadro). E' "
                  "por isso que ele fica congelado: sem congelar, esse modelo "
                  "iria para o overlay como erro sistematico (ver docstring).")

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


def save_calibration(final, active, errors, obj, kin, aux, camera_name=""):
    """Grava o stereo da camera auxiliar ativa em stereo_calibration_aux{N}.npz.

    `camera_name` e' o NOME do Windows da camera deste indice (`camera_names`):
    o indice do OpenCV e' a POSICAO na enumeracao do DirectShow e muda quando uma
    webcam entra/sai do USB, entao sem o nome o arquivo `_aux0` pode ser
    carregado pela camera errada -- e o unico sintoma seria um RMS lindo com a
    sobreposicao centenas de px fora (foi o que aconteceu em 28/09/2026, com um
    npz gravado em OUTRA maquina). O campo e' OPCIONAL na leitura: npz antigo
    (sem ele) segue sendo aceito por placa + quadrado + RMS.
    """
    output = output_file(AUX_CAMERA_INDEX)
    np.savez(
        output,
        # `camera_name` e' a unica forma de dizer A QUAL camera o npz pertence.
        camera_name=str(camera_name or ""),
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
        # DE ONDE veio o modelo do Kinect gravado neste npz: "congelado"
        # (kinect_intrinsics.npz, o caminho recomendado) ou "sessao" (estimado
        # nesta calibracao -- ver o item 2 do docstring). `kinect_deviation_px`
        # e' o quanto o modelo da sessao divergia do congelado (-1 = nao medido):
        # e' a estimativa do erro sistematico que o overlay teria.
        kinect_intrinsics_source=("congelado" if final.get("kinect_frozen")
                                  else "sessao"),
        kinect_deviation_px=(-1.0 if final.get("kinect_deviation_px") is None
                             else float(final["kinect_deviation_px"][0])),
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


def motivo_do_sensor_mudo(espera_s, tentativas):
    """Texto unico para "abriu e nao entregou frame" -- usado por todos os callers.

    As tres causas TEM sintomas diferentes e a mensagem separa: (1) outro programa
    com o sensor aberto (o caso do Kinect Studio, MEDIDO em 07/10/2026); (2) o
    link USB caiu para 2.0 / faltou banda, em que o driver carrega, o runtime abre
    e nada transmite; (3) energia -- sem a fonte propria o sensor nao acorda.
    Nao adianta mandar "confira o cabo" sem dizer o que muda em cada caso.
    """
    return (
        "O Kinect ABRIU mas NAO entregou frame em %.0f s (tentei reabrir %d "
        "vezes). Ele e' obrigatorio (o par e' Kinect RGB + webcam auxiliar). O "
        "driver esta' de pe' -- o que falta e' o sensor TRANSMITIR. Em ordem de "
        "probabilidade:\n"
        "    1) algum programa esta' com o sensor: FECHE e confira na barra do "
        "Windows o que ficou vivo -- %s. 'Fechei o Kinect Studio' nao basta se o "
        "watch/gravador dele continuar aberto; se aparecer 'usbccgp' ou o sensor "
        "sem frame, DESLIGUE e RELIGUE a fonte do Kinect.\n"
        "    2) link USB caiu para 2.0 (o runtime abre e nada sai): use porta "
        "USB 3.0 AZUL direto na placa-mae -- sem hub, sem extensao, sem painel "
        "frontal (o v2 precisa das duas faixas de banda; as webcams continuam "
        "funcionando porque sao outra coisa).\n"
        "    3) energia: a fonte propria do Kinect tem de estar ligada ANTES do "
        "USB; sem ela o sensor aparece enumerado e mudo.\n"
        "  Diagnostico rapido (nao exige tabuleiro): "
        ".venv\\Scripts\\python.exe tools\\checa_kinect.py\n"
        "  Se o proprio Kinect Studio mostra imagem e o SDK nao, o passo que "
        "decide e' o `sensor.IsAvailable` (que o pykinect2 nunca le') e o "
        "checklist do que ja' foi descartado esta' em "
        "`tools\\checklist_kinect_mudo.py` -- suspensao seletiva de USB, "
        "porta/controladora, fonte, reinstalar o dispositivo e o verificador "
        "oficial da Microsoft."
        % (espera_s, tentativas, ", ".join(APLICATIVOS_QUE_SEGURAM_O_KINECT)))


def abre_kinect(espera_s=ESPERA_KINECT_S, tentativas=None):
    """Abre o Kinect RGB e ESPERA o primeiro frame antes de seguir.

    O Kinect e' OBRIGATORIO: o par estereo e' (Kinect RGB, auxiliar), nao ha'
    como calibrar so' com a auxiliar. Sem esta espera o sintoma e' cruel, e foi
    o que aconteceu na bancada: se o Kinect nao entrega frame, o loop principal
    faz `continue` ANTES do `cv2.imshow` e NENHUMA JANELA APARECE. O programa
    fica mudo, com cara de travado, sem dizer que o problema e' a camera.
    Aqui ele falha ALTO e diz exatamente o que conferir.

    `tentativas` = quantas vezes REABRIR o runtime antes de desistir. Reabrir
    recupera o sensor que ficou mudo depois de outro programa soltar a camera
    (medido em 07/10/2026, ver `TENTATIVAS_KINECT`). Esperas curtas (teste, sonda)
    NUNCA reabrem: com `espera_s <= 2` vale uma tentativa so', para o teste nao
    virar uma espera de minutos e para o numero de consultas ao fake ficar
    previsivel.
    """
    if tentativas is None:
        tentativas = TENTATIVAS_KINECT if espera_s > 2.0 else 1
    tentativas = max(1, int(tentativas))
    print("  Kinect: abrindo o RGB (pode levar alguns segundos)...")
    ultimo_erro = None
    for tentativa in range(1, tentativas + 1):
        try:
            runtime = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        except Exception as exc:
            raise RuntimeError(
                "Nao consegui abrir o Kinect v2. Confira: cabo USB 3.0 (porta "
                "azul) DIRETO na maquina (sem hub/extensor), SDK do Kinect v2 "
                "instalado, e nenhum outro programa (Kinect Studio, ferramenta "
                "de ground truth) usando a camera. Erro: %s" % (exc,)) from exc
        inicio = time.time()
        while time.time() - inicio < espera_s:
            if runtime.has_new_color_frame():
                if tentativa > 1:
                    print("  Kinect: primeiro frame na %da tentativa (%.1f s) -- "
                          "o sensor estava preso e a REABERTURA resolveu."
                          % (tentativa, time.time() - inicio))
                else:
                    print("  Kinect: primeiro frame em %.1f s"
                          % (time.time() - inicio))
                return runtime
            time.sleep(0.005)
        ultimo_erro = motivo_do_sensor_mudo(espera_s, tentativa)
        try:
            runtime.close()
        except Exception:                       # noqa: BLE001 - fechar e' higiene
            pass
        if tentativa < tentativas:
            print("  Kinect: sem frame em %.0f s (tentativa %d/%d) -- fechando e "
                  "reabrindo o runtime..." % (espera_s, tentativa, tentativas))
    raise RuntimeError(ultimo_erro)


def calibrate_one(aux_index, ref_kinect=None, salvar_intrinsecas=False):
    """Calibra o par (Kinect RGB, webcam auxiliar `aux_index`).

    `ref_kinect` = (matriz, distorcao) do MODELO CONGELADO do Kinect: quando vem,
    o Kinect NAO e' estimado nem refinado nesta sessao (ver o item 2 do
    docstring) e o npz grava esse modelo, junto com o desvio medido da sessao.
    `salvar_intrinsecas` = ao resolver, tenta GRAVAR o modelo congelado a partir
    das poses desta sessao (so' passa quando o tabuleiro preenche o quadro).

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
                object_points, kinect_points, auxiliary_points, kinect_size,
                auxiliary_size, ref_kinect=ref_kinect
            )
            print_solve_report(final, errors, active, kinect_spans, tilt_ratios)
            if salvar_intrinsecas:
                talvez_salvar_intrinsecas_kinect(final_obj, final_kin, kinect_size)
            if final["stereo_rms"] < best_rms:
                best_rms = final["stereo_rms"]
                save_calibration(final, active, errors, final_obj, final_kin,
                                 final_aux, camera_name=nome_real or "")
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
    pronto, mensagem = aplica_tabuleiro(args.checkerboard, args.quadrado)
    if not pronto:
        print(f"ERRO: {mensagem}.")
        return 2
    print(f"Tabuleiro: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS} cantos "
          f"internos, quadrado de {SQUARE_SIZE_M * 1000:g} mm")
    DETECTION_WIDTHS = tuple(int(v) for v in args.largura_deteccao)
    DETECTION_CLAHE_CLIP = float(args.clahe)
    DETECTION_SB_FLAGS = 0 if args.rapido else cv2.CALIB_CB_EXHAUSTIVE
    print(f"Detector: larguras {DETECTION_WIDTHS} (0 = cheia) | "
          f"CLAHE {DETECTION_CLAHE_CLIP:g} | flags {DETECTION_SB_FLAGS}")
    #: dict.fromkeys preserva a ordem e remove indices repetidos.
    if args.aux_index is None:
        args.aux_index = indices_padrao_auxiliares()
        print(f"Webcams auxiliares desta maquina (pelo NOME): {args.aux_index} "
              "-- use --aux-index N para forcar outros indices")
    indices = list(dict.fromkeys(int(i) for i in args.aux_index))
    # --- MODELO CONGELADO do Kinect (ver o item 2 do docstring) -------------
    referencia = None
    if args.sem_intrinsecas_kinect:
        print("Modelo do Kinect: --sem-intrinsecas-kinect -> cada sessao "
              "re-estima o Kinect (comportamento antigo).")
    else:
        matriz_ref, dist_ref, info_ref = carrega_intrinsecas_kinect(
            args.intrinsecas_kinect)
        if matriz_ref is None:
            print(f"Modelo do Kinect: SEM modelo congelado ({info_ref['motivo']}).")
            print("  O Kinect sera' estimado/refinado NESTA sessao. E' assim que "
                  "cada sessao termina com um modelo do Kinect diferente -- e esse "
                  "erro NAO aparece no RMS (que so' mede a coerencia interna do "
                  "npz), ele aparece como dezenas de px no overlay. Para congelar "
                  "(recomendado, UMA vez por bancada):")
            print("    python freeze_kinect_direto.py     <- arranjo Kinect-only:"
                  " o tabuleiro chega a 0,3 m e o span passa (RECOMENDADO)")
            print("    python stereo_calibration.py --salvar-intrinsecas-kinect"
                  "   (o mesmo, com a auxiliar na sessao)")
        else:
            referencia = (matriz_ref, dist_ref)
            print(f"Modelo do Kinect CONGELADO: {info_ref['arquivo']}")
            if info_ref["rms"] is not None:
                print("  RMS %.3f px | %s poses | tabuleiro a %.0f%% da diagonal "
                      "| resolucao %s | congelado em %s"
                      % (info_ref["rms"], info_ref["poses"],
                         (info_ref["span"] or 0.0) * 100.0,
                         "x".join(str(v) for v in info_ref["kinect_size"])
                         if info_ref["kinect_size"] else "?",
                         info_ref["congelado_em"]))
            if info_ref.get("aviso_placa"):
                print(f"  AVISO de placa: {info_ref['aviso_placa']}")
            print("  -> o Kinect NAO sera' re-estimado: o stereo resolve so' R/T")
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
            if calibrate_one(aux_index, ref_kinect=referencia,
                             salvar_intrinsecas=args.salvar_intrinsecas_kinect) != 0:
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
