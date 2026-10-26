"""Ferramenta autonoma de calibracao/diagnostico do ground truth 3D da mao.

Este programa era o `main()` de `kinect_imu_groundtruth.py`; foi movido para
`tools/` (itens #9/#10 da revisao) para deixar a biblioteca como biblioteca --
antes, importar o modulo trazia junto um programa de 700+ linhas.

Uso (da raiz do projeto, que e' onde ficam as calibracoes):

    python tools/kinect_groundtruth_tool.py [--aux-index N] [--aux-cameras "N,M"]
                                            [--salvar-quadros S]
                                            [--escala-janelas F]
                                            [--imu-portas "P_D,P_E"]

`--aux-index` escolhe a webcam auxiliar ATIVA (a das teclas K/L/H, do trim e da
calibracao do esqueleto): a calibracao e' POR INDICE (`stereo_calibration_aux{N}
.npz`), entao o indice escolhido define camera e calibracao juntas. ATENCAO: o
indice do OpenCV e' a POSICAO da camera na enumeracao do DirectShow e MUDA quando
uma camera entra/sai do USB -- foi assim que "a auxiliar" virou a webcam do
LAPTOP (a janela mostrava o rosto de quem esta' no laptop). Quem CONFERE o mapa
no dia, pelo NOME do Windows, e' tools/diagnostico_cameras.py --salvar (imprime
`idx N: <nome>` e grava cam_idx{N}.png). Este programa avisa quando o indice
escolhido e' a do laptop.

`--aux-cameras` (vazio = 'auto') liga as OUTRAS auxiliares -- por padrao todas as
webcams MENOS a do laptop, identificadas pelo NOME. Todas as auxiliares abertas
tem o esqueleto da mao DESENHADO no video do Kinect, cada uma na sua cor e com
rotulo: na janela aparecem, no MESMO quadro, o esqueleto do KINECT (verde) + o de
CADA webcam. E' o teste visual da calibracao de cada camera e a base do par que
SUBSTITUI o Kinect: quando o Kinect perde a mao, as DUAS auxiliares, sozinhas,
triangulam a posicao 3D (colunas `hand_source`/`aux_pair_*` do CSV dizem qual
fonte foi usada -- `trianguladoAux` = par de auxiliares).

`--salvar-quadros S` grava _aux_overlay.png / _kinect_overlay.png a cada S
segundos com os quadros JA' sobrepostos (e imprime o desvio vermelho-verde de
cada mao em px) -- serve para conferir depois se a projecao caiu na mao.

JANELAS DE VIDEO: uma por camera, todas visiveis ao mesmo tempo. Sao 4 fluxos
(RGB e depth do Kinect + uma janela por webcam auxiliar) e a grade e' calculada
a partir do tamanho REAL da tela, com a escala de exibicao reduzida o necessario
para todas caberem (o WINDOW_SCALE de 0.5 e' o teto; em telas com DPI alto ela
cai para ~0.4). `--escala-janelas F` fixa a escala (0.5 = janelas maiores, mas
voce arrasta as que ficarem uma sobre a outra). O titulo de cada janela tem o
indice e o NOME do Windows da camera; a camera ATIVA (teclas K/L/H) diz "ATIVA"
no quadro, porque trim, log de desvio e calibracao do esqueleto valem so' nela.

Teclas: C origem | B bias do acelerometro | O alinhamento da palma (IMU <-> camera,
por luva) | D diagnostico da palma (3 poses programadas) | T diagnostico do
tabuleiro | K trim fixo do overlay | H calibracao do esqueleto da mao | L log de
desvio | R reset | ESC sair. Precisa do Kinect v2 (SDK 2.0) e, opcionalmente, da
webcam auxiliar e do ESP32/UDP.

FOCO DA JANELA: o waitKey do OpenCV so' entrega tecla com uma janela de VIDEO
do programa em foco -- se a tecla O "nao faz nada" e o console nao imprime
"Captura do alinhamento da palma por 14 s", clique primeiro no video (o foco
estava no terminal/VS Code, que engole a tecla).

VETOR DA PALMA (por que a tecla O existe): a seta vermelha da palma e' prevista
pelo IMU da LUVA, e o referencial do IMU (vertical do acelerometro, mas "norte"
arbitrario -- nao ha' magnetometro) NAO e' o da camera, e ainda muda com a
MONTAGEM de cada luva. Compor as rotacoes sem converter da' um vetor certo na
pose de calibracao e errado conforme a mao gira -- e o erro e' DIFERENTE em cada
mao (era a queixa "a luva da mao esquerda da' valores de rotacao errados").
A tecla O resolve isso MEDINDO: durante ~14 s gire a mao (palma virada para o
Kinect, sem virar o dorso) e o programa ajusta o alinhamento por luva
(`solve_palm_alignment`: Kabsch alternado em `v_i = T R_i f`), imprime o residuo
em graus e grava `imu_palm_alignment_{right,left}.json`, recarregado nas proximas
sessoes. O erro ao vivo ("Palma R/L: erro Kinect x IMU") fica no video, no CSV
(coluna `palm_err_deg`) e no log da tecla L -- e' o numero que diz se aquela luva
esta' boa. Residuo alto em um lado so' costuma ser luva trocada: use
`--imu-portas "4210,4211"` (ordem DIREITA,ESQUERDA) para casar com o firmware.

DIAGNOSTICO DA PALMA (tecla D): quando o erro ao vivo esta' alto e a tecla O nao
resolve, o numero sozinho nao diz se o problema e' SISTEMATICO (T/f errados,
arquivo de outra luva, normal do Kinect com sinal trocado) ou DEPENDENTE DA
ROTACAO (o alinhamento "decora" a pose da tecla O). A tecla D mede o erro contra
um MOVIMENTO PROGRAMADO de 4 fases (21 s): PREPARO (3 s, palmas para o Kinect,
parado), MESA (6 s, as duas palmas na BORDA da mesa, dorso para cima), REZANDO
(6 s, palmas uma para a outra MAS SEM SE ENCOSTAREM -- ~15 cm) e KINECT (6 s,
palmas abertas para o Kinect). As maos tem de ficar DENTRO do quadro do Kinect
nas quatro fases E SEM UMA COBRIR A OUTRA: a verdade e' a normal medida pelo
depth; medido em 19/10/2026 que, com as maos realmente juntas, a mao coberta
perde ~90% das amostras (e o par de cameras paralelas tambem passa a ver so' um
lado). A instrucao da fase aparece no video; D de novo (ou ESC) encerra e
resume o que ja' foi coletado. No comeco a ferramenta avisa, POR LUVA, se esta'
usando alinhamento SALVO (`imu_palm_alignment_<lado>.json`) ou o bootstrap de 1
amostra -- se avisar que nao ha' arquivo, rode a tecla O antes (foi a causa
medida em 19/10/2026: erro de 20-140 graus com a seta rodando sem alinhamento).
O CSV (`palm_diag_<data>_<hora>.csv`) grava uma linha por (frame, lado) com a
normal do Kinect, a previsao COM alinhamento (a seta), a previsao SEM
alinhamento e os dois erros; o resumo sai em
`palm_diag_<data>_<hora>.csv_resumo.txt` (tabela por fase/lado + veredito) e a
analise completa, com refit do modelo, em
`python tools\\analisa_diag_palma.py [csv]` (sem argumento pega o CSV COM
amostras mais recente, ignorando sessao abortada de 2 s).

Nao ha copia de codigo: o corpo abaixo e' o mesmo de antes e usa os nomes do
modulo importado (constantes, classes e helpers), copiados para o namespace
deste arquivo logo apos o import.
"""
from __future__ import annotations

import os
import sys

# A biblioteca vive na raiz do projeto (um nivel acima de tools/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kinect_imu_groundtruth as _gt  # noqa: E402

# Copia TODO o namespace do modulo (inclusive nomes com "_"), para o corpo
# abaixo enxergar exatamente os mesmos nomes de antes do movimento.
globals().update({nome: valor for nome, valor in vars(_gt).items()
                  if not nome.startswith("__")})

# Trava de exposicao/foco e o modo MJPEG das webcams (module proprio:
# camera_names.py e' deliberadamente "sem cv2"; camera_uvc.py e' o espelho
# enxuto dele -- la' esta' medido que o bruto cai a 8-10 fps, o MJPEG da 30 e
# que a ORDEM dos set's decide se a stream fica MJPG ou volta pra YUY2).
from camera_uvc import (define_modo, le_fourcc, parse_controle_por_camera,  # noqa: E402
                           trava_exposicao, valor_por_camera)

# Reancoragem do YAW DO BOOT (item 97): o yaw absoluto do IMU e' arbitrario a
# cada boot (filtro de 6 eixos sem magnetometro), entao o T de um alinhamento
# salvo so' vale no boot em que foi resolvido. O `AncoradorYawBoot` resolve o
# delta de yaw nas primeiras amostras com a palma "fora da vertical" e corrige
# o T EM MEMORIA (o JSON salvo continua valendo para qualquer boot).
from imu import AncoradorYawBoot  # noqa: E402

# Ritual de CALIBRACAO DOS IMUs no inicio da sessao (item 101): aviso por
# participante + reset do ESP32 pela serial (luvas paradas) + conferencia do
# bias parado pelo UDP. O bias do giroscopio e' calibrado no BOOT e depende de
# a luva estar parada (medido 21/10/2026: -12,5 dps = 434 graus/min quando o
# boot pegou a luva em movimento; reboot parado = -0,20 graus/min).
import imu_reset  # noqa: E402

#: Landmark do MediaPipe Hands usado como POSICAO DA MAO (9 = centro da palma,
#: conforme a metodologia; 0 = pulso). Mesmo valor do paradigma
#: (`eeg_motor_paradigm.HAND_TRACK_LANDMARK`), mas definido aqui: a biblioteca
#: nao usa `HAND_TRACK_LANDMARK` (ela projeta os 21 nos), quem escolhe o no'
#: rastreado sao os programas -- e a coluna `aux_pair_*_m` do CSV grava ESTE no'.
HAND_TRACK_LANDMARK = 9


def desenhar_vetor_palma_seguro(tracker, frame, landmarks, direcao, **kwargs):
    """Desenha a seta da palma sem derrubar a sessao por erro de desenho.

    O desenho nao e' critico para a aquisicao: se a projecao do vetor falhar
    (ponto fora do FOV, inf/NaN, cv2), avisa UMA vez e segue o loop. Sem isso
    um unico frame ruim (mao no limite do quadro, item 121) matava o tool no
    meio de um J/V e perdia a sessao inteira.
    """
    if tracker is None or frame is None or landmarks is None or direcao is None:
        return
    try:
        tracker.draw_palm_vector_custom(frame, landmarks, direcao, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- desenho nunca aborta aquisicao
        if not getattr(tracker, "_aviso_desenho_vetor", False):
            tracker._aviso_desenho_vetor = True
            print(f"[desenho] seta da palma ignorada neste frame: {exc}",
                  flush=True)

def desenhar_guia_plano_palma(tracker, frame, origem_camera, normal_camera,
                               dedos_camera=None, tamanho_m=0.16,
                               cor=(255, 0, 255), cor_dedos=(255, 255, 0)):
    """Desenha o PLANO-ALVO da calibracao geometrica (tecla X) no video.

    O quadrado representa o plano que a FOLHA/PALMA deve "vestir": o usuario
    gira a mao ate o plano dela ficar paralelo ao quadrado. A seta MAGENTA e'
    a normal-alvo (a direcao EXATA que entra na verdade) e a seta CIANO e' a
    referencia de dedos (so' para padronizar a rolagem do punho; nao entra na
    verdade). Tudo e' projetado com `camera_para_pixel_color`, a MESMA funcao
    do esqueleto -- se falhar num frame, retorna False e a aquisicao segue.
    """
    if tracker is None or frame is None or origem_camera is None \
            or normal_camera is None:
        return False
    try:
        origem = np.asarray(origem_camera, np.float64).reshape(3)
        normal = np.asarray(normal_camera, np.float64).reshape(3)
        norma = np.linalg.norm(normal)
        if not np.isfinite(origem).all() or norma < 1e-9:
            return False
        normal = normal / norma
        dedos = None
        if dedos_camera is not None:
            dedos = np.asarray(dedos_camera, np.float64).reshape(3)
            dedos = dedos - float(np.dot(dedos, normal)) * normal
            if np.linalg.norm(dedos) < 1e-6:
                dedos = None
        if dedos is None:
            aux = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(aux, normal))) > 0.9:
                aux = np.array([1.0, 0.0, 0.0])
            dedos = aux - float(np.dot(aux, normal)) * normal
        dedos = dedos / np.linalg.norm(dedos)
        lado_plano = np.cross(normal, dedos)
        meio = 0.5 * float(tamanho_m)
        cantos = [origem + sx * meio * dedos + sy * meio * lado_plano
                  for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pixels = [tracker.camera_para_pixel_color(ponto)
                  for ponto in cantos]
        if all(pixel is not None for pixel in pixels):
            cv2.polylines(frame, [np.array(pixels, np.int32)], True, cor, 2,
                          cv2.LINE_AA)
            # diagonais suaves para o usuario perceber o plano no espaco
            cv2.line(frame, tuple(int(v) for v in pixels[0]),
                     tuple(int(v) for v in pixels[2]), cor, 1, cv2.LINE_AA)
            cv2.line(frame, tuple(int(v) for v in pixels[1]),
                     tuple(int(v) for v in pixels[3]), cor, 1, cv2.LINE_AA)
        # seta da normal-alvo (magenta) -- direcao exata da verdade
        fim_normal = tracker.camera_para_pixel_color(
            origem + 0.16 * normal)
        inicio = tracker.camera_para_pixel_color(origem)
        if fim_normal is not None and inicio is not None:
            cv2.arrowedLine(frame, tuple(int(v) for v in inicio),
                            tuple(int(v) for v in fim_normal), cor, 3,
                            cv2.LINE_AA, tipLength=0.25)
        # seta de referencia dos dedos (ciano)
        fim_dedos = tracker.camera_para_pixel_color(origem + 0.12 * dedos)
        if fim_dedos is not None and inicio is not None:
            cv2.arrowedLine(frame, tuple(int(v) for v in inicio),
                            tuple(int(v) for v in fim_dedos), cor_dedos, 2,
                            cv2.LINE_AA, tipLength=0.3)
        return True
    except Exception:  # noqa: BLE001 -- desenho nunca aborta aquisicao
        return False


# --- alinhamento IMU <-> camera da palma (por luva) --------------------------
#: Arquivo do alinhamento resolvido de cada luva. Fica na raiz (junto das
#: calibracoes) e e' POR LADO: cada luva tem a sua montagem, e o alinhamento de
#: uma nao vale para a outra (era o bug do vetor da palma da mao esquerda).
PALM_ALIGNMENT_FILES = {"right": "imu_palm_alignment_right.json",
                        "left": "imu_palm_alignment_left.json"}
#: Duracao da captura do alinhamento (tecla O): tempo para girar a mao devagar
#: por varios angulos com a palma virada para a camera.
PALM_ALIGNMENT_CAPTURE_SECONDS = 14.0
#: Duracao da captura da tecla J (plano do dorso). Mais longa que a O de
#: proposito: o protocolo e' girar devagar e PARAR ~1 s em cada uma das ~6
#: orientacoes -- com a mao parada o ruido do Kinect nao se mistura com o
#: movimento e a verdade do plano fica limpa (item 105).
PALM_PLANO_CAPTURE_SECONDS = 24.0
#: Minimo de amostras do J (plano do dorso).
PALM_PLANO_MIN_AMOSTRAS = 5
#: Tolerancia (graus) da MEDIANA ROBUSTA das normais: quadros mais longe que
#: isso da mediana sao OUTLIERS e nao derrubam o gate (item 112; medido no log
#: do usuario: 5-8% dos quadros com salto >45 graus zeravam o J/V).
PLANO_ROBUSTO_TOLERANCIA_DEG = 10.0
#: Fracao minima dos quadros da janela que precisa concordar com a mediana.
PLANO_ROBUSTO_FRACAO_MIN = 0.70
#: Quadros consecutivos estaveis exigidos antes de usar/coletar a verdade.
PLANO_QUADROS_ESTAVEIS_MIN = 5
#: Janela da mediana robusta (~1,5 s a 10 fps).
PLANO_JANELA_QUADROS = 15
#: Erro (px) do diagnostico do tabuleiro (tecla T) a partir do qual a calibracao
#: gravada e' considerada de OUTRA montagem. Por que existe: o RMS do arquivo
#: so' mede a coerencia INTERNA dele -- um stereo feito em outra maquina/sala e'
#: aceito com RMS 1 px e projeta centenas de px fora daqui (medido 28/09/2026:
#: a calibracao do repositorio deu 491 px nesta bancada, enquanto uma homografia
#: nova do mesmo par fechou em 0,3 px). Para comparacao, o esperado com a
#: calibracao CERTA e' a casa dos poucos px (RMS do stereo ~1 px).
LIMITE_PX_CALIBRACAO_SUSPEITA = 25.0
#: Mediana (px) acima da qual a CORRESPONDENCIA dos cantos (a homografia dos
#: MESMOS cantos, ajustada na hora) ja' esta' ruim -- nesse caso o problema e' a
#: captura (borrao/reflexo/tabuleiro que mudou entre as duas fotos), nao a
#: calibracao gravada. O tabuleiro e' plano: com correspondencia correta a
#: homografia fecha em fracao de px (medido 0,3 px na bancada de 28/09/2026).
LIMITE_PX_CORRESPONDENCIA = 5.0
#: Intervalo minimo entre amostras da captura (evita 30 amostras iguais).
PALM_ALIGNMENT_SAMPLE_INTERVAL = 0.10
#: Variacao minima (graus) entre a amostra e a ANTERIOR do mesmo lado: sem isso
#: a captura aceita 100 amostras paradas (postura constante nao informa nada).
PALM_ALIGNMENT_MIN_STEP_DEG = 4.0

# --- DIAGNOSTICO DA PALMA por MOVIMENTO PROGRAMADO (tecla D) -----------------
#: Fases do diagnostico: (nome, segundos, instrucao), NESTA ordem.
#:
#: Por que POSES CONHECIDAS e nao "gire a mao como quiser" (tecla O): o erro ao
#: vivo (`palm_err_deg`) diz QUE a seta esta' errada, nunca POR QUE -- era o
#: buraco do item 74 ("a calibracao da palma esta' dando ruim"). Com tres poses
#: fixas o erro vira um TESTE que separa as duas causas possiveis:
#:
#:   - erro parecido nas TRES poses -> problema SISTEMATICO: `T`/`f` errados,
#:     arquivo `imu_palm_alignment_<lado>.json` de outra luva, ou normal do
#:     Kinect com plano/sinal trocados (nesse caso a seta erra sempre igual);
#:   - erro que cresce quando a mao gira (MESA/REZANDO) e fecha em KINECT ->
#:     o alinhamento DECORA a pose da tecla O (que e' "palma para o Kinect"),
#:     ou seja, o modelo/captura estao ruins;
#:   - PREPARO (mesma pose de KINECT, no inicio) mede o RUIDO da normal antes de
#:     qualquer expectativa: se PREPARO e KINECT divergem, a mao nao ficou
#:     parada (ou o Kinect esta' ruidoso) e o resto do numero nao vale.
#:
#: A mao tem de ficar DENTRO do quadro do Kinect nas quatro fases: a verdade e'
#: a normal medida pelo depth, e fora do quadro nao ha' linha nenhuma.
#:
#: NAO ENCOSTE AS MAOS (medido na bancada, 19/10/2026, sessao 15:38:36): com as
#: maos REALMENTE juntas ("rezando"), uma mao COBRE a outra para o Kinect -- na
#: fase REZANDO a esquerda teve 19 amostras brutas (4 depois do settle) contra
#: 56 da direita, e as cameras auxiliares (o par paralelo) tambem passaram a ver
#: SO' um lado (a queixa do usuario: "quando junto as duas maos, as cameras
#: paralelas so' pegam a mao de um dos lados"). A pose continua valendo (e' a
#: unica que poe a normal na HORIZONTAL, quebrando a "decoracao" da pose da
#: tecla O) com as maos a ~15 cm uma da outra. Mesma razao para a MESA: mao na
#: BORDA, inteira no quadro -- mesa/corpo tapando a mao nao tem amostra.
PALM_DIAG_PHASES = (
    ("PREPARO", 3.0,
     "Maos na frente do corpo, palmas viradas para o Kinect, dedos para cima, "
     "PARADO."),
    ("MESA", 6.0,
     "Apoie as DUAS palmas na BORDA da mesa (dorso para cima, dedos para "
     "frente), com as maos INTEIRAS no quadro e nada tapando, sem mexer."),
    ("REZANDO", 6.0,
     "Maos na frente do peito, palmas viradas UMA PARA A OUTRA, SEM SE "
     "ENCOSTAREM (uns 15 cm, como segurando uma bola), dedos para cima, sem "
     "mexer."),
    ("KINECT", 6.0,
     "Palmas abertas viradas para o Kinect, dedos para cima, sem mexer."),
)
#: Amostras do INICIO de cada fase ficam FORA do resumo (elas entram no CSV):
#: o experimentador leva ~1 s para CHEGAR na pose e as primeiras amostras de uma
#: transicao (mao ainda girando) envenenariam a mediana da fase.
PALM_DIAG_SETTLE_S = 1.5
#: Limite (graus) da mediana por fase. E' o MESMO criterio do residuo do
#: alinhamento (PALM_ALIGNMENT_MAX_RESIDUAL_DEG): acima disso a pose nao fecha.
PALM_DIAG_MAX_ERR_DEG = PALM_ALIGNMENT_MAX_RESIDUAL_DEG
#: Prefixo dos arquivos do diagnostico: palm_diag_<data>_<hora>.csv (+ _meta.json
#: e <csv>_resumo.txt).
PALM_DIAG_CSV_PREFIX = "palm_diag_"
#: Intervalo minimo entre linhas do CSV, POR LADO (20 Hz). O laco roda a ~30 fps
#: e quatro fases em 21 s nao precisam de 600 linhas por lado para descrever uma
#: pose PARADA (o CSV principal ja' grava todo frame).
PALM_DIAG_SAMPLE_INTERVAL = 0.05
#: Abaixo disto numa fase o veredito avisa "a mao saiu do quadro" em vez de
#: fingir que a estatistica daquela pose vale.
PALM_DIAG_MIN_SAMPLES_FASE = 10


def palm_alignment_path(lado):
    """Caminho do JSON do alinhamento da luva `lado` ('right'/'left')."""
    return Path(PALM_ALIGNMENT_FILES.get(lado, f"imu_palm_alignment_{lado}.json"))


def load_palm_alignment(fusion, lado, avisar=print):
    """Carrega o alinhamento salvo da luva `lado` (False se nao houver/valido)."""
    caminho = palm_alignment_path(lado)
    if not caminho.exists():
        return False
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        aplicado = fusion.set_palm_alignment(
            dados["rotation"], dados.get("body"), dados.get("residual_deg"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        avisar(f"IMU {lado}: alinhamento em {caminho.name} ilegivel ({exc}); "
               "rode a tecla O de novo")
        return False
    if aplicado:
        avisar(f"IMU {lado}: alinhamento da palma carregado de {caminho.name} "
               f"(residuo {dados.get('residual_deg')})")
    return aplicado


def save_palm_alignment(fusion, lado, avisar=print):
    """Grava o alinhamento atual da luva `lado` (False se nao ha' o que gravar).

    O ARQUIVO ANTERIOR vai para `.json.bak` ANTES de gravar (mesma politica de
    `tools/analisa_diag_palma.py --salvar`). Por que: em 21/10/2026 a captura da
    tecla O sobrescreveu o T/f do REFIT de 19/10 (erro ao vivo 2-13 graus) por um
    ajuste que errava ~100 graus quando o punho girava, e o anterior ficou
    irrecuperavel -- o analisador so' faz backup do que encontra no disco e
    `save_palm_alignment` nao fazia nenhum. Com o backup aqui, NENHUMA captura da
    O apaga um refit validado (e o aviso mostra o residuo anterior x o novo).
    """
    estado = fusion.palm_alignment_state()
    if estado is None:
        return False
    caminho = palm_alignment_path(lado)
    backup = caminho.with_suffix(".json.bak")
    anterior = None
    if caminho.exists():
        try:
            anterior = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            anterior = None
    try:
        caminho.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    except OSError as exc:
        avisar(f"IMU {lado}: nao consegui gravar {caminho.name} ({exc})")
        return False
    if anterior is not None:
        try:
            backup.write_text(json.dumps(anterior, indent=2), encoding="utf-8")
            avisar(f"IMU {lado}: {caminho.name} anterior (residuo "
                   f"{anterior.get('residual_deg')}) guardado em {backup.name}")
        except OSError as exc:
            avisar(f"IMU {lado}: AVISO backup de {caminho.name} falhou ({exc})")
    return True


def finalizar_calibracao_geometrica(calib, fusion, participante="",
                                     avisar=print):
    """Resolve T/f da calibracao geometrica, grava o piloto e talvez o oficial.

    Sempre grava `calibracao_geometrica_<lado>_<data>.json` (dados brutos +
    solucao + validacoes) para auditoria/reanalise. So' substitui o alinhamento
    OFICIAL (`imu_palm_alignment_<lado>.json`) se o gate passar; o arquivo
    anterior vira `.json.bak` (mesma politica do `save_palm_alignment`).
    Devolve dict com `ok`, `salvou_oficial`, `resolucao` e `arquivo_piloto`.
    """
    resolucao = calib.resultado()
    if "erro" in resolucao:
        avisar(f"[GEO {calib.lado}] {resolucao['erro']} -- alinhamento NAO "
               "alterado")
        return {"ok": False, "salvou_oficial": False,
                "resolucao": resolucao}
    erro_geo_max = resolucao.get("erro_geometricas_max")
    gate = True
    motivos = []
    if resolucao["n_takes"] < CALIB_GEO_MIN_TAKES + 1:
        gate = False
        motivos.append(f"takes {resolucao['n_takes']} < "
                       f"{CALIB_GEO_MIN_TAKES + 1} (6 geometricas + mira)")
    if resolucao["residual_deg"] > CALIB_GEO_MAX_RESIDUO_DEG:
        gate = False
        motivos.append(f"residuo {resolucao['residual_deg']:.1f} > "
                       f"{CALIB_GEO_MAX_RESIDUO_DEG:g}")
    if erro_geo_max is not None and erro_geo_max > CALIB_GEO_MAX_POSE_DEG:
        gate = False
        motivos.append(f"pior pose geometrica {erro_geo_max:.1f} > "
                       f"{CALIB_GEO_MAX_POSE_DEG:g}")
    # A pose de MIRA e a validacao dela FORA do ajuste sao a unica prova de
    # que o T/f generaliza para a direcao absoluta (as poses geometricas so'
    # pinam a rotacao). Sem elas, NAO sobrescrever o alinhamento oficial.
    if resolucao.get("validacao_mira") is None:
        gate = False
        motivos.append("sem validacao independente da mira (pule <= 2 poses "
                       "geometricas)")
    else:
        _val_mira = resolucao["validacao_mira"]
        if _val_mira["mediana_deg"] > CALIB_GEO_MAX_VALIDACAO_MIRA_DEG:
            gate = False
            motivos.append(f"validacao da mira {_val_mira['mediana_deg']:.1f} > "
                           f"{CALIB_GEO_MAX_VALIDACAO_MIRA_DEG:g}")
    if (resolucao.get("loo_mediana_deg") is not None
            and resolucao["loo_mediana_deg"] > CALIB_GEO_MAX_LOO_MEDIANA_DEG):
        gate = False
        motivos.append(f"validacao cruzada (LOO) "
                       f"{resolucao['loo_mediana_deg']:.1f} > "
                       f"{CALIB_GEO_MAX_LOO_MEDIANA_DEG:g}")
    avisar("")
    avisar(f"===== CALIBRACAO GEOMETRICA -- {calib.lado.upper()} =====")
    avisar(f"takes: {resolucao['n_takes']} | residuo ponderado: "
           f"{resolucao['residual_deg']:.2f} graus")
    for linha in resolucao.get("por_pose", ()):
        avisar(f"  {linha['pose']:<32} QA {linha['qa_deg']:5.1f} | "
               f"erro {linha['erro_deg']:5.2f} graus")
    if resolucao.get("validacao_mira"):
        val = resolucao["validacao_mira"]
        avisar(f"  validacao independente (mira, fora do ajuste): "
               f"{val['mediana_deg']:.2f} graus (n={val['n']})")
    if resolucao.get("loo_mediana_deg") is not None:
        avisar(f"  validacao cruzada (deixa-um-de-fora): mediana "
               f"{resolucao['loo_mediana_deg']:.2f} / max "
               f"{resolucao['loo_max_deg']:.2f} graus")
    quando = time.strftime("%Y-%m-%d %H:%M:%S")
    nome = time.strftime(f"calibracao_geometrica_{calib.lado}_%Y%m%d_%H%M%S.json")
    payload = {
        "modo": "calibracao_geometrica",
        "lado": calib.lado,
        "participante": participante or "",
        "quando": quando,
        "posicoes": [
            {"posicao": pose["pose"], "tipo": pose["tipo"],
             "alvo_camera": pose["alvo"], "verdade_camera": pose["verdade"],
             "qa_deg": pose["qa_deg"], "rotacao": pose["rotacao"],
             "posicao_mao_camera": pose["posicao"],
             "normal_medida_camera": pose["normal_medida"]}
            for pose in calib.capturas
        ],
        "solucao": {chave: valor for chave, valor in resolucao.items()
                    if chave in ("rotation", "body", "residual_deg")},
        "diagnostico": {chave: valor for chave, valor in resolucao.items()
                        if chave not in ("rotation", "body")},
        "constantes": {
            "TAKE_SEC": CALIB_GEO_TAKE_SEC,
            "QA_MAX_DEG": CALIB_GEO_QA_MAX_DEG,
            "PESO_GEOMETRICA": CALIB_GEO_PESO_GEOMETRICA,
            "PESO_POSICAO": CALIB_GEO_PESO_POSICAO,
        },
    }
    arquivo = Path(nome)
    try:
        arquivo.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        avisar(f"  dados completos: {arquivo.name}")
    except OSError as exc:
        avisar(f"  AVISO: nao gravei {arquivo.name} ({exc})")
    salvou_oficial = False
    if gate:
        fusion.set_palm_alignment(resolucao["rotation"], resolucao["body"],
                                  resolucao["residual_deg"])
        salvou_oficial = save_palm_alignment(fusion, calib.lado, avisar=avisar)
        avisar(f"  ALINHAMENTO OFICIAL {calib.lado.upper()} "
               + ("gravado" if salvou_oficial else "NAO gravado")
               + " (gate passou)")
    else:
        avisar("  ALINHAMENTO OFICIAL NAO alterado (" + "; ".join(motivos) + ")")
    return {"ok": True, "salvou_oficial": salvou_oficial,
            "resolucao": resolucao, "arquivo_piloto": str(arquivo)}


class CapturaAlinhamentoPalma:
    """Coleta amostras (atitude do IMU + normal da palma do Kinect) por lado.

    Por que existe: a direcao da palma prevista pelo IMU precisa do referencial
    da luva (mundo do IMU -> camera), que NAO e' o mesmo nas duas luvas e nao da'
    para adivinhar -- so' medir. Enquanto a captura esta' ligada (tecla O), cada
    amostra aceita guarda a rotacao do ESP32 e a normal da palma medida pelo
    Kinect (depth) daquele lado; no fim, `solve_palm_alignment` resolve T e f.

    Regras: amostra so' entra se a normal existir (Kinect enxergando a mao), se
    passou o intervalo minimo e se a mao MUDOU de postura desde a ultima
    (`PALM_ALIGNMENT_MIN_STEP_DEG`) -- girar a mao e' o que informa a rotacao de
    alinhamento.
    """

    def __init__(self, segundos=PALM_ALIGNMENT_CAPTURE_SECONDS):
        self.duracao_s = float(segundos)
        self.fim = 0.0
        self.ativa = False
        self.amostras = {"right": [], "left": []}   # [(rotacao (3,3), normal (3,))]
        self._ultima_amostra = {"right": None, "left": None}
        self._ultimo_tempo = {"right": 0.0, "left": 0.0}

    def iniciar(self, agora):
        self.ativa = True
        self.fim = agora + self.duracao_s
        self.amostras = {"right": [], "left": []}
        self._ultima_amostra = {"right": None, "left": None}
        self._ultimo_tempo = {"right": 0.0, "left": 0.0}

    def parar(self):
        self.ativa = False

    @property
    def restante_s(self):
        return max(0.0, self.fim - time.monotonic())

    def contar(self, lado):
        return len(self.amostras.get(lado, ()))

    def coletar(self, lado, agora, sample, normal_camera):
        """Tenta guardar uma amostra do lado. Devolve True se aceitou."""
        if not self.ativa or sample is None or normal_camera is None:
            return False
        if agora - self._ultimo_tempo[lado] < PALM_ALIGNMENT_SAMPLE_INTERVAL:
            return False
        normal = np.asarray(normal_camera, np.float64).reshape(3)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
            return False
        anterior = self._ultima_amostra[lado]
        if anterior is not None and \
                angle_between(anterior, normal) < PALM_ALIGNMENT_MIN_STEP_DEG:
            return False
        rotacao = rotation_matrix(sample.roll_deg, sample.pitch_deg,
                                  sample.yaw_deg)
        self.amostras[lado].append((rotacao, normal / np.linalg.norm(normal)))
        self._ultima_amostra[lado] = normal / np.linalg.norm(normal)
        self._ultimo_tempo[lado] = agora
        return True

    def resolver(self, lado, min_amostras=None):
        """(T, f, residuo) do lado, ou None se faltam amostras validas."""
        minimo = (PALM_ALIGNMENT_MIN_SAMPLES if min_amostras is None
                  else int(min_amostras))
        amostras = self.amostras.get(lado, ())
        if len(amostras) < minimo:
            return None
        return solve_palm_alignment([r for r, _v in amostras],
                                    [v for _r, v in amostras])


def cabecalho_palm_diag():
    """Colunas do CSV do diagnostico da palma (tecla D), uma linha por
    (frame, lado) -- e so' quando a mao E a normal do Kinect existem.

    ORDEM = a mesma da linha gravada em `DiagnosticoPalma.registrar`; quem le'
    por NOME (tools/analisa_diag_palma.py) nao depende dela, mas a
    documentacao da sessao depende: (nx,ny,nz) e' a normal do KINECT (a
    verdade), (prev_*) e' a previsao COM alinhamento (`T R f`, a seta do
    video) e (raw_*) e' a previsao SEM alinhamento (`R f`, T = identidade).
    Ter as DUAS previsoes na mesma linha e' o que permite responder, offline,
    se o alinhamento melhorou alguma coisa (era a suspeita do item 74: "o tool
    nem esta' usando o arquivo").
    """
    return ["t_s", "fase_idx", "fase_nome", "t_fase_s", "lado",
            "roll_deg", "pitch_deg", "yaw_deg",
            "nx", "ny", "nz",
            "prev_x", "prev_y", "prev_z", "err_deg",
            "raw_x", "raw_y", "raw_z", "err_raw_deg",
            "alinhado",
            # --- EVIDENCIA CRUA (acrescentada no item 78) -------------------
            # Por que: a sessao real de 19/10/2026 15:55:51 provou que a
            # ATITUDE do IMU nao fecha (a MESMA pose medida no inicio e no fim
            # difere 74-171 graus no IMU e 11-20 no Kinect) -- o firmware
            # integra o yaw com `yaw += gz*dt` (eixo Z do CORPO: so' vale com o
            # sensor nivelado). Para PROVAR isso e testar uma correcao offline
            # (yaw com o giro completo: -sin(pitch)gx + cos(pitch)sin(roll)gy +
            # cos(pitch)cos(roll)gz) o CSV precisa do giroscopio/acelerometro
            # CRUS e do timestamp do ESP32 (dt pelo relogio do sensor).
            # Os 4 pontos 3D crus do frame da palma existem para diagnostico do
            # SINAL da normal: `palm_normal_from_points` FORCA nz<0 (so' vale na
            # pose "palma para o Kinect"); com os pontos da' para reconstruir a
            # normal anatomica (por mao) offline, sem chute.
            "imu_timestamp_us",
            "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps",
            "wrist_x", "wrist_y", "wrist_z",
            "index_x", "index_y", "index_z",
            "pinky_x", "pinky_y", "pinky_z",
            "middle_x", "middle_y", "middle_z"]


def _valor_csv_diag(texto, padrao=float("nan")):
    """float() tolerante para o CSV do diagnostico: vazio/ilegivel -> NaN.

    Nunca 0: "nao deu para medir" e "erro zero" sao coisas MUITO diferentes
    neste diagnostico (um erro 0 graus seria o alinhamento perfeito).
    """
    try:
        return float(texto)
    except (TypeError, ValueError):
        return padrao


def _estatisticas_diag(valores):
    """n/media/mediana/desvio-padrao/p90/min/max (graus) de uma lista."""
    dados = np.asarray([valor for valor in valores
                        if valor is not None and np.isfinite(valor)], dtype=float)
    if dados.size == 0:
        return {"n": 0}
    return {
        "n": int(dados.size),
        "media": float(np.mean(dados)),
        "mediana": float(np.median(dados)),
        "desvio": float(np.std(dados)),
        "p90": float(np.percentile(dados, 90)),
        "minimo": float(np.min(dados)),
        "maximo": float(np.max(dados)),
    }


def veredito_palm_diag(estat_por_fase, bruto, limite=None, minimo=None):
    """Linhas do veredito da palma para UM lado.

    `estat_por_fase` = {fase: estatisticas de `err_deg`} (SEM as amostras do
    settle); `bruto` = estatisticas de `err_raw_deg` do lado (o erro que a seta
    teria SEM alinhamento nenhum). `tools/analisa_diag_palma.py` tem uma copia
    literal desta funcao: ele NAO pode importar este arquivo (o import carrega
    OpenCV/pykinect) -- se um criterio mudar aqui, mude la' tambem.

    A ordem das causas e' a ordem em que uma esconde a outra: poucas amostras
    invalidam qualquer conclusao; "o alinhamento nao esta' aplicado" explica
    erro grande em TODAS as poses; so' depois faz sentido separar sistematico
    (T/f) de dependente da rotacao (modelo/captura).
    """
    limite = PALM_DIAG_MAX_ERR_DEG if limite is None else float(limite)
    minimo = PALM_DIAG_MIN_SAMPLES_FASE if minimo is None else int(minimo)
    notas = []
    poses = [fase for fase in ("MESA", "REZANDO", "KINECT")
             if estat_por_fase.get(fase, {}).get("n")]
    if not poses:
        return ["sem amostras validas: nenhuma pose programada foi medida"]
    for fase, dados in estat_por_fase.items():
        if 0 < dados.get("n", 0) < minimo:
            notas.append(f"sem amostras suficientes na fase {fase} "
                         f"({dados['n']} < {minimo}): a mao saiu do quadro OU "
                         "a outra mao cobriu esta (nao encoste as maos)")
    mediana = {fase: estat_por_fase[fase]["mediana"] for fase in poses}
    medio = float(np.mean([estat_por_fase[fase]["media"] for fase in poses]))
    bruto_medio = (bruto or {}).get("media")
    if bruto_medio is not None and np.isfinite(bruto_medio) \
            and bruto_medio <= medio + 2.0:
        notas.append("o alinhamento NAO esta' sendo aplicado (sem "
                     "imu_palm_alignment_<lado>.json / bootstrap de 1 amostra): "
                     f"erro SEM alinhamento {bruto_medio:.1f} graus x "
                     f"{medio:.1f} com ele")
    ok = {fase: mediana[fase] <= limite for fase in poses}
    if len(poses) == 3 and all(ok.values()):
        notas.append("ALINHAMENTO OK nas 3 poses programadas")
    elif ok.get("KINECT") and (not ok.get("MESA", True)
                               or not ok.get("REZANDO", True)):
        notas.append("o alinhamento DECORA a pose da tecla O: T/f estao errados "
                     "(captura O quase parada ou sempre na mesma orientacao) -> "
                     "refaca O girando a mao devagar")
    elif not ok.get("KINECT", True):
        notas.append("nem a pose da calibracao fecha: conferir "
                     "enquadramento/profundidade da mao, luva trocada "
                     "(--imu-portas) ou normal do Kinect ruidosa")
    notas.append("medianas: " + ", ".join(
        f"{fase} {mediana[fase]:.1f}" for fase in poses) + " graus")
    return notas


def pontos_palma_camera(tracker, landmarks):
    """4 pontos 3D (camera space) do frame da palma: punho e 3 MCPs.

    PUNHO (0), MCP do indicador (5), MCP do medio (9), MCP do minimo (17) --
    EVIDENCIA CRUA do diagnostico da palma (item 78 do historico).

    Por que existe: a normal pronta (`palm_normal_from_points`) FORCA o sinal
    para nz<0, o que so' e' a direcao da PALMA na pose "palma para o Kinect".
    Nas outras poses (MESA com a palma para baixo, REZANDO com as palmas uma
    para a outra) esse sinal vira ruido e o modelo `v = T R f` nao fecha. Com os
    pontos crus da' para reconstruir a normal VERDADEIRA offline e decidir o
    sinal POR MAO (direita x esquerda) com prova, sem chute.

    Devolve {nome: (3,)} ou None se algum ponto nao tiver profundidade valida
    (mao fora do quadro/borda): a linha do diagnostico sai sem os pontos, nunca
    com zero disfarcado de medida.
    """
    indices = {"wrist": 0, "index": 5, "middle": 9, "pinky": 17}
    pontos = {}
    try:
        for nome, indice in indices.items():
            ponto = tracker._landmark_camera_point(landmarks[indice])
            if ponto is None:
                return None
            ponto = np.asarray(ponto, np.float64).reshape(3)
            if not np.isfinite(ponto).all():
                return None
            pontos[nome] = ponto
    except (AttributeError, IndexError, TypeError):
        return None
    return pontos


def _colunas_evidencia_diag(sample, pontos):
    """Colunas CRUAS do CSV do diagnostico: giroscopio/acelerometro/pontos 3D.

    Vazio quando o dado nao existe (nunca 0.0 disfarcado: no giroscopio um zero
    significa "parado" e no timestamp "sem amostra"). Serve para (a) PROVAR a
    inconsistencia de yaw do IMU com o relogio do proprio sensor e (b) testar
    uma correcao de atitude offline (item 78).
    """
    campos = {}
    aceleracao = (np.asarray(getattr(sample, "accel_g", None), np.float64).reshape(-1)
                  if sample is not None else np.zeros(0))
    giroscopio = (np.asarray(getattr(sample, "gyro_dps", None), np.float64).reshape(-1)
                  if sample is not None else np.zeros(0))
    campos["imu_timestamp_us"] = (
        int(getattr(sample, "timestamp_us", 0) or 0)
        if sample is not None else "")
    for indice, chave in enumerate(("ax_g", "ay_g", "az_g")):
        campos[chave] = (f"{aceleracao[indice]:.5f}"
                         if aceleracao.size == 3 and np.isfinite(aceleracao).all()
                         else "")
    for indice, chave in enumerate(("gx_dps", "gy_dps", "gz_dps")):
        campos[chave] = (f"{giroscopio[indice]:.3f}"
                         if giroscopio.size == 3 and np.isfinite(giroscopio).all()
                         else "")
    for nome in ("wrist", "index", "pinky", "middle"):
        ponto = None if pontos is None else pontos.get(nome)
        ponto = (np.asarray(ponto, np.float64).reshape(3)
                 if ponto is not None else None)
        for eixo, sufixo in enumerate("xyz"):
            campos[f"{nome}_{sufixo}"] = (
                f"{ponto[eixo]:.5f}"
                if ponto is not None and np.isfinite(ponto).all() else "")
    return campos


class DiagnosticoPalma:
    """Diagnostico da palma por MOVIMENTO PROGRAMADO (tecla D): CSV + resumo.

    Por que existe (item 74 do historico -- "a calibracao da palma esta'
    dando ruim"): `palm_err_deg` diz QUE a seta esta' errada, mas nao POR QUE.
    Sem um protocolo de poses conhecidas nao da' para separar as duas causas:

      (a) SISTEMATICA -- `T`/`f` (ou o arquivo `imu_palm_alignment_<lado>.json`
          carregado) estao errados, e entao o erro e' grande em QUALQUER pose;
      (b) DEPENDENTE DA ROTACAO -- o modelo/captura sao ruins e o vetor acerta
          so' a pose "palma para o Kinect", que e' a unica que a tecla O
          costuma ver (captura quase parada "decora" essa pose).

    O que este objeto grava, uma linha por (frame, lado): a normal da palma
    medida pelo KINECT (a verdade), a previsao COM alinhamento (`T R f`, a seta
    do video), a previsao SEM alinhamento (`R f`, T = identidade) e os dois
    erros em graus. O resumo por (fase, lado) e o veredito saem no fim (a
    tecla D de novo encerra antes e gera o resumo do que ja' foi coletado).
    A analise mais completa e' offline, em `tools/analisa_diag_palma.py`.

    Robustez: o arquivo e' fechado em `finally` (nunca fica aberto se algo
    estourar no meio de um frame), cada linha e' gravada com flush (dá' para
    ler o CSV com o diagnostico ainda rodando) e `cancelar()` fecha sem resumo
    (saida por ESC).
    """

    def __init__(self, fases=None, relogio=time.monotonic, avisar=print):
        #: Clock injetavel: os testes avancam o tempo sem dormir 21 s.
        self.fases = tuple(PALM_DIAG_PHASES if fases is None else fases)
        self.relogio = relogio
        self.avisar = avisar
        self.ativa = False
        self.t_inicio = 0.0
        self.caminho_csv = None
        self.caminho_meta = None
        self.caminho_resumo = None
        #: Linhas ja' gravadas (dicts com as MESMAS chaves do cabecalho): o
        #: resumo e' calculado daqui, sem reler o arquivo (o CSV pode estar
        #: aberto em outra maquina/editor e travado no Windows).
        self.linhas = []
        self._arquivo = None
        self._escritor = None
        self._ultimo = {"right": -1e9, "left": -1e9}

    # ------------------------------------------------------------ propriedades
    @property
    def duracao_total_s(self):
        """Duracao programada das fases, em segundos (21 s com as 4 padrao)."""
        return float(sum(float(segundos) for _nome, segundos, _instrucao
                         in self.fases))

    @property
    def restante_total_s(self):
        """Quanto falta do diagnostico inteiro (0 quando nao esta' ativo)."""
        if not self.ativa:
            return 0.0
        return max(0.0, self.duracao_total_s - (self.relogio() - self.t_inicio))

    def fase_atual(self, agora):
        """(idx, nome, t_fase_s, restante_s, instrucao) ou None se terminou.

        `agora` e' um instante de `time.monotonic()` (o do laco), passado de
        fora para o mesmo `now` valer para o CSV principal e para o
        diagnostico -- dois `time.monotonic()` diferentes fariam a linha do
        diagnostico cair numa fase e a do CSV na anterior.
        """
        if not self.ativa:
            return None
        decorrido = float(agora) - self.t_inicio
        if decorrido < 0.0:
            decorrido = 0.0
        inicio = 0.0
        for indice, (nome, segundos, instrucao) in enumerate(self.fases):
            fim = inicio + float(segundos)
            if decorrido < fim:
                return (indice, nome, decorrido - inicio, fim - decorrido,
                        instrucao)
            inicio = fim
        return None

    # ------------------------------------------------------------- ciclo de vida
    def iniciar(self, agora, fusions):
        """Liga o diagnostico e cria `palm_diag_<data>_<hora>.csv` + `_meta.json`.

        Grava no meta o que NAO esta' no CSV e e' decisivo para interpretar o
        resultado: as fases e suas instrucoes, o settle, o limite de erro, o
        estado de alinhamento de cada lado (`palm_alignment_state()`, None =
        bootstrap de 1 amostra) e se havia `imu_palm_alignment_<lado>.json` --
        com data/hora do arquivo, para pegar o caso "o arquivo e' de outra
        sessao/outra luva" (um dos modos de falha do item 74).

        Devolve False se ja' estava ativo (nao cria um segundo par de arquivos).
        """
        if self.ativa:
            return False
        agora = float(agora)
        carimbo = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        base = PALM_DIAG_CSV_PREFIX + carimbo
        sufixo = 1
        # Duas sessoes iniciadas no mesmo segundo nao podem dividir arquivo.
        while Path(base + ".csv").exists() or Path(base + "_meta.json").exists():
            sufixo += 1
            base = f"{PALM_DIAG_CSV_PREFIX}{carimbo}_{sufixo}"
        self.caminho_csv = Path(base + ".csv")
        self.caminho_meta = Path(base + "_meta.json")
        self.caminho_resumo = Path(base + ".csv_resumo.txt")
        self.linhas = []
        self._ultimo = {"right": -1e9, "left": -1e9}
        meta = {
            "pc_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "csv": self.caminho_csv.name,
            "fases": [{"nome": nome, "segundos": float(segundos),
                       "instrucao": instrucao}
                      for nome, segundos, instrucao in self.fases],
            "settle_s": float(PALM_DIAG_SETTLE_S),
            "limite_erro_graus": float(PALM_DIAG_MAX_ERR_DEG),
            "duracao_total_s": self.duracao_total_s,
            "amostras_minimas_fase": int(PALM_DIAG_MIN_SAMPLES_FASE),
            "lados": {},
        }
        for lado in ("right", "left"):
            fusion = (fusions or {}).get(lado)
            estado = (fusion.palm_alignment_state()
                      if fusion is not None else None)
            caminho = palm_alignment_path(lado)
            existe = caminho.exists()
            mtime = None
            if existe:
                try:
                    mtime = time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(caminho.stat().st_mtime))
                except OSError:
                    existe = False
            meta["lados"][lado] = {
                "alinhamento": estado,
                "arquivo": caminho.name,
                "arquivo_existe": bool(existe),
                "arquivo_mtime": mtime,
            }
        try:
            self._arquivo = open(self.caminho_csv, "w", newline="",
                                 encoding="utf-8")
            self._escritor = csv.writer(self._arquivo)
            self._escritor.writerow(cabecalho_palm_diag())
            self._arquivo.flush()
            with self.caminho_meta.open("w", encoding="utf-8") as arquivo_meta:
                json.dump(meta, arquivo_meta, indent=2)
        except OSError as exc:
            self._fechar()
            self.avisar(f"DIAGNOSTICO DA PALMA: nao consegui criar os arquivos "
                        f"({exc}); diagnostico NAO ligado")
            self.ativa = False
            self._arquivo = None
            self._escritor = None
            return False
        self.t_inicio = agora
        self.ativa = True
        fases_txt = " | ".join(f"{indice + 1}) {nome} {float(segundos):g}s"
                               for indice, (nome, segundos, _instrucao)
                               in enumerate(self.fases))
        self.avisar(
            f"DIAGNOSTICO DA PALMA LIGADO -> {self.caminho_csv.name} "
            f"({self.duracao_total_s:g} s, {len(self.fases)} fases): {fases_txt}\n"
            "  Mantenha as DUAS maos DENTRO do quadro do Kinect e faca a pose "
            "pedida no video; D de novo encerra e resume o que ja' foi coletado.")
        # Estado do alinhamento ANTES da coleta: medido em 19/10/2026 que o
        # operador fazia as 21 s e so' no fim descobria que NAO havia
        # `imu_palm_alignment_<lado>.json` nenhum -- a seta estava no bootstrap
        # de 1 amostra (T = identidade) e o erro de 20-140 graus nao tinha nada
        # a ver com o modelo. Dizer isso no INICIO deixa escolher: rodar a
        # tecla O antes (mede a seta de producao) ou seguir assim mesmo (o
        # diagnostico separa bruto x calibrado e o refit offline ainda serve).
        for lado in ("right", "left"):
            dados_lado = meta["lados"][lado]
            estado_lado = dados_lado.get("alinhamento") or {}
            residual = estado_lado.get("residual_deg")
            if dados_lado.get("arquivo_existe") and estado_lado:
                self.avisar(
                    f"  {lado}: alinhamento SALVO ({dados_lado['arquivo']}, "
                    f"residuo {residual} graus) -- a seta aqui e' a de producao")
            elif estado_lado:
                self.avisar(
                    f"  {lado}: SEM arquivo de alinhamento -- a seta esta' no "
                    "BOOTSTRAP de 1 amostra (vai errar): rode a tecla O antes se "
                    "quiser medir a seta calibrada; sem O, o refit do analisador "
                    "ainda mostra o melhor alinhamento possivel")
            else:
                self.avisar(f"  {lado}: sem alinhamento NENHUM (a seta nem "
                            "aparece): rode a tecla O antes")
        for indice, (nome, segundos, instrucao) in enumerate(self.fases):
            self.avisar(f"  fase {indice + 1}/{len(self.fases)} {nome} "
                        f"({float(segundos):g} s): {instrucao}")
        return True

    def registrar(self, lado, agora, sample, normal_camera, fusion, pontos=None):
        """Grava 1 linha do CSV (False quando nao deu para medir/ainda nao e' hora).

        `normal_camera` e' `palm_normal_camera` do Kinect (a verdade) e `sample`
        a amostra do IMU daquele lado. As duas previsoes sao calculadas AQUI
        (nao vem do `mao[...]` do laco) para a linha do diagnostico ser
        justamente "T R f" e "R f", independentes do que a tela desenhou.
        """
        if not self.ativa or self._escritor is None:
            return False
        agora = float(agora)
        fase = self.fase_atual(agora)
        if fase is None:
            return False
        # 20 Hz por lado: sem isto o CSV teria uma linha por frame para uma
        # pose PARADA (o CSV principal ja' guarda todo frame).
        if agora - self._ultimo.get(lado, -1e9) < PALM_DIAG_SAMPLE_INTERVAL:
            return False
        if sample is None or normal_camera is None:
            return False
        normal = np.asarray(normal_camera, np.float64).reshape(3)
        norma = np.linalg.norm(normal)
        if not np.isfinite(normal).all() or norma < 1e-9:
            return False
        normal = normal / norma
        indice, nome, t_fase, _restante, _instrucao = fase
        previsto = fusion.palm_direction(sample) if fusion is not None else None
        if previsto is not None:
            previsto = np.asarray(previsto, np.float64).reshape(3)
            norma_prev = np.linalg.norm(previsto)
            previsto = (previsto / norma_prev
                        if np.isfinite(previsto).all() and norma_prev > 1e-9
                        else None)
        rotacao = rotation_matrix(sample.roll_deg, sample.pitch_deg,
                                  sample.yaw_deg)
        referencia = (None if fusion is None
                      else getattr(fusion, "palm_reference_body", None))
        bruto = None
        if referencia is not None:
            bruto = rotacao @ np.asarray(referencia, np.float64).reshape(3)
            norma_bruta = np.linalg.norm(bruto)
            bruto = (bruto / norma_bruta
                     if np.isfinite(bruto).all() and norma_bruta > 1e-9
                     else None)
        erro = None if previsto is None else angle_between(normal, previsto)
        erro_bruto = None if bruto is None else angle_between(normal, bruto)
        linha = {
            "t_s": f"{agora:.3f}",
            "fase_idx": indice,
            "fase_nome": nome,
            "t_fase_s": f"{t_fase:.3f}",
            "lado": lado,
            "roll_deg": f"{float(sample.roll_deg):.3f}",
            "pitch_deg": f"{float(sample.pitch_deg):.3f}",
            "yaw_deg": f"{float(sample.yaw_deg):.3f}",
            "nx": f"{normal[0]:.6f}", "ny": f"{normal[1]:.6f}",
            "nz": f"{normal[2]:.6f}",
            "prev_x": "" if previsto is None else f"{previsto[0]:.6f}",
            "prev_y": "" if previsto is None else f"{previsto[1]:.6f}",
            "prev_z": "" if previsto is None else f"{previsto[2]:.6f}",
            "err_deg": "" if erro is None else f"{float(erro):.3f}",
            "raw_x": "" if bruto is None else f"{bruto[0]:.6f}",
            "raw_y": "" if bruto is None else f"{bruto[1]:.6f}",
            "raw_z": "" if bruto is None else f"{bruto[2]:.6f}",
            "err_raw_deg": ("" if erro_bruto is None
                            else f"{float(erro_bruto):.3f}"),
            "alinhado": 1 if referencia is not None else 0,
        }
        # Evidencia crua (item 78): giroscopio/acelerometro/timestamp do ESP32 e
        # os 4 pontos 3D da palma -- sem isso nao da' para provar a
        # inconsistencia de yaw nem reconstruir o sinal da normal offline.
        linha.update(_colunas_evidencia_diag(sample, pontos))
        try:
            self._escritor.writerow([linha[coluna]
                                     for coluna in cabecalho_palm_diag()])
            self._arquivo.flush()
        except (OSError, ValueError) as exc:
            # Disco cheio/arquivo travado nao pode derrubar a sessao de
            # aquisicao: avisa e DESLIGA o diagnostico (o CSV principal segue).
            self.avisar(f"DIAGNOSTICO DA PALMA: falha ao gravar em "
                        f"{self.caminho_csv.name} ({exc}); desligando")
            self.cancelar()
            return False
        self.linhas.append(linha)
        self._ultimo[lado] = agora
        return True

    # --------------------------------------------------------------- resumo
    def _estatisticas_por_fase(self):
        """({(fase, lado): estatisticas de err_deg}, {lado: estatisticas de raw})."""
        grupos = {}
        brutos = {"right": [], "left": []}
        for linha in self.linhas:
            lado = linha.get("lado")
            if lado not in brutos:
                continue
            bruto = _valor_csv_diag(linha.get("err_raw_deg"))
            if np.isfinite(bruto):
                brutos[lado].append(bruto)
            t_fase = _valor_csv_diag(linha.get("t_fase_s"))
            if not np.isfinite(t_fase) or t_fase < PALM_DIAG_SETTLE_S:
                continue
            valor = _valor_csv_diag(linha.get("err_deg"))
            if not np.isfinite(valor):
                continue
            chave = (linha.get("fase_nome"), lado)
            grupos.setdefault(chave, []).append(valor)
        return ({chave: _estatisticas_diag(valores)
                 for chave, valores in grupos.items()},
                {lado: _estatisticas_diag(valores)
                 for lado, valores in brutos.items()})

    def _linha_tabela(self, lado, fase, dados):
        if not dados.get("n"):
            return (f"{lado:5s} | {fase:8s} | {0:4d} | {'-':>7s} | {'-':>7s} | "
                    f"{'-':>6s} | {'-':>6s} | {'-':>6s} | {'-':>6s}")
        return (f"{lado:5s} | {fase:8s} | {dados['n']:4d} | "
                f"{dados['media']:7.2f} | {dados['mediana']:7.2f} | "
                f"{dados['desvio']:6.2f} | {dados['p90']:6.2f} | "
                f"{dados['minimo']:6.2f} | {dados['maximo']:6.2f}")

    def _texto_resumo(self, estat, brutos):
        """Tabela ASCII + veredito por lado (o que vai para o .txt e a tela)."""
        nome_csv = self.caminho_csv.name if self.caminho_csv else "?"
        cabecalho = (f"{'lado':5s} | {'fase':8s} | {'n':>4s} | {'media':>7s} | "
                     f"{'mediana':>7s} | {'std':>6s} | {'p90':>6s} | "
                     f"{'min':>6s} | {'max':>6s}")
        linha_sep = "-" * len(cabecalho)
        linhas = [
            f"DIAGNOSTICO DA PALMA -- {nome_csv}",
            f"settle {PALM_DIAG_SETTLE_S:g} s (amostras do inicio de cada fase "
            f"ficam FORA do resumo, mas no CSV) | limite por fase "
            f"{PALM_DIAG_MAX_ERR_DEG:g} graus",
            "",
            "ERRO AO VIVO (Kinect x IMU) POR FASE E LADO",
            cabecalho,
            linha_sep,
        ]
        for lado in ("right", "left"):
            for fase, _segundos, _instrucao in self.fases:
                linhas.append(self._linha_tabela(lado, fase,
                                                 estat.get((fase, lado), {})))
            linhas.append(linha_sep)
        brutos_txt = " | ".join(
            f"{lado} {dados['media']:.1f} graus"
            for lado, dados in brutos.items() if dados.get("n"))
        if brutos_txt:
            linhas.append("erro SEM alinhamento (R f, T = identidade), sessao "
                          "inteira: " + brutos_txt)
        linhas.append("")
        linhas.append("VEREDITO")
        for lado in ("right", "left"):
            estat_lado = {fase: estat[(fase, lado)] for fase, _s, _i
                          in self.fases if (fase, lado) in estat}
            if not estat_lado:
                linhas.append(f"  {lado}: sem amostras (a mao nao apareceu no "
                              "Kinect durante o diagnostico)")
                continue
            for nota in veredito_palm_diag(estat_lado, brutos.get(lado)):
                linhas.append(f"  {lado}: {nota}")
        linhas.append("")
        linhas.append("Analise completa (refit do modelo, execucao das poses): "
                      "python tools\\analisa_diag_palma.py " + nome_csv)
        return "\n".join(linhas) + "\n"

    def finalizar(self, resumo_txt=None):
        """Fecha o CSV, calcula o resumo e grava `<csv>_resumo.txt`.

        `resumo_txt` sobrescreve o caminho do resumo (None = `<csv>_resumo.txt`).
        Devolve o caminho gravado (None se nao havia nada coletado). Depois
        disto `ativa` e' False e o objeto PODE ser reiniciado por `iniciar`.

        Parada antecipada (D de novo) tambem passa por aqui: perder 15 s de
        coleta so' porque o experimentador quis parar antes seria pior que um
        resumo de amostras parciais -- a coluna `n` diz quanto vale.
        """
        self.ativa = False
        caminho_resumo = (Path(resumo_txt) if resumo_txt is not None
                          else self.caminho_resumo)
        try:
            self._fechar()
            estat, brutos = self._estatisticas_por_fase()
            if not estat and not any(dados.get("n") for dados in brutos.values()):
                self.avisar("DIAGNOSTICO DA PALMA: nenhuma amostra coletada "
                            "(a mao/normal nao apareceu no Kinect); resumo NAO "
                            "gerado")
                return None
            texto = self._texto_resumo(estat, brutos)
            gravado = None
            if caminho_resumo is not None:
                try:
                    caminho_resumo.write_text(texto, encoding="utf-8")
                    gravado = caminho_resumo
                except OSError as exc:
                    self.avisar(f"DIAGNOSTICO DA PALMA: nao consegui gravar "
                                f"{caminho_resumo} ({exc})")
            print(texto, flush=True)
            if gravado is not None:
                self.avisar(f"resumo gravado em {gravado}")
            if self.caminho_csv is not None:
                self.avisar(f"CSV do diagnostico: {self.caminho_csv}")
            if self.caminho_meta is not None and self.caminho_meta.exists():
                self.avisar(f"meta do diagnostico: {self.caminho_meta}")
            return gravado
        finally:
            # Nunca deixar arquivo aberto, mesmo se o resumo estourar no meio
            # (o finally roda DEPOIS do return: `_fechar` e' idempotente).
            self._fechar()
            self._escritor = None
            self.ativa = False

    def cancelar(self):
        """Fecha os arquivos SEM gerar resumo (saida por ESC)."""
        self.ativa = False
        self._fechar()
        self._escritor = None

    def _fechar(self):
        """Fecha o CSV se estiver aberto (idempotente, nunca levanta)."""
        if self._arquivo is not None:
            try:
                self._arquivo.close()
            except OSError:
                pass
            finally:
                self._arquivo = None


class GradeDeJanelas:
    """Cria e POSICIONA as janelas do OpenCV sem uma tapar a outra.

    Por que existe (bancada, 25/09/2026): com o Kinect e as DUAS webcams ligadas
    sao QUATRO fluxos de video (RGB e depth do Kinect + uma janela por webcam).
    Faltavam DUAS coisas:

      1. a janela da webcam nao-ATIVA nao existia -- ela aparecia apenas como
         esqueleto desenhado no RGB do Kinect, e nao havia como conferir foco,
         enquadramento, luz nem se a mao/tabuleiro estavam em quadro NAQUELA
         camera (era o "nao aparece o video da terceira camera");
      2. o Windows abre todas as janelas na MESMA posicao, empilhadas, de modo
         que mesmo as que existem ficam escondidas atras das outras.

    Aqui cada janela e' criada com WINDOW_NORMAL (redimensionavel com o mouse),
    ajustada ao tamanho da imagem e colocada num slot de
    `posicoes_das_janelas` (grade sem sobreposicao; cascata se a grade nao
    couber na tela). A grade e' refeita quando uma janela NOVA entra (ex.: as de
    calibracao, teclas T/K); depois disso o programa NAO mexe mais nas posicoes,
    entao arrastar uma janela e' permitido e permanece.
    """

    def __init__(self, cv2_mod, escala=None, tela=None, avisar=print):
        self.cv2 = cv2_mod
        self.tela = tuple(tela) if tela is not None else tamanho_da_tela()
        #: Escala de EXIBICAO usada por quem chama (reduzida automaticamente
        #: quando o padrao nao faz as janelas caberem -- ver `escala_para_grade`).
        self.escala = float(WINDOW_SCALE if escala is None else escala)
        self.avisar = avisar
        self.tamanhos = {}          # titulo -> (largura, altura) exibidos
        self.posicoes = {}          # titulo -> (x, y)
        self.modo = "sem-tela"      # "grade" | "cascata" | "sem-tela"
        self.criadas = 0
        self._avisou_sem_tela = False

    # ------------------------------------------------------------------ uso
    def mostrar(self, titulo, imagem):
        """Mostra `imagem` (JA' no tamanho de exibicao) na janela `titulo`."""
        if titulo not in self.tamanhos:
            self.tamanhos[titulo] = (int(imagem.shape[1]),
                                     int(imagem.shape[0]))
            self._criar(titulo)
            self._reposicionar()
        self.cv2.imshow(titulo, imagem)

    def redimensionar(self, quadro):
        """Aplica a escala de exibicao a um quadro em tamanho cheio."""
        if quadro is None:
            return None
        if abs(self.escala - 1.0) < 1e-9:
            return quadro
        return self.cv2.resize(quadro, None, fx=self.escala, fy=self.escala)

    def destruir(self):
        """Fecha as janelas que ESTE objeto criou."""
        for titulo in list(self.tamanhos):
            try:
                self.cv2.destroyWindow(titulo)
            except Exception:                    # noqa: BLE001 - ja' fechada
                pass
        self.tamanhos.clear()
        self.posicoes.clear()

    # ------------------------------------------------------------- interno
    def _criar(self, titulo):
        largura, altura = self.tamanhos[titulo]
        try:
            self.cv2.namedWindow(titulo, self.cv2.WINDOW_NORMAL)
            self.cv2.resizeWindow(titulo, largura, altura)
        except Exception:                        # noqa: BLE001 - backend sem
            pass
        self.criadas += 1

    def _reposicionar(self):
        """Coloca TODAS as janelas de novo (grade/cascata) -- so' quando muda."""
        largura_tela, altura_tela = self.tela
        posicoes, modo = posicoes_das_janelas(list(self.tamanhos.values()),
                                              int(largura_tela),
                                              int(altura_tela))
        self.modo = modo
        if posicoes is None:
            if not self._avisou_sem_tela:        # avisa UMA vez
                self._avisou_sem_tela = True
                self.avisar("[janelas] nao deu para ler o tamanho da tela: as "
                            "janelas podem ficar uma sobre a outra (arraste-as)")
            return
        for titulo, (x, y) in zip(self.tamanhos, posicoes):
            try:
                self.cv2.moveWindow(titulo, int(x), int(y))
            except Exception:                    # noqa: BLE001 - janela fechada
                continue
        self.posicoes = dict(zip(self.tamanhos, posicoes))
        if modo == "cascata":
            self.avisar(
                f"[janelas] a grade NAO cabe na tela {largura_tela}x{altura_tela} "
                f"com escala {self.escala:g}: as {len(self.tamanhos)} janelas "
                "foram postas em CASCATA (nenhuma fica exatamente atras da "
                "outra; arraste) -- ou rode sem --escala-janelas, que o padrao "
                "automatico escolhe a maior escala que faz todas caberem")
        else:
            self.avisar(f"[janelas] tela {largura_tela}x{altura_tela} | escala "
                        f"{self.escala:g} | {len(self.tamanhos)} janelas em "
                        "grade, sem sobreposicao")


def rotular_camera_auxiliar(quadro, titulo, linha_status, linha_stereo=None,
                            cor_stereo=(0, 255, 255)):
    """Escreve o rotulo de uma janela de webcam auxiliar JA' REDUZIDA.

    Desenhar DEPOIS da reducao (e nao antes) mantem o texto legivel quando a
    escala de exibicao cai -- em telas com DPI alto ela vai a ~0.4 e um texto
    desenhado no quadro cheio encolheria junto com a imagem.
    """
    cv2.putText(quadro, titulo, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1)
    cv2.putText(quadro, linha_status, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)
    if linha_stereo:
        cv2.putText(quadro, linha_stereo, (10, 66), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, cor_stereo, 1)


def parse_cli():
    """Le as opcoes de linha de comando.

    Existe para resolver um problema pratico: o indice OpenCV da webcam NAO e'
    o mesmo em toda maquina (nesta, 1 = webcam do laptop). Sem escolher o
    indice, o programa abre a camera errada e a calibracao salva -- que e' POR
    INDICE (`stereo_calibration_aux{idx}.npz`) -- nao corresponde ao que a
    camera ve.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=("Ground truth 3D bimanual: Kinect v2 + webcam auxiliar + "
                     "luvas ESP32 (UDP 4210/4211)."),
    )
    parser.add_argument(
        "--aux-index", type=int, default=None, metavar="N",
        help=("indice OpenCV da webcam auxiliar ATIVA (a das teclas K/L/H: trim, "
              "log de desvio e calibracao do esqueleto). PADRAO = AUTOMATICO: a "
              "lista sai do NOME do Windows (todas as externas, as que tem "
              "stereo_calibration_aux{N}.npz primeiro) -- o indice NAO e' "
              "estavel: e' a posicao na enumeracao do DirectShow e muda quando "
              "uma camera entra/sai do USB (medido em 28/09/2026: nesta maquina "
              "so' existem 0 = SIGMA-W780M e 1 = Trust USB Camera, e o antigo "
              "padrao 2 abortava o programa). Fixe um valor so' para forcar uma "
              "camera: a calibracao usada e' stereo_calibration_aux{N}.npz e "
              "quem e' quem se confere com tools\\diagnostico_cameras.py "
              "--salvar."),
    )
    parser.add_argument(
        "--aux-cameras", default="", metavar="\"N,M\"",
        help=("indices das OUTRAS webcams auxiliares a abrir e DESENHAR no video "
              "do Kinect (cada uma com o seu esqueleto, na sua cor: e' a "
              "verificacao visual de que as duas enxergam a mao). Vazio = 'auto': "
              "todas as webcams MENOS a do laptop, pelo NOME do Windows. Se uma "
              "delas nao abrir, o programa avisa e segue com as que abriram."),
    )
    parser.add_argument(
        "--aux-flip-lateralidade", default="", metavar='"N,M"',
        help=("inverte o rotulo de lateralidade (MediaPipe) das webcams "
              "auxiliares listadas -- para quando as maos aparecerem TROCADAS "
              "numa camera em relacao ao Kinect (o driver dela pode ja' entregar "
              "o fluxo espelhado; com dois MODELOS de webcam isso acontece numa "
              "so'). '2' = so' a camera 2; '2,0' = as duas; 'limpar' = esquece "
              "o que foi aprendido. O NORMAL E' NAO PRECISAR: o arbitro "
              "geometrico descobre sozinho (e grava em aux_handedness_flip.json) "
              "assim que as DUAS maos aparecem no quadro -- ver a linha "
              "'Lateralidade aux N' no video."),
    )
    parser.add_argument(
        "--salvar-quadros", type=float, default=0.0, metavar="S",
        help=("grava _aux_overlay.png e _kinect_overlay.png a cada S segundos "
              "(0 = desligado). Serve para conferir DEPOIS se a projecao "
              "vermelha caiu na mao, sem depender de olhar a janela."),
    )
    parser.add_argument(
        "--log-overlay", action="store_true",
        help=("liga o log de desvio da mao (overlay_deviation_log.csv) desde o "
              "INICIO, sem precisar apertar L: 1 linha/frame com o deslocamento "
              "de cada um dos 21 nos (n{n}_dev_px/_dx/_dy) e o contexto "
              "(profundidade, borda, velocidade, palm_err_deg). Use para "
              "conferir DEPOIS, com calma, onde a projecao vermelha saiu da "
              "mao (offset grotesco em alguma posicao)."),
    )
    parser.add_argument(
        "--exposicao", type=str, default=None,
        metavar='"N" ou "i:N,i:N"',
        help=("FIXA o tempo de exposicao das webcams em unidades do DRIVER (o "
              "que cv2.CAP_PROP_EXPOSURE le; no DirectShow costuma ser log2 de "
              "segundos: -7 ~ 1/128 s). POR CAMERA: aceita um valor unico ('-6' "
              "= todas) ou por indice ('0:-6,1:-4'). Por que: medido 19/10/2026 "
              "que as duas webcams pedem valores DIFERENTES na mesma cena (no -7 "
              "a Trust ficou 46 por cento escura e a SIGMA boa). Sem este flag a camera "
              "so' e' congelada na exposicao atual; com ele voce ENCURTA o tempo "
              "(mata o borrado no movimento) -- exige mais LUZ."),
    )
    parser.add_argument(
        "--ganho-cam", type=str, default=None,
        metavar='"N" ou "i:N,i:N"',
        help=("ganho das webcams em unidades do DRIVER, mesma sintaxe do "
              "--exposicao (valor unico ou '0:N,1:N'). Medido na bancada: o "
              "driver da SIGMA aceita o valor mas QUASE NAO muda a imagem -- o "
              "que resolve luz baixa e' exposicao + luz fisica; fica aqui para "
              "as cameras cujo ganho funciona."),
    )
    parser.add_argument(
        "--fps-cam", type=int, default=None, metavar="N",
        help=("pede um fps fixo para as webcams (ex.: 60 se a camera suportar "
              "720p60). Menos tempo entre quadros = menos deslocamento por "
              "frame no movimento rapido. Sem flag, o fps fica no atual."),
    )
    parser.add_argument(
        "--sem-trava-exposicao", action="store_true",
        help=("desliga a trava de auto-exposicao/autofocus (padrao: LIGADA -- "
              "a imagem para de variar durante a sessao e o foco nao respira). "
              "So' para comparar A/B ou usar o controle automatico de luz."),
    )
    parser.add_argument(
        "--escala-janelas", type=float, default=None, metavar="F",
        help=("escala das janelas de video (padrao: automatica -- o maior valor "
              "<= 0.5 que faz TODAS as janelas caberem na tela; em telas com "
              "DPI alto ela cai para ~0.4). Fixe um valor (ex.: 0.5) se preferir "
              "janelas maiores e arrastar voce mesmo."),
    )
    parser.add_argument(
        "--imu-portas", default="4210,4211", metavar='"P_D,P_E"',
        help=("portas UDP das luvas, na ordem DIREITA,ESQUERDA (padrao "
              "'4210,4211'). Aceita tambem o formato 'lado:porta' (ex.: "
              "'1:4211,2:4210'). Serve para quando o firmware das luvas esta' "
              "invertido: e' assim que se confere se a luva que aparece como "
              "'esquerda' e' mesmo a mao esquerda."),
    )
    parser.add_argument(
        "--sem-ancora-yaw", action="store_true",
        help=("desliga a REANCORAGEM AUTOMATICA do yaw do boot. Com alinhamento "
              "salvo, o tool resolve nas primeiras amostras (palma fora da "
              "vertical) o giro de yaw que o IMU ganha a cada BOOT -- o filtro "
              "de 6 eixos nao tem magnetometro, o yaw absoluto e' arbitrario -- "
              "e corrige o T do alinhamento EM MEMORIA (o JSON salvo nao muda e "
              "continua valendo para os outros boots). Desligue para comparar "
              "com o comportamento antigo (seta certa na MESA e ~100 graus fora "
              "com a palma de frente)."),
    )
    parser.add_argument(
        "--participante", default="",
        help=("identificacao do participante mostrada no AVISO de CALIBRACAO "
              "dos IMUs no inicio da sessao (item 101)."),
    )
    parser.add_argument(
        "--sem-calibra-imu", action="store_true",
        help=("pula o ritual de CALIBRACAO dos IMUs do inicio da sessao (reset "
              "pela serial + conferencia do bias parado). Use so' para comparar "
              "ou quando as luvas ja' foram reiniciadas paradas."),
    )
    parser.add_argument(
        "--porta-imu-serial", default="",
        help=("porta serial do ESP32 das luvas para o reset de calibracao "
              "(padrao: detecta o CP210x). Ex.: COM4."),
    )
    parser.add_argument(
        "--log-palma", action="store_true",
        help=("liga o LOG DO VETOR DA PALMA (1 linha/frame/mao em "
              "palma_vetor_log_<stamp>.csv: mapeamento/depth, normais, "
              "spread, erro x IMU, sync) para diagnosticar seta ruim ou "
              "sumindo -- tecla G tambem liga/desliga no video)."),
    )
    return parser.parse_args()


def cabecalho_overlay_log():
    """Cabecalho do overlay_deviation_log.csv (tecla L OU --log-overlay).

    Existe para os DOIS caminhos escreverem o MESMO cabecalho: a lista estava
    so' dentro da tecla L, e copiar ela em dois lugares e' como o arquivo sai
    com colunas divergentes dependendo de quem abriu.

    ORDEM = ORDEM DA LINHA escrita no loop (fixas -> nos -> palma). Antes a
    palma vinha ANTES dos nos no cabecalho, mas a linha grava os nos PRIMEIRO e
    a palma POR ULTIMO (ver o bloco `row = [...]` em main): o DictReader lia
    `n0_dev_px` na posicao do `n2_dx` (e o `dev_px` NEGATIVAMENTE, que e'
    impossivel -- e' magnitude). Quem le por NOME (analise_estatistica.py e o
    proprio analyzer) ficava com colunas trocadas. Corrigido 08/10/2026.
    """
    header = ["t_s", "lado", "z_usado_m", "z_bruto_m", "cx_px",
              "cy_px", "borda_0c_1b", "vel_px_s", "note"]
    for n in range(21):
        header += [f"n{n}_dev_px", f"n{n}_dx", f"n{n}_dy"]
    header += ["palm_kin_x", "palm_kin_y", "palm_kin_z",
               "palm_imu_x", "palm_imu_y", "palm_imu_z",
               "palm_err_deg"]
    return header


def aplica_flip_lateralidade(texto):
    """Aplica `--aux-flip-lateralidade` (forcar inversao ou limpar o aprendido).

    Existe para o caso em que o usuario JA' SABE qual camera esta' invertida (o
    arbitro so' descobre com as DUAS maos no quadro, comparando com o Kinect):
    aqui ele fixa na mao, e o valor fica GRAVADO para as proximas sessoes --
    inclusive para o protocolo, que so' tem o rotulo do MediaPipe.

    Devolve {indice: True} do que foi forcado (vazio quando nada foi pedido).
    """
    pedido = str(texto or "").strip().lower()
    if not pedido:
        return {}
    if pedido in ("limpar", "esquecer", "esquece", "off", "none"):
        antes = limpa_aux_handedness_flip()
        print("--aux-flip-lateralidade limpar: "
              + (f"inversoes esquecidas {sorted(antes)}" if antes
                 else "nao havia inversao gravada")
              + " -- a lateralidade volta a ser a do rotulo do MediaPipe")
        return {}
    invertidas = {}
    for parte in pedido.replace(";", ",").split(","):
        parte = parte.strip()
        if parte.startswith("aux"):
            parte = parte[3:].lstrip("=: ")
        if not parte.lstrip("-").isdigit():
            raise SystemExit(
                f"--aux-flip-lateralidade invalido ('{parte}'): use indices das "
                "webcams (ex.: '2' ou '2,0') ou 'limpar'.")
        indice = int(parte)
        set_aux_handedness_flip(
            indice, True, silencioso=True,
            motivo="pedido na linha de comando (--aux-flip-lateralidade)")
        invertidas[indice] = True
    print(f"--aux-flip-lateralidade: rotulo do MediaPipe INVERTIDO para as cameras "
          f"{sorted(invertidas)} (gravado em {AUX_HANDEDNESS_FLIP_FILE.name}; "
          "'limpar' desfaz)")
    return invertidas


def _sem_numeros(texto):
    """Texto com os NUMEROS trocados por '#'.

    A chave de estado NAO pode carregar o valor medido (\"projecao a 302 px\"): ele
    muda a cada frame e imprimiria a MESMA linha para sempre. O que interessa
    para o usuario ver e' a NATUREZA do motivo (margem pequena, projecao longe,
    lado repetido) -- e essa muda de verdade.
    """
    return "".join("#" if caractere.isdigit() else caractere
                   for caractere in (texto or ""))


#: Intervalo minimo (s) entre duas impressoes do veredito de lateralidade. Sem
#: ele, um motivo que oscila (margem pequena <-> projecao longe) imprimia duas
#: linhas por frame durante toda a sessao.
#: 2 s era inundacao (item 75): o veredito REAL alterna frame a frame
#: (indecisa <-> geometria) durante o movimento e o pendente rearmava a cada
#: virada -- 122 linhas [lateralidade] + 71 [gate-tri] em minutos de uso.
#: 10 s entre reemissoes + ESTABILIDADE (abaixo) para o estado PEGAR antes.
INTERVALO_LOG_LATERALIDADE_S = 10.0
#: Um estado novo so' arma impressao depois de persistir isto sem piscar --
#: mata o flicker indecisa<->geometria sem esconder mudanca que PEGA.
ESTABILIDADE_LOG_LATERALIDADE_S = 3.0
#: Uma linha [status] consolidada (taxa IMU por porta, alinhamento O SIM/NAO,
#: erro da palma ao vivo) -- a saude do IMU num lugar so' no terminal.
STATUS_LOG_INTERVALO_S = 5.0

#: Arquivo do diagnostico "SEM vermelho" (oclusao 14/10): estado de cada frame
#: em que a mao EXISTE mas o overlay nao nasce. Escrito em arquivo porque a
#: sidebar do PowerShell nao da' Ctrl+C/V -- o experimentador roda e o arquivo
#: e' lido direto. Truncado na primeira recusa de cada sessao.
DIAG_OVERLAY_PATH = Path(__file__).resolve().parent.parent / "overlay_diag.log"

#: --- GATES da triangulacao (picos do overlay em movimento rapido) ------------
#: (1) Erro de reprojecao mediano em px. As webcams NAO sao sincronizadas com o
#: Kinect: com a mao rapida, `aux_px` e `kin_px` vem de momentos DIFERENTES, os
#: raios nao fecham e o erro de reprojecao SOBE. Acima disto a medicao e'
#: DESCARTADA e o overlay cai no depth do Kinect (suave) em vez de projetar um
#: 3D errado. 0 = desligado.
TRI_ERR_MAX_PX = 6.0
#: (2) Velocidade maxima plausivel da mao (m/s) entre DOIS aceites seguidos. Um
#: pico de dessincronia implica um SALTO de velocidade absurda e e' SEGURADO (o
#: ultimo ponto bom fica; o dt ate' ele so' cresce, entao movimento legitimo
#: rapido passa no frame seguinte enquanto o pico segue rejeitado). 0 = desligado.
TRI_VEL_MAX_M_S = 5.0
#: (o TRI_HOLD_MAX_S por IDADE foi removido em 09/10: a regra certa virou
#: comparar o erro projetado da bruta x do hold e usar o MENOR -- ver o bloco
#: `else` da triangulacao; idade batia errado: hold "fresco" de 0,25 s deu
#: med 30 px contra 10 px da bruta rejeitada na bancada da esquerda.)


def avalia_triangulacao(tri3d, tri_err, ref_anterior, dt,
                        max_err_px=None, max_vel_m_s=None):
    """(aceita, categoria, motivo, palm) de UMA triangulacao, sob os DOIS gates.

    `ref_anterior` = (palm_xyz, t) do ultimo ACEITE (None no 1o frame); `dt` =
    segundos desde esse aceite (None se desconhecido). `palm` devolvido e' o no'
    9 (ou a mediana) -- o chamador guarda (palm, agora) SO' quando aceita.

    Nos DOIS rejeitos a referencia NAO avanca: assim o `dt` do proximo frame e'
    maior, o movimento legitimo rapido passa na hora seguinte, e um pico (salto)
    segue rejeitado ate' a mao voltar perto do ultimo ponto aceito.

    Funcao PURA (sem estado, sem impressao) -- e' o que o teste exercita.
    `max_err_px`/`max_vel_m_s` sao 0 para desligar cada gate.
    """
    if max_err_px is None:
        max_err_px = TRI_ERR_MAX_PX
    if max_vel_m_s is None:
        max_vel_m_s = TRI_VEL_MAX_M_S
    tri = np.asarray(tri3d, np.float64).reshape(-1, 3)
    palm = (tri[9] if tri.shape[0] > 9 and np.isfinite(tri[9]).all()
            else np.nanmedian(tri, axis=0))
    # --- Gate 1: erro de reprojecao (dessincronia aux x Kinect) ------------
    if max_err_px > 0 and tri_err is not None:
        err = np.asarray(tri_err, np.float64).ravel()
        err = err[np.isfinite(err)]
        if err.size:
            med = float(np.median(err))
            if med > max_err_px:
                return False, "reproj", f"reprojecao {med:.1f} px", None
    # --- Gate 2: velocidade (pico) -----------------------------------------
    if max_vel_m_s > 0 and ref_anterior is not None and dt and dt > 1e-3:
        ant = np.asarray(ref_anterior[0], np.float64)
        if np.isfinite(palm).all() and np.isfinite(ant).all():
            vel = float(np.linalg.norm(palm - ant) / dt)
            if vel > max_vel_m_s:
                return False, "vel", f"velocidade {vel:.1f} m/s", None
    return True, "ok", None, palm


def estado_lateralidade(relatorio):
    """Chave de estado do veredito de lateralidade, por camera.

    Serve para o tool imprimir UMA vez quando o veredito de uma camera MUDA (ela
    passou a ser medida pela geometria? o arbitro aprendeu a inversao dela?). O
    estado e' o que muda quando algo relevante acontece: a fonte do lado, se o
    rotulo estava invertido e a NATUREZA do motivo de nao decidir (sem os numeros
    -- ver `_sem_numeros`).
    """
    return {int(indice): (info.get("fonte"), info.get("inverteu"),
                          _sem_numeros(info.get("detalhe")))
            for indice, info in (relatorio or {}).items()}


def _ema(anterior, novo, alfa=0.2):
    """Media movel exponencial das metricas de desempenho (None = primeira)."""
    return (float(novo) if anterior is None
            else (1.0 - alfa) * float(anterior) + alfa * float(novo))


def abre_camera_auxiliar(indice, tamanho, exposicao=None, fps=None,
                         trava_expo=True, ganho=None):
    """Abre a webcam auxiliar no MESMO backend que a calibracao usou.

    O DSHOW vem primeiro: `stereo_calibration.py` calibra por ele (AUX_BACKENDS)
    e ele abre em ~1 s, contra os 20-40 s do MSMF. Se o driver nao entregar
    frame, cai para o MSMF -- nesta maquina os dois backends enumeram os
    MESMOS indices, entao trocar de backend nao troca de camera.

    `trava_expo` (padrao LIGADO) congela auto-exposicao/autofocus via
    `camera_uvc.trava_exposicao`, que IMPRIME o que a camera obedeceu e o que
    ignorou; com `exposicao`/`fps` dados, tempo/fps tambem sao FIXADOS
    (unidades do driver -- ver `--exposicao`/`--fps-cam`).
    """
    for backend, nome in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
        for tentativa, mjpeg in enumerate((True, False)):
            captura = cv2.VideoCapture(indice, backend)
            if not captura.isOpened():
                captura.release()
                if tentativa == 0:
                    print(f"Camera auxiliar {indice}: backend {nome} nao abriu.")
                break                      # o backend nao abre: nao repete
            #: MJPEG: no padrao (720p bruto/YUY2) esta bancada mediu 8-10 fps
            #: (100-125 ms/quadro = os picos de movimento); MJPEG entrega 30 na
            #: MESMA resolucao. `define_modo` faz W/H ANTES do fourcc -- ordem
            #: inversa renegocia e volta pra YUY2 (medido em 10/10). O modo que
            #: VAI valer e' o que `le_fourcc` LER da stream, nao o pedido.
            define_modo(captura, tamanho[0], tamanho[1], mjpeg=mjpeg)
            captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if trava_expo:
                trava_exposicao(captura, exposicao=exposicao, ganho=ganho,
                                fps=fps, rotulo=f"aux {indice}")
            for _ in range(4):
                captura.read()
            ok, quadro = captura.read()
            if ok and quadro is not None:
                real = le_fourcc(captura)
                if mjpeg and real not in ("?", "MJPG"):
                    #: pediu MJPEG e a stream NAO ficou MJPG -> desiste e
                    #: reabre no modo padrao (fallback daqui abaixo)
                    captura.release()
                    print(f"Camera auxiliar {indice}: stream ficou em "
                          f"'{real}' (esperava MJPG); reabrindo no padrao...")
                    continue
                modo = (real if real != "?" else
                        ("MJPEG" if mjpeg else "bruto"))
                print(
                    f"Camera auxiliar {indice}: backend {nome} OK "
                    f"({quadro.shape[1]}x{quadro.shape[0]}, {modo})."
                )
                return captura
            captura.release()
            if mjpeg:
                print(f"Camera auxiliar {indice}: MJPEG sem frame; "
                      f"reabrindo no modo padrao...")
    raise RuntimeError(
        f"Nao consegui abrir a camera auxiliar de indice {indice} (DSHOW nem "
        "MSMF). Rode `python tools\\diagnostico_cameras.py --salvar` para ver "
        "qual indice e' qual webcam."
    )

def _indices_auxiliares(args):
    """Indices das webcams a abrir: a ATIVA (--aux-index) + as OUTRAS.

    `--aux-cameras` vazio (ou 'auto') resolve pelo NOME do Windows, deixando a
    webcam do LAPTOP de fora (ver `indices_auxiliares_automaticos`). A camera
    ativa vem PRIMEIRO: e' ela que fica com as teclas K/L/H e o overlay
    vermelho-verde ja' validado.
    """
    pedido = str(args.aux_cameras).strip()
    if pedido in ("", "auto"):
        outros = indices_auxiliares_automaticos()
    else:
        outros = tuple(int(indice) for indice in pedido.split(",")
                       if indice.strip().lstrip("-").isdigit())
    # A ATIVA vem do global (o `main` resolve o automatico ANTES daqui); o
    # `args.aux_index` so' manda quando o usuario fixou um valor.
    ativa = (int(AUX_CAMERA_INDEX) if args.aux_index is None
             else int(args.aux_index))
    ordem = [ativa]
    for indice in outros:
        if int(indice) not in ordem:
            ordem.append(int(indice))
    return ordem

def nome_gravado_no_npz(alignment, aux_index=None):
    """Nome da camera para a qual o npz carregado foi GRAVADO (None se nao consta).

    Existe para a mensagem poder dizer *qual* camera o arquivo julga ser: em
    28/09/2026 a bancada estava com uma calibracao de outra maquina e isso nao
    aparecia em lugar nenhum (o RMS do arquivo nao sabe de quem ele e').
    """
    modelos = getattr(alignment, "aux_models", None) or {}
    if aux_index is not None and int(aux_index) in modelos:
        return modelos[int(aux_index)].get("camera_name")
    return getattr(alignment, "stereo_camera_name", None)


def diagnostico_tabuleiro(color, color_for_calibration, auxiliary_frame, tracker,
                          alignment, aux_ok=True, aux_index=None):
    """Erro ponta-a-ponta da sobreposicao no tabuleiro (o diagnostico da tecla T).

    O tabuleiro e' uma superficie rigida com correspondencia EXATA entre as duas
    cameras: por isso ele separa erro de projecao/calibracao dos efeitos
    especificos da mao. Imprime o resumo e devolve os numeros num dicionario (ou
    None quando o diagnostico nao pode ser feito).

    Esta aqui, e nao dentro do loop de teclas, para poder ser testada sem
    hardware -- dois bugs reais nasceram deste bloco: o `find_board` que nao
    achava a placa com a mesa cheia e o `errors[24]`, que era o "centro" da placa
    ANTIGA de 11x7 = 77 cantos e estourava IndexError com a atual (5x3 = 15).

    Alem do erro ponta-a-ponta, separa DUAS causas que pedem consertos opostos:
    `homografia_nova_px` (homografia ajustada AGORA com os MESMOS cantos, logo
    mede a correspondencia/deteccao) e o nome da camera gravada no npz. Erro
    grande com correspondencia boa = calibracao de outra montagem (RECALIBRE);
    erro grande sem correspondencia = captura ruim (melhore luz/postura antes).
    `hasattr` no `camera_name_conflict`: as classes falsas dos testes nao
    implementam a checagem.
    """
    stats = tracker.depth_stats()
    if stats is None:
        print("Diagnostico depth: NENHUM frame de profundidade recebido")
    else:
        count, dmin, dmax, frac = stats
        print(
            f"Diagnostico depth: {count}/{tracker.last_depth.size} px "
            f"validos ({frac:.0%}) | min {dmin:.0f} max {dmax:.0f} mm | "
            f"frames recebidos ate agora: {tracker.depth_frame_index}"
        )
    if color is None or not aux_ok:
        print("Diagnostico: sem frame simultaneo das duas cameras")
        return None
    if not alignment.stereo_usable:
        print("Diagnostico: calibracao stereo nao carregada")
        return None
    clean_color = (
        color_for_calibration
        if color_for_calibration is not None
        else cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
    )
    if clean_color.ndim == 3 and clean_color.shape[2] == 4:
        clean_color = cv2.cvtColor(clean_color, cv2.COLOR_BGRA2BGR)
    board_kinect = alignment.find_board(clean_color)
    board_aux = alignment.find_board(auxiliary_frame)
    if board_kinect is None or board_aux is None:
        # Diz QUAL camera nao viu: a mensagem antiga ("nao visto nas duas
        # cameras") mandava procurar o problema no tabuleiro quando o caso comum
        # e' o tabuleiro estar FORA do enquadramento de UMA delas (a auxiliar e'
        # mais fechada que o Kinect).
        print(
            "Diagnostico: tabuleiro NAO visto em "
            f"Kinect={'OK' if board_kinect is not None else 'nao'}"
            f" | auxiliar (indice {aux_index})="
            f"{'OK' if board_aux is not None else 'nao'}"
            " -- se o tabuleiro esta' na mesa, confira o que cada camera enxerga "
            "(diagnostico_tabuleiro.py) e se ele cabe no quadro")
        return None
    print("Diagnostico: processando tabuleiro...")
    # O npz pode ser de OUTRA camera (o indice do OpenCV e' a posicao na
    # enumeracao do USB): nesse caso nenhum numero abaixo vale. So' os npz
    # gravados a partir de 29/09/2026 dizem de quem sao (ver camera_name_conflict).
    conflito = None
    if hasattr(alignment, "camera_name_conflict"):
        conflito = alignment.camera_name_conflict(aux_index)
    if conflito:
        print(f"Diagnostico: este npz foi GRAVADO PARA OUTRA CAMERA "
              f"('{conflito}' x '{_gt.nome_da_camera_do_indice(aux_index)}' no "
              f"indice {aux_index} agora): os numeros abaixo saem de um par que "
              "nao e' este -- recalibre (python stereo_calibration.py)")
    depths = np.full(len(board_kinect), np.nan, np.float32)
    for index, (corner_x, corner_y) in enumerate(board_kinect.reshape(-1, 2)):
        depth_pixel = tracker._color_to_depth_pixel(int(corner_x),
                                                    int(corner_y))
        if depth_pixel is None:
            continue
        depth_mm = tracker._median_depth(*depth_pixel,
                                        radius=DEPTH_PATCH_RADIUS * 3)
        if depth_mm is not None:
            depths[index] = depth_mm / 1000.0
    valid = np.isfinite(depths)
    print(f"Diagnostico: {int(valid.sum())}/{len(depths)} cantos com "
          "profundidade valida do Kinect")
    if int(valid.sum()) < 5:
        print(
            "Diagnostico: tabuleiro fora do alcance do depth do Kinect; afaste "
            "o tabuleiro para >= 0.8 m e mantenha-o dentro do FOV do depth "
            "(evite as bordas extremas do video)")
        return None
    depths = np.where(valid, depths, float(np.nanmedian(depths)))
    # PAREAMENTO DOS CANTOS: o tabuleiro e' simetrico, entao a ordem que cada
    # detector devolve nao diz qual canto fisico e' o primeiro -- e cada camera
    # escolhe o seu. Projetar as listas cruas assim compara cantos que NAO se
    # correspondem (centenas de px com a calibracao perfeita). Aqui as 4
    # simetrias da grade sao testadas e fica a de MENOR erro, que e' o que o
    # solver da calibracao tambem faz (`solve_with_repair`): com a calibracao
    # certa o pareamento correto e' ordens de grandeza melhor que os outros.
    candidatos = []
    for nome, indices in _gt.simetrias_do_tabuleiro().items():
        cantos = board_aux.reshape(-1, 2)[indices]
        proj = np.asarray(
            alignment.auxiliary_to_kinect_stereo(cantos, depths), np.float64
        ).reshape(-1, 2)
        erros_c = np.linalg.norm(proj - board_kinect.reshape(-1, 2), axis=1)
        candidatos.append({
            "nome": nome,
            "media_px": float(np.nanmedian(erros_c)),
            "cantos": cantos,
            "projetado": proj,
            "erros": erros_c,
        })
    candidatos.sort(key=lambda item: item["media_px"])
    escolhido = candidatos[0]
    alternativas = ", ".join(
        "%s %.0f" % (item["nome"], item["media_px"]) for item in candidatos[1:])
    print(f"Diagnostico: pareamento dos cantos {escolhido['nome']} "
          f"({escolhido['media_px']:.0f} px) | as outras dariam: {alternativas}")
    projected = escolhido["projetado"]
    errors = escolhido["erros"]
    board_aux = escolhido["cantos"]
    pareamento = escolhido["nome"]
    alternativas_px = {item["nome"]: item["media_px"] for item in candidatos}
    finite = np.isfinite(errors)
    finite_err = errors[finite]
    # A CORRESPONDENCIA EM SI ESTA' BOA? Ajusta uma homografia NOVA entre os
    # cantos DESTE par, agora: o tabuleiro e' plano, entao correspondencia
    # correta = homografia quase perfeita. Separar as duas coisas importa porque
    # o conserto e' oposto -- correspondencia ruim pede MELHOR CAPTURA,
    # calibracao de outra montagem pede RECALIBRAR (medido 28/09/2026: 214 px
    # com o npz gravado x 0,3 px com a homografia do mesmo par).
    homografia_nova = None
    if int(finite.sum()) >= 4:
        cantos_aux32 = board_aux.reshape(-1, 1, 2).astype(np.float32)
        cantos_kin32 = board_kinect.reshape(-1, 1, 2).astype(np.float32)
        matriz, _ = cv2.findHomography(cantos_aux32, cantos_kin32, cv2.RANSAC,
                                      3.0)
        if matriz is not None:
            reescalonado = cv2.perspectiveTransform(cantos_aux32,
                                                    matriz).reshape(-1, 2)
            residuos = np.linalg.norm(
                reescalonado - board_kinect.reshape(-1, 2), axis=1)
            homografia_nova = float(np.nanmedian(residuos))
            if homografia_nova <= LIMITE_PX_CORRESPONDENCIA:
                print(
                    "Diagnostico: deteccao/correspondencia OK -- a homografia "
                    f"dos MESMOS cantos, ajustada agora, fecha em "
                    f"{homografia_nova:.2f} px (entao um erro grande no resumo "
                    "e' da CALIBRACAO, nao do detector)")
            else:
                print(
                    "Diagnostico: a propria CORRESPONDENCIA esta' ruim "
                    f"({homografia_nova:.1f} px entre os mesmos cantos): "
                    "tabuleiro borrado/pequeno/reflexivo, ou ele mudou entre as "
                    "duas capturas (elas nao sao sincronizadas)")
    correspondencia_ok = (homografia_nova is not None
                          and homografia_nova <= LIMITE_PX_CORRESPONDENCIA)
    differences = projected - board_kinect.reshape(-1, 2)
    dx = float(np.nanmean(differences[:, 0]))
    dy = float(np.nanmean(differences[:, 1]))
    # Centro x bordas pela GEOMETRIA do tabuleiro (ver _gt.grupos_centro_borda):
    # antes eram os indices da placa antiga (errors[24],
    # errors[:8]/errors[-8:]) e a tecla T estourava IndexError com a placa atual.
    indice_centro, bordas, raios = _gt.grupos_centro_borda(board_kinect)
    erro_centro = float(errors[indice_centro])
    erro_bordas = float("nan")
    if bordas.any():
        erro_bordas = float(np.nanmean(errors[bordas]))
    slope = None
    if finite_err.size >= 2:
        r_median = max(1e-9, float(np.median(raios[finite])))
        slope = float(np.polyfit(raios[finite] / r_median, finite_err, 1)[0])
    print(
        f"Diagnostico tabuleiro: {np.nanmean(errors):.1f} px media | "
        f"mediana {float(np.nanmedian(errors)):.0f} | "
        f"max {np.nanmax(errors):.0f} | centro {erro_centro:.0f} | "
        f"bordas ({int(bordas.sum())} de {len(errors)}) {erro_bordas:.0f} | "
        f"dx {dx:.0f} dy {dy:.0f} | "
        f"slope {slope if slope is not None else '--'}"
    )
    if float(np.nanmedian(errors)) > LIMITE_PX_CALIBRACAO_SUSPEITA:
        if correspondencia_ok:
            # O tabuleiro e' o teste ponta-a-ponta da calibracao: erro nesta
            # ordem (dezenas/centenas de px) com a correspondencia BOA nao e'
            # ruido de deteccao, e' calibracao de OUTRA montagem -- medido em
            # 28/09/2026: a gravada no repositorio (feita na maquina antiga)
            # deu 491 px aqui, e 214 px nesta bancada, enquanto uma homografia
            # NOVA do MESMO par fechou em 0,3 px.
            gravado = nome_gravado_no_npz(alignment, aux_index)
            print(
                "Diagnostico: ATENCAO -- erro muito acima do esperado para esta "
                "montagem (o RMS do arquivo so' mede a coerencia INTERNA dele, "
                "nao se ele e' deste lugar). RECALIBRE nesta maquina: "
                "python stereo_calibration.py (e confira o esqueleto vermelho "
                "sobre a mao com as teclas K/L/H)"
                + (f" [este npz diz ter sido gravado para '{gravado}']"
                   if gravado else "")
            )
            print("Diagnostico: causas comuns, na ordem em que vale conferir:")
            print("   1. a webcam auxiliar saiu do lugar/zoom ou trocou de "
                  "porta USB (o INDICE muda: confira "
                  "python tools\\diagnostico_cameras.py);")
            print("   2. o npz e' de outra maquina/bancada;")
            print("   3. placa/quadrado diferente do atual (o npz grava os dois).")
        else:
            # Correspondencia ruim: recaribrar agora seria trabalho perdido --
            # o mesmo erro grande apareceria no arquivo novo.
            print(
                "Diagnostico: ATENCAO -- antes de recalibrar, melhore a "
                "CAPTURA: o tabuleiro nao fecha nem com uma homografia ajustada "
                "agora (luz difusa apontada para o tabuleiro, tabuleiro apoiado "
                "e PARADO nas duas capturas, sem reflexo, e maior no quadro)."
            )
    return {
        "cantos": len(errors),
        "media_px": float(np.nanmean(errors)),
        "mediana_px": float(np.nanmedian(errors)),
        "max_px": float(np.nanmax(errors)),
        "centro_px": erro_centro,
        "bordas_px": erro_bordas,
        "bordas_cantos": int(bordas.sum()),
        "dx_px": dx,
        "dy_px": dy,
        "slope": slope,
        "homografia_nova_px": homografia_nova,
        "correspondencia_ok": correspondencia_ok,
        "cantos_com_profundidade": int(valid.sum()),
        "pareamento_cantos": pareamento,
        "pareamento_px": alternativas_px,
    }


def main():
    args = parse_cli()
    # O indice escolhido vale para a camera aberta E para o arquivo de
    # calibracao (stereo_calibration_aux{N}.npz) -- os dois andam juntos.
    # Sem `--aux-index`, a camera ATIVA vem do NOME do Windows
    # (`indice_auxiliar_automatico`): a constante 2 do modulo era o default
    # antigo e abortava o programa nesta maquina, onde so' existem 0 e 1.
    global AUX_CAMERA_INDEX
    if args.aux_index is None:
        AUX_CAMERA_INDEX = indice_auxiliar_automatico()
        print(f"Webcam auxiliar ATIVA: indice {AUX_CAMERA_INDEX} escolhido "
              f"AUTOMATICAMENTE ({nome_auxiliar(AUX_CAMERA_INDEX)}); use "
              "--aux-index N para forcar outra")
    else:
        AUX_CAMERA_INDEX = int(args.aux_index)
        externas = indices_auxiliares_automaticos()
        if externas and AUX_CAMERA_INDEX not in externas:
            print(f"AVISO: o indice {AUX_CAMERA_INDEX} nao esta' entre as "
                  f"cameras EXTERNAS desta maquina ({externas}): pode ser a "
                  "webcam do laptop ou um indice que nao existe. Confira com "
                  "tools\\diagnostico_cameras.py --salvar")
    # --- BIMANUAL: um receptor de IMU POR mao -------------------------------
    # Direita -> porta 4210, esquerda -> 4211 (convencao do projeto, imu.py),
    # ajustavel por `--imu-portas` quando o firmware das luvas estiver invertido.
    # O lado 1 do formato 'lado:porta' e' a DIREITA (ARM_SIDE_CODE).
    try:
        portas = dict(parse_portas_imu(args.imu_portas))
    except ValueError as exc:
        raise SystemExit(
            f"--imu-portas invalido ({exc}); use '4210,4211' (direita,esquerda) "
            "ou '1:4210,2:4211'.")
    # --- CALIBRACAO DOS IMUs NO INICIO DA SESSAO (item 101) -----------------
    # O bias do giroscopio E OS OFFSETS DO ACELEROMETRO sao calibrados NO BOOT
    # (itens 101/124): as luvas tem de estar PARADAS e DEITADAS na mesa (uma
    # face do modulo IMU paralela a mesa) para o acelerometro medir um eixo
    # com +-1 g e os outros com ~0. Medido em 21/10/2026: boot com a luva em
    # movimento deixou -12,5 dps no gy da direita = 434 graus/min de deriva de
    # yaw; reiniciando parado caiu para -0,20 graus/min. O ritual avisa o
    # participante, reinicia o ESP32 pela serial e CONFERE o bias parado pelo
    # UDP -- antes dos receptores abrirem as portas. Falha aqui nao derruba a
    # sessao (so' avisa). O boot agora calibra 2 s por MPU + Wi-Fi; 15 s de
    # espera dao margem (antes 10 s podia cortar no meio da calibracao).
    if not getattr(args, "sem_calibra_imu", False):
        try:
            args.imu_calibracao = imu_reset.calibra_imu(
                participante=getattr(args, "participante", ""),
                portas=tuple(portas.values()),
                porta_serial=(getattr(args, "porta_imu_serial", "") or None),
                segundos_boot=15.0,
                confirmar=True)
        except Exception as exc:                    # noqa: BLE001 - nao derrubar
            print(f"AVISO: calibracao dos IMUs falhou ({type(exc).__name__}: "
                  f"{exc}); siga e confira o bias manualmente")
            args.imu_calibracao = None
    else:
        print("Calibracao dos IMUs PULADA (--sem-calibra-imu): confira o bias "
              "parado manualmente (tools\\deriva_estatica_imu.py).")
    receivers = {
        "right": ImuReceiver(portas.get(1, 4210)),
        "left": ImuReceiver(portas.get(2, 4211)),
    }
    print(f"IMU: direita na porta {receivers['right'].port} | esquerda na porta "
          f"{receivers['left'].port}")
    print("TECLAS (C/B/M/O/D/G/J/K/H/L/R/V/X/N): clique ANTES em uma janela de VIDEO -- sem "
          "foco no video o OpenCV nao recebe a tecla (a tecla O parece nao "
          "funcionar e o console fica mudo).")
    print("  M = MEDIR NIVEL DA MESA (plano do Kinect, compensa inclinacao)")
    print("  J = CALIBRAR por 4 POSES (TAKES: 2 s PARADO "
              "grava sozinho, sem tempo fixo; mao a ~50 cm "
              "da lente mirando a lente: centro/esquerda/"
              "direita/baixo) | V = VERIFICAR | "
              "G = log do vetor")
    for receiver in receivers.values():
        receiver.start()
    # --- webcams auxiliares: a ATIVA (--aux-index) + as OUTRAS (--aux-cameras) ---
    # A resolucao vem ANTES do tracker/alignment porque os dois precisam da
    # lista: o tracker cria UM detector MediaPipe por camera (o VideoMode nao
    # admite intercalar fluxos) e o alignment carrega a calibracao de cada uma
    # (as vistas por indice sao o que permite desenhar as DUAS no mesmo frame).
    # --- LATERALIDADE: o forcado/limpo na linha de comando vale ANTES de criar o
    # tracker (o detector le' a inversao no momento de cada deteccao).
    aplica_flip_lateralidade(args.aux_flip_lateralidade)
    aux_indices = _indices_auxiliares(args)
    tracker = KinectHandTracker(aux_indices=tuple(aux_indices))
    # Carrega SO a calibracao do indice escolhido: os npz sao POR INDICE
    # (stereo_calibration_aux{N}.npz) e usar o arquivo de outra camera projeta o
    # esqueleto do Kinect em qualquer lugar da imagem.
    alignment = CameraAlignment(aux_indices=aux_indices)
    # Arbitro de lateralidade: decide o lado das maos auxiliares pela GEOMETRIA
    # (punho 3D do Kinect + homografia da mao de cada camera) e aprende/grava a
    # inversao de rotulo por camera. Sem ele, uma auxiliar com o rotulo trocado
    # casa as maos erradas -- e o casamento por lado e' o que liga cada mao ao
    # seu IMU, ao seu overlay e a sua triangulacao.
    arbitro = ArbitroDeLateralidade()
    # Nome REAL do Windows (nao o apelido estatico): e' o que diz se o indice
    # escolhido e' a webcam do laptop -- o erro que faz a janela mostrar o rosto
    # de quem esta' no laptop no lugar da bancada.
    nome_camera = nome_auxiliar(AUX_CAMERA_INDEX)
    arquivo_calibracao = aux_stereo_file(AUX_CAMERA_INDEX)
    print(
        f"Auxiliar ativa: indice {AUX_CAMERA_INDEX} ({nome_camera}) | "
        f"calibracao: "
        f"{arquivo_calibracao.name if arquivo_calibracao else 'AUSENTE'}"
    )
    aviso = aviso_camera_do_laptop(AUX_CAMERA_INDEX)
    if aviso:
        print(aviso)
    print("Mapa de cameras desta maquina (nome = verdade, indice muda):")
    for linha in mapa_de_cameras_resumo():
        print("  " + linha)
    # Inversao de lateralidade vigente: foi um BUG MUDO (as maos trocadas numa
    # camera sem nada avisar), entao ela tem de aparecer ao abrir o programa.
    for indice in aux_indices:
        if aux_handedness_flip(indice):
            print(f"Lateralidade da camera auxiliar {indice}: rotulo do MediaPipe "
                  "INVERTIDO (aprendido ou forcado -- gravado em "
                  f"{AUX_HANDEDNESS_FLIP_FILE.name})")
    # --- webcams auxiliares: a ATIVA (--aux-index) + as OUTRAS (--aux-cameras) --
    # Todas sao abertas e DESENHADAS no video do Kinect: o objetivo e' ver, no
    # MESMO quadro, o esqueleto do Kinect (verde) e o de CADA auxiliar (na cor
    # de cada uma) -- e', ao mesmo tempo, o teste visual da calibracao de cada
    # camera e a base do par que substitui o Kinect quando ele perde a mao.
    requested_size = alignment.calibrated_auxiliary_size or (1280, 720)
    # Exposicao/ganho POR CAMERA (medido 19/10/2026: as duas webcams pedem
    # valores DIFERENTES na mesma cena -- no -7 a Trust ficava 46% escura e a
    # SIGMA boa; um valor unico escurece uma delas). Sintaxe: "N" ou
    # "indice:N,indice:N" (ex.: --exposicao "0:-6,1:-4").
    exposicoes_aux = parse_controle_por_camera(args.exposicao)
    ganhos_aux = parse_controle_por_camera(args.ganho_cam)
    auxiliares = {}
    for indice in aux_indices:
        try:
            captura = abre_camera_auxiliar(
                indice, requested_size,
                exposicao=valor_por_camera(exposicoes_aux, indice),
                fps=args.fps_cam,
                trava_expo=not args.sem_trava_exposicao,
                ganho=valor_por_camera(ganhos_aux, indice))
        except RuntimeError as exc:                 # nao abriu: avisa e segue
            print(f"AVISO: webcam auxiliar {indice} nao abriu ({exc})")
            continue
        auxiliares[int(indice)] = captura
        obtido = (int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Camera auxiliar {indice} ({nome_auxiliar(indice)}): "
              f"solicitado {requested_size[0]}x{requested_size[1]}, obtido "
              f"{obtido[0]}x{obtido[1]}")
    if int(AUX_CAMERA_INDEX) not in auxiliares:
        if auxiliares:
            abriram = sorted(auxiliares)
            raise RuntimeError(
                f"A webcam auxiliar ATIVA (indice {AUX_CAMERA_INDEX}) nao "
                f"abriu -- abriram: {abriram}. Rode com --aux-index "
                f"{abriram[0]} (ou --aux-cameras \"{','.join(str(i) for i in reversed(abriram))}\" "
                "para trocar qual e' a ativa) e confira o NOME de cada indice "
                "com tools\\diagnostico_cameras.py --salvar: a calibracao e' "
                "POR INDICE (stereo_calibration_aux{N}.npz).")
        raise RuntimeError(
            f"Nenhuma webcam auxiliar abriu (indices pedidos: {aux_indices}). "
            "Confira o cabo/hub USB e rode tools\\diagnostico_cameras.py "
            "--salvar -- o indice MUDA quando uma camera entra ou sai do USB.")
    # Nome do Windows por camera (UMA vez: a enumeracao COM custa caro e o nome
    # nao muda durante a sessao). Entra no titulo de cada janela de video.
    nomes_aux = {int(indice): nome_auxiliar(indice) for indice in auxiliares}
    auxiliary = auxiliares[int(AUX_CAMERA_INDEX)]
    # --- JANELAS: uma por camera, posicionadas para TODAS aparecerem ----------
    # Sao 4 fluxos (RGB e depth do Kinect + uma janela por webcam auxiliar) e
    # antes so' a camera ATIVA tinha janela -- as outras existiam apenas como
    # esqueleto no RGB do Kinect ("nao aparece o video da terceira camera").
    # A escala de exibicao e' ajustada para TODAS caberem na tela: o WINDOW_SCALE
    # de 0.5 e' o teto e, com DPI alto (esta maquina: 1536x960 virtuais num
    # monitor 1920x1080 a 125%), cai para o maior valor que couber.
    tela = tamanho_da_tela()
    caixas_base = [(int(tracker.color_width), int(tracker.color_height)),
                   (int(tracker.depth_width), int(tracker.depth_height))]
    for index_camera in sorted(auxiliares):
        captura_base = auxiliares[index_camera]
        caixas_base.append((int(captura_base.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(captura_base.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    if args.escala_janelas is not None:
        escala_janelas = float(args.escala_janelas)
        origem_escala = "fixada em --escala-janelas"
    else:
        escala_janelas = escala_para_grade(caixas_base, *tela)
        origem_escala = ("automatica (maior que faz as "
                         f"{len(caixas_base)} janelas caberem)")
    janelas = GradeDeJanelas(cv2, escala=escala_janelas, tela=tela)
    print(f"[janelas] escala de exibicao {janelas.escala:g} -- {origem_escala} "
          f"| quadros base: {caixas_base}")
    # Snapshot periodico dos quadros COM overlay (--salvar-quadros): permite
    # conferir depois, com calma, se a projecao vermelha caiu na mao.
    snapshot_period = float(args.salvar_quadros)
    next_snapshot = (
        time.monotonic() + snapshot_period if snapshot_period > 0 else None
    )
    estado_stereo, motivo_stereo = alignment.stereo_status()
    if alignment.stereo_rotation is not None:
        print(
            f"Calibracao stereo: {estado_stereo} ({motivo_stereo}) | "
            f"tam. Kinect={alignment.calibrated_kinect_size} tam. aux={alignment.calibrated_auxiliary_size}"
        )
    else:
        print(f"Calibracao stereo: {estado_stereo} ({motivo_stereo}); "
              "sobreposicao desligada")
    # --- BIMANUAL: uma fusao (origem/bias/velocidade) POR mao ---------------
    fusions = {"right": PositionFusion(), "left": PositionFusion()}
    # Calibracao de IMU por lado: cada mao entra em modo bias/calibracao
    # INDEPENDENTEMENTE (a que esta' visivel e parada calibra primeiro).
    imu_calibration_active = {"right": True, "left": True}
    # --- ALINHAMENTO DA PALMA (IMU <-> camera), POR LUVA --------------------
    # Carrega o que ja' foi resolvido com a tecla O em outra sessao. Sem arquivo
    # vale o comportamento antigo (ancoragem de UMA amostra), que so' acerta na
    # pose de calibracao -- era o bug do vetor da mao esquerda.
    alinhamento_carregado = {}
    for lado in ("right", "left"):
        alinhamento_carregado[lado] = load_palm_alignment(fusions[lado], lado)
        if not alinhamento_carregado[lado]:
            print(f"IMU {lado}: sem alinhamento da palma salvo -- aperte O para "
                  "medir (gire a mao com a palma virada para o Kinect)")
    # --- REANCORAGEM DO YAW DO BOOT (item 97) ------------------------------
    # O yaw ABSOLUTO do IMU e' arbitrario a cada boot (filtro de 6 eixos, sem
    # magnetometro); o `T` do alinhamento salvo absorve o yaw do boot em que foi
    # resolvido. Sem reancorar, a seta acerta a MESA (normal ~vertical, insensivel
    # ao yaw) e erra ~100 graus com a palma de frente para o sensor -- o sintoma
    # "IMU da mao direita invertido" que sobrevivia a refits. Aqui o delta e'
    # resolvido com as primeiras amostras em que a palma nao esta' ~vertical e o
    # T e' corrigido EM MEMORIA (o JSON salvo nao muda: continua valendo para
    # qualquer boot). `--sem-ancora-yaw` desliga (para comparar o antigo).
    ancoras_yaw = {}
    if not getattr(args, "sem_ancora_yaw", False):
        for lado in ("right", "left"):
            if not alinhamento_carregado[lado]:
                continue
            ancoras_yaw[lado] = AncoradorYawBoot(fusions[lado], lado=lado)
            print(f"IMU {lado}: alinhamento salvo carregado; reancorando o YAW "
                  "DO BOOT automaticamente (mova a mao -- a palma nao pode ficar "
                  "o tempo todo apoiada/vertical para o yaw ser observavel)")
    captura_palma = CapturaAlinhamentoPalma()
    # Captura de alinhamento pela VERDADE do PLANO MEDIO do dorso (item 102,
    # tecla J): mesma mecanica da O, mas cada amostra usa a normal do plano
    # punho+4 MCPs (mais estavel que o produto vetorial de 3 pontos).
    captura_palma_plano = CapturaAlinhamentoPalma(
        PALM_PLANO_CAPTURE_SECONDS)
    # --- LOG DO VETOR DA PALMA (item 111) ------------------------------
    log_palma = None
    log_palma_ativo = False
    if getattr(args, "log_palma", False):
        _nome_log = time.strftime("palma_vetor_log_%Y%m%d_%H%M%S.csv")
        log_palma = LogVetorPalma(_nome_log)
        log_palma_ativo = True
        print(f"[log-palma] gravando vetor da palma em {_nome_log}")
    plano_display = {"right": None, "left": None}
    #: Ultima normal ROBUSTA aceita por lado (sinal/continuidade da mediana).
    plano_robusto_anterior = {"right": None, "left": None}
    falhas_plano_contagem = {"right": 0, "left": 0}
    frame_contador = 0
    # VERIFICACAO GUIADA DA PALMA (item 103, tecla V): 6 poses com a verdade do
    # plano do dorso -> certificado FECHADO/NAO FECHADO pose a pose. E' a prova
    # objetiva de que o vetor derivado do IMU coincide com a direcao anatomica.
    verificacao_palma = VerificacaoPosesPalma()
    #: CALIBRACAO/VERIFICACAO POR 3 POSES FISICAS (item 114): a verdade
    #: e' geometrica (palma p/ Kinect, p/ baixo na mesa, p/ cima), sem
    #: depender de landmark/depth/sinal do Kinect.
    poses_fisicas = ColetorPosesFisicas()
    poses_fisicas_fase_vista = None
    #: CALIBRACAO GEOMETRICA (item 127, tecla X): uma mao por vez; None = off.
    calib_geo = None
    #: Throttle do aviso "mao/plano nao detectado" do modo geometrico (5 s).
    ultimo_aviso_mao_geo = 0.0
    # --- NIVEL DA MESA (item 115): referencia de gravidade medida pelo depth.
    # O usuario confirmou o Kinect nivelado em relacao a mesa; medimos o
    # PLANO DA MESA e usamos a normal como referencia das poses 2/3, o que
    # compensa qualquer inclinacao residual do sensor ("alinhar ele antes").
    mesa_normal_camera = None
    mesa_plano_info = None
    mesa_auto_tentativas = 0
    #: Sentido da normal medida que corresponde a "para CIMA" (+1/-1);
    #: resolvido pelo J comparando o residuo das duas hipoteses.
    mesa_sinal = 1.0
    #: Verdades por lado definidas no J (pose1 geometrica + angulos
    #: relativos do Kinect); o V usa exatamente estas (item 116).
    poses_verdade_por_lado = {}
    #: Melhor calibracao da SESSAO por lado (nao deixa uma tentativa pior
    #: sobrescrever a melhor; item 116).
    melhor_palma_sessao = {"right": None, "left": None}
    caminho_mesa = Path("mesa_kinect.json")
    try:
        if caminho_mesa.exists():
            _estado_mesa = json.loads(caminho_mesa.read_text(encoding="utf-8"))
            _normal_mesa = np.asarray(_estado_mesa.get("normal"),
                                      np.float64).reshape(3)
            _norma_mesa = float(np.linalg.norm(_normal_mesa))
            if np.isfinite(_normal_mesa).all() and _norma_mesa > 1e-6:
                mesa_normal_camera = _normal_mesa / _norma_mesa
                mesa_plano_info = {
                    "tilt_deg": float(_estado_mesa.get("tilt_deg")
                                       or 0.0),
                    "inliers": int(_estado_mesa.get("inliers") or 0),
                    "fracao": float(_estado_mesa.get("fracao") or 0.0),
                }
                mesa_sinal = float(_estado_mesa.get("sinal") or 1.0)
                _verdades_mesa = _estado_mesa.get("poses_verdade") or {}
                for _lado_mesa in ("right", "left"):
                    _lista_verdades = _verdades_mesa.get(_lado_mesa)
                    if not _lista_verdades:
                        continue
                    _convertidas = []
                    for _item, (_nome_pose, _v) in zip(
                            _lista_verdades, POSES_FISICAS_CALIBRACAO):
                        try:
                            _vec = np.asarray(_item[1], np.float64)
                        except (TypeError, ValueError, IndexError):
                            _vec = np.asarray(_item, np.float64)
                        _vec = _vec.reshape(3)
                        _norma = float(np.linalg.norm(_vec))
                        if np.isfinite(_vec).all() and _norma > 1e-9:
                            _convertidas.append(
                                (_nome_pose, tuple(_vec / _norma)))
                    if len(_convertidas) == len(POSES_FISICAS_CALIBRACAO):
                        poses_verdade_por_lado[_lado_mesa] = tuple(
                            _convertidas)
                print(f"[nivel] plano da mesa carregado de "
                      f"{caminho_mesa.name} (tilt "
                      f"{mesa_plano_info['tilt_deg']:.1f} graus; tecle M "
                      "para medir de novo)")
    except (OSError, ValueError, TypeError):
        mesa_normal_camera = None
        mesa_plano_info = None
    verif_fase_vista = None
    #: RMS do plano do dorso durante a captura J (controle de qualidade: luz
    #: estourada/landmarks ruins inflam o RMS e descartam amostras).
    plano_rms_captura = {"right": [], "left": []}
    #: Ultima normal do PLANO valida por lado: mantem a CONTINUIDADE do
    #: sinal anatomico quando a referencia transversal x longitudinal
    #: degenera (o sinal segue o quadro anterior em vez de virar moeda --
    #: item 105).
    ultimo_plano_camera = {"right": None, "left": None}
    #: Historico curto das normais do plano: o vetor instantaneo cambaleia
    #: com o ruido do depth; a media com spread baixo e' a verdade estavel
    #: usada na calibracao/video (item 105).
    hist_plano_camera = {"right": [], "left": []}
    #: Quadros consecutivos com a normal do plano "parada" (spread <= gate).
    #: So' depois de PLANO_QUADROS_ESTAVEIS_MIN a verdade entra na
    #: calibracao (item 110).
    estabilidade_plano_contagem = {"right": 0, "left": 0}
    # Diagnostico da palma por MOVIMENTO PROGRAMADO (tecla D): grava
    # palm_diag_<data>_<hora>.csv e, no fim, o resumo por (fase, lado). E' o
    # log que faltava para separar "calibracao sistematicamente errada" de
    # "modelo so' vale na pose da tecla O" (item 74).
    diag_palma = DiagnosticoPalma()
    #: Ultima fase do diagnostico JA' ANUNCIADA (indice): a instrucao sai uma
    #: vez por fase, nao a cada frame (e o texto aparece tambem no video).
    diag_fase_vista = None

    # Camera calibration is performed only by stereo_calibration.py.
    camera_calibrating = False
    bias_calibration_until = 0.0
    calibration_until = time.monotonic() + CALIBRATION_SECONDS
    # Calibração do esqueleto (mão): captura nó-a-nó auxiliar <-> Kinect.
    hand_calib_active = False
    hand_calib_samples_aux = []
    hand_calib_samples_kin = []
    hand_calib_samples_depths = []
    hand_calib_count = 0
    last_hand_sample_time = 0.0
    # Log de diagnostico do overlay (tecla L): 1 linha/frame em
    # overlay_deviation_log.csv. Colunas: t, z_usado, z_bruto, cx, cy,
    # borda(0 centro..1 borda), velocidade_px_s, + por no (dev,dx,dy).
    overlay_logging = False
    overlay_log_file = None
    # Medicao do trim fixo (tecla K): coleta ~3 s com a mao parada.
    trim_measure_until = 0.0
    trim_measure_samples = []
    #: Velocidade da palma POR MAO (o log agora grava as DUAS maos por frame --
    #: um `overlay_prev_palm` unico misturaria os lados no calculo de vel).
    overlay_prev_palm = {"right": None, "left": None}
    overlay_prev_time = {"right": 0.0, "left": 0.0}
    #: Contador do diagnostico "SEM vermelho" (oclusao 14/10: 567 linhas de par
    #: no CSV e ZERO no overlay -- imprime a causa na 1a vez e 1 em cada 30).
    sem_vermelho = {}
    # --log-overlay: liga o log de desvio DESDE O INICIO (sem apertar L). Mesma
    # abertura da tecla L, so' que no setup. `overlay_log_file` ja' e' fechado no
    # `finally` entao a saida limpa funciona igual. -- 08/10/2026.
    if getattr(args, "log_overlay", False):
        overlay_logging = True
        overlay_log_file = open(OVERLAY_LOG_PATH, "w", newline="", encoding="utf-8")
        overlay_log_file.write(",".join(cabecalho_overlay_log()) + "\n")
        print(f"Log overlay LIGADO (--log-overlay) -> {OVERLAY_LOG_PATH.name} "
              "(1 linha/mao por frame)")
    # --- BIMANUAL: EMA de profundidade POR lado ------------------------------
    # O alignment guarda UMA EMA (legado unimanual); o tool mantem a versao
    # por lado e publica no alignment a EMA do lado ANTES de cada desenho.
    ema_por_lado = {"right": None, "left": None}
    ema_trust_por_lado = {"right": 0, "left": 0}
    #: Gates da triangulacao (ver `avalia_triangulacao`): referencia do ultimo
    #: aceite POR LADO (palm, t), o ultimo esqueleto 3D ACEITO (para MANTER no
    #: lugar do pico -- ver bloco da triangulacao), contadores e anti-spam.
    tri_ref = {"right": None, "left": None}
    tri_ultima = {"right": None, "left": None}
    tri_gate_cont = {"aceitas": 0, "reproj": 0, "vel": 0, "mantidas": 0,
                     "cruas": 0}
    tri_gate_pend = False
    tri_gate_log_em = 0.0
    #: Estado do veredito de lateralidade ja' IMPRESSO (o tool imprime so' quando
    #: MUDA -- ver `estado_lateralidade`) e o que ficou PENDENTE de sair (uma
    #: mudanca dentro do intervalo anti-spam sai no proximo instante permitido:
    #: e' lembrada, nunca perdida).
    lateralidade_anterior = {}
    lateralidade_estado_atual = {}
    lateralidade_desde_em = 0.0
    lateralidade_impressa_em = 0.0
    status_log_em = 0.0
    status_log_t0 = 0.0
    status_counts = {"right": None, "left": None}
    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["pc_time_s", "hand_side", "imu_timestamp_us", "kinect_x_m", "kinect_y_m", "kinect_z_m", "fused_x_m", "fused_y_m", "fused_z_m", "roll_deg", "pitch_deg", "yaw_deg", "elbow_angle_deg", "hand_source", "aux_pair_x_m", "aux_pair_y_m", "aux_pair_z_m", "palm_err_deg"])


    #: Metricas de DESEMPENHO do laco: o terminal dizia so' a taxa do IMU. Sem o
    #: fps do laco nao da' para separar "camera lenta" de "processamento lento".
    #: MEDIDO nas cameras (19/10/2026): 30 fps cada uma e 30 fps ATE' COM AS DUAS
    #: juntas em 720p MJPG -- o atraso que se VE e' este laco (Kinect + 2x
    #: MediaPipe + desenho). Medias moveis de tempo em ms.
    desempenho = {"t_anterior": None, "loop_ms": None, "kinect_ms": None,
                  "aux_ms": None}
    try:
        while True:
            frame_contador += 1
            now = time.monotonic()
            if desempenho["t_anterior"] is not None:
                desempenho["loop_ms"] = _ema(
                    desempenho["loop_ms"],
                    (now - desempenho["t_anterior"]) * 1000.0)
            desempenho["t_anterior"] = now
            color, depth = tracker.frames()
            if (mesa_normal_camera is None and mesa_auto_tentativas < 3
                    and frame_contador % 80 == 40):
                mesa_auto_tentativas += 1
                _info_mesa_auto = tracker.medir_plano_mesa()
                if (_info_mesa_auto is not None
                        and _info_mesa_auto["tilt_deg"] <= 85.0):
                    mesa_normal_camera = _info_mesa_auto["normal"]
                    mesa_plano_info = _info_mesa_auto
                    print(f"[nivel] plano da MESA medido: tilt "
                          f"{_info_mesa_auto['tilt_deg']:.1f} graus | "
                          f"inliers "
                          f"{_info_mesa_auto['fracao'] * 100:.0f}% | "
                          "usando como referencia das poses 2/3 "
                          "(Kinect inclinado em relacao a mesa: a "
                          "calibracao COMPENSA com o plano medido)",
                          flush=True)
            # --- TODAS as auxiliares: le, casa a resolucao e detecta POR CAMERA --
            # Cada camera tem o seu detector/timestamp (VideoMode do MediaPipe) e
            # o seu tamanho de frame; a camera ATIVA e' restaurada no fim do laco
            # porque os campos vivos do alignment (e as teclas K/L/H) sao dela.
            aux_frames, aux_por_camera = {}, {}
            t_aux = time.monotonic()
            for indice, captura in auxiliares.items():
                ok_lido, quadro = captura.read()
                if not ok_lido or quadro is None:
                    aux_por_camera[indice] = []
                    continue
                quadro = transform_auxiliary_frame(quadro)
                alignment.set_active_aux(indice)
                alignment.set_auxiliary_size(quadro)
                aux_frames[indice] = quadro
                aux_por_camera[indice] = tracker.detect_auxiliary_hands_sides(
                    quadro, aux_index=indice)
            desempenho["aux_ms"] = _ema(
                desempenho["aux_ms"], (time.monotonic() - t_aux) * 1000.0)
            alignment.set_active_aux(int(AUX_CAMERA_INDEX))
            auxiliary_frame = aux_frames.get(int(AUX_CAMERA_INDEX))
            auxiliary_ok = auxiliary_frame is not None
            auxiliary_for_calibration = (None if not auxiliary_ok
                                         else auxiliary_frame.copy())
            color_for_calibration = None if color is None else color.copy()
            t_kinect = time.monotonic()
            display, kinect_hands = tracker.detect_hands_all(color, depth)
            desempenho["kinect_ms"] = _ema(
                desempenho["kinect_ms"], (time.monotonic() - t_kinect) * 1000.0)
            # --- LATERALIDADE: a GEOMETRIA manda, o rotulo e' palpite ----------
            # Roda DEPOIS do Kinect (a verdade geometrica -- punho em 3D -- vem
            # dele) e ANTES de tudo que casa as maos por lado. Motivo: o rotulo
            # do MediaPipe pode sair TROCADO numa auxiliar (o driver dela pode
            # ja' entregar o fluxo espelhado; com dois MODELOS de webcam isso
            # acontece numa so'), e o casamento por lado e' o que liga cada mao
            # ao IMU certo, ao overlay certo e a triangulacao certa. O arbitro
            # tambem APRENDE e GRAVA a inversao daquela camera
            # (aux_handedness_flip.json) para o protocolo/sand/demo, que so'
            # tem o rotulo.
            aux_por_camera, relatorio_lateralidade = arbitro.arbitrar(
                alignment, kinect_hands, aux_por_camera,
                (int(tracker.color_width), int(tracker.color_height)))
            # Imprime quando o veredito de alguma camera MUDA (ela passou a ser
            # medida pela geometria? o rotulo dela estava invertido?): e' assim que
            # o experimentador ve no terminal o que o arbitro esta' fazendo.
            # Tres protecoes contra inundacao (item 75): (1) o estado NAO carrega
            # os numeros do motivo (a projecao em px muda sempre -- ver
            # `_sem_numeros`); (2) o estado novo precisa PEGAR -- persistir
            # ESTABILIDADE sem piscar -- antes de valer (no movimento o veredito
            # real alterna indecisa<->geometria frame a frame); (3) intervalo
            # minimo entre impressoes.
            agora_lateralidade = estado_lateralidade(relatorio_lateralidade)
            if agora_lateralidade != lateralidade_estado_atual:
                lateralidade_estado_atual = agora_lateralidade
                lateralidade_desde_em = time.monotonic()
            if (agora_lateralidade != lateralidade_anterior
                    and time.monotonic() - lateralidade_desde_em
                    >= ESTABILIDADE_LOG_LATERALIDADE_S
                    and time.monotonic() - lateralidade_impressa_em
                    >= INTERVALO_LOG_LATERALIDADE_S):
                for linha in arbitro.resumo(nomes_aux):
                    print("[lateralidade] " + linha, flush=True)
                lateralidade_anterior = agora_lateralidade
                lateralidade_impressa_em = time.monotonic()
            aux_entries = aux_por_camera.get(int(AUX_CAMERA_INDEX), [])
            auxiliary_hands = [entrada["landmarks"] for entrada in aux_entries]
            # Por lado e POR CAMERA: e' o que permite (a) casar cada mao com o
            # seu IMU e (b) triangular o par de auxiliares quando o Kinect cego.
            aux_lado_por_camera = {
                indice: {entrada["side"]: entrada["landmarks"]
                         for entrada in entradas}
                for indice, entradas in aux_por_camera.items()
            }
            # A mao auxiliar saiu do frame? Reseta o EMA de profundidade
            # estimada (modelo 3D). Sem isso, o EMA rejeitaria o primeiro
            # estimate pos-retorno como "salto >30 cm" e a mao vermelha
            # ficaria CONGELADA na profundidade antiga (desalinhada).
            if not auxiliary_hands and alignment._hand_depth_ema_initialized:
                alignment.reset_hand_depth_ema()
            overlay_deviation = None
            depth_text = ""
            # --- BIMANUAL: casamento mao Kinect <-> mao auxiliar PELO LADO ---
            # O rotulo de lateralidade do MediaPipe (com o espelho corrigido)
            # define qual IMU alimenta qual mao: right -> 4210, left -> 4211.
            kin_por_lado = {}
            aux_por_lado = {}
            for lado in ("right", "left"):
                kin_por_lado[lado] = next(
                    (m for m in kinect_hands if m["side"] == lado), None
                )
                aux_por_lado[lado] = next(
                    (e for e in aux_entries if e["side"] == lado), None
                )
            # Mao "primaria" (direita, senao a primeira visivel): serve para
            # os recursos de mao unica (H calib esqueleto, K trim, L log).
            primaria = kin_por_lado["right"] or kin_por_lado["left"]
            lado_primario = primaria["side"] if primaria else None
            landmarks = primaria["landmarks"] if primaria else None
            aux_primaria = (
                aux_por_lado[lado_primario]["landmarks"]
                if (lado_primario and aux_por_lado[lado_primario]) else None
            )
            # Captura de amostras para a calibração do esqueleto (mão)
            if hand_calib_active and landmarks is not None and aux_primaria is not None:
                if (
                    hand_calib_count < HAND_CALIBRATION_SAMPLES_REQUIRED
                    and now - last_hand_sample_time > 2.0
                ):
                    # Só aceita amostra se o Kinect depth é confiável
                    # (>=14 nós com depth válido). Caso contrário, pula
                    # e tenta de novo no próximo frame (sem contar).
                    sample_depths = tracker.landmark_depths(landmarks).astype(np.float64)
                    valid_depth = np.isfinite(sample_depths) & (sample_depths > 0.1)
                    if valid_depth.sum() < 14:
                        print(
                            "Amostra IGNORADA: poucos nós com depth válido "
                            f"({valid_depth.sum()}/21). Mão mais centrada/"
                            "afastada do Kinect."
                        )
                        last_hand_sample_time = now
                    else:
                        hand_calib_samples_aux.append(
                            np.array(
                                [(lm.x, lm.y) for lm in aux_primaria],
                                dtype=np.float64,
                            )
                        )
                        hand_calib_samples_kin.append(
                            np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float64)
                        )
                        hand_calib_samples_depths.append(sample_depths)
                        hand_calib_count = len(hand_calib_samples_aux)
                        last_hand_sample_time = now
                        print(
                            f"Amostra de mao: {hand_calib_count}/"
                            f"{HAND_CALIBRATION_SAMPLES_REQUIRED} (varie posicao, "
                            "profundidade e inclinacao da mao)"
                        )
                    if hand_calib_count >= HAND_CALIBRATION_SAMPLES_REQUIRED:
                        all_aux = np.concatenate(hand_calib_samples_aux)
                        span = np.ptp(all_aux, axis=0)
                        if np.hypot(*span) < HAND_CALIBRATION_MIN_SPAN:
                            print(
                                "Calibração de mano: pouca varredura (a mano quase "
                                "sempre na mesma zona). Refaça a captura (tecla P) "
                                "moviendo a mano por TODO o frame."
                            )
                            hand_calib_active = False
                        else:
                            ok = alignment.solve_hand_landmark_calibration(
                                hand_calib_samples_aux,
                                hand_calib_samples_kin,
                                samples_depths=hand_calib_samples_depths,
                            )
                            if ok:
                                print(
                                    "Calibração de esqueleto guardada em "
                                    f"{HAND_CALIBRATION_FILE.name}; aplique-se "
                                    "automaticamente no overlay."
                                )
                            else:
                                print("Falha ao resolver a calibração de esqueleto.")
                            hand_calib_active = False
            # --- BIMANUAL: triangulacao/overlay POR mao (casadas pelo LADO) ---
            # REGRA DE DESENHO (a mesma do tool unimanual): a transformacao
            # aux->Kinect e 100% calibracao (stereo 3D + profundidade
            # estimada por calibracao). Nada aqui "segue a mao verde".
            resultados_por_lado = {}
            for lado in ("right", "left"):
                khand = kin_por_lado[lado]
                ahand = aux_por_lado[lado]
                if khand is None and ahand is None:
                    resultados_por_lado[lado] = None
                    continue
                landmarks_lado = khand["landmarks"] if khand else None
                aux_hand = ahand["landmarks"] if ahand else None
                mao = {
                    "landmarks": landmarks_lado,
                    "position": khand["position"] if khand else None,
                    "aux_hand": aux_hand,
                    "triangulated_3d": None,
                    "overlay_deviation": None,
                    "overlay_red": None,
                    "overlay_green": None,
                    "depth_note": "",
                    "z_m": float("nan"),
                    "sample": None,
                    "fused": None,
                    "aux_pair": None,
                    # Vetor da palma: normal medida pelo Kinect, direcao prevista
                    # pelo IMU e o ERRO entre as duas (diagnostico por luva).
                    "palm_normal_kinect": None,
                    "palm_direction": None,
                    "palm_error_deg": None,
                }
                # --- Prioridade 1: triangulacao estereo (espaco vetorial 3D) ---
                # Com as DUAS cameras vendo a mao, o esqueleto 3D (21,3) em
                # metros e MEDIDO pela interseccao de raios (calibracao
                # stereo). A profundidade deixa de ser estimada por escala
                # aparente e o overlay fica imune a inclinacao da mao.
                #: True quando a Prioridade 1 RODOU mas foi REJEITADA pelos gates
                #: (reprojecao/velocidade): nesse caso NAO se cai no par de
                #: auxiliares (tambem dessincronizado) -- o overlay cai no depth
                #: do Kinect. Quando a Prioridade 1 nem roda (Kinect cego), o par
                #: segue como antes (e' a funcao dele).
                tri_descartada = False
                if (
                    alignment.stereo_usable
                    and landmarks_lado is not None
                    and len(landmarks_lado) == 21
                    and aux_hand is not None
                ):
                    aux_px = [
                        (lm.x * alignment.auxiliary_width, lm.y * alignment.auxiliary_height)
                        for lm in aux_hand
                    ]
                    kin_px = [
                        (lm.x * tracker.color_width, lm.y * tracker.color_height)
                        for lm in landmarks_lado
                    ]
                    tri3d, tri_err = alignment.triangulate_hand_points(aux_px, kin_px)
                    if tri3d is not None and np.isfinite(tri3d).any():
                        ref = tri_ref.get(lado)
                        dt_tri = ((now - ref[1])
                                  if (ref is not None and ref[1] is not None)
                                  else None)
                        aceita, categoria, motivo_gate, palm = avalia_triangulacao(
                            tri3d, tri_err, ref, dt_tri)
                        if aceita:
                            mao["triangulated_3d"] = tri3d
                            tri_ultima[lado] = tri3d
                            tri_ref[lado] = (palm, now)
                            tri_gate_cont["aceitas"] += 1
                            mao["tri_gate"] = "aceita"
                        else:
                            #: ESCOLHA POR MENOR ERRO PROJETADO (mesma metrica px):
                            #: `tri_err` ja' mede o erro da BRUTA; do HOLD projeta-
                            #: se agora (`project_camera_points`) e mede contra
                            #: `kin_px`. Vence quem projetar MAIS PERTO da mao do
                            #: Kinect. Idade do hold deixou de decidir (bancada
                            #: 09/10: hold "fresco" 0,25 s deu med 30 px e a bruta
                            #: rejeitada 10 px; no pico catastrofico ~200 px a
                            #: bruta e' pior que o hold -- a comparacao resolve OS
                            #: DOIS sem limiar achado a mao). Depth do Kinect
                            #: continua fora (medido 87-105 px, dx -218).
                            hold = tri_ultima.get(lado)
                            err_crua = float("inf")
                            if tri_err is not None:
                                e_c = np.asarray(tri_err, np.float64).ravel()
                                e_c = e_c[np.isfinite(e_c)]
                                if e_c.size:
                                    err_crua = float(np.median(e_c))
                            err_hold = float("inf")
                            if hold is not None:
                                try:
                                    px_hold = alignment.project_camera_points(hold)
                                    dh = np.linalg.norm(
                                        np.asarray(px_hold, np.float64)
                                        - np.asarray(kin_px, np.float64), axis=1)
                                    dh = dh[np.isfinite(dh)]
                                    if dh.size:
                                        err_hold = float(np.median(dh))
                                except Exception:
                                    err_hold = float("inf")
                            tri_descartada = True
                            tri_gate_cont[categoria] += 1
                            tri_gate_pend = True
                            if hold is not None and err_hold <= err_crua:
                                #: hold projeta mais perto -> mantem
                                mao["triangulated_3d"] = hold
                                mao["tri_gate"] = categoria
                                tri_gate_cont["mantidas"] += 1
                            else:
                                #: bruta projeta mais perto -> usa a bruta mesmo
                                #: rejeitada pelo gate (`-crua` no note)
                                mao["triangulated_3d"] = tri3d
                                mao["tri_gate"] = f"{categoria}-crua"
                                tri_gate_cont["cruas"] += 1
                        if (tri_gate_pend and now - tri_gate_log_em
                                >= INTERVALO_LOG_LATERALIDADE_S):
                            print(f"[gate-tri] recusadas "
                                  f"reproj={tri_gate_cont['reproj']} "
                                  f"vel={tri_gate_cont['vel']} | mantidas "
                                  f"{tri_gate_cont['mantidas']} / cruas "
                                  f"{tri_gate_cont['cruas']} (aceitas "
                                  f"{tri_gate_cont['aceitas']})",
                                  flush=True)
                            tri_gate_log_em = now
                            tri_gate_pend = False
                # --- Prioridade 2: o PAR DE AUXILIARES (Kinect cego) -----------
                # Quando o Kinect perde a mao (ou nao triangula com ele), as DUAS
                # auxiliares, sozinhas, medem a mao nos 3 eixos pela calibracao de
                # cada uma com o Kinect -- e' o objetivo do projeto: o par
                # SUBSTITUI o Kinect, nao so' o apoia.
                if mao["triangulated_3d"] is None and not tri_descartada:
                    deteccoes = {
                        indice: pares.get(lado)
                        for indice, pares in aux_lado_por_camera.items()
                        if pares.get(lado) is not None
                    }
                    pontos, info = triangulate_aux_pairs(alignment, deteccoes)
                    if pontos is not None:
                        mao["triangulated_3d"] = pontos
                        mao["aux_pair"] = info
                resultados_por_lado[lado] = mao
            # O alignment guarda o esqueleto 3D da mao PRIMARIA (consumidores
            # externos checam last_triangulated_3d_time para frescor).
            prim_mao = resultados_por_lado.get(lado_primario) if lado_primario else None
            if prim_mao is not None and prim_mao["triangulated_3d"] is not None:
                alignment.last_triangulated_3d = prim_mao["triangulated_3d"]
                alignment.last_triangulated_3d_time = now
            else:
                alignment.last_triangulated_3d = None
                alignment.last_triangulated_3d_time = None
            for lado in ("right", "left"):
                mao = resultados_por_lado[lado]
                if mao is None:
                    continue
                landmarks_lado = mao["landmarks"]
                aux_hand = mao["aux_hand"]
                depth_note = "Kinect"
                kinect_depth_m = None
                if landmarks_lado is not None:
                    kinect_depth_m = tracker.landmark_depths(landmarks_lado)
                    if not np.isfinite(np.asarray(kinect_depth_m, np.float32)).any():
                        kinect_depth_m = None
                if kinect_depth_m is None and aux_hand is not None:
                    # Norma de desenho: o overlay NAO depende do tracking do
                    # Kinect. Se o Kinect perdeu a mao, a profundidade vem de
                    # fontes independentes do tracking verde, por prioridade:
                    # (a) esqueleto do SDK (medida REAL de depth — o Z vem do
                    #     body/depth index, nao segue tracking nenhum);
                    # (b) modelo 3D calibrado da vista auxiliar (escala
                    #     aparente, calibracao fixa do .npz);
                    # (c) amostragem da cena onde a palma projeta.
                    # A roxa/vermelha fica onde a calibracao diz.
                    sdk_anchor = tracker.body_hand_anchor()
                    if sdk_anchor is not None:
                        kinect_depth_m = [sdk_anchor[2]]
                        depth_note = "esqueletoSDK"
                        # Atualiza a memoria de profundidade (mesma EMA do
                        # detect_hand) para alimentar os fallbacks seguintes.
                        if tracker.last_hand_depth_m is None:
                            tracker.last_hand_depth_m = sdk_anchor[2]
                        else:
                            tracker.last_hand_depth_m = (
                                0.8 * tracker.last_hand_depth_m
                                + 0.2 * sdk_anchor[2]
                            )
                    if kinect_depth_m is None:
                        hand_aux = aux_hand
                        aux_normalized = np.array(
                            [(lm.x, lm.y) for lm in hand_aux], dtype=np.float64
                        )
                        model_depth = alignment.get_smoothed_hand_depth(aux_normalized)
                        if model_depth is not None:
                            kinect_depth_m = [model_depth]
                            depth_note = "modelo3D"
                        else:
                            # Sem calibracao 3D da mao: amostra a cena onde a
                            # palma auxiliar projeta (ainda usa a transformacao
                            # calibrada, nao o tracking verde).
                            guess = tracker.last_hand_depth_m or NOMINAL_HAND_DEPTH_M
                            kinect_depth_m = [
                                tracker.depth_at_projected(
                                    alignment, hand_aux[9], guess
                                )
                            ]
                            depth_note = "amostrada"
                if kinect_depth_m is None:
                    if tracker.last_hand_depth_m is not None:
                        kinect_depth_m = [tracker.last_hand_depth_m]
                        depth_note = "ultima valida"
                    else:
                        kinect_depth_m = [NOMINAL_HAND_DEPTH_M]
                        depth_note = "nominal"

                # --- Profundidade unica robusta, EMA adaptativa (POR LADO) --
                # Mesmo algoritmo do tool unimanual; a EMA agora vive em
                # ema_por_lado/ema_trust_por_lado e e' publicada no
                # alignment antes de desenhar CADA mao.
                depths_for_projection = kinect_depth_m
                if kinect_depth_m is not None:
                    depths_arr = np.asarray(kinect_depth_m, np.float32).reshape(-1)
                    valid_depths = depths_arr[
                        np.isfinite(depths_arr)
                        & (depths_arr > 0.1)
                        & (depths_arr <= (DEPTH_MAX_MM + 100) / 1000.0)
                    ]
                    if valid_depths.size > 0:
                        single_depth = float(np.median(valid_depths))
                    else:
                        single_depth = tracker.last_hand_depth_m or NOMINAL_HAND_DEPTH_M
                    single_depth = float(np.clip(single_depth, 0.4, 3.0))
                    # confidence: 2 = medida real do Kinect (MediaPipe+depth ou
                    # esqueleto do SDK; pode bootstrap),
                    # 1 = estimacao da vista auxiliar (modelo 3D calibrado ou
                    # amostragem da cena), 0 = chute nominal.
                    confidence = 0
                    if depth_note in ("Kinect", "esqueletoSDK"):
                        confidence = 2
                    elif depth_note in ("modelo3D", "amostrada"):
                        confidence = 1
                    ema = ema_por_lado[lado]
                    ema_trust = ema_trust_por_lado[lado]
                    if ema is None or not np.isfinite(ema):
                        ema, ema_trust = single_depth, confidence
                    elif confidence >= 2 and ema_trust < 2:
                        # Primeira medida real do Kinect: bootstrap imediato
                        # para eliminar o transitorio do arranque.
                        ema, ema_trust = single_depth, 2
                    else:
                        delta = float(abs(single_depth - ema))
                        if delta > 0.50:
                            # Salto grande em UM frame: impossivel para uma
                            # mao real. NAO segue o pico; medida real (o
                            # bootstrap inicial foi um fundo) recupera por
                            # deriva minima (0.05).
                            alpha = 0.05 if confidence >= 2 else 0.0
                        else:
                            # Adaptativa: segue rapido quando Z se move (mao
                            # se aproximando), suave quando esta quieta.
                            alpha = float(np.clip(0.12 + 2.5 * delta, 0.12, 0.60))
                            if confidence < 2:
                                alpha = min(alpha, 0.20)
                        ema = (1.0 - alpha) * ema + alpha * single_depth
                        ema_trust = max(ema_trust, confidence)
                    ema_por_lado[lado] = float(ema)
                    ema_trust_por_lado[lado] = int(ema_trust)
                    alignment._kinect_depth_ema = float(ema)
                    alignment._kinect_depth_ema_trust = int(ema_trust)
                    depths_for_projection = [float(ema)]
                if mao["triangulated_3d"] is not None:
                    # A triangulacao (medida real em 3D) tem prioridade sobre
                    # qualquer profundidade estimada/EMA para o desenho.
                    tri = np.asarray(mao["triangulated_3d"], np.float64)
                    palm_z = float(tri[9, 2])
                    if not np.isfinite(palm_z):
                        palm_z = float(np.nanmedian(tri[:, 2]))
                    depths_for_projection = [palm_z]
                    # "trianguladoAux" = medido pelo PAR de auxiliares (Kinect
                    # cego); "triangulado" = Kinect + auxiliar. O nome aparece no
                    # texto da janela e no CSV -- a analise precisa saber a fonte.
                    depth_note = ("trianguladoAux" if mao.get("aux_pair")
                                  else "triangulado")

                # --- Overlay vermelho POR mao -----------------------------
                # O draw compara a projecao com o esqueleto verde do
                # Kinect; passamos UMA mao por chamada (e o
                # last_kinect_hands correspondente) para o desvio casar
                # exatamente o lado desta mao.
                if display is not None and (
                    alignment.stereo_usable or alignment.hand_landmark_ready
                ):
                    anteriores = tracker.last_kinect_hands
                    tracker.last_kinect_hands = (
                        [landmarks_lado] if landmarks_lado is not None else []
                    )
                    try:
                        bruto = tracker.draw_auxiliary_hands_on_color(
                            display,
                            [aux_hand] if aux_hand is not None else [],
                            alignment,
                            depths_for_projection,
                            mao["triangulated_3d"],
                        )
                    finally:
                        tracker.last_kinect_hands = anteriores
                    #: O motivo do draw e' POR MAO: `last_draw_recusa` e' um
                    #: atributo unico e havia DUAS chamadas por frame (right,
                    #: left) -- a segunda sobrescrevia e o diag misturava os
                    #: lados (linhas "None" x "nenhuma mao desenhavel" no mesmo
                    #: frame, medido em 16/10). Guarda na propria mao.
                    mao["draw_recusa"] = getattr(tracker,
                                                 "last_draw_recusa", None)
                    overlay_red = overlay_green = None
                    if bruto is not None and len(bruto) == 5:
                        dev, dx, dy, overlay_red, overlay_green = bruto
                        # Sem verde (Kinect cego) o desvio e indefinido, mas
                        # a projecao vermelha segue sendo desenhada e logada.
                        mao["overlay_deviation"] = (
                            (dev, dx, dy) if dev is not None else None
                        )
                    mao["overlay_red"] = overlay_red
                    mao["overlay_green"] = overlay_green
                else:
                    mao["draw_recusa"] = "sem display/stereo"
                #: O estado do gate (aceita/reproj/vel) entra no `note` para o
                #: overlay_deviation_log.csv DISTINGUIR aceita de mantida (hold)
                #: -- foi o que faltou para diagnosticar a mao esquerda (item 51).
                tri_gate = mao.get("tri_gate")
                mao["depth_note"] = (f"{depth_note}:{tri_gate}"
                                     if tri_gate else depth_note)
                mao["z_m"] = (
                    float(np.nanmedian(np.asarray(kinect_depth_m, np.float32)))
                    if kinect_depth_m is not None else float("nan")
                )
                mao["z_used"] = (
                    float(np.nanmedian(np.asarray(depths_for_projection, np.float32)))
                    if depths_for_projection is not None else float("nan")
                )
            # Fim do loop por lado. Os blocos seguintes (trim K, log L,
            # textos) continuam usando a mao PRIMARIA (compatibilidade).

            # --- Os esqueletos das OUTRAS auxiliares (os TRES no mesmo quadro) --
            # A camera ATIVA ja' foi desenhada em vermelho no loop por lado (com
            # o par vermelho/verde medido pelas teclas K/L); aqui entram as
            # demais, cada uma na SUA cor e com rotulo, para o experimentador ver
            # de uma vez: esqueleto do Kinect (verde) + webcam aux 1 + webcam
            # aux 2. E' a verificacao visual da calibracao das duas.
            if display is not None and len(auxiliares) > 1:
                outras = {}
                lados_outras = {}
                for indice, entradas in aux_por_camera.items():
                    if int(indice) == int(AUX_CAMERA_INDEX):
                        continue
                    visiveis = [e for e in entradas if e.get("landmarks")]
                    if visiveis:
                        outras[int(indice)] = [e["landmarks"]
                                               for e in visiveis]
                        lados_outras[int(indice)] = [e.get("side")
                                                     for e in visiveis]
                if outras:
                    # Paleta da BIBLIOTECA (CORES_AUXILIARES e' atributo de
                    # KinectHandTracker, nao nome do modulo): a cor 0 (vermelha)
                    # fica com a camera ATIVA, que ja' foi desenhada no loop por
                    # lado -- entao as outras comecam na 1 (magenta).
                    cores = {
                        indice: KinectHandTracker.CORES_AUXILIARES[
                            (posicao + 1) % len(KinectHandTracker.CORES_AUXILIARES)]
                        for posicao, indice in enumerate(sorted(outras))
                    }
                    #: REFERENCIA POR MAO, nao so' a PRIMARIA: antes todas as
                    #: maos das outras cameras recebiam o 3D/z da PRIMARIA e,
                    #: com as DUAS maos na tela, a mao NAO primaria (a esquerda,
                    #: tipicamente) era desenhada na profundidade/3D da outra --
                    #: e' o "esqueleto roxo offset na mao esquerda" da bancada
                    #: de 09/10. Agora cada mao leva o triangulated_3d/z_used do
                    #: SEU lado (resultados_por_lado), como o vermelho ja' fazia
                    #: no loop por lado.
                    tri_por_mao, z_por_mao = {}, {}
                    for indice, lados in lados_outras.items():
                        tri_por_mao[indice] = [
                            (resultados_por_lado.get(lado) or {})
                            .get("triangulated_3d") for lado in lados]
                        z_por_mao[indice] = [
                            (resultados_por_lado.get(lado) or {})
                            .get("z_used") for lado in lados]
                    tracker.draw_auxiliary_cameras_on_color(
                        display, outras, alignment,
                        depths=z_por_mao,
                        triangulados=tri_por_mao,
                        cores=cores,
                    )
                    #: RE-TRECO da camera ATIVA: as outras cameras pintam
                    #: DEPOIS do vermelho (laco por lado, la' tras) e o ROXO
                    #: COBRIA o vermelho quando as projecoes coincidem -- que,
                    #: com calibracao boa, e' quase sempre (16/10: o log PROVAVA
                    #: que os pixels do vermelho existiam na oclusao e o usuario
                    #: via SO' o roxo). Repinta a ativa por cima com os MESMOS
                    #: pixels ja' medidos -- sem recalcular nada. A ativa e' a
                    #: validada (teclas K/L/H), logo ela fica por cima.
                    for _lado_topo in ("right", "left"):
                        _mao_topo = resultados_por_lado.get(_lado_topo)
                        if (_mao_topo is not None
                                and _mao_topo.get("overlay_red") is not None):
                            desenhar_esqueleto_projetado(
                                display, _mao_topo["overlay_red"])
            overlay_red = overlay_green = None
            overlay_deviation = None
            triangulated_3d = None
            if prim_mao is not None:
                overlay_red = prim_mao["overlay_red"]
                overlay_green = prim_mao["overlay_green"]
                overlay_deviation = prim_mao["overlay_deviation"]
                triangulated_3d = prim_mao["triangulated_3d"]
            # --- Medicao do trim fixo (tecla K, 3 s, mao PARADA) ------------
            # Coleta (verde - vermelho) por frame e, ao fim, grava a
            # traducao constante em overlay_trim.json. O offset medido e
            # somado ao trim atual (o vermelho desenhado ja inclui o trim).
            # (BIMANUAL: mede na mao PRIMARIA.)
            if trim_measure_until > 0.0:
                if now < trim_measure_until:
                    if overlay_red is not None and overlay_green is not None:
                        diffs = np.asarray(overlay_green, np.float64) - np.asarray(
                            overlay_red, np.float64
                        )
                        valid = np.isfinite(diffs).all(axis=1)
                        if valid.any():
                            trim_measure_samples.append(
                                diffs[valid].mean(axis=0) + alignment.overlay_trim
                            )
                else:
                    trim_measure_until = 0.0
                    if len(trim_measure_samples) >= 20:
                        samples = np.asarray(trim_measure_samples, np.float64)
                        med = np.median(samples, axis=0)
                        spread = np.linalg.norm(samples - med[None, :], axis=1)
                        keep = spread <= max(15.0, float(np.median(spread)) * 3.0)
                        new_trim = samples[keep].mean(axis=0) if keep.any() else med
                        alignment.overlay_trim = np.asarray(new_trim, np.float64)
                        try:
                            with open(OVERLAY_TRIM_PATH, "w", encoding="utf-8") as fh:
                                json.dump(
                                    {
                                        "dx": float(new_trim[0]),
                                        "dy": float(new_trim[1]),
                                        "samples": int(len(samples)),
                                    },
                                    fh,
                                )
                            print(
                                f"Trim do overlay salvo: dx {new_trim[0]:+.1f} px, "
                                f"dy {new_trim[1]:+.1f} px ({len(samples)} frames)"
                            )
                        except OSError as exc:
                            print(
                                f"Trim medido (falha ao salvar: {exc}): "
                                f"dx {new_trim[0]:+.1f}, dy {new_trim[1]:+.1f}"
                            )
                    else:
                        print(
                            "Trim cancelado: poucos frames validos "
                            f"({len(trim_measure_samples)}). Mao precisa estar "
                            "visivel nas DUAS cameras e parada."
                        )
                    trim_measure_samples = []
            #: O log NAO pode depender do prim_mao (que vem SO' da mao do
            #: Kinect): no teste de OCLUSAO (14/10) o Kinect cego zerava o
            #: prim_mao e o log sumia EXATAMENTE nas 74 frames de fallback --
            #: o vermelho continuava desenhado (usuario confirmou) e o CSV
            #: principal gravava, so o overlay_deviation_log perdia. Por isso
            #: o bloco abre com `or overlay_logging` e o depth_text (que sim
            #: precisa do Kinect) fica guardado por dentro.
            if prim_mao is not None or overlay_logging:
                if prim_mao is not None:
                    mark = "" if prim_mao["depth_note"] in ("Kinect", "esqueletoSDK") else "!"
                    depth_text = (
                        f"Z {prim_mao['z_m']:.2f} m ({prim_mao['depth_note']}{mark})"
                    )
                # --- Log de diagnostico do overlay (tecla L liga/desliga) --------
                # Grava UMA LINHA POR MAO por frame em overlay_deviation_log.csv
                # com o desvio vermelho-verde por no + contexto (profundidade,
                # posicao na imagem, velocidade da mao). Serve para classificar o
                # erro: estatico (bias) x movimento (sincronia) x bordas
                # (distorcao). Frames SEM mao verde sao gravados com as colunas
                # de desvio vazias (so a trajetoria vermelha + contexto).
                # (ANTES: so a mao PRIMARIA -- o veredito das DUAS maos ao mesmo
                #  tempo ficava com metade dos dados; agora a coluna `lado`
                #  identifica e as duas entram, com velocidade POR MAO.)
                if overlay_logging:
                    for lado_log in ("right", "left"):
                        mao_log = resultados_por_lado.get(lado_log)
                        if mao_log is None:
                            continue
                        if mao_log.get("overlay_red") is None:
                            #: Diagnostico de OCLUSAO (14/10): o CSV teve 567
                            #: linhas de PAR e o overlay ZERO -- o vermelho nao
                            #: saia. O estado (e o motivo do DRAW, lido de
                            #: tracker.last_draw_recusa) vai para o ARQUIVO
                            #: overlay_diag.log (a sidebar nao da' Ctrl+C/V)
                            #: e para o console (1a vez + 1 em cada 30).
                            primeira = not sem_vermelho
                            oculto = sem_vermelho.get(lado_log, 0) + 1
                            sem_vermelho[lado_log] = oculto
                            estado = (
                                f"3D={'sim' if mao_log.get('triangulated_3d') is not None else 'nao'}"
                                f" fonte={mao_log.get('depth_note') or '?'}"
                                f" aux_hand={'sim' if mao_log.get('aux_hand') else 'nao'}"
                                f" display={'sim' if display is not None else 'nao'}"
                                f" stereo={'sim' if (alignment.stereo_usable or alignment.hand_landmark_ready) else 'nao'}"
                                f" recusa_draw={mao_log.get('draw_recusa')}"
                            )
                            try:
                                with open(DIAG_OVERLAY_PATH,
                                          "w" if primeira else "a",
                                          encoding="utf-8") as fh_diag:
                                    fh_diag.write(
                                        f"{time.strftime('%H:%M:%S')} "
                                        f"{lado_log} {estado}\n")
                            except Exception:
                                pass
                            if oculto == 1 or oculto % 30 == 0:
                                print(f"[overlay] {lado_log} SEM vermelho "
                                      f"({oculto}x): {estado}", flush=True)
                            continue
                        try:
                            red = np.asarray(
                                mao_log["overlay_red"], np.float64
                            ).reshape(-1, 2)
                            grn = (
                                np.asarray(
                                    mao_log["overlay_green"], np.float64
                                ).reshape(-1, 2)
                                if mao_log["overlay_green"] is not None
                                else None
                            )
                            n_nodes = (21 if grn is None else
                                       min(21, red.shape[0], grn.shape[0]))
                            palm_ref = red if grn is None else grn
                            palm = palm_ref[9] if n_nodes > 9 else palm_ref[0]
                            cx = float(palm[0]) if np.isfinite(palm).all() else float("nan")
                            cy = float(palm[1]) if np.isfinite(palm).all() else float("nan")
                            edge = float("nan")
                            if np.isfinite([cx, cy]).all():
                                edge = float(
                                    max(
                                        cx / tracker.color_width,
                                        (tracker.color_width - cx) / tracker.color_width,
                                        cy / tracker.color_height,
                                        (tracker.color_height - cy) / tracker.color_height,
                                    )
                                )
                            z_used = float(mao_log["z_used"])
                            z_raw = float(mao_log["z_m"])
                            palm_now = np.array([cx, cy], np.float64)
                            speed = float("nan")
                            prev_palm = overlay_prev_palm.get(lado_log)
                            prev_time = overlay_prev_time.get(lado_log, 0.0)
                            if (np.isfinite(palm_now).all()
                                    and prev_palm is not None):
                                dt = (now - prev_time
                                      if prev_time else float("nan"))
                                if dt and dt > 1e-3:
                                    speed = float(
                                        np.linalg.norm(palm_now - prev_palm)
                                        / dt)
                            overlay_prev_palm[lado_log] = (
                                palm_now if np.isfinite(palm_now).all()
                                else None)
                            overlay_prev_time[lado_log] = now
                            row = [f"{now:.3f}", lado_log,
                                   f"{z_used:.3f}", f"{z_raw:.3f}",
                                   f"{cx:.1f}", f"{cy:.1f}", f"{edge:.3f}",
                                   f"{speed:.1f}" if np.isfinite(speed) else "",
                                   mao_log["depth_note"]]
                            for n in range(n_nodes):
                                if (
                                    grn is not None
                                    and np.isfinite(red[n]).all()
                                    and np.isfinite(grn[n]).all()
                                ):
                                    d = grn[n] - red[n]
                                    row.append(f"{np.hypot(*d):.1f}")
                                    row.append(f"{d[0]:+.1f}")
                                    row.append(f"{d[1]:+.1f}")
                                else:
                                    row += ["", "", ""]
                            # VETOR DA PALMA desta mao: normal do Kinect,
                            # direcao prevista pelo IMU da luva e o erro entre
                            # as duas. O numero que diz se o alinhamento daquela
                            # luva esta' bom (sem alinhamento cresce ao girar).
                            normal_palma = mao_log.get("palm_normal_kinect")
                            direcao_palma = mao_log.get("palm_direction")
                            if normal_palma is not None:
                                row += [f"{valor:+.4f}" for valor in
                                        np.asarray(normal_palma, np.float64)]
                            else:
                                row += ["", "", ""]
                            if direcao_palma is not None:
                                row += [f"{valor:+.4f}" for valor in
                                        np.asarray(direcao_palma, np.float64)]
                            else:
                                row += ["", "", ""]
                            erro_palma = mao_log.get("palm_error_deg")
                            row.append("" if erro_palma is None
                                       else f"{erro_palma:.2f}")
                            overlay_log_file.write(",".join(row) + "\n")
                        except (ValueError, IndexError, TypeError, KeyError):
                            pass
            # --- UMA JANELA DE VIDEO POR WEBCAM AUXILIAR ----------------------
            # Antes SO' a camera ATIVA tinha janela: o video das outras nao
            # existia em lugar nenhum (elas apareciam apenas como esqueleto
            # desenhado no RGB do Kinect), e sem o video nao da' para conferir
            # foco, enquadramento, luz nem se a mao/tabuleiro estao em quadro na
            # camera que vai entrar na triangulacao -- foi a queixa "nao aparece
            # o video da terceira camera" da bancada de 25/09/2026. Agora cada
            # camera aberta tem a SUA janela (titulo com o indice e o NOME do
            # Windows) e a ATIVA leva "ATIVA: K/L/H" no quadro, porque as teclas
            # de trim/log/calibracao do esqueleto valem SO' nela.
            for indice_camera in sorted(aux_frames):
                frame_camera = aux_frames[indice_camera]
                camera_display = (
                    cv2.flip(frame_camera, 1)
                    if AUXILIARY_DISPLAY_MIRROR
                    else frame_camera.copy()
                )
                maos_da_camera = [entrada["landmarks"] for entrada in
                                  aux_por_camera.get(indice_camera, [])]
                tracker.draw_auxiliary_hands(camera_display, maos_da_camera)
                ativa = int(indice_camera) == int(AUX_CAMERA_INDEX)
                linha_stereo, cor_stereo = None, (0, 255, 255)
                if ativa:
                    estado_stereo, motivo_stereo = alignment.stereo_status(
                        int(indice_camera))
                    linha_stereo = f"stereo {estado_stereo}: {motivo_stereo}"
                    cor_stereo = ((0, 255, 255) if alignment.stereo_usable
                                  else (0, 0, 255))
                # Reduz ANTES de escrever o rotulo: assim o texto nao encolhe
                # junto com a imagem (ver `rotular_camera_auxiliar`).
                pequeno = janelas.redimensionar(camera_display)
                rotular_camera_auxiliar(
                    pequeno,
                    f"Camera auxiliar {indice_camera} "
                    f"({nomes_aux.get(int(indice_camera), '?')})"
                    + (" | ATIVA: K/L/H (trim, log, esqueleto)" if ativa else ""),
                    f"maos: {len(maos_da_camera)} | "
                    f"{camera_display.shape[1]}x{camera_display.shape[0]}",
                    linha_stereo, cor_stereo)
                janelas.mostrar(
                    f"Camera auxiliar {indice_camera} - tracking", pequeno)
                if ativa:
                    # O snapshot e' da camera ATIVA (as teclas K/L/H sao dela).
                    auxiliary_display = camera_display
            if auxiliary_ok and next_snapshot is not None \
                    and time.monotonic() >= next_snapshot:
                cv2.imwrite("_aux_overlay.png", auxiliary_display)
                if display is not None:
                    cv2.imwrite("_kinect_overlay.png", display)
                desvios = []
                for lado, mao_lado in resultados_por_lado.items():
                    # Sem mao detectada daquele lado o dict e' None: o
                    # snapshot tem de dizer isso, nao estourar (era
                    # AttributeError: 'NoneType' object has no attribute
                    # 'get' -- o programa morria no primeiro snapshot com
                    # uma mao so' em quadro).
                    if mao_lado is None:
                        desvios.append(f"{lado}: sem mao detectada")
                        continue
                    dados = mao_lado.get("overlay_deviation")
                    if dados is None:
                        desvios.append(f"{lado}: sem par vermelho/verde")
                    else:
                        desvios.append(
                            f"{lado}: dev {dados[0]:.0f} px "
                            f"(dx {dados[1]:+.0f}, dy {dados[2]:+.0f})"
                        )
                print(
                    "Snapshot salvo (_aux_overlay.png / _kinect_overlay.png) "
                    f"| maos na auxiliar: {len(auxiliary_hands)} | "
                    + " | ".join(desvios),
                    flush=True,
                )
                next_snapshot = time.monotonic() + snapshot_period
            # --- BIMANUAL: IMU + fusao POR mao ---------------------------------
            # Cada lado usa o SEU receptor (right=4210, left=4211) e a SUA
            # fusao; a mao e' localizada pela lateralidade do MediaPipe.
            # A IMU e' lida SEMPRE (mesmo sem mao visivel): a luva continua
            # enviando e o CSV precisa do dado para analise posterior.
            amostras_por_lado = {}
            for lado in ("right", "left"):
                mao = resultados_por_lado[lado]
                sample_lado = receivers[lado].get_latest()
                amostras_por_lado[lado] = sample_lado
                if poses_fisicas.ativa and poses_fisicas.modo_takes:
                    if poses_fisicas.modo == "verificar":
                        rotacao_take = (None if sample_lado is None
                                        else rotation_matrix(
                                            sample_lado.roll_deg,
                                            sample_lado.pitch_deg,
                                            sample_lado.yaw_deg))
                        poses_fisicas.atualizar_lado(
                            lado, now, rotacao_take, None)
                elif poses_fisicas.ativa:
                    rotacao_pose = (None if sample_lado is None
                                    else rotation_matrix(
                                        sample_lado.roll_deg,
                                        sample_lado.pitch_deg,
                                        sample_lado.yaw_deg))
                    poses_fisicas.coletar(lado, now, rotacao_pose)
                if mao is None:
                    if log_palma is not None and log_palma_ativo:
                        log_palma.registrar(
                            t_mono=now, lado=lado, frame=frame_contador,
                            mao_detectada=0,
                            sync_color_depth_ms=getattr(
                                tracker, "sync_color_depth_ms", None),
                            log_ligado=1)
                    continue
                mao["sample"] = sample_lado
                fusion = fusions[lado]
                camera_position_lado = mao["position"]
                landmarks_lado = mao["landmarks"]
                if poses_fisicas.ativa and poses_fisicas.modo_takes:
                    if poses_fisicas.modo == "calibrar":
                        try:
                            _posicao_take = tracker._landmark_camera_point(
                                landmarks_lado[9], lado)
                        except (AttributeError, IndexError, TypeError):
                            _posicao_take = None
                        _rotacao_take = (None if sample_lado is None
                                         else rotation_matrix(
                                             sample_lado.roll_deg,
                                             sample_lado.pitch_deg,
                                             sample_lado.yaw_deg))
                        poses_fisicas.atualizar_lado(
                            lado, now, _rotacao_take, _posicao_take)
                elif poses_fisicas.ativa:
                    _fase_posicao = poses_fisicas.fase_atual(now)
                    if _fase_posicao is not None:
                        try:
                            _posicao_modelo = tracker._landmark_camera_point(
                                landmarks_lado[9], lado)
                        except (AttributeError, IndexError, TypeError):
                            _posicao_modelo = None
                        if _posicao_modelo is not None:
                            poses_fisicas.coletar_posicao_palma(
                                lado, _fase_posicao[0], _posicao_modelo)
                # Normal da palma medida pelo Kinect (depth): e' a referencia do
                # vetor da palma -- usada para ancorar, para a captura do
                # alinhamento (tecla O) e para o ERRO AO VIVO (Kinect x IMU).
                # `lado` liga a orientacao ANATOMICA (item 79): sem ele a lib
                # forcaria o sinal para a camera (nz<0), o que so' e' a palma na
                # pose "palma virada para o Kinect" -- nas outras o sinal vira
                # ruido e corrompe a captura O e o diagnostico.
                normal_lado = (
                    None if landmarks_lado is None
                    else tracker.palm_normal_camera(landmarks_lado, lado)
                )
                mao["palm_normal_kinect"] = normal_lado
                # Reancoragem do yaw do boot (item 97): alimenta o ancorador
                # ANTES de calcular a seta, para o frame que ancora ja' sair com
                # o T corrigido. Nao precisa de pose especial: o delta de yaw e'
                # o mesmo em todas as orientacoes; so' exige normais validas.
                ancora_yaw = ancoras_yaw.get(lado)
                if ancora_yaw is not None and normal_lado is not None:
                    rotacao_imu = (None if sample_lado is None
                                   else rotation_matrix(sample_lado.roll_deg,
                                                        sample_lado.pitch_deg,
                                                        sample_lado.yaw_deg))
                    if ancora_yaw.amostra(rotacao_imu, normal_lado) is not None:
                        mao["yaw_boot_delta_deg"] = ancora_yaw.delta_deg
                mao["palm_direction"] = (
                    fusion.palm_direction(sample_lado)
                    if landmarks_lado is not None else None
                )
                mao["palm_error_deg"] = (
                    None
                    if (mao["palm_direction"] is None or normal_lado is None)
                    else angle_between(normal_lado, mao["palm_direction"])
                )
                # --- PLANO MEDIO DO DORSO (item 102, tecla J) ----------------
                # Verdade mais estavel que o produto vetorial de 3 pontos: o
                # ruido de profundidade de um no' girava a normal inteira. A
                # normal do plano (punho + 4 MCPs) e' o que o usuario pediu
                # ("plano medio entre os nos do dorso") e serve para comparar
                # ao vivo com o vetor do IMU e para refazer T/f.
                plano_lado = (
                    None if landmarks_lado is None
                    else tracker.palm_plane_camera(
                        landmarks_lado, lado,
                        normal_anterior=ultimo_plano_camera[lado],
                        rms_max_m=0.015))
                if plano_lado is None:
                    normal_plano_lado, rms_plano_lado = None, None
                else:
                    normal_plano_lado, rms_plano_lado = plano_lado
                plano_estavel = None
                spread_plano = None
                fracao_robusta = 0.0
                if normal_plano_lado is not None:
                    ultimo_plano_camera[lado] = normal_plano_lado
                    hist_plano_camera[lado].append(normal_plano_lado)
                    if len(hist_plano_camera[lado]) > PLANO_JANELA_QUADROS:
                        hist_plano_camera[lado] = (
                            hist_plano_camera[lado][-PLANO_JANELA_QUADROS:])
                    plano_estavel, fracao_robusta = normal_robusta(
                        hist_plano_camera[lado],
                        normal_anterior=plano_robusto_anterior[lado],
                        tolerancia_deg=PLANO_ROBUSTO_TOLERANCIA_DEG,
                        minimo_fracao=PLANO_ROBUSTO_FRACAO_MIN)
                    if plano_estavel is not None:
                        plano_robusto_anterior[lado] = plano_estavel
                        if (normal_plano_lado is not None
                                and np.isfinite(normal_plano_lado).all()):
                            spread_plano = angle_between(
                                normal_plano_lado, plano_estavel)
                (estabilidade_plano_contagem[lado],
                 falhas_plano_contagem[lado]) = atualizar_estabilidade(
                    estabilidade_plano_contagem[lado],
                    falhas_plano_contagem[lado], plano_estavel)
                if plano_estavel is not None:
                    plano_display[lado] = (plano_estavel, now)
                if poses_fisicas.ativa and plano_estavel is not None:
                    _fase_normal = poses_fisicas.fase_atual(now)
                    if _fase_normal is not None:
                        poses_fisicas.coletar_normal_kinect(
                            lado, _fase_normal[0], plano_estavel)
                # --- CALIBRACAO GEOMETRICA (item 127, tecla X) -------------
                # Só alimenta o lado que está sendo calibrado. A normal medida
                # entra apenas como QA de execucao; a VERDADE e' o alvo exato
                # (ou, na pose de mira, -normalize(posicao da mao)).
                if (calib_geo is not None and calib_geo.ativa
                        and calib_geo.lado == lado):
                    _rot_geo = (None if sample_lado is None
                                else rotation_matrix(sample_lado.roll_deg,
                                                     sample_lado.pitch_deg,
                                                     sample_lado.yaw_deg))
                    _pos_geo = None
                    if landmarks_lado is not None:
                        try:
                            _pos_geo = tracker._landmark_camera_point(
                                landmarks_lado[HAND_TRACK_LANDMARK], lado)
                        except (AttributeError, IndexError, TypeError):
                            _pos_geo = None
                    _nrm_geo = (plano_estavel if plano_estavel is not None
                                else normal_lado)
                    # O PLANO-GUIA magenta e' ancorado na mao detectada: sem
                    # landmark+depth nao ha' origem para desenhar (o usuario
                    # remoto achou que era bug; item 130). Avisa no console
                    # (throttle 5 s) e a tela mostra um banner.
                    if (_pos_geo is None or _nrm_geo is None) \
                            and now - ultimo_aviso_mao_geo > 5.0:
                        ultimo_aviso_mao_geo = now
                        print(
                            f"[GEO {lado}] mao/plano NAO detectado (landmark "
                            "ou depth ausente): o plano MAGENTA so' aparece "
                            "ancorado na MAO -- mostre a mao e o dorso para o "
                            "Kinect; o take nao avanca sem isso.", flush=True)
                    _evento_geo = calib_geo.amostra(
                        now, _rot_geo, _pos_geo, _nrm_geo)
                    if _evento_geo is not None:
                        if _evento_geo["tipo"] == "take":
                            _cap_geo = _evento_geo["captura"]
                            print(
                                f"[GEO {lado}] TAKE "
                                f"{_cap_geo['indice'] + 1}/"
                                f"{_evento_geo['total']} GRAVADO: "
                                f"{_cap_geo['pose']} (QA "
                                f"{_cap_geo['qa_deg']:.1f} graus)",
                                flush=True)
                        elif _evento_geo["tipo"] == "pulada":
                            print(f"[GEO {lado}] pose PULADA: "
                                  f"{_evento_geo['pose']}", flush=True)
                    if calib_geo.completa:
                        _fechou_geo = finalizar_calibracao_geometrica(
                            calib_geo, fusions[lado],
                            participante=getattr(args, "participante", ""))
                        _lado_feito_geo = calib_geo.lado
                        calib_geo = None
                        if (_fechou_geo.get("ok")
                                and _lado_feito_geo == "right"):
                            calib_geo = CalibracaoGeometrica("left")
                            print(
                                "AGORA A MAO ESQUERDA: repita as mesmas poses "
                                "(o modo NAO grava nada da mao esquerda antes "
                                "de voce alinhar a pose 1).", flush=True)
                        else:
                            print("Calibracao geometrica encerrada.",
                                  flush=True)
                mao["palm_normal_plano"] = normal_plano_lado
                mao["palm_plano_estavel"] = plano_estavel
                mao["palm_plano_rms_m"] = rms_plano_lado
                mao["palm_plano_spread_deg"] = spread_plano
                mao["palm_erro_plano_deg"] = (
                    None
                    if (plano_estavel is None
                        or mao["palm_direction"] is None)
                    else angle_between(plano_estavel,
                                       mao["palm_direction"])
                )
                estavel_agora = (plano_estavel is not None
                                 and estabilidade_plano_contagem[lado]
                                 >= PLANO_QUADROS_ESTAVEIS_MIN)
                if captura_palma_plano.ativa and estavel_agora:
                    captura_palma_plano.coletar(lado, now, sample_lado,
                                                plano_estavel)
                    if rms_plano_lado is not None:
                        plano_rms_captura[lado].append(float(rms_plano_lado))
                if verificacao_palma.ativa and estavel_agora:
                    rotacao_verif = (None if sample_lado is None
                                     else rotation_matrix(
                                         sample_lado.roll_deg,
                                         sample_lado.pitch_deg,
                                         sample_lado.yaw_deg))
                    verificacao_palma.coletar(lado, now, rotacao_verif,
                                              plano_estavel)
                # --- LOG DO VETOR DA PALMA (item 111) ---------------------
                if log_palma is not None and log_palma_ativo:
                    diag_frame = getattr(tracker, "diag", None) or {}
                    profundidades = diag_frame.get("profundidades") or []
                    posicao_palma = mao.get("position")
                    normal_plano_log = mao.get("palm_normal_plano")
                    normal_estavel_log = mao.get("palm_plano_estavel")
                    palm_dir_log = mao.get("palm_direction")

                    def _eixo(vetor, indice):
                        try:
                            return float(np.asarray(vetor).reshape(3)[indice])
                        except (TypeError, ValueError, IndexError):
                            return None

                    log_palma.registrar(
                        t_mono=now, lado=lado, frame=frame_contador,
                        mao_detectada=1,
                        mapeamento_falhas=diag_frame.get("mapeamento"),
                        depth_falhas=diag_frame.get("depth"),
                        fallbacks=diag_frame.get("fallbacks"),
                        resgatados=diag_frame.get("resgatados"),
                        pontos_3d_ok=diag_frame.get("pontos"),
                        depth_palma_mm=(None if posicao_palma is None
                                        else float(posicao_palma[2]) * 1000.0),
                        depth_min_mm=(min(profundidades)
                                      if profundidades else None),
                        depth_max_mm=(max(profundidades)
                                      if profundidades else None),
                        normal_plano_x=_eixo(normal_plano_log, 0),
                        normal_plano_y=_eixo(normal_plano_log, 1),
                        normal_plano_z=_eixo(normal_plano_log, 2),
                        normal_estavel_x=_eixo(normal_estavel_log, 0),
                        normal_estavel_y=_eixo(normal_estavel_log, 1),
                        normal_estavel_z=_eixo(normal_estavel_log, 2),
                        spread_deg=spread_plano,
                        rms_mm=(None if rms_plano_lado is None
                                else float(rms_plano_lado) * 1000.0),
                        estavel_contagem=estabilidade_plano_contagem[lado],
                        robust_inliers=fracao_robusta,
                        palm_dir_x=_eixo(palm_dir_log, 0),
                        palm_dir_y=_eixo(palm_dir_log, 1),
                        palm_dir_z=_eixo(palm_dir_log, 2),
                        erro_plano_deg=mao.get("palm_erro_plano_deg"),
                        erro_palma_deg=mao.get("palm_error_deg"),
                        sync_color_depth_ms=getattr(
                            tracker, "sync_color_depth_ms", None),
                        seta_visivel=(1 if normal_estavel_log is not None
                                      else 0),
                        log_ligado=1)

                # Diagnostico da palma (tecla D): a linha do CSV do diagnostico
                # e' gravada AQUI, logo depois do erro ao vivo -- as tres coisas
                # tem de ser do MESMO frame e da MESMA amostra de IMU, senao a
                # comparacao "vivo x bruto x refit" mede o atraso, nao o erro.
                if diag_palma.ativa:
                    # Os 4 pontos 3D crus so' sao calculados DURANTE o
                    # diagnostico (4 lookups de depth extras por frame/mao): e'
                    # a evidencia do sinal da normal (item 78), nao precisa
                    # existir na aquisicao normal.
                    pontos_diag = (None if landmarks_lado is None
                                   else pontos_palma_camera(tracker,
                                                            landmarks_lado))
                    diag_palma.registrar(lado, now, sample_lado, normal_lado,
                                         fusion, pontos=pontos_diag)
                if captura_palma.ativa:
                    captura_palma.coletar(lado, now, sample_lado, normal_lado)
                if imu_calibration_active[lado]:
                    fusion.add_bias_sample(sample_lado)
                    palm_normal = normal_lado
                    if (
                        now >= calibration_until
                        and sample_lado is not None
                        and camera_position_lado is not None
                        and palm_normal is not None
                    ):
                        # Com alinhamento CARREGADO, `palm_reference_body` ja' veio
                        # do ajuste de varias amostras: reancorar com UMA amostra
                        # (ruidosa) so' pioraria. So' ancora quando nao ha' nada.
                        if not alinhamento_carregado[lado]:
                            fusion.calibrate_palm_direction(sample_lado,
                                                            palm_normal)
                        fusion.origin = camera_position_lado.copy()
                        fusion.position[:] = 0.0
                        fusion.velocity[:] = 0.0
                        imu_calibration_active[lado] = False
                        print(f"IMU {lado}: direcao da palma "
                              + ("mantida do alinhamento salvo."
                                 if alinhamento_carregado[lado]
                                 else "ancorada em 1 amostra so' (bootstrap "
                                      "ruidoso, NAO e' o alinhamento da tecla "
                                      "O) -- aperte O com a mao visivel e "
                                      "girando para a seta ficar confiavel."))
                elif now < bias_calibration_until:
                    fusion.add_bias_sample(sample_lado)
                if imu_calibration_active[lado]:
                    mao["fused"] = fusion.position.copy()
                else:
                    mao["fused"] = fusion.update(sample_lado, camera_position_lado)
            # --- MODO TAKES (item 120): 2 s parado = take ---------------
            if poses_fisicas.ativa and poses_fisicas.modo_takes:
                _evento_take = poses_fisicas.processar_takes(now)
                if _evento_take is not None:
                    if _evento_take.get("rejeitado"):
                        print(f"  TAKE {_evento_take['rejeitado']} RECUSADO: "
                              "pose muito parecida com a anterior "
                              f"({_evento_take['angulo']:.1f} graus < "
                              f"{_evento_take['minimo']:g}). Desloque a "
                              "mao ~30-40 cm mais para o lado/baixo "
                              "(ate' perto da borda do quadro) e "
                              "segure PARADO de novo.", flush=True)
                    if _evento_take.get("take"):
                        print(f"  TAKE {_evento_take['take']}/"
                              f"{_evento_take['total']} gravado! Pode ir "
                              "para a proxima posicao no seu tempo.",
                              flush=True)
                    if _evento_take.get("pronto"):
                        print("  todos os takes gravados; "
                              "finalizando...", flush=True)
            # --- fim da captura do alinhamento da palma (tecla O) -------------
            if captura_palma.ativa and captura_palma.restante_s <= 0.0:
                captura_palma.parar()
                print("Captura do alinhamento da palma terminada.")
                for lado in ("right", "left"):
                    amostras_lado = captura_palma.contar(lado)
                    resolvido = captura_palma.resolver(lado)
                    if resolvido is None:
                        print(f"  {lado}: amostras insuficientes "
                              f"({amostras_lado}; precisa de "
                              f"{PALM_ALIGNMENT_MIN_SAMPLES} com a mao girando) "
                              "-- alinhamento NAO atualizado")
                        continue
                    t_lado, f_lado, residuo = resolvido
                    fusions[lado].set_palm_alignment(t_lado, f_lado, residuo)
                    gravou = save_palm_alignment(fusions[lado], lado)
                    veredito = ("OK" if residuo <= PALM_ALIGNMENT_MAX_RESIDUAL_DEG
                                else "RESIDUO ALTO")
                    print(f"  {lado}: {amostras_lado} amostras | residuo "
                          f"{residuo:.2f} graus ({veredito}) | "
                          + (f"gravado em {palm_alignment_path(lado).name}"
                             if gravou else "NAO gravado"))
                    amostras_rms = plano_rms_captura.get(lado) or []
                    if amostras_rms:
                        print(f"    {lado}: qualidade do plano na captura: RMS "
                              f"mediana "
                              f"{float(np.median(amostras_rms)) * 1000:.1f} mm "
                              f"({len(amostras_rms)} quadros) -- RMS alto = luz "
                              "estourada/landmarks ruins, refaca com menos luz")
                    if residuo > PALM_ALIGNMENT_MAX_RESIDUAL_DEG:
                        print(f"    {lado}: residuo acima de "
                              f"{PALM_ALIGNMENT_MAX_RESIDUAL_DEG:g} graus -- "
                              "repita girando a mao SEM virar a palma para o "
                              "dorso e mantenha a mao dentro do quadro; se "
                              "persistir, confira se as luvas nao estao "
                              "trocadas (--imu-portas)")
            # --- POSES FISICAS (tecla J calibrar / V verificar, item 114) ---
            if poses_fisicas.ativa:
                if poses_fisicas.modo_takes:
                    _terminou_poses = poses_fisicas.takes_completos()
                else:
                    _terminou_poses = poses_fisicas.fase_atual(now) is None
                if _terminou_poses:
                    modo_pose = poses_fisicas.modo
                    poses_fisicas.parar()
                    poses_fisicas_fase_vista = None
                    if modo_pose == "calibrar":
                        if mesa_normal_camera is not None:
                            resultado_poses = poses_fisicas.calibrar_por_posicoes()
                            if resultado_poses.get("faltam_posicoes"):
                                print("  ERRO: o Kinect nao mediu a POSICAO "
                                      "da mao em todas as poses (faltam: "
                                      f"{resultado_poses['faltam_posicoes']}). "
                                      "NADA sera gravado -- mantenha as "
                                      "DUAS maos visiveis, segure ~2 s em "
                                      "cada pose e refaca o J.")
                            mesa_sinal = float(
                                resultado_poses.get("sinal_mesa") or 1.0)
                            poses_verdade_por_lado = {
                                _lado_v: tuple(_dados_v.get("poses_usadas"))
                                for _lado_v, _dados_v
                                in resultado_poses.get("lados", {}).items()
                                if _dados_v.get("poses_usadas")}
                            print("CALIBRACAO POR POSES FISICAS "
                                  "terminada (verdade: DIRECAO MAO->LENTE "
                                  f"medida pelo Kinect nas "
                                  f"{len(poses_fisicas.poses)} posicoes):")
                            for _lado_diag in ("right", "left"):
                                _ang = resultado_poses.get(
                                    "angulos", {}).get(_lado_diag)
                                if _ang:
                                    _exp12 = _angulo_entre(
                                        poses_fisicas.poses[0][1],
                                        poses_fisicas.poses[1][1])
                                    _exp13 = _angulo_entre(
                                        poses_fisicas.poses[0][1],
                                        poses_fisicas.poses[2][1])
                                    _exp23 = _angulo_entre(
                                        poses_fisicas.poses[1][1],
                                        poses_fisicas.poses[2][1])
                                    print(f"  {_lado_diag}: angulos "
                                          "medidos entre poses "
                                          f"(1-2, 1-3, 2-3) = "
                                          f"{_ang[0]:.1f}, {_ang[1]:.1f}, "
                                          f"{_ang[2]:.1f} graus "
                                          f"(esperado ~{_exp12:.0f}, "
                                          f"~{_exp13:.0f}, ~{_exp23:.0f})")
                                    if abs(_ang[2] - 180.0) > 12.0:
                                        print("    AVISO: as poses de "
                                              "mesa nao ficaram "
                                              "opostas; usar o apoio "
                                              "das maos ajuda")
                                else:
                                    _modo_verdade_lado = resultado_poses.get(
                                        "modo_verdade", "")
                                    if _modo_verdade_lado == "posicao_da_mao":
                                        print(f"  {_lado_diag}: verdade = "
                                              "POSICAO da mao nas 3 poses "
                                              "(direcao mao->lente medida pelo Kinect)")
                                    else:
                                        print(f"  {_lado_diag}: Kinect nao "
                                              "mediu as normais das 3 poses "
                                              "(mao fora de quadro?) -- usando o "
                                              "plano da mesa como verdade; mantenha as "
                                              "maos visiveis durante o J")
                        else:
                            resultado_poses = poses_fisicas.calibrar()
                            print("CALIBRACAO POR POSES FISICAS "
                                  "terminada (referencia: Kinect "
                                  "NIVELADO):")
                        #: guarda as verdades/angulos p/ o V usar as mesmas
                        try:
                            _estado_salvar = {}
                            if caminho_mesa.exists():
                                _estado_salvar = json.loads(
                                    caminho_mesa.read_text(encoding="utf-8"))
                            _estado_salvar["poses_verdade"] = {
                                _lado_json: [[_nome_json,
                                              [float(c) for c in _vec_json]]
                                             for _nome_json, _vec_json in
                                             poses_verdade_por_lado
                                             .get(_lado_json, ())]
                                for _lado_json in ("right", "left")
                                if poses_verdade_por_lado.get(_lado_json)
                            }
                            _estado_salvar["angulos"] = {
                                _lado_json: [float(v) for v in _ang_json]
                                for _lado_json, _ang_json in
                                resultado_poses.get("angulos", {}).items()}
                            _estado_salvar["sinal"] = float(mesa_sinal)
                            _estado_salvar["quando"] = time.strftime(
                                "%Y-%m-%d %H:%M:%S")
                            if mesa_plano_info is not None:
                                _estado_salvar["normal"] = [float(c) for c in
                                                             mesa_normal_camera]
                                _estado_salvar["tilt_deg"] = float(
                                    mesa_plano_info.get("tilt_deg") or 0.0)
                                _estado_salvar["inliers"] = int(
                                    mesa_plano_info.get("inliers") or 0)
                                _estado_salvar["fracao"] = float(
                                    mesa_plano_info.get("fracao") or 0.0)
                            caminho_mesa.write_text(
                                json.dumps(_estado_salvar, indent=2),
                                encoding="utf-8")
                            print("  verdades das poses gravadas em "
                                  f"{caminho_mesa.name}")
                        except (OSError, TypeError, ValueError) as exc:
                            print("  nao gravou as verdades em "
                                  f"{caminho_mesa.name}: {exc}")
                        for lado in ("right", "left"):
                            if melhor_palma_sessao.get(lado) is None:
                                melhor_palma_sessao[lado] = getattr(
                                    fusions[lado],
                                    "palm_alignment_residual_deg", None)
                            dados_pose = resultado_poses["lados"].get(lado, {})
                            _espalhamento = dados_pose.get(
                                "espalhamento_min_deg")
                            _espalhamento_max = dados_pose.get(
                                "espalhamento_max_deg")
                            if _espalhamento is not None:
                                _texto_espalhamento = (
                                    f"  {lado}: espalhamento "
                                    f"min {_espalhamento:.1f} / "
                                    f"max {_espalhamento_max:.1f} "
                                    "graus "
                                    f"(minimo {POSES_FISICAS_SPREAD_MIN_DEG:g}, "
                                    f"maximo >= "
                                    f"{POSES_FISICAS_SPREAD_MAX_MIN_DEG:g})")
                                if _espalhamento < POSES_FISICAS_SPREAD_MIN_DEG:
                                    _texto_espalhamento += (
                                        " -- POSES MUITO PARECIDAS: "
                                        "leve a mao mais para os "
                                        "lados/baixo e refaca")
                                elif (_espalhamento_max is not None
                                      and _espalhamento_max
                                      < POSES_FISICAS_SPREAD_MAX_MIN_DEG):
                                    _texto_espalhamento += (
                                        " -- LECQUE ESTREITO: va mais "
                                        "para a esquerda/direita/baixo "
                                        "e refaca")
                                print(_texto_espalhamento)
                            if dados_pose.get("sem_amostras"):
                                faltam = dados_pose.get("poses_insuficientes")
                                print(f"  {lado}: amostras insuficientes "
                                      f"(poses {faltam}) -- alinhamento "
                                      "NAO atualizado; segure cada pose "
                                      "PARADA ~2 s")
                                continue
                            t_pose = dados_pose["T"]
                            f_pose = dados_pose["f"]
                            residuo_pose = dados_pose["residuo"]
                            fusions[lado].set_palm_alignment(
                                t_pose, f_pose, residuo_pose)
                            _espalhamento_pose = dados_pose.get(
                                "espalhamento_min_deg")
                            _espalhamento_ok = (
                                _espalhamento_pose is None
                                or _espalhamento_pose >=
                                POSES_FISICAS_SPREAD_MIN_DEG)
                            _melhor = melhor_palma_sessao.get(lado)
                            _melhora = (_melhor is None
                                       or residuo_pose < _melhor)
                            if _espalhamento_ok and _melhora:
                                gravou_pose = save_palm_alignment(
                                    fusions[lado], lado)
                                melhor_palma_sessao[lado] = residuo_pose
                                _texto_gravacao = (
                                    f"gravado em "
                                    f"{palm_alignment_path(lado).name}"
                                    if gravou_pose else "NAO gravado")
                            elif not _espalhamento_ok:
                                _texto_gravacao = (
                                    "NAO gravado (poses muito "
                                    "parecidas: espalhamento "
                                    f"{_espalhamento_pose:.1f} < "
                                    f"{POSES_FISICAS_SPREAD_MIN_DEG:g} "
                                    "graus -- leve a mao mais para os "
                                    "lados/baixo e refaca)")
                            else:
                                _texto_gravacao = (
                                    "NAO gravado (pior que o melhor da "
                                    f"sessao: {_melhor:.2f} graus)")
                            print(f"  {lado}: {dados_pose['amostras']} "
                                  f"amostras | residuo {residuo_pose:.2f} "
                                  f"graus | {_texto_gravacao}")
                            for indice_pose, (nome_pose, _vetor_pose) in enumerate(
                                    POSES_FISICAS_CALIBRACAO):
                                erro_pose = dados_pose["poses"].get(
                                    indice_pose)
                                if erro_pose is not None:
                                    print(f"    {indice_pose + 1}. "
                                          f"{nome_pose}: erro "
                                          f"{erro_pose:.1f} graus")
                        print("  CERTIFICADO: "
                              + ("FECHADO (3 poses dentro do limite)"
                                 if resultado_poses["fechado"] else
                                 "NAO FECHADO -- repita as poses com "
                                 "mais calma (parado ~2 s em cada)"))
                    else:
                        _verdades_v = dict(poses_verdade_por_lado or {})
                        if any(len(_lista) != len(poses_fisicas.poses)
                               for _lista in _verdades_v.values()):
                            print("  AVISO: as verdades gravadas sao de "
                                  "outro numero de poses (rode J de "
                                  "novo com as 4 poses); usando as "
                                  "nominais")
                            _verdades_v = {}
                        relatorio_pose = poses_fisicas.verificar(
                            fusions, poses_por_lado=(_verdades_v or None))
                        print("VERIFICACAO POR POSES FISICAS terminada "
                              f"(limite {POSES_FISICAS_LIMITE_POSE1_DEG:g} graus "
                              "na pose 1 e "
                              f"{POSES_FISICAS_LIMITE_MESA_DEG:g} nas demais):")
                        for lado in ("right", "left"):
                            dados_pose = relatorio_pose["lados"].get(
                                lado, {})
                            print(f"  {lado}:")
                            if dados_pose.get("sem_alinhamento"):
                                print("    SEM ALINHAMENTO CARREGADO")
                                continue
                            for indice_pose, (nome_pose, _vetor_pose) in enumerate(
                                    POSES_FISICAS_CALIBRACAO):
                                stats_pose = dados_pose["poses"].get(
                                    indice_pose, {})
                                if stats_pose.get("media") is None:
                                    print(f"    {indice_pose + 1}. "
                                          f"{nome_pose}: SEM "
                                          "ESTABILIDADE/AMOSTRAS "
                                          f"(n={stats_pose.get('n', 0)}) "
                                          "-- FAIL")
                                else:
                                    print(f"    {indice_pose + 1}. "
                                          f"{nome_pose}: "
                                          f"n={stats_pose['n']} | media "
                                          f"{stats_pose['media']:.1f} | "
                                          f"p90 {stats_pose['p90']:.1f} | "
                                          f"max {stats_pose['max']:.1f} "
                                          "graus (limite "
                                          f"{stats_pose.get('limite_graus', 12.0):.0f}) "
                                          "-- "
                                          + ("PASS" if stats_pose["passou"]
                                             else "FAIL"))
                        if relatorio_pose["fechado"]:
                            print("  CERTIFICADO: FECHADO")
                        else:
                            ruins = "; ".join(
                                f"{lado} {indice} ({nome}, "
                                + ("sem amostras" if media is None
                                   else f"{media:.1f} graus") + ")"
                                for lado, indice, nome, media
                                in relatorio_pose["poses_ruins"])
                            print("  CERTIFICADO: NAO FECHADO -- poses "
                                  f"ruins: {ruins}")
            # --- fim da captura do alinhamento por PLANO DO DORSO (tecla J) ---
            if captura_palma_plano.ativa and captura_palma_plano.restante_s <= 0.0:
                captura_palma_plano.parar()
                print("Captura do alinhamento por PLANO DO DORSO terminada.")
                for lado in ("right", "left"):
                    amostras_lado = captura_palma_plano.contar(lado)
                    if amostras_lado < PALM_PLANO_MIN_AMOSTRAS:
                        print(f"  {lado}: amostras insuficientes "
                              f"({amostras_lado}; precisa de "
                              f"{PALM_PLANO_MIN_AMOSTRAS} com a mao "
                              "PARADA ~1,5 s em orientacoes diferentes) -- "
                              "alinhamento NAO atualizado; veja o "
                              "palma_vetor_log_*.csv (tecla G)")
                        continue
                    resolvido = captura_palma_plano.resolver(
                        lado, min_amostras=PALM_PLANO_MIN_AMOSTRAS)
                    if resolvido is None:
                        print(f"  {lado}: refit falhou (dados "
                              "degenerados) -- alinhamento NAO atualizado")
                        continue
                    t_lado, f_lado, residuo = resolvido
                    erros_antes = []
                    for rotacao_amostra, normal_amostra in captura_palma_plano.amostras[lado]:
                        vetor = (fusions[lado].imu_to_camera @ rotacao_amostra
                                 @ fusions[lado].palm_reference_body)
                        norma = float(np.linalg.norm(vetor))
                        if norma > 1e-9:
                            erros_antes.append(angle_between(
                                normal_amostra, vetor / norma))
                    erro_antes = (float(np.mean(erros_antes))
                                  if erros_antes else float("nan"))
                    fusions[lado].set_palm_alignment(t_lado, f_lado, residuo)
                    gravou = save_palm_alignment(fusions[lado], lado)
                    veredito = ("OK" if residuo <= PALM_ALIGNMENT_MAX_RESIDUAL_DEG
                                else "RESIDUO ALTO")
                    print(f"  {lado}: {amostras_lado} amostras | residuo do "
                          f"PLANO {residuo:.2f} graus ({veredito}) | erro do "
                          f"alinhamento ANTERIOR {erro_antes:.1f} graus | "
                          + (f"gravado em {palm_alignment_path(lado).name}"
                             if gravou else "NAO gravado"))
                    erros_novos = []
                    for rotacao_amostra, normal_amostra in captura_palma_plano.amostras[lado]:
                        vetor = (fusions[lado].imu_to_camera
                                 @ rotacao_amostra
                                 @ fusions[lado].palm_reference_body)
                        norma_vetor = float(np.linalg.norm(vetor))
                        if norma_vetor > 1e-9:
                            erros_novos.append(angle_between(
                                normal_amostra, vetor / norma_vetor))
                    if erros_novos:
                        mediana_nova = float(np.median(erros_novos))
                        p90_nova = float(np.percentile(erros_novos, 90))
                        print(f"    {lado}: erro do modelo NOVO nessas "
                              f"amostras: mediana {mediana_nova:.1f} | "
                              f"p90 {p90_nova:.1f} graus")
                        if mediana_nova > PALM_ALIGNMENT_MAX_RESIDUAL_DEG:
                            print(f"    {lado}: ATENCAO -- erro alto nas "
                                  "amostras: capture de novo com a mao "
                                  "PARADA ~1,5 s em cada orientacao")
                    amostras_rms = plano_rms_captura.get(lado) or []
                    if amostras_rms:
                        print(f"    {lado}: qualidade do plano na captura: RMS "
                              f"mediana "
                              f"{float(np.median(amostras_rms)) * 1000:.1f} mm "
                              f"({len(amostras_rms)} quadros) -- RMS alto = luz "
                              "estourada/landmarks ruins, refaca com menos luz")
            # --- VERIFICACAO GUIADA DA PALMA (tecla V, item 103) --------------
            if verificacao_palma.ativa:
                fase_verif = verificacao_palma.fase_atual(now)
                if fase_verif is None:
                    verificacao_palma.parar()
                    relatorio_verif = verificacao_palma.relatorio(fusions)
                    print("VERIFICACAO DA PALMA terminada (limite "
                          f"{relatorio_verif['limite_graus']:g} "
                          "graus/pose):")
                    for linha_cert in formatar_certificado(relatorio_verif):
                        print(linha_cert)
                    verif_fase_vista = None
                elif fase_verif[0] != verif_fase_vista:
                    verif_fase_vista = fase_verif[0]
                    print(f"VERIFICACAO PALMA pose {fase_verif[0] + 1}/"
                          f"{len(verificacao_palma.poses)}: {fase_verif[1]} "
                          f"({fase_verif[2]:.1f} s)", flush=True)
            # --- DIAGNOSTICO DA PALMA (tecla D): fase e fim --------------------
            # Roda UMA vez por iteracao (depois do laco dos lados): trocar a
            # fase no meio do laco faria uma mao gravar na fase nova e a outra
            # na antiga no MESMO frame. A instrucao sai no terminal UMA vez por
            # fase -- repetir a cada frame afogaria as outras linhas (e o
            # experimentador esta' olhando a mao, nao o console).
            if diag_palma.ativa:
                fase_diag = diag_palma.fase_atual(now)
                if fase_diag is None:
                    diag_palma.finalizar()
                    diag_fase_vista = None
                elif fase_diag[0] != diag_fase_vista:
                    diag_fase_vista = fase_diag[0]
                    print(f"DIAG PALMA fase {fase_diag[0] + 1}/"
                          f"{len(diag_palma.fases)} {fase_diag[1]} "
                          f"({fase_diag[3]:.1f} s): {fase_diag[4]}",
                          flush=True)
            # Compatibilidade: variaveis "atuais" apontam para a mao primaria.
            lado_atual = lado_primario or "right"
            prim_mao = resultados_por_lado.get(lado_atual)
            fusion = fusions[lado_atual]
            sample = prim_mao["sample"] if prim_mao else None
            fused = (
                prim_mao["fused"] if prim_mao and prim_mao["fused"] is not None
                else fusion.position.copy()
            )
            camera_position = prim_mao["position"] if prim_mao else None
            palm_direction = prim_mao["palm_direction"] if prim_mao else None
            if camera_calibrating:
                if color_for_calibration is not None and auxiliary_ok:
                    calibration_kinect = cv2.cvtColor(color_for_calibration, cv2.COLOR_BGRA2BGR)
                    kinect_found = alignment.draw_board_detection(calibration_kinect)
                    auxiliary_found = alignment.draw_board_detection(auxiliary_frame)
                    auxiliary_calibration_display = (
                        cv2.flip(auxiliary_frame, 1)
                        if AUXILIARY_DISPLAY_MIRROR
                        else auxiliary_frame.copy()
                    )
                    cv2.putText(
                        calibration_kinect,
                        f"Tabuleiro: {'OK' if kinect_found else 'nao encontrado'} | ESPACO captura",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0) if kinect_found else (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        auxiliary_calibration_display,
                        f"Tabuleiro: {'OK' if auxiliary_found else 'nao encontrado'} | ESPACO captura",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0) if auxiliary_found else (0, 0, 255),
                        2,
                    )
                    # As janelas de calibracao sao da camera ATIVA (e' ela que
                    # tem o tabuleiro enquadrado no modo T): o titulo leva o
                    # indice para nao confundir com as janelas de tracking.
                    janelas.mostrar(
                        f"Calibracao {AUX_CAMERA_INDEX} - Kinect",
                        janelas.redimensionar(calibration_kinect))
                    janelas.mostrar(
                        f"Calibracao {AUX_CAMERA_INDEX} - camera auxiliar",
                        janelas.redimensionar(auxiliary_calibration_display))
            # Textos de IMU/camera POR lado (uma linha por mao). A IMU aparece
            # mesmo sem mao visivel (a luva pode estar fora do quadro).
            partes_imu = []
            for lado in ("right", "left"):
                amostra_lado = amostras_por_lado.get(lado)
                if amostra_lado is None:
                    partes_imu.append(f"IMU {lado[0].upper()} aguardando UDP")
                else:
                    partes_imu.append(
                        f"IMU {lado[0].upper()} R/P/Y {amostra_lado.roll_deg:.1f}/"
                        f"{amostra_lado.pitch_deg:.1f}/{amostra_lado.yaw_deg:.1f}"
                    )
            imu_text = " | ".join(partes_imu)
            camera_text = " | ".join(
                (
                    f"{lado[0].upper()}: XYZ {mao['position'][0]:.3f} "
                    f"{mao['position'][1]:.3f} {mao['position'][2]:.3f} m"
                    if mao is not None and mao["position"] is not None
                    else f"{lado[0].upper()}: sem mao ({tracker.last_hand_reason})"
                )
                for lado, mao in resultados_por_lado.items()
            )
            if camera_calibrating:
                calibration_text = f"CALIBRACAO: tabuleiro visivel nas duas cameras, ESPACO ({len(alignment.auxiliary_points)}/{CALIBRATION_SAMPLES_REQUIRED})"
            elif any(imu_calibration_active.values()):
                remaining = max(0.0, calibration_until - now)
                faltando = "/".join(
                    lado[0].upper() for lado, ativo in imu_calibration_active.items() if ativo
                )
                calibration_text = f"CALIBRANDO IMU {faltando}: mao visivel e parada ({remaining:.1f} s)"
            elif captura_palma.ativa:
                calibration_text = (
                    f"CAPTURA ALINHAMENTO DA PALMA: gire a mao devagar com a "
                    f"palma virada para o Kinect ({captura_palma.restante_s:.1f} s) "
                    f"| R {captura_palma.contar('right')} amostras | "
                    f"L {captura_palma.contar('left')} amostras")
            else:
                calibration_text = ("C origem | B bias | O alinhamento palma | "
                                    "R reset | H cal mao | P limpa | T tabuleiro "
                                    "| ESC sair")
            # Uma linha [status] a cada STATUS_LOG_INTERVALO_S: a saude do IMU
            # num lugar so' (taxa por porta, alinhamento O SIM/NAO -- so' conta
            # com residual do solve, nao o bootstrap -- e erro da palma ao vivo).
            if now - status_log_em >= STATUS_LOG_INTERVALO_S:
                taxas = []
                for lado in ("right", "left"):
                    recebidas = receivers[lado].count
                    anterior = status_counts[lado]
                    if anterior is None or status_log_t0 <= 0.0:
                        taxas.append("-- Hz")
                    else:
                        dt_status = max(now - status_log_t0, 1e-3)
                        taxas.append(
                            f"{(recebidas - anterior) / dt_status:.0f} Hz")
                    status_counts[lado] = recebidas
                alinhamentos = []
                erros_palma = []
                erros_plano = []
                estabilidade_plano = []
                for lado in ("right", "left"):
                    sigla = lado[0].upper()
                    estado = fusions[lado].palm_alignment_state()
                    tem = (estado is not None
                           and estado.get("residual_deg") is not None)
                    alinhamentos.append(f"{sigla} {'SIM' if tem else 'NAO'}")
                    mao_l = resultados_por_lado.get(lado)
                    erro = (None if mao_l is None
                            else mao_l.get("palm_error_deg"))
                    erros_palma.append(
                        f"{sigla} " + ("--" if erro is None
                                       else f"{erro:.1f} graus"))
                    erro_plano = (None if mao_l is None
                                  else mao_l.get("palm_erro_plano_deg"))
                    erros_plano.append(
                        f"{sigla} " + ("--" if erro_plano is None
                                       else f"{erro_plano:.1f} graus"))
                    rms_p = (None if mao_l is None
                             else mao_l.get("palm_plano_rms_m"))
                    spread_p = (None if mao_l is None
                                else mao_l.get("palm_plano_spread_deg"))
                    estabilidade_plano.append(
                        f"{sigla} " + ("--" if rms_p is None
                                       else f"{rms_p*1000:.1f}mm"
                                       + ("/--" if spread_p is None
                                          else f"/{spread_p:.1f}graus")))
                falta_o = "NAO" in " ".join(alinhamentos)
                print(f"[status] IMU R {taxas[0]} | L {taxas[1]} | "
                      f"alinhamento O: {' / '.join(alinhamentos)} | "
                      f"erro palma: {' | '.join(erros_palma)} | "
                      f"erro plano: {' | '.join(erros_plano)} | "
                      f"estab. plano: {' | '.join(estabilidade_plano)}"
                      + (f" | nivel mesa: "
                         f"{mesa_plano_info['tilt_deg']:.1f} graus"
                         if mesa_plano_info is not None else "")
                      + (" | SEM ALINHAMENTO -- aperte O (14 s girando a mao)"
                         if falta_o else ""),
                      flush=True)
                loop_ms = desempenho["loop_ms"] or float("nan")
                sinc_ms = getattr(tracker, "sync_color_depth_ms", None)
                sinc_txt = ("--" if sinc_ms is None
                            else "%.0f ms" % sinc_ms)
                print("[desempenho] loop %.1f fps (%.0f ms) | "
                      "Kinect+MediaPipe %.0f ms | auxiliares+MediaPipe %.0f ms"
                      " | sync color-depth %s"
                      % ((1000.0 / loop_ms) if loop_ms else float("nan"),
                         loop_ms, desempenho["kinect_ms"] or float("nan"),
                         desempenho["aux_ms"] or float("nan"), sinc_txt),
                      flush=True)
                status_log_em = now
                status_log_t0 = now
            if display is not None:
                for lado, mao in resultados_por_lado.items():
                    if mao is None or mao["landmarks"] is None:
                        continue
                    # Comparacao AO VIVO (item 102, tecla J): VERMELHO = seta do
                    # IMU (T R f); VERDE = normal do PLANO MEDIO do dorso
                    # medida pelo Kinect (a verdade mais estavel). O rotulo traz
                    # o erro entre as duas em graus.
                    if mao["palm_direction"] is not None:
                        desenhar_vetor_palma_seguro(
                            tracker, display, mao["landmarks"],
                            mao["palm_direction"])
                    normal_plano = mao.get("palm_plano_estavel")
                    rotulo = "plano dorso"
                    if normal_plano is None:
                        mantido = plano_display.get(lado)
                        if (mantido is not None
                                and (now - mantido[1]) <= 1.5):
                            normal_plano = mantido[0]
                            rotulo = "plano dorso (mantido)"
                        else:
                            normal_plano = mao.get("palm_normal_plano")
                    if normal_plano is not None:
                        if mao["palm_direction"] is not None:
                            rotulo += (" (err %.0f)" % angle_between(
                                normal_plano, mao["palm_direction"]))
                        desenhar_vetor_palma_seguro(
                            tracker, display, mao["landmarks"], normal_plano,
                            cor=(0, 255, 0), rotulo=rotulo)
            if display is not None and verificacao_palma.ativa:
                fase_verif_display = verificacao_palma.fase_atual(now)
                if fase_verif_display is not None:
                    cv2.putText(
                        display,
                        f"VERIFICACAO {fase_verif_display[0] + 1}/"
                        f"{len(verificacao_palma.poses)}: "
                        f"{fase_verif_display[1]} "
                        f"({fase_verif_display[2]:.1f} s)",
                        (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)

            if display is not None:
                _texto_amostras = None
                if captura_palma_plano.ativa:
                    _texto_amostras = (
                        'J (plano do dorso): R '
                        + str(captura_palma_plano.contar('right'))
                        + ' | L '
                        + str(captura_palma_plano.contar('left'))
                        + ' amostras')
                elif verificacao_palma.ativa:
                    _fase_amostras = verificacao_palma.fase_atual(now)
                    if _fase_amostras is not None:
                        _texto_amostras = (
                            'V ' + str(_fase_amostras[0] + 1) + '/'
                            + str(len(verificacao_palma.poses))
                            + ' | R '
                            + str(verificacao_palma.contar('right'))
                            + ' | L '
                            + str(verificacao_palma.contar('left'))
                            + ' amostras')
                if _texto_amostras is not None:
                    cv2.putText(display, _texto_amostras, (12, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (255, 255, 0), 2)
            if display is not None and poses_fisicas.ativa:
                if poses_fisicas.modo_takes:
                    _modo_txt = ("CALIBRAR" if poses_fisicas.modo
                                  == "calibrar" else "VERIFICAR")
                    _indice_take = min(poses_fisicas.take_indice(),
                                      len(poses_fisicas.poses) - 1)
                    _nome_take = poses_fisicas.poses[_indice_take][0]
                    cv2.putText(
                        display,
                        f"{_modo_txt} TAKES {_indice_take}/"
                        f"{len(poses_fisicas.poses)} -- "
                        "2 s PARADO = TAKE (sem tempo fixo)",
                        (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 255, 255), 2)
                    cv2.putText(display, _nome_take[:76], (12, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 0), 2)
                    for _ordem, _lado_txt in enumerate(("right", "left")):
                        _estado_txt = poses_fisicas.estado_lado(
                                        _lado_txt, now)
                        if not _estado_txt["movimento_ok"]:
                            _msg_estado = "MOVA / va para a proxima posicao"
                        elif _estado_txt["estavel"]:
                            _msg_estado = "PARADO OK -- gravando take"
                        else:
                            _msg_estado = (
                                f"parado {_estado_txt['janela_s']:.1f}/"
                                f"{POSES_FISICAS_TAKE_SEC:.0f} s")
                        cv2.putText(
                            display,
                            f"{_lado_txt.upper()}: {_msg_estado}",
                            (12, 218 + _ordem * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 0), 2)
                else:
                    fase_pose_display = poses_fisicas.fase_atual(now)
                    if fase_pose_display is not None:
                        modo_display = ("CALIBRAR" if poses_fisicas.modo
                                       == "calibrar" else "VERIFICAR")
                        cv2.putText(
                            display,
                            f"{modo_display} POSES FISICAS "
                            f"{fase_pose_display[0] + 1}/"
                            f"{len(poses_fisicas.poses)} -- SEGURE PARADO "
                            f"({fase_pose_display[2]:.1f} s)",
                            (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
                        cv2.putText(display, fase_pose_display[1][:78],
                                    (12, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 0), 2)
                        cv2.putText(
                            display,
                            f"amostras R "
                            f"{poses_fisicas.contar('right')} | L "
                            f"{poses_fisicas.contar('left')}",
                            (12, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)
            # --- OVERLAY DA CALIBRACAO GEOMETRICA (item 127, tecla X) -----
            if display is not None and calib_geo is not None \
                    and calib_geo.ativa:
                _prog_geo = calib_geo.progresso(now)
                _pose_geo = _prog_geo["pose"]
                _lado_geo = calib_geo.lado
                _mao_geo = resultados_por_lado.get(_lado_geo)
                _origem_geo = None
                if (_mao_geo is not None
                        and _mao_geo.get("landmarks") is not None):
                    try:
                        _origem_geo = tracker._landmark_camera_point(
                            _mao_geo["landmarks"][HAND_TRACK_LANDMARK],
                            _lado_geo)
                    except (AttributeError, IndexError, TypeError):
                        _origem_geo = None
                if (_pose_geo is not None and _origem_geo is not None
                        and _pose_geo.get("tipo") == "geometrica"):
                    desenhar_guia_plano_palma(
                        tracker, display, _origem_geo,
                        _pose_geo.get("alvo"), _pose_geo.get("dedos"))
                elif _pose_geo is not None:
                    # Sem mao/depth a projecao 3D nao existe: explica na tela
                    # em vez de simplesmente "nao desenhar nada" (confundiu no
                    # teste remoto). O guia volta assim que a mao aparece.
                    cv2.putText(
                        display,
                        "MAO NAO DETECTADA -- mostre a MAO + PROFUNDIDADE "
                        "para o Kinect; o PLANO MAGENTA e' ancorado na mao",
                        (12, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
                _lado_txt_geo = ("DIREITA" if _lado_geo == "right"
                                 else "ESQUERDA")
                _indice_geo = min(_prog_geo["indice"] + 1,
                                  _prog_geo["total"])
                cv2.putText(
                    display,
                    f"GEO {_lado_txt_geo} {_indice_geo}/"
                    f"{_prog_geo['total']} -- alinhe o PLANO ao quadro "
                    "MAGENTA",
                    (12, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 255, 255), 2)
                if _pose_geo is not None:
                    cv2.putText(display, _pose_geo["nome"][:78],
                                (12, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 0), 2)
                    _qa_geo_txt = ("--" if _prog_geo["qa_deg"] is None
                                   else f"{_prog_geo['qa_deg']:.1f}")
                    _cor_qa_geo = ((0, 255, 0)
                                   if (_prog_geo["qa_deg"] is not None
                                       and _prog_geo["qa_deg"]
                                       <= CALIB_GEO_QA_MAX_DEG)
                                   else (0, 165, 255))
                    cv2.putText(
                        display,
                        f"QA {_qa_geo_txt} graus (max "
                        f"{CALIB_GEO_QA_MAX_DEG:g}) | "
                        f"{_prog_geo['status']} | "
                        f"{_prog_geo['tempo_ok']:.1f}/"
                        f"{CALIB_GEO_TAKE_SEC:.1f} s",
                        (12, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        _cor_qa_geo, 2)
                    cv2.putText(display, _pose_geo["instrucao"][:92],
                                (12, 346), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)
                cv2.putText(display, "N: pular pose | X: cancelar",
                            (12, 374), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1)
            depth_status = "depth: sem frame"
            stats = tracker.depth_stats()
            if stats is not None:
                count, dmin, dmax, frac = stats
                health = "OK" if frac > 0.30 else ("FRACO" if frac > 0.02 else "FALHA")
                depth_status = f"depth {health}: {frac:.0%} valido ({dmin:.0f}-{dmax:.0f} mm)"
            if alignment.stereo_usable:
                if alignment.hand_3d_ready:
                    overlay_text = (
                        f"sobreposicao mano 3D (z-est) | "
                        f"maos aux: {len(auxiliary_hands)}"
                    )
                elif alignment.hand_landmark_ready:
                    overlay_text = (
                        f"sobreposicao mano 2D (RMS {alignment.hand_landmark_rms:.4f} norm) | "
                        f"maos aux: {len(auxiliary_hands)}"
                    )
                else:
                    overlay_text = (
                        f"sobreposicao stereo (RMS {alignment.stereo_rms:.2f}) | "
                        f"{depth_text} | maos aux: {len(auxiliary_hands)}"
                    )
                if overlay_deviation is not None:
                    dev, dx, dy = overlay_deviation
                    overlay_text += (
                        f" | desvio vermelho-verde {dev:.0f} px "
                        f"(dx {dx:+.0f}, dy {dy:+.0f})"
                    )
            elif alignment.hand_3d_ready:
                overlay_text = (
                    f"sobreposicao mano 3D | maos aux: {len(auxiliary_hands)}"
                )
            elif alignment.hand_landmark_ready:
                overlay_text = (
                    f"sobreposicao mano (RMS {alignment.hand_landmark_rms:.4f} norm) | "
                    f"maos aux: {len(auxiliary_hands)}"
                )
            elif alignment.stereo_rms is not None:
                overlay_text = (
                    f"sobreposicao OFF: RMS {alignment.stereo_rms:.2f} px > "
                    f"{STEREO_OVERLAY_MAX_RMS_PX:.0f} (recalibre)"
                )
            else:
                overlay_text = "sobreposicao OFF: sem calibracao stereo valida"
            lines = [
                imu_text,
                camera_text,
                # Lateralidade por camera: diz se o lado de cada mao auxiliar foi
                # MEDIDO pela geometria (com o erro em px) ou se o programa esta'
                # apenas acreditando no rotulo do MediaPipe -- e avisa quando o
                # rotulo de uma camera estava invertido.
                *arbitro.resumo(nomes_aux),
                *(
                    f"Fusao {lado[0].upper()} XYZ {mao['fused'][0]:.3f} "
                    f"{mao['fused'][1]:.3f} {mao['fused'][2]:.3f} m"
                    for lado, mao in resultados_por_lado.items()
                    if mao is not None and mao["fused"] is not None
                ),
                # Erro ao vivo do vetor da palma (Kinect x IMU): e' o numero que
                # diz se o alinhamento daquela luva esta' bom. Sem alinhamento
                # (so' a ancoragem de 1 amostra) ele cresce quando a mao gira.
                *(
                    f"Palma {lado[0].upper()}: erro Kinect x IMU "
                    f"{mao['palm_error_deg']:.1f} graus"
                    for lado, mao in resultados_por_lado.items()
                    if mao is not None and mao.get("palm_error_deg") is not None
                ),
                calibration_text,
                overlay_text,
                depth_status,
            ]
            if display is not None:
                for index, text in enumerate(lines):
                    cv2.putText(display, text, (20, 35 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                # DIAGNOSTICO DA PALMA (tecla D): a fase TAMBEM vai para o
                # video, em texto grande e no rodape (para nao competir com as
                # linhas de status). Por que no video e nao so' no terminal:
                # durante o diagnostico o experimentador esta' de frente para o
                # Kinect, longe do teclado -- e a janela do video e' a que
                # precisa estar em FOCO para o waitKey entregar D/ESC. Desenhado
                # ANTES do `janelas.mostrar`, sobre o display em tamanho cheio:
                # depois do redimensionamento o texto encolheria junto.
                if diag_palma.ativa:
                    fase_video = diag_palma.fase_atual(now)
                    if fase_video is not None:
                        altura_texto = display.shape[0]
                        cv2.putText(
                            display,
                            f"DIAG PALMA fase {fase_video[0] + 1}/"
                            f"{len(diag_palma.fases)} {fase_video[1]} "
                            f"{fase_video[3]:.1f}s",
                            (20, altura_texto - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        # Instrucao em UMA linha curta: o texto completo (longo)
                        # ja' saiu no terminal ao trocar de fase.
                        instrucao_curta = fase_video[4]
                        if len(instrucao_curta) > 72:
                            instrucao_curta = instrucao_curta[:69] + "..."
                        cv2.putText(display, instrucao_curta,
                                    (20, altura_texto - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                janelas.mostrar("Kinect RGB + esqueleto + IMU",
                                janelas.redimensionar(display))
                depth_view_frame = tracker.depth_view()
                if depth_view_frame is not None:
                    janelas.mostrar("Kinect depth",
                                    janelas.redimensionar(depth_view_frame))
            # CSV: UMA linha POR mao (lado identifica a origem). Colunas sem
            # medida ficam VAZIAS (nunca 0.0 disfarcado); a IMU e' gravada
            # mesmo quando a mao nao foi vista pelas cameras.
            for lado in ("right", "left"):
                mao = resultados_por_lado[lado]
                amostra = (
                    mao["sample"] if mao is not None and mao["sample"] is not None
                    else amostras_por_lado.get(lado)
                )
                aux_pair_3d = None
                if mao is not None and mao.get("aux_pair"):
                    pontos = mao.get("triangulated_3d")
                    if pontos is not None:
                        aux_pair_3d = pontos
                writer.writerow([
                    now,
                    lado,
                    amostra.timestamp_us if amostra is not None else "",
                    *(mao["position"] if (mao is not None and mao["position"] is not None) else ["", "", ""]),
                    *(mao["fused"] if (mao is not None and mao["fused"] is not None) else ["", "", ""]),
                    amostra.roll_deg if amostra is not None else "",
                    amostra.pitch_deg if amostra is not None else "",
                    amostra.yaw_deg if amostra is not None else "",
                    "",
                    mao["depth_note"] if mao is not None else "",
                    *(np.asarray(aux_pair_3d)[HAND_TRACK_LANDMARK]
                      if aux_pair_3d is not None else ["", "", ""]),
                    (mao["palm_error_deg"] if (mao is not None
                                               and mao.get("palm_error_deg")
                                               is not None) else ""),
                ])
            csv_file.flush()
            key = cv2.waitKey(1) & 0xFF
            if camera_calibrating and key == ord(" "):
                if color_for_calibration is not None and auxiliary_ok:
                    calibration_kinect = cv2.cvtColor(color_for_calibration, cv2.COLOR_BGRA2BGR)
                    if alignment.add_sample(auxiliary_for_calibration, calibration_kinect):
                        print(f"Amostra de calibracao aceita: {len(alignment.auxiliary_points)}/{CALIBRATION_SAMPLES_REQUIRED}")
                        if len(alignment.auxiliary_points) >= CALIBRATION_SAMPLES_REQUIRED:
                            alignment.solve()
                            camera_calibrating = False
                            cv2.destroyWindow("Calibracao - Kinect")
                            cv2.destroyWindow("Calibracao - camera auxiliar")
                            print(f"Calibracao salva em {CALIBRATION_FILE}")
                continue
            if camera_calibrating:
                continue
            if key == ord("c"):
                for lado in ("right", "left"):
                    mao = resultados_por_lado[lado]
                    if mao is not None and mao["position"] is not None:
                        fusions[lado].origin = mao["position"].copy()
                        fusions[lado].position[:] = 0.0
                        fusions[lado].velocity[:] = 0.0
            elif key == ord("b"):
                for lado in ("right", "left"):
                    fusions[lado].bias_samples.clear()
                bias_calibration_until = time.monotonic() + 2.0
                print("Calibrando bias do acelerometro por 2 segundos. Mantenha a mao parada.")
            elif key == ord("o"):
                # Alinhamento IMU <-> camera da palma, POR LUVA (ver
                # `CapturaAlinhamentoPalma`). E' o que faz o vetor da palma
                # seguir a mao depois que ela gira -- e o que corrige a luva
                # esquerda, cuja montagem tem outro referencial.
                if captura_palma.ativa:
                    captura_palma.parar()
                    print("Captura do alinhamento da palma cancelada (nada foi "
                          "gravado).")
                else:
                    if calib_geo is not None:
                        calib_geo.parar()
                        calib_geo = None
                        print("Calibracao geometrica cancelada pela tecla O.")
                    captura_palma.iniciar(now)
                    print(
                        f"Captura do alinhamento da palma por "
                        f"{PALM_ALIGNMENT_CAPTURE_SECONDS:g} s. Gire a mao "
                        "DEVAGAR (palma virada para o Kinect, sem virar o dorso) "
                        "em varias direcoes; as duas maos podem ser capturadas "
                        "ao mesmo tempo. O alinhamento e' gravado por luva no "
                        "fim; O de novo cancela."
                    )
            elif key == ord("r"):
                for lado in ("right", "left"):
                    fusions[lado].reset()
            elif key == ord("h"):
                hand_calib_active = not hand_calib_active
                if hand_calib_active:
                    print(
                        "Calibracao de esqueleto (man): capturando. Segure a man "
                        "visibile nas DUAS cameras e VARIE posicao/profundidade/"
                        "inclinacao; H para pausar, P para descartar."
                    )
                else:
                    print("Captura de esqueleto pausada.")
            elif key == ord("k"):
                trim_measure_until = now + 3.0
                trim_measure_samples = []
                print(
                    "Medindo trim do overlay (3 s): segure a mao PARADA e "
                    "visivel nas DUAS cameras..."
                )
            elif key == ord("l"):
                overlay_logging = not overlay_logging
                if overlay_logging:
                    overlay_log_file = open(OVERLAY_LOG_PATH, "w", newline="", encoding="utf-8")
                    overlay_log_file.write(",".join(cabecalho_overlay_log()) + "\n")
                    overlay_prev_palm = {"right": None, "left": None}
                    overlay_prev_time = {"right": now, "left": now}
                    print(f"Log overlay LIGADO -> {OVERLAY_LOG_PATH.name} "
                          "(1 linha/mao por frame)")
                else:
                    if overlay_log_file is not None:
                        overlay_log_file.close()
                        overlay_log_file = None
                    print(f"Log overlay DESLIGADO (arquivo em {OVERLAY_LOG_PATH.name})")
            elif key == ord("m"):
                # NIVEL DA MESA (item 115): mede o plano da mesa pelo depth.
                _info_mesa = tracker.medir_plano_mesa()
                if _info_mesa is None:
                    print("[nivel] NAO consegui medir a mesa: aponte o "
                          "Kinect para a mesa SEM maos/objetos na frente "
                          "e tecle M de novo")
                else:
                    mesa_normal_camera = _info_mesa["normal"]
                    mesa_plano_info = _info_mesa
                    _nx, _ny, _nz = (float(v) for v in _info_mesa["normal"])
                    print(f"[nivel] plano da MESA: tilt "
                          f"{_info_mesa['tilt_deg']:.1f} graus | "
                          f"inliers {_info_mesa['fracao'] * 100:.0f}% | "
                          f"erro {_info_mesa['erro_m'] * 1000:.1f} mm | "
                          f"normal ({_nx:+.3f}, {_ny:+.3f}, {_nz:+.3f})")
                    print("  referencia das poses 2/3 = plano da mesa "
                          "medido (compensa inclinacao do sensor)")
                    print(f"  interpretacao: se a mesa for HORIZONTAL, o "
                          f"Kinect esta' inclinado "
                          f"{_info_mesa['tilt_deg']:.1f} graus em relacao "
                          "a ela (a calibracao ja' compensa; alinhar so' "
                          "e' necessario se voce quiser)")
                    if _info_mesa["tilt_deg"] > 2.0:
                        print(f"  ATENCAO: inclinacao "
                              f"{_info_mesa['tilt_deg']:.1f} graus em "
                              "relacao a camera; a calibracao COMPENSA "
                              "com o plano medido, mas se quiser alinhar "
                              "fisicamente ajuste o suporte ate < 1 grau")
                    try:
                        caminho_mesa.write_text(json.dumps({
                            "normal": [float(v) for v in
                                       _info_mesa["normal"]],
                            "tilt_deg": float(_info_mesa["tilt_deg"]),
                            "inliers": int(_info_mesa["inliers"]),
                            "fracao": float(_info_mesa["fracao"]),
                            "sinal": float(mesa_sinal),
                            "poses_verdade": {
                                _lado_json: [[_nome_json, [float(c)
                                              for c in _vec_json]]
                                             for _nome_json, _vec_json in
                                             poses_verdade_por_lado
                                             .get(_lado_json, ())]
                                for _lado_json in ("right", "left")
                                if poses_verdade_por_lado.get(_lado_json)
                            },
                            "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }, indent=2), encoding="utf-8")
                        print(f"  gravado em {caminho_mesa.name}")
                    except OSError as exc:
                        print(f"  nao gravou {caminho_mesa.name}: {exc}")
            elif key == ord("g"):
                # LOG DO VETOR DA PALMA (item 111).
                if log_palma is None:
                    _nome_log = time.strftime(
                        "palma_vetor_log_%Y%m%d_%H%M%S.csv")
                    log_palma = LogVetorPalma(_nome_log)
                    log_palma_ativo = True
                    print(f"[log-palma] gravando em {_nome_log}")
                else:
                    log_palma_ativo = not log_palma_ativo
                    print("[log-palma] " + ("LIGADO" if log_palma_ativo
                                            else "DESLIGADO"))
            elif key == ord("v"):
                # VERIFICACAO CONTRA AS POSES FISICAS (item 114): o alinhamento
                # carregado e' comparado com as normais CONHECIDAS das 3 poses.
                if poses_fisicas.ativa:
                    poses_fisicas.parar()
                    print("Verificacao por poses fisicas cancelada.")
                else:
                    if calib_geo is not None:
                        calib_geo.parar()
                        calib_geo = None
                        print("Calibracao geometrica cancelada pelo V.")
                    captura_palma_plano.parar()
                    verificacao_palma.parar()
                    poses_fisicas = ColetorPosesFisicas()
                    poses_fisicas.iniciar_takes(now, modo="verificar")
                    poses_fisicas_fase_vista = None
                    print(
                        "VERIFICACAO POR TAKES -- "
                        f"{len(poses_fisicas.poses)} posicoes, segure "
                        "PARADO em cada "
                        "uma, com a PALMA MIRANDO A LENTE e a mao PERTO do "
                        "sensor (~40-50 cm), SEM TEMPO FIXO: 2 s PARADO = "
                        "TAKE. Desloque a mao ~30-40 cm: 1) CENTRO; 2) BEM "
                        "A ESQUERDA; 3) BEM A DIREITA; 4) BEM PARA BAIXO. "
                        "A verdade e a DIRECAO mao->lente medida "
                        "pelo Kinect (mesmas verdades do J). No fim sai PASS/FAIL e "
                        "o CERTIFICADO; V de novo cancela.")
            elif key == ord("j"):
                # CALIBRACAO POR TAKES (itens 114/120): 4 posicoes espalhadas
                # com a palma mirando a lente; 2 s PARADO = take (sem tempo
                # fixo). No fim resolve T/f pela direcao mao->lente e grava.
                if poses_fisicas.ativa:
                    poses_fisicas.parar()
                    print("Calibracao por poses fisicas cancelada.")
                else:
                    if calib_geo is not None:
                        calib_geo.parar()
                        calib_geo = None
                        print("Calibracao geometrica cancelada pelo J.")
                    captura_palma_plano.parar()
                    verificacao_palma.parar()
                    # O J resolve T/f para o boot ATUAL com a verdade geometrica
                    # (mao->lente). Uma ancoragem de yaw pendente nao pode
                    # sobrepor esse T depois (item 126): cancela agora.
                    for _lado_ancora, _ancora in ancoras_yaw.items():
                        if not _ancora.pronto:
                            _ancora.cancelar()
                    poses_fisicas = ColetorPosesFisicas()
                    poses_fisicas.iniciar_takes(now, modo="calibrar")
                    print("  modo TAKES: 2 s com a mao PARADA grava o "
                          "take; a tela guia a posicao e o status R/L "
                          "(sem tempo fixo).")
                    poses_fisicas_fase_vista = None
                    print(
                        "CALIBRACAO POR TAKES -- "
                        f"{len(poses_fisicas.poses)} posicoes. A verdade e' FISICA, "
                        "nao vem de planura: e a DIRECAO da sua mao ate a "
                        "LENTE, medida pelo Kinect. SAO 4 POSICOES BEM "
                        "ESPALHADAS: nao basta inclinar a palma, DESLOQUE a "
                        "mao ~30-40 cm para os lados/baixo ate' perto da "
                        "borda do quadro (se uma pose ficar parecida com a "
                        "anterior o TAKE e' RECUSADO na hora). NAO TEM TEMPO "
                        "FIXO: leve o tempo que quiser; sempre com a PALMA "
                        "MIRANDO A LENTE e a mao PERTO do sensor (~40-50 cm). "
                        "Quando a mao ficar PARADA por 2 s o TAKE e' gravado "
                        "automaticamente; depois: 1) CENTRO; 2) BEM A "
                        "ESQUERDA; 3) BEM A DIREITA; 4) BEM PARA BAIXO. "
                        "Mantenha as DUAS maos visiveis. J de novo cancela.")
            elif key == ord("x"):
                # CALIBRACAO GEOMETRICA (item 127): verdade EXATA da camera
                # (alvos X/Y/Z e diagonais), uma mao por vez, com plano-guia
                # no video + QA de execucao; a ultima pose ancora a direcao
                # absoluta pela posicao da mao (mira deslocada).
                if calib_geo is not None:
                    calib_geo.parar()
                    calib_geo = None
                    print("Calibracao geometrica cancelada.")
                else:
                    poses_fisicas.parar()
                    poses_fisicas_fase_vista = None
                    captura_palma_plano.parar()
                    verificacao_palma.parar()
                    if diag_palma.ativa:
                        diag_palma.finalizar()
                    for _lado_ancora, _ancora in ancoras_yaw.items():
                        if not _ancora.pronto:
                            _ancora.cancelar()
                    calib_geo = CalibracaoGeometrica("right")
                    print(
                        "CALIBRACAO GEOMETRICA -- MAO DIREITA PRIMEIRO "
                        "(uma mao por vez).\n"
                        "  - Deixe a OUTRA mao fora do quadro/parada;\n"
                        "  - Cole uma FOLHA RIGIDA (papelao/plastico) na palma "
                        "para alinhar o plano com o quadro MAGENTA;\n"
                        "  - O quadro/setas do video sao o ALVO EXATO no "
                        "referencial da camera; a seta CIANO guia os dedos;\n"
                        "  - Alinhe, fique PARADO 2 s com QA <= "
                        f"{CALIB_GEO_QA_MAX_DEG:g} graus; o take e' gravado "
                        "automaticamente;\n"
                        "  - 8 poses geometricas + 1 MIRA DESLOCADA "
                        f"(~20 cm para a esquerda) por mao;\n"
                        "  - N pula uma pose; X cancela. No fim resolve T/f "
                        "hibrido e grava o JSON piloto (o oficial so' se o "
                        "gate passar).")
            elif key == ord("n") and calib_geo is not None:
                _ev_pular = calib_geo.pular_pose()
                if _ev_pular is not None:
                    print(f"[GEO {calib_geo.lado}] pose pulada: "
                          f"{_ev_pular['pose']}")
            elif key == ord("p"):
                hand_calib_active = False
                hand_calib_samples_aux = []
                hand_calib_samples_kin = []
                hand_calib_samples_depths = []
                hand_calib_count = 0
                print("Captura de esqueleto resetada.")
            elif key == ord("t"):
                diagnostico_tabuleiro(
                    color, color_for_calibration, auxiliary_frame, tracker,
                    alignment, aux_ok=auxiliary_ok, aux_index=AUX_CAMERA_INDEX)
            elif key == ord("d"):
                # DIAGNOSTICO DA PALMA (item 74): liga/desliga a coleta das 4
                # poses programadas. Por que uma tecla e nao um flag de linha de
                # comando: o diagnostico tem de comecar com a mao JA' em quadro
                # e da' para repetir quantas vezes quiser na mesma sessao, sem
                # reiniciar Kinect+IMU (o custo de reiniciar e' alto e a pose
                # muda). D de novo encerra ANTES e gera o resumo do que ja' foi
                # coletado (a coluna `n` de cada fase diz o quanto vale).
                if diag_palma.ativa:
                    diag_palma.finalizar()
                    diag_fase_vista = None
                else:
                    if calib_geo is not None:
                        calib_geo.parar()
                        calib_geo = None
                        print("Calibracao geometrica cancelada pelo D.")
                    diag_palma.iniciar(now, fusions)
                    diag_fase_vista = None
            elif key == 27:
                break
    finally:
        if log_palma is not None:
            log_palma.fechar()
        # DIAGNOSTICO DA PALMA: escolhemos FINALIZAR (nao cancelar) na saida --
        # o experimentador pode ter saido no meio de uma sessao de 21 s, e o que
        # ja' foi coletado custou poses reais. `finalizar` fecha o CSV no
        # `finally` dele e nunca deixa arquivo aberto; se o resumo falhar, o CSV
        # continua valido (flush por linha).
        if diag_palma.ativa:
            diag_palma.finalizar()
        if overlay_log_file is not None:
            overlay_log_file.close()
        for receiver in receivers.values():
            receiver.stop()
        tracker.close()
        auxiliary.release()
        csv_file.close()
        # Fecha SO' as janelas que ESTE programa criou (a GradeDeJanelas sabe
        # quais sao): um destroyAllWindows fecharia tambem janelas de outro
        # programa que dividisse a sessao do OpenCV.
        janelas.destruir()


if __name__ == "__main__":
    main()
