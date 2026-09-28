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
por luva) | T diagnostico do tabuleiro | K trim fixo do overlay | H calibracao do
esqueleto da mao | L log de desvio | R reset | ESC sair. Precisa do Kinect v2
(SDK 2.0) e, opcionalmente, da webcam auxiliar e do ESP32/UDP.

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

#: Landmark do MediaPipe Hands usado como POSICAO DA MAO (9 = centro da palma,
#: conforme a metodologia; 0 = pulso). Mesmo valor do paradigma
#: (`eeg_motor_paradigm.HAND_TRACK_LANDMARK`), mas definido aqui: a biblioteca
#: nao usa `HAND_TRACK_LANDMARK` (ela projeta os 21 nos), quem escolhe o no'
#: rastreado sao os programas -- e a coluna `aux_pair_*_m` do CSV grava ESTE no'.
HAND_TRACK_LANDMARK = 9

# --- alinhamento IMU <-> camera da palma (por luva) --------------------------
#: Arquivo do alinhamento resolvido de cada luva. Fica na raiz (junto das
#: calibracoes) e e' POR LADO: cada luva tem a sua montagem, e o alinhamento de
#: uma nao vale para a outra (era o bug do vetor da palma da mao esquerda).
PALM_ALIGNMENT_FILES = {"right": "imu_palm_alignment_right.json",
                        "left": "imu_palm_alignment_left.json"}
#: Duracao da captura do alinhamento (tecla O): tempo para girar a mao devagar
#: por varios angulos com a palma virada para a camera.
PALM_ALIGNMENT_CAPTURE_SECONDS = 14.0
#: Intervalo minimo entre amostras da captura (evita 30 amostras iguais).
PALM_ALIGNMENT_SAMPLE_INTERVAL = 0.10
#: Variacao minima (graus) entre a amostra e a ANTERIOR do mesmo lado: sem isso
#: a captura aceita 100 amostras paradas (postura constante nao informa nada).
PALM_ALIGNMENT_MIN_STEP_DEG = 4.0


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
    """Grava o alinhamento atual da luva `lado` (False se nao ha' o que gravar)."""
    estado = fusion.palm_alignment_state()
    if estado is None:
        return False
    caminho = palm_alignment_path(lado)
    try:
        caminho.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    except OSError as exc:
        avisar(f"IMU {lado}: nao consegui gravar {caminho.name} ({exc})")
        return False
    return True


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

    def resolver(self, lado):
        """(T, f, residuo) do lado, ou None se faltam amostras validas."""
        amostras = self.amostras.get(lado, ())
        if len(amostras) < PALM_ALIGNMENT_MIN_SAMPLES:
            return None
        return solve_palm_alignment([r for r, _v in amostras],
                                    [v for _r, v in amostras])


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
        "--aux-index", type=int, default=AUX_CAMERA_INDEX, metavar="N",
        help=("indice OpenCV da webcam auxiliar ATIVA (a das teclas K/L/H: trim, "
              "log de desvio e calibracao do esqueleto). ATENCAO: o indice NAO "
              "e' estavel -- e' a posicao na enumeracao do DirectShow e muda "
              "quando uma camera entra/sai do USB (medido em 23/09/2026: com as "
              "externas ligadas 2 = bancada / 0 = ampla / 1 = webcam do LAPTOP; "
              "sem elas, 0 = webcam do LAPTOP). A calibracao usada e' "
              "stereo_calibration_aux{N}.npz. Confira quem e' quem PELO NOME "
              "com tools\\diagnostico_cameras.py --salvar."),
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
        "--salvar-quadros", type=float, default=0.0, metavar="S",
        help=("grava _aux_overlay.png e _kinect_overlay.png a cada S segundos "
              "(0 = desligado). Serve para conferir DEPOIS se a projecao "
              "vermelha caiu na mao, sem depender de olhar a janela."),
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
    return parser.parse_args()


def abre_camera_auxiliar(indice, tamanho):
    """Abre a webcam auxiliar no MESMO backend que a calibracao usou.

    O DSHOW vem primeiro: `stereo_calibration.py` calibra por ele (AUX_BACKENDS)
    e ele abre em ~1 s, contra os 20-40 s do MSMF. Se o driver nao entregar
    frame, cai para o MSMF -- nesta maquina os dois backends enumeram os
    MESMOS indices, entao trocar de backend nao troca de camera.
    """
    for backend, nome in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
        captura = cv2.VideoCapture(indice, backend)
        if not captura.isOpened():
            captura.release()
            print(f"Camera auxiliar {indice}: backend {nome} nao abriu.")
            continue
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, tamanho[0])
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, tamanho[1])
        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(4):
            captura.read()
        ok, quadro = captura.read()
        if ok and quadro is not None:
            print(
                f"Camera auxiliar {indice}: backend {nome} OK "
                f"({quadro.shape[1]}x{quadro.shape[0]})."
            )
            return captura
        captura.release()
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
    ordem = [int(args.aux_index)]
    for indice in outros:
        if int(indice) not in ordem:
            ordem.append(int(indice))
    return ordem

def main():
    args = parse_cli()
    # O indice escolhido vale para a camera aberta E para o arquivo de
    # calibracao (stereo_calibration_aux{N}.npz) -- os dois andam juntos.
    global AUX_CAMERA_INDEX
    AUX_CAMERA_INDEX = int(args.aux_index)
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
    receivers = {
        "right": ImuReceiver(portas.get(1, 4210)),
        "left": ImuReceiver(portas.get(2, 4211)),
    }
    print(f"IMU: direita na porta {receivers['right'].port} | esquerda na porta "
          f"{receivers['left'].port}")
    for receiver in receivers.values():
        receiver.start()
    # --- webcams auxiliares: a ATIVA (--aux-index) + as OUTRAS (--aux-cameras) ---
    # A resolucao vem ANTES do tracker/alignment porque os dois precisam da
    # lista: o tracker cria UM detector MediaPipe por camera (o VideoMode nao
    # admite intercalar fluxos) e o alignment carrega a calibracao de cada uma
    # (as vistas por indice sao o que permite desenhar as DUAS no mesmo frame).
    aux_indices = _indices_auxiliares(args)
    tracker = KinectHandTracker(aux_indices=tuple(aux_indices))
    # Carrega SO a calibracao do indice escolhido: os npz sao POR INDICE
    # (stereo_calibration_aux{N}.npz) e usar o arquivo de outra camera projeta o
    # esqueleto do Kinect em qualquer lugar da imagem.
    alignment = CameraAlignment(aux_indices=aux_indices)
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
    # --- webcams auxiliares: a ATIVA (--aux-index) + as OUTRAS (--aux-cameras) --
    # Todas sao abertas e DESENHADAS no video do Kinect: o objetivo e' ver, no
    # MESMO quadro, o esqueleto do Kinect (verde) e o de CADA auxiliar (na cor
    # de cada uma) -- e', ao mesmo tempo, o teste visual da calibracao de cada
    # camera e a base do par que substitui o Kinect quando ele perde a mao.
    requested_size = alignment.calibrated_auxiliary_size or (1280, 720)
    auxiliares = {}
    for indice in aux_indices:
        try:
            captura = abre_camera_auxiliar(indice, requested_size)
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
        raise RuntimeError(
            f"A webcam auxiliar ATIVA (indice {AUX_CAMERA_INDEX}) nao abriu. "
            "Rode tools\\diagnostico_cameras.py --salvar para ver o NOME de cada "
            "indice e escolha uma camera EXTERNA com --aux-index.")
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
    captura_palma = CapturaAlinhamentoPalma()

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
    overlay_prev_palm = None
    overlay_prev_time = 0.0
    # --- BIMANUAL: EMA de profundidade POR lado ------------------------------
    # O alignment guarda UMA EMA (legado unimanual); o tool mantem a versao
    # por lado e publica no alignment a EMA do lado ANTES de cada desenho.
    ema_por_lado = {"right": None, "left": None}
    ema_trust_por_lado = {"right": 0, "left": 0}
    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["pc_time_s", "hand_side", "imu_timestamp_us", "kinect_x_m", "kinect_y_m", "kinect_z_m", "fused_x_m", "fused_y_m", "fused_z_m", "roll_deg", "pitch_deg", "yaw_deg", "elbow_angle_deg", "hand_source", "aux_pair_x_m", "aux_pair_y_m", "aux_pair_z_m", "palm_err_deg"])


    try:
        while True:
            now = time.monotonic()
            color, depth = tracker.frames()
            # --- TODAS as auxiliares: le, casa a resolucao e detecta POR CAMERA --
            # Cada camera tem o seu detector/timestamp (VideoMode do MediaPipe) e
            # o seu tamanho de frame; a camera ATIVA e' restaurada no fim do laco
            # porque os campos vivos do alignment (e as teclas K/L/H) sao dela.
            aux_frames, aux_por_camera = {}, {}
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
            alignment.set_active_aux(int(AUX_CAMERA_INDEX))
            auxiliary_frame = aux_frames.get(int(AUX_CAMERA_INDEX))
            auxiliary_ok = auxiliary_frame is not None
            aux_entries = aux_por_camera.get(int(AUX_CAMERA_INDEX), [])
            auxiliary_hands = [entrada["landmarks"] for entrada in aux_entries]
            # Por lado e POR CAMERA: e' o que permite (a) casar cada mao com o
            # seu IMU e (b) triangular o par de auxiliares quando o Kinect cego.
            aux_lado_por_camera = {
                indice: {entrada["side"]: entrada["landmarks"]
                         for entrada in entradas}
                for indice, entradas in aux_por_camera.items()
            }
            auxiliary_for_calibration = (None if not auxiliary_ok
                                         else auxiliary_frame.copy())
            color_for_calibration = None if color is None else color.copy()
            display, kinect_hands = tracker.detect_hands_all(color, depth)
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
                        mao["triangulated_3d"] = tri3d
                # --- Prioridade 2: o PAR DE AUXILIARES (Kinect cego) -----------
                # Quando o Kinect perde a mao (ou nao triangula com ele), as DUAS
                # auxiliares, sozinhas, medem a mao nos 3 eixos pela calibracao de
                # cada uma com o Kinect -- e' o objetivo do projeto: o par
                # SUBSTITUI o Kinect, nao so' o apoia.
                if mao["triangulated_3d"] is None:
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
                mao["depth_note"] = depth_note
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
                for indice, entradas in aux_por_camera.items():
                    if int(indice) == int(AUX_CAMERA_INDEX):
                        continue
                    maos = [e["landmarks"] for e in entradas if e.get("landmarks")]
                    if maos:
                        outras[int(indice)] = maos
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
                    z_primario = None
                    tri_primario = None
                    if prim_mao is not None:
                        tri_primario = prim_mao["triangulated_3d"]
                        if np.isfinite(prim_mao.get("z_used", np.nan)):
                            z_primario = [float(prim_mao["z_used"])]
                    tracker.draw_auxiliary_cameras_on_color(
                        display, outras, alignment,
                        depths={i: z_primario for i in outras},
                        triangulados={i: tri_primario for i in outras},
                        cores=cores,
                    )
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
            if prim_mao is not None:
                mark = "" if prim_mao["depth_note"] in ("Kinect", "esqueletoSDK") else "!"
                depth_text = (
                    f"Z {prim_mao['z_m']:.2f} m ({prim_mao['depth_note']}{mark})"
                )
                # --- Log de diagnostico do overlay (tecla L liga/desliga) --------
                # Grava 1 linha/frame em overlay_deviation_log.csv com o desvio
                # vermelho-verde por no + contexto (profundidade, posicao na
                # imagem, velocidade da mao). Serve para classificar o erro:
                # estatico (bias) x movimento (sincronia) x bordas (distorcao).
                # Frames SEM mao verde sao gravados com as colunas de desvio
                # vazias (so a trajetoria vermelha + contexto).
                # (BIMANUAL: loga a mao PRIMARIA; coluna 'lado' identifica.)
                if overlay_logging and prim_mao["overlay_red"] is not None:
                    try:
                        red = np.asarray(
                            prim_mao["overlay_red"], np.float64
                        ).reshape(-1, 2)
                        grn = (
                            np.asarray(
                                prim_mao["overlay_green"], np.float64
                            ).reshape(-1, 2)
                            if prim_mao["overlay_green"] is not None
                            else None
                        )
                        n_nodes = 21 if grn is None else min(21, red.shape[0], grn.shape[0])
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
                        z_used = float(prim_mao["z_used"])
                        z_raw = float(prim_mao["z_m"])
                        palm_now = np.array([cx, cy], np.float64)
                        speed = float("nan")
                        if np.isfinite(palm_now).all() and overlay_prev_palm is not None:
                            dt = now - overlay_prev_time if overlay_prev_time else float("nan")
                            if dt and dt > 1e-3:
                                speed = float(np.linalg.norm(palm_now - overlay_prev_palm) / dt)
                        overlay_prev_palm = palm_now if np.isfinite(palm_now).all() else None
                        overlay_prev_time = now
                        row = [f"{now:.3f}", lado_primario or "",
                               f"{z_used:.3f}", f"{z_raw:.3f}",
                               f"{cx:.1f}", f"{cy:.1f}", f"{edge:.3f}",
                               f"{speed:.1f}" if np.isfinite(speed) else "",
                               prim_mao["depth_note"]]
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
                        # VETOR DA PALMA do lado primario: normal do Kinect,
                        # direcao prevista pelo IMU da luva e o erro entre as
                        # duas. E' o numero que diz se o alinhamento daquela luva
                        # esta' bom (sem alinhamento ele cresce ao girar a mao).
                        normal_palma = prim_mao.get("palm_normal_kinect")
                        direcao_palma = prim_mao.get("palm_direction")
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
                        erro_palma = prim_mao.get("palm_error_deg")
                        row.append("" if erro_palma is None
                                   else f"{erro_palma:.2f}")
                        overlay_log_file.write(",".join(row) + "\n")
                    except (ValueError, IndexError, TypeError):
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
                if mao is None:
                    continue
                mao["sample"] = sample_lado
                fusion = fusions[lado]
                camera_position_lado = mao["position"]
                landmarks_lado = mao["landmarks"]
                # Normal da palma medida pelo Kinect (depth): e' a referencia do
                # vetor da palma -- usada para ancorar, para a captura do
                # alinhamento (tecla O) e para o ERRO AO VIVO (Kinect x IMU).
                normal_lado = (
                    None if landmarks_lado is None
                    else tracker.palm_normal_camera(landmarks_lado)
                )
                mao["palm_normal_kinect"] = normal_lado
                mao["palm_direction"] = (
                    fusion.palm_direction(sample_lado)
                    if landmarks_lado is not None else None
                )
                mao["palm_error_deg"] = (
                    None
                    if (mao["palm_direction"] is None or normal_lado is None)
                    else angle_between(normal_lado, mao["palm_direction"])
                )
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
                                 else "calibrada (1 amostra)."))
                elif now < bias_calibration_until:
                    fusion.add_bias_sample(sample_lado)
                if imu_calibration_active[lado]:
                    mao["fused"] = fusion.position.copy()
                else:
                    mao["fused"] = fusion.update(sample_lado, camera_position_lado)
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
                    if residuo > PALM_ALIGNMENT_MAX_RESIDUAL_DEG:
                        print(f"    {lado}: residuo acima de "
                              f"{PALM_ALIGNMENT_MAX_RESIDUAL_DEG:g} graus -- "
                              "repita girando a mao SEM virar a palma para o "
                              "dorso e mantenha a mao dentro do quadro; se "
                              "persistir, confira se as luvas nao estao "
                              "trocadas (--imu-portas)")
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
            if display is not None:
                for lado, mao in resultados_por_lado.items():
                    if mao is None:
                        continue
                    if mao["palm_direction"] is not None and mao["landmarks"] is not None:
                        tracker.draw_palm_vector(
                            display, mao["landmarks"], mao["palm_direction"]
                        )
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
                    header = ["t_s", "lado", "z_usado_m", "z_bruto_m", "cx_px",
                              "cy_px", "borda_0c_1b", "vel_px_s", "note",
                              "palm_kin_x", "palm_kin_y", "palm_kin_z",
                              "palm_imu_x", "palm_imu_y", "palm_imu_z",
                              "palm_err_deg"]
                    for n in range(21):
                        header += [f"n{n}_dev_px", f"n{n}_dx", f"n{n}_dy"]
                    overlay_log_file.write(",".join(header) + "\n")
                    overlay_prev_palm = None
                    overlay_prev_time = now
                    print(f"Log overlay LIGADO -> {OVERLAY_LOG_PATH.name} (1 linha/frame)")
                else:
                    if overlay_log_file is not None:
                        overlay_log_file.close()
                        overlay_log_file = None
                    print(f"Log overlay DESLIGADO (arquivo em {OVERLAY_LOG_PATH.name})")
            elif key == ord("p"):
                hand_calib_active = False
                hand_calib_samples_aux = []
                hand_calib_samples_kin = []
                hand_calib_samples_depths = []
                hand_calib_count = 0
                print("Captura de esqueleto resetada.")
            elif key == ord("t"):
                # Saude do fluxo de profundidade (o ground truth depende dele)
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
                # Diagnostico: erro ponta-a-ponta da sobreposicao no tabuleiro
                # (superficie rigida com correspondencia exata; separa erro de
                # projecao/calibracao de efeitos especificos da mao).
                if color is None or not auxiliary_ok:
                    print("Diagnostico: sem frame simultaneo das duas cameras")
                elif not alignment.stereo_usable:
                    print("Diagnostico: calibracao stereo nao carregada")
                else:
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
                        print("Diagnostico: tabuleiro nao visto nas duas cameras")
                    else:
                        print("Diagnostico: processando tabuleiro...")
                        depths = np.full(len(board_kinect), np.nan, np.float32)
                        for index, (corner_x, corner_y) in enumerate(
                            board_kinect.reshape(-1, 2)
                        ):
                            depth_pixel = tracker._color_to_depth_pixel(
                                int(corner_x), int(corner_y)
                            )
                            if depth_pixel is None:
                                continue
                            depth_mm = tracker._median_depth(
                                *depth_pixel, radius=DEPTH_PATCH_RADIUS * 3
                            )
                            if depth_mm is not None:
                                depths[index] = depth_mm / 1000.0
                        valid = int(np.isfinite(depths).sum())
                        total = len(depths)
                        print(
                            f"Diagnostico: {valid}/{total} cantos com profundidade "
                            "valida do Kinect"
                        )
                        if depths is not None:
                            valid = np.isfinite(depths)
                            if valid.sum() < 5:
                                print(
                                    "Diagnostico: tabuleiro fora do alcance do "
                                    "depth do Kinect; afaste o tabuleiro para "
                                    ">= 0.8 m e mantenha-o dentro do FOV do "
                                    "depth (evite as bordas extremas do video)"
                                )
                            else:
                                depths = np.where(
                                    valid, depths, float(np.nanmedian(depths))
                                )
                                projected = np.asarray(
                                    alignment.auxiliary_to_kinect_stereo(
                                        board_aux, depths
                                    ),
                                    np.float64,
                                ).reshape(-1, 2)
                                differences = projected - board_kinect.reshape(-1, 2)
                                errors = np.linalg.norm(differences, axis=1)
                                finite_err = errors[np.isfinite(errors)]
                                dx = float(np.nanmean(differences[:, 0]))
                                dy = float(np.nanmean(differences[:, 1]))
                                centro = np.mean(board_kinect.reshape(-1, 2), axis=0)
                                radius = np.linalg.norm(
                                    board_kinect.reshape(-1, 2) - centro, axis=1
                                )
                                slope = None
                                if finite_err.size >= 2:
                                    r_median = max(1e-9, float(np.median(radius)))
                                    slope = float(
                                        np.polyfit(
                                            radius / r_median,
                                            finite_err,
                                            1,
                                        )[0]
                                    )
                                print(
                                    f"Diagnostico tabuleiro: {np.nanmean(errors):.1f} px media | "
                                    f"mediana {float(np.nanmedian(errors)):.0f} | max {np.nanmax(errors):.0f} | "
                                    f"centro {errors[24]:.0f} | bordas "
                                    f"{np.nanmean(np.concatenate((errors[:8], errors[-8:]))):.0f} | "
                                    f"dx {dx:.0f} dy {dy:.0f} | slope {slope if slope is not None else '--'}"
                                )
            elif key == 27:
                break
    finally:
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
