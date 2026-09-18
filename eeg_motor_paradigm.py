"""Paradigma EEG Execucao + Imaginacao Motora (ME/MI) - g.Nautilus + ground truth.

Programa 1 (gravacao offline). Um UNICO processo integra:

  1. Pipeline g.Pype: GNautilus 32 canais @ 500 Hz (ou Generator sintetico)
     -> Bandpass 1-30 Hz -> Notch 50/60 Hz -> RecordingMuxer -> CSV.
  2. Paradigma Qt (thread principal), protocolo final da dissertacao:
     [PREPARACAO ~10 min: EEG, calibracao do Kinect e teste de sinal; ESPACO
      inicia] -> [BASELINE 5 min: 2 min olhos abertos, 2 min olhos fechados,
      1 min repouso ativo] -> calibracao de origem (10 s) -> contagem ->
      [5 BLOCOS de 50 trials com 2 min de pausa entre eles] -> encerramento.
     Cada trial tem 12 s:
       REPOUSO 2 s (mao no home + ponto de fixacao)
       CUE + ME 4 s (imagem do objeto - garrafa/bola/caneta - + seta do lado
         da mao; o participante EXECUTA alcance/preensao)
       PAUSA + VIDEO 2 s (video do proprio movimento em camera lenta 0,5x,
         com a trajetoria 3D desenhada sobre o quadro - priming visual)
       MI 4 s (IMAGINA o mesmo movimento; o MESMO cue permanece na tela)
     Condicoes: 3 objetos x 2 maos = 6, balanceadas ao longo da sessao.
  3. Thread de tracking: reutiliza KinectHandTracker/CameraAlignment de
     kinect_imu_groundtruth (triangulacao stereo MediaPipe->3D + cadeia de
     fallback) e o ImuReceiver de imu.py (ESP32, UDP 4210),
     publicando no estado thread-safe, a cada frame: a posicao 3D da mao
     (landmark 9 = centro da palma, metros no frame do Kinect), roll/pitch/yaw
     do ESP32, a geometria do BRACO (ombro/cotovelo/punho + angulos, com
     cinematica inversa de elos rigidos), a ORIENTACAO DA PALMA e o CLIPE do
     movimento em camera lenta para a fase de priming.

O CSV final tem: Time, EEG_Ch01..ChNN, KT_x_m, KT_y_m, KT_z_m,
IMU_L_* (11 colunas) + IMU_R_* (11 colunas) + IMU_hand, KTT_valid,
ARM_* (18 colunas), PALM_* (11 colunas), Marker.

OBS: EEG_ChNN sai em MICROVOLTS (uV). O no CountsToMicrovolts converte os
counts do ADC de 24 bits com a sensibilidade do aparelho (LSB =
sensibilidade/2**24 nV) e com a calibracao Factor/Offset de get_scaling();
use --sem-escala-uv se quiser os counts crus de volta.

KT_*  = mao 3D (landmark 9, m, frame Kinect; RELATIVO a origem),
IMU_* = DOIS MPU6050, um em cada mao (ver docs/IMU_DUPLO_ESP32_DESENHO.md):
        IMU_L_* = esquerda e IMU_R_* = direita, cada bloco com
        rpy (angulos do ESP32, RELATIVO a origem apos a calibracao) +
        acc_*_g (aceleracao BRUTA em g, o que o filtro de Kalman consome) +
        pos_*_m (posicao FUSIONADA acelerometro+cameras, m, relativa a origem) +
        valid + ZERO_lock_L/R (=1 quando a zeragem daquele lado esta ativa:
        camera valida + mao parada, bias corrigido e posicao/velocidade
        ancoradas na camera). Cada sensor tem o SEU bias e a SUA zeragem.
        IMU_hand = qual mao o trial estava executando (1 direita, 2 esquerda,
        0 desconhecido), a mesma convencao de KT_hand/ARM_side. Com UM so' IMU
        (sessao antiga ou bancada), o bloco preenchido e' o do lado ativo -- ou
        o lado declarado em `--imu-lado` quando a mao e' desconhecida -- e o
        outro lado sai NaN com valid=0,
KTT_valid = 1 se a medida 3D e do frame atual (0 = fallback/sem medida),
ARM_* = ombro/cotovelo/punho do esqueleto do SDK, corrigidos pela cinematica
        inversa de elos rigidos (ombro->cotovelo e cotovelo->punho constantes,
        calibrados na origem e guardados em arm_model.json), com angulo do
        cotovelo, elevacao/azimute do braco, elos efetivos e diagnostico da IK
        (ARM_ik: 0 = esqueleto cru sem modelo, 1 = IK com dobramento medido,
        2 = IK com dobramento de referencia, 3 = IK sem referencia;
        ARM_clamped: 1 = punho medido fora do alcance dos elos),
PALM_* = orientacao da palma pelo Kinect (normal do plano + direcao dos dedos,
        vetores unitarios, e angulos elev/azim/pitch/yaw). E independente do
        IMU: serve de referencia absoluta para a orientacao da mao (o IMU_*
        continua sendo a orientacao medida pelo ESP32 preso a mao).

Codigos de marcador (convensao BCI Competition IV 2a estendida):

  32766 = inicio da sessao           768 = inicio do trial (repouso/home)
  32765 = fim da sessao              769 = cue MAO ESQUERDA
  276   = baseline olhos abertos     770 = cue MAO DIREITA
  277   = baseline olhos fechados    811..816 = condicoes (objeto x mao):
  784   = baseline repouso ativo        811 garrafa+direita, 812 garrafa+esq,
  500/501 = inicio/fim da origem         813 bola+direita, 814 bola+esquerda,
  778/779 = inicio/fim da ME             815 caneta+direita, 816 caneta+esq
  780/781 = inicio/fim da MI
  790/791 = inicio/fim do bloco      792/793 = inicio/fim da pausa
  794   = inicio do video em 0,5x

Os codigos de mao (769/770), ME (778) e condicao (811-816) sao emitidos com
offsets de 50/100 ms no inicio do cue para nao colidirem na coluna Marker
(que guarda um codigo por amostra). Per-trial metadata (objeto, mao, bloco)
fica no JSON de eventos.

Uso tipico (g.Nautilus real, sempre com o .venv do projeto e no terminal do
VS Code - o g.Pype exige IDE suportada):
    .venv\\Scripts\\python.exe eeg_motor_paradigm.py --source gnautilus
    .venv\\Scripts\\python.exe eeg_motor_paradigm.py --impedancia  (checar antes)

Teste rapido sem hardware:
    python eeg_motor_paradigm.py --source generator --blocos 1 --trials-por-bloco 2
    python eeg_motor_paradigm.py --source generator --sem-kinect
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import threading
import time
from collections import deque

import numpy as np
import gpype as gp
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter,
                           QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget)
from gpype.backend.core.io_node import IONode

from gpype.common.constants import Constants
from imu import ImuReceiver

try:
    import winsound

    def _beep(freq=880, ms=250):
        winsound.Beep(int(freq), int(ms))
except ImportError:  # fora do Windows
    def _beep(freq=880, ms=250):
        print(f"[beep] {freq} Hz por {ms} ms")

PORT_IN = gp.Constants.Defaults.PORT_IN
PORT_OUT = gp.Constants.Defaults.PORT_OUT

# =============================================================================
# Protocolo (constantes do experimento)
# =============================================================================
FS = 500.0                 # taxa de amostragem do g.Nautilus (Hz)
CHANNEL_COUNT = 32         # canais de EEG
FRAME_SIZE = 1             # amostras por frame (muxer exige 1)
HOLD_SEC = 0.5             # duracao do codigo na coluna Marker (s)
# Taxa de amostragem do CSV de movimento (arquivo separado). O EEG fica em
# 500 Hz e as cameras entregam ~30 Hz: repetir as colunas de movimento em cada
# amostra de EEG era o maior consumidor de disco do projeto sem ganho de
# informacao (a banda do movimento e' ~15 Hz, logo 30 Hz basta).
MOTION_HZ = 30.0
# False = movimento em arquivo proprio (padrao, recomendado). True = movimento
# embutido no CSV do EEG (formato antigo, 74 colunas).
MOTION_IN_EEG_CSV = False
#: Versao do protocolo/arquivo. Gravada no JSON da sessao: se o protocolo
#: mudar, as sessoes antigas continuam identificaveis.
PROTOCOL_VERSION = "1.0"
TRIALS_PER_CLASS = 16      # (legado) trials por classe do protocolo antigo

# --- Protocolo final (metodologia da dissertacao) ---------------------------
# Cada trial tem 12 s:
#   [REPOUSO 2 s: mao na posicao inicial (home) + ponto de fixacao]
#   [CUE + ME 4 s: imagem do objeto (garrafa/bola/caneta) + lado da mao;
#    o participante EXECUTA alcance/preensao -> Kinect grava a trajetoria]
#   [PAUSA + VIDEO 2 s: video do proprio movimento em camera lenta (0,5x)]
#   [MI 4 s: IMAGINA o mesmo movimento; apenas EEG e' registrado]
HOME_SEC = 2.0
CUE_ME_SEC = 4.0
PAUSE_VIDEO_SEC = 2.0
MI_SEC = 4.0
TRIAL_SEC = HOME_SEC + CUE_ME_SEC + PAUSE_VIDEO_SEC + MI_SEC      # 12 s

BLOCKS = 5                        # blocos por sessao
TRIALS_PER_BLOCK = 50             # trials por bloco (10 min por bloco)
BLOCK_PAUSE_SEC = 120.0           # pausa entre blocos (SPACE pula)
BASELINE_EYES_OPEN_SEC = 120.0    # baseline: 2 min olhos abertos
BASELINE_EYES_CLOSED_SEC = 120.0  # baseline: 2 min olhos fechados
BASELINE_REST_SEC = 60.0          # baseline: 1 min repouso ativo

SLOWMO_SPEED = 0.5                # camera lenta do video de priming
SLOWMO_CAPTURE_FPS = 30.0         # taxa de captura dos quadros do video
SLOWMO_BUFFER_SEC = 4.0           # buffer circular de quadros (s) - PRECISA
                                  # cobrir: (ate' o fim da ME) + SLOWMO_SPAN_SEC
                                  # quando o clipe e' ancorado no ONSET. Com o
                                  # onset ~0,3-1,5 s apos a cue e a pausa
                                  # comecando a 4 s, [onset, onset+1 s] cabe em
                                  # [pedido-4 s, pedido]. RAM: 4 s x 30 fps x
                                  # 640x360x3 B ~ 83 MB.
SLOWMO_SPAN_SEC = 1.0             # trecho do movimento mostrado (s reais)
SLOWMO_END_LAG_SEC = 1.0          # ignora o fim da ME (mao ja parada)
#: Deteccao do INICIO DO MOVIMENTO (onset) pela velocidade do punho 3D.
#: Motivo cientifico: o planejamento motor comeca 500-1000 ms ANTES do
#: movimento visivel (potencial de prontidao) e, ~500-350 ms antes, o cortex
#: pre-motor / area motora suplementar organizam a sequencia do movimento.
#: Por isso a janela de interesse do EEG deve comecar ~0,5 s ANTES do onset
#: detectado, e nao num instante fixo pos-cue:
#:     python sand_traj_treino.py --data ... --event-code 795 \
#:            --window-start-sec -0.5 --window-sec 2.0
MOVE_ONSET_SPEED_MPS = 0.030      # limiar de velocidade do punho (m/s)
MOVE_ONSET_CONSEC = 3             # amostras acima do limiar (~100 ms a 30 fps)
MOVE_ONSET_MIN_TRAVEL_M = 0.005   # deslocamento minimo desde a cue (anti-ruido)
#: Landmark do MediaPipe Hands usado como posicao da mao (9 = centro da palma,
#: conforme a metodologia de aquisicao; 0 = pulso).
HAND_TRACK_LANDMARK = 9
OBJECTS = ("garrafa", "bola", "caneta")
HANDS = ("esquerda", "direita")

REST_EYES_OPEN = BASELINE_EYES_OPEN_SEC
REST_EYES_CLOSED = BASELINE_EYES_CLOSED_SEC
COUNTDOWN = 5              # contagem regressiva antes do primeiro trial (s)
ORIGIN_CAL_SEC = 10.0      # duracao da calibracao de origem (s)
BEEP_LEAD_SEC = 0.0        # beep antes da cue (protocolo final: sem beep)
ITI_MIN_SEC = 0.0          # sem ITI: o repouso do proximo trial e' a pausa
ITI_MAX_SEC = 0.0
#: Vigia do link EEG (#15): sem amostras novas por este tempo = link perdido.
LINK_TIMEOUT_SEC = 1.5
#: Painel de impedancias da preparacao (#23/#24): limite de contato bom (kOhm)
#: e intervalo entre medicoes (s).
IMPEDANCE_LIMIT_KOHM = 50.0
IMPEDANCE_PERIOD_S = 1.5

#: Codigos de marcador (coluna Marker do CSV).
CODE_SESSION = 32766
CODE_SESSION_END = 32765
CODE_EYES_OPEN = 276
CODE_EYES_CLOSED = 277
CODE_BASELINE_REST = 784
CODE_TRIAL = 768
CODE_ORIGIN_START = 500
CODE_ORIGIN_END = 501
CODE_ME_START = 778
CODE_ME_END = 779
CODE_MI_START = 780
CODE_MI_END = 781
CODE_BLOCK_START = 790
CODE_BLOCK_END = 791
CODE_PAUSE_START = 792
CODE_PAUSE_END = 793
CODE_SLOWMO_START = 794
#: INICIO DO MOVIMENTO (onset) detectado pelo Kinect durante a ME. Usado para
#: ancorar a janela de EEG 0,5 s ANTES do movimento (planejamento motor) e o
#: clipe do priming no inicio real do movimento.
CODE_MOVE_ONSET = 795
#: Vigia do link (#15): queda/retorno da entrega de amostras do amplificador.
CODE_LINK_LOST = 899
CODE_LINK_OK = 898
#: Codigo da MAO (compativel com o dataset anterior: 769 esq / 770 dir)
CODE_HAND = {"esquerda": 769, "direita": 770}
#: Codigo da CONDICAO (objeto x mao), emitido ~100 ms depois do codigo da mao
#: para nao colidir na coluna Marker (um codigo por amostra).
CODE_CONDITION = {
    ("garrafa", "direita"): 811,
    ("garrafa", "esquerda"): 812,
    ("bola", "direita"): 813,
    ("bola", "esquerda"): 814,
    ("caneta", "direita"): 815,
    ("caneta", "esquerda"): 816,
}
CODE_NAMES = {
    CODE_SESSION: "inicio da sessao",
    CODE_SESSION_END: "fim da sessao",
    CODE_EYES_OPEN: "baseline olhos abertos",
    CODE_EYES_CLOSED: "baseline olhos fechados",
    CODE_LINK_LOST: "LINK EEG PERDIDO (sem amostras novas)",
    CODE_LINK_OK: "link EEG recuperado",
    CODE_MOVE_ONSET: "INICIO DO MOVIMENTO (onset detectado pelo Kinect)",
    CODE_BASELINE_REST: "baseline repouso ativo",
    CODE_TRIAL: "inicio do trial (repouso/home)",
    CODE_ORIGIN_START: "inicio calibracao de origem",
    CODE_ORIGIN_END: "fim calibracao de origem",
    CODE_ME_START: "inicio fase ME (execucao)",
    CODE_ME_END: "fim ME (pausa + video)",
    CODE_MI_START: "inicio fase MI (imaginacao)",
    CODE_MI_END: "fim MI (fim do trial)",
    CODE_BLOCK_START: "inicio do bloco",
    CODE_BLOCK_END: "fim do bloco",
    CODE_PAUSE_START: "inicio pausa entre blocos",
    CODE_PAUSE_END: "fim pausa entre blocos",
    CODE_SLOWMO_START: "inicio do video em camera lenta",
    769: "cue mao esquerda",
    770: "cue mao direita",
    811: "condicao garrafa + mao direita",
    812: "condicao garrafa + mao esquerda",
    813: "condicao bola + mao direita",
    814: "condicao bola + mao esquerda",
    815: "condicao caneta + mao direita",
    816: "condicao caneta + mao esquerda",
}

# Estados de fase compartilhados com a thread de tracking.
PHASE_IDLE = ""
PHASE_ORIGIN = "origem"
PHASE_HOME = "home"
PHASE_ME = "ME"
PHASE_VIDEO = "video"
PHASE_MI = "MI"

CUE_LABELS = {
    "esquerda": "Mao ESQUERDA",
    "direita": "Mao DIREITA",
    "circulo": "AMBAS as maos",
    "nada": "",
}

# ---------------------------------------------------------------------------
# Colunas de movimento anexadas ao CSV (ordem = ordem no CSV).
# Ordem: KT (3) + bloco do IMU de cada lado (11 + 11) + IMU_hand (1) +
# KTT_valid (1) + ARM_* (18) + PALM_* (11) + KT_hand/KT_src/KT_onset (3).
# ---------------------------------------------------------------------------
KT_COLUMNS = ["KT_x_m", "KT_y_m", "KT_z_m"]
def _imu_block(sufixo):
    """Colunas do bloco de UM lado (11), na ordem em que a linha e' montada.

    orientacao (rpy, graus, relativa a origem) -> aceleracao BRUTA (g) ->
    posicao fusionada (m, relativa a origem) -> validade -> zeragem.

    A aceleracao bruta passou a ser gravada porque e' ela que o filtro de Kalman
    do tempo real consome (m/s^2 apos rodar para o referencial da origem); sem
    ela nao daria para validar/reprocessar o filtro offline no dado real.
    """
    return [
        f"IMU_{sufixo}_roll_deg", f"IMU_{sufixo}_pitch_deg",
        f"IMU_{sufixo}_yaw_deg",
        f"IMU_{sufixo}_acc_x_g", f"IMU_{sufixo}_acc_y_g", f"IMU_{sufixo}_acc_z_g",
        f"IMU_{sufixo}_pos_x_m", f"IMU_{sufixo}_pos_y_m", f"IMU_{sufixo}_pos_z_m",
        f"IMU_{sufixo}_valid", f"ZERO_lock_{sufixo}",
    ]


#: DOIS IMUs (um por mao; ver docs/IMU_DUPLO_ESP32_DESENHO.md): um bloco por lado
#: ("L" = esquerda, "R" = direita) e `IMU_hand` dizendo QUAL lado o trial estava
#: executando (mesma convencao de ARM_side/KT_hand: 1 direita, 2 esquerda, 0
#: desconhecido). Cada sensor tem o seu bias e a sua zeragem, e o lado que nao
#: executa serve de referencia de repouso (taxa de falso movimento).
#: A posicao fusionada e' relativa a origem (como KT_*); a aceleracao e' a
#: BRUTA do MPU6050, em g.
IMU_COLUMNS = _imu_block("L") + _imu_block("R") + ["IMU_hand"]
# Braco com cinematica inversa de elos rigidos (ArmLinkModel em
# kinect_imu_groundtruth): juntas em metros RELATIVAS a origem + angulos
# articulares + comprimentos de elo efetivos + diagnostico da IK.
ARM_COLUMNS = [
    "ARM_shoulder_x_m", "ARM_shoulder_y_m", "ARM_shoulder_z_m",
    "ARM_elbow_x_m", "ARM_elbow_y_m", "ARM_elbow_z_m",
    "ARM_wrist_x_m", "ARM_wrist_y_m", "ARM_wrist_z_m",
    "ARM_elbow_angle_deg",     # interno no cotovelo: 180 = esticado
    "ARM_shoulder_elev_deg",   # elevacao do braco (ombro->cotovelo)
    "ARM_shoulder_azim_deg",   # azimute do braco no plano horizontal
    "ARM_upper_len_m",         # elo efetivo ombro->cotovelo
    "ARM_fore_len_m",          # elo efetivo cotovelo->punho
    "ARM_valid",               # 1 = geometria do braco do frame atual
    "ARM_ik",                  # 0 cru | 1 dobramento medido | 2 bend_ref | 3 sem ref
    "ARM_clamped",             # 1 = punho medido fora do alcance dos elos
    "ARM_side",                # 0 desconhecido | 1 direita | 2 esquerda
]
# Orientacao da palma medida pelo Kinect (independente do IMU): normal do
# plano da palma e direcao dos dedos (vetores unitarios em camera space).
PALM_COLUMNS = [
    "PALM_nx", "PALM_ny", "PALM_nz",        # normal (sinal para a camera)
    "PALM_dirx", "PALM_diry", "PALM_dirz",  # direcao dos dedos
    "PALM_elev_deg",   # +90 = palma virada para cima, -90 = para baixo
    "PALM_azim_deg",   # +90 = normal virada para a direita
    "PALM_pitch_deg",  # +90 = dedos apontando para cima
    "PALM_yaw_deg",    # +90 = dedos para a direita, 0 = dedos para frente
    "PALM_valid",
]
MOTION_COLUMNS = (KT_COLUMNS + IMU_COLUMNS + ["KTT_valid"]
                  + ARM_COLUMNS + PALM_COLUMNS
                  + ["KT_hand", "KT_src", "KT_onset"])
N_MOTION_COLS = len(MOTION_COLUMNS)
#: Codigo numerico da FONTE da posicao 3D (coluna KT_src), para a analise
#: saber se cada amostra veio de MEDIDA real (triangulacao/depth do Kinect ou
#: esqueleto do SDK) ou de ESTIMATIVA (modelo/ultima valida/nominal).
#: 0 = sem medida.
KT_SRC_CODE = {
    "triangulado": 1, "kinect": 2, "esqueletosdk": 3, "modelo3d": 4,
    "amostrada": 5, "ultima": 6, "nominal": 7, "imu": 8, "imuonly": 8,
}


def kt_src_code(source):
    """Traduz a string de origem ('triangulado_laptop', ...) para o codigo."""
    chave = str(source or "").split("_")[0].strip().lower().rstrip("!")
    return int(KT_SRC_CODE.get(chave, 0))

# Lado do braco a publicar, a partir do cue da trial. Sem lado definido
# (circulo = ambas as maos, repouso, calibracao) o lado e escolhido pela mao
# que estiver se movendo mais rapido (ver TrackingThread._fastest_side).
ARM_CUE_SIDE = {"esquerda": "left", "direita": "right"}
ARM_SIDE_CODE = {"left": 2, "right": 1}

# Tempo maximo para a webcam auxiliar abrir. cv2.VideoCapture pode bloquear
# dezenas de segundos com o barramento USB saturado (o Kinect v2 ocupa muita
# banda); sem este limite a thread de tracking inteira ficaria parada antes do
# loop, ou seja: sem KT_*, sem ARM_* e sem publicar o ESP32 no CSV, em
# silencio. Passado o tempo, a sessao segue SEM a segunda vista (stereo off).
AUX_OPEN_TIMEOUT_SEC = 8.0
# Distancia (m) acima da qual o punho medido (triangulado/MediaPipe) e
# considerado de OUTRA mao / medida ruim e o pulso do esqueleto prevalece.
# Protege a IK de receber um alvo impossivel quando a deteccao troca de mao.
ARM_WRIST_TRUST_M = 0.55
# Velocidade (m/s) minima para considerar um lado "em movimento" na escolha
# automatica da mao ativa quando a trial nao define lado (cue circulo/nada).
ARM_ACTIVE_SPEED_MPS = 0.05

# =============================================================================
# Estado compartilhado (thread de tracking <-> muxer/controlador Qt)
# =============================================================================
def _nan3():
    """Vetor 3D "sem medida" (NaN), usado nos campos do SharedState."""
    return np.full(3, np.nan)


class SharedState:
    """Estado thread-safe entre a thread de tracking e o resto do programa."""

    def __init__(self):
        self.lock = threading.Lock()
        self.phase = PHASE_IDLE          # PHASE_* : o overlay segue a fase
        self.cue = ""                    # tipo de cue atual (texto no overlay)
        self.message = ""                # mensagem livre no overlay
        # --- diagnostico de aquisicao ao vivo (#15 e #23/#24) ---
        self.link_ok = 1                 # 0 = sem amostras de EEG ha > 1.5 s
        self.link_text = ""              # motivo/idade da ultima perda
        self.impedance_values = None     # ultima medida (kOhm por canal)
        self.impedance_text = ""         # tabela formatada (kOhm) p/ a tela
        self.motion = {                  # ultima medida publicada
            "x": np.nan, "y": np.nan, "z": np.nan,
            "roll": np.nan, "pitch": np.nan, "yaw": np.nan,
            "valid": 0, "src": "",
            "hand": 0, "src_code": 0, "onset": 0,
            "esp_ts_us": np.nan, "pc_time": np.nan,
            # --- fusao IMU + cameras (zeragem) ---
            "imu_pos": _nan3(), "imu_vel": np.nan, "zero_lock": 0,
            # --- braco (cinematica inversa de elos rigidos) ---
            "arm_shoulder": _nan3(), "arm_elbow": _nan3(), "arm_wrist": _nan3(),
            "arm_elbow_angle": np.nan, "arm_shoulder_elev": np.nan,
            "arm_shoulder_azim": np.nan,
            "arm_upper_len": np.nan, "arm_fore_len": np.nan,
            "arm_valid": 0, "arm_ik": 0, "arm_clamped": 0, "arm_side": 0,
            # --- orientacao da palma (Kinect) ---
            "palm_normal": _nan3(), "palm_dir": _nan3(),
            "palm_elev": np.nan, "palm_azim": np.nan,
            "palm_pitch": np.nan, "palm_yaw": np.nan, "palm_valid": 0,
        }
        self.origin_xyz = None           # (x, y, z) absoluto da origem (m)
        self.origin_rpy = None           # (roll, pitch, yaw) da origem (deg)
        self.origin_ready = False
        self.origin_collect_until = None # monotonic ate quando coletar
        self._origin_samples = []
        # Elos do braco: coletados junto com a origem (mesmo periodo de 10 s).
        self.measure_arm = False
        self.arm_model_done = False
        self.arm_cue_side = ""           # lado do cue corrente ("left"/"right")
        # IK do braco: a thread de tracking informa se esta ligada e se ja
        # existe modelo de elos calibrado (arm_model.json).
        self.arm_ik_enabled = True
        self.arm_model_ready = False
        # Detalhes do modelo de elos publicado no JSON de eventos (L1/L2/lado).
        self.arm_model_info = {}
        # Video do movimento em camera lenta (priming visual da pausa):
        # o controlador PEDE o clipe ao fim da ME e o tracking publica os
        # quadros (ja duplicados para 0,5x) para a janela de estimulos tocar.
        self.video_frames = []
        self.video_ready = False
        self._video_pos = 0
        self.slowmo_request = None       # monotonic do pedido pendente
        self.quit_requested = False      # tecla 'q' no overlay

    # --------------------------------------------------------------- video
    def request_slowmo(self, destino=None):
        """Pede ao tracking o clipe do movimento executado (ao fim da ME).

        `destino` e' o caminho do arquivo onde o clipe sera' salvo (#7); None
        apenas toca o clipe na tela, sem gravar em disco.
        """
        with self.lock:
            self.video_frames = []
            self.video_ready = False
            self._video_pos = 0
            self.slowmo_request = (time.monotonic(), destino)

    def take_slowmo_request(self):
        """Consome o pedido pendente: (monotonic, destino) ou None."""
        with self.lock:
            pedido = self.slowmo_request
            self.slowmo_request = None
            return pedido

    def publish_video(self, frames):
        with self.lock:
            self.video_frames = list(frames)
            self.video_ready = True
            self._video_pos = 0

    def take_video_frame(self):
        """Proximo quadro do clipe, ou None se acabou/ainda nao esta pronto."""
        with self.lock:
            if self._video_pos >= len(self.video_frames):
                return None
            frame = self.video_frames[self._video_pos]
            self._video_pos += 1
            return frame

    def clear_video(self):
        with self.lock:
            self.video_frames = []
            self.video_ready = False
            self._video_pos = 0

    @property
    def needs_arm_calibration(self):
        """True quando a proxima calibracao de origem deve medir os elos."""
        with self.lock:
            return bool(self.arm_ik_enabled and not self.arm_model_ready)

    def set_phase(self, phase, cue=None, message=""):
        """Atualiza fase/cue. `cue=None` mantem o cue atual: a MI acontece
        logo apos a ME, com a seta ainda na tela, entao o lado escolhido para
        a publicacao do braco (arm_cue_side) deve persistir na MI."""
        with self.lock:
            self.phase = phase
            if cue is not None:
                self.cue = cue
                self.arm_cue_side = ARM_CUE_SIDE.get(cue, "")
            self.message = message

    def start_origin_collection(self, seconds, measure_arm=False):
        """Inicia a coleta da origem. `measure_arm` tambem calibra os elos do
        braco (a mao fica parada na mesa: e o momento ideal para a mediana)."""
        with self.lock:
            self.origin_ready = False
            self._origin_samples = []
            self.arm_model_done = False
            self.origin_collect_until = time.monotonic() + seconds
        self.measure_arm = bool(measure_arm)

    def finish_origin_collection(self, timeout=2.0):
        """Espera a thread de tracking resumir a origem coletada."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.origin_ready:
                    return True
            time.sleep(0.05)
        return False

    def publish_motion(self, **kwargs):
        with self.lock:
            self.motion.update(kwargs)

    def snapshot(self):
        with self.lock:
            return dict(self.motion), self.phase, self.cue, self.message

    def set_link_status(self, ok, text=""):
        """Publica o estado do link de EEG (usado pelo LinkWatchdog)."""
        with self.lock:
            self.link_ok = int(bool(ok))
            self.link_text = text

    def set_impedance(self, values, text=""):
        """Publica a ultima medida de impedancia (kOhm por canal)."""
        with self.lock:
            self.impedance_values = (None if values is None
                                     else [float(v) for v in values])
            self.impedance_text = text

    def impedance_snapshot(self):
        with self.lock:
            return self.impedance_values, self.impedance_text

    def origin_snapshot(self):
        with self.lock:
            return self.origin_xyz, self.origin_rpy

    def request_quit(self):
        with self.lock:
            self.quit_requested = True

    @property
    def quit_flag(self):
        with self.lock:
            return self.quit_requested


class MovementOnsetDetector:
    """Detecta o INICIO DO MOVIMENTO (onset) pela velocidade do punho 3D.

    Motivo cientifico (protocolo): o planejamento motor comeca 500-1000 ms ANTES
    do movimento visivel (potencial de prontidao) e ~500-350 ms antes o cortex
    pre-motor / area motora suplementar organizam a sequencia do movimento. Por
    isso a janela de EEG de interesse deve comecar ~0,5 s ANTES do onset
    detectado -- e nao num instante fixo pos-cue (que depende de quanto o
    participante demora para reagir).

    Como funciona:
      - so' vale na fase ME (o chamador zera a deteccao nas fases HOME/origem);
      - calcula a velocidade (m/s) entre amostras consecutivas do punho 3D;
      - exige `consec` amostras consecutivas acima de `speed_mps` E um
        deslocamento acumulado >= `min_travel_m` desde o inicio da ME (o que
        rejeita ruido/piscadas de triangulacao com a mao parada);
      - ao disparar, guarda o instante (relogio monotonic) e o numero de
        amostras desde a cue, e nunca dispara de novo na mesma trial.
    """

    def __init__(self, speed_mps=MOVE_ONSET_SPEED_MPS,
                 consec=MOVE_ONSET_CONSEC,
                 min_travel_m=MOVE_ONSET_MIN_TRAVEL_M):
        self._speed = float(speed_mps)
        self._consec = int(consec)
        self._min_travel = float(min_travel_m)
        self.reset()

    def reset(self):
        """Reinicia para uma nova trial (chamar nas fases HOME/origem)."""
        self.armed = False          # ja' viu a fase ME?
        self.detected = False
        self.onset_time = None      # time.monotonic() do onset
        self.onset_index = 0        # amostras desde o inicio da ME
        self._run = 0
        self._last_time = None
        self._last_position = None
        self._first_position = None
        self.samples = 0

    def update(self, time_s, position, phase_me):
        """Recebe (t, posicao 3D ou None, esta' na fase ME?).

        Devolve True EXATAMENTE na amostra em que o onset e' detectado.
        """
        if not phase_me:
            return False
        if self.detected:
            return False
        if not self.armed:
            self.armed = True
            self._run = 0
            self._last_time = None
            self._last_position = None
            self._first_position = None
        self.samples += 1
        if position is None or time_s is None:
            self._last_time, self._last_position = None, None
            return False
        position = np.asarray(position, np.float64).reshape(3)
        if not np.isfinite(position).all():
            self._last_time, self._last_position = None, None
            return False
        if self._first_position is None:
            self._first_position = position.copy()
        travel = float(np.linalg.norm(position - self._first_position))
        speed = 0.0
        if self._last_time is not None and self._last_position is not None:
            dt = float(time_s) - float(self._last_time)
            if dt > 1e-6:
                speed = float(np.linalg.norm(position - self._last_position)) / dt
        self._last_time, self._last_position = float(time_s), position
        if speed >= self._speed and travel >= self._min_travel:
            self._run += 1
        else:
            self._run = 0
        if self._run >= self._consec:
            self.detected = True
            self.onset_time = float(time_s)
            self.onset_index = self.samples
            return True
        return False


class TrackingThread(threading.Thread):
    """Rodape de kinect_imu_groundtruth rodando em thread propria.

    A cada frame publica no SharedState: posicao 3D do punho (m, frame
    Kinect; no 0 da triangulacao stereo, com cadeia de fallback: triangulado
    -> depth da palma -> esqueleto do SDK -> NaN) e a ultima orientacao do
    ESP32 (UDP 4210). Durante MI/origem desenha o overlay calibrado (vermelho)
    na janela de tracking; nas demais fases mostra a janela escurecida com o
    rotulo da fase (a participante olha apenas a tela de cue).
    """

    WINDOW_NAME = "Tracking (overlay Kinect) - MI"

    def __init__(self, state, imu, with_kinect=True, use_ik=True,
                 aux_indices=(0, 1), use_zeroing=True, marker_queue=None,
                 use_onset=True):
        super().__init__(daemon=True)
        self.state = state
        self.imu = imu
        self.marker_queue = marker_queue    # p/ publicar o marcador 795 (onset)
        self.with_kinect = with_kinect
        self.use_ik = use_ik
        self.aux_indices = tuple(int(index) for index in aux_indices)
        self.use_zeroing = bool(use_zeroing)
        #: Deteccao do INICIO DO MOVIMENTO (ancora da janela e do priming).
        self.onset_detector = MovementOnsetDetector() if use_onset else None
        self.onset_time = None              # monotonic do onset da trial atual
        self._onset_flag_frames = 0
        self.running = True
        self.mode = "kinect" if with_kinect else "imu_only"
        self.tracker = None
        self.alignment = None
        self.auxiliary = None               # primeira camera (compatibilidade)
        self.auxiliaries = {}               # indice -> (capture, resolucao)
        self.arm_model = None
        self.fusion = None                  # gt.PositionFusion (zeragem)
        self._arm_prev = {}      # posicao/tempo do punho por lado (velocidade)
        self._triangulated = None
        self.cv2 = None
        self.gt = None
        # Hook opcional: callable(display, state) chamado a cada frame antes do
        # imshow. Usado pelo Programa 2 (sand_traj_tempo_real.py) para desenhar
        # a trajetoria prevista pelo SAND sem alterar o programa de gravacao.
        self.extra_draw = None
        # Video do movimento (priming da pausa): buffer circular dos ultimos
        # quadros do RGB do Kinect (bem reduzidos) + historico da trajetoria
        # 3D do punho para desenhar o caminho sobre o video.
        self._frame_buffer = deque(maxlen=int(
            SLOWMO_CAPTURE_FPS * SLOWMO_BUFFER_SEC))
        self._traj_history = deque(maxlen=900)
        self._clip_size = (640, 360)
        # Estatisticas dos clipes de priming gravados em disco (#7).
        self._clips_saved = 0
        self._clips_bytes = 0

    @property
    def clips_saved(self):
        """Quantos clipes de priming foram gravados (#7)."""
        return self._clips_saved

    @property
    def clips_bytes(self):
        """Tamanho total (bytes) dos clipes de priming gravados (#7)."""
        return self._clips_bytes

    # ------------------------------------------------------------------ setup
    def _setup_tracking(self):
        import cv2
        self.cv2 = cv2
        import kinect_imu_groundtruth as gt
        self.gt = gt
        self.tracker = gt.KinectHandTracker(aux_indices=self.aux_indices)
        self.alignment = gt.CameraAlignment(self.aux_indices)
        if self.use_ik:
            self.arm_model = gt.ArmLinkModel()
            if self.arm_model.ready:
                print(f"[braco] modelo carregado de {self.arm_model.path.name}: "
                      f"L1={self.arm_model.l1:.3f} m, L2={self.arm_model.l2:.3f} m "
                      f"(lado {self.arm_model.side or '-'}) -> IK ativa",
                      flush=True)
            else:
                print("[braco] sem modelo de elos: a calibracao de origem "
                      "medira ombro/cotovelo/punho e criara arm_model.json",
                      flush=True)
        else:
            print("[braco] IK desativada (--sem-ik): ARM_* vem do esqueleto cru",
                  flush=True)
        self.state.arm_ik_enabled = bool(self.use_ik)
        self.state.arm_model_ready = bool(
            self.arm_model is not None and self.arm_model.ready)
        self.state.arm_model_info = self._arm_model_info()
        if self.use_zeroing:
            self.fusion = gt.PositionFusion()
            print("[zeragem] fusao IMU+cameras ativa (bias + ancora espacial)",
                  flush=True)
        requested_default = self.alignment.calibrated_auxiliary_size or (1280, 720)
        # Abre TODAS as webcams configuradas (laptop, usb, ...) com timeout
        # por camera. Cada indice usa a propria calibracao (stereo/landmarks) e
        # a propria resolucao de calibracao quando existir.
        for index in self.aux_indices:
            model = self.alignment.aux_models.get(int(index)) or {}
            requested = model.get("auxiliary_size") or requested_default
            capture, obtained = self._open_auxiliary(cv2, int(index),
                                                     requested)
            self.auxiliaries[int(index)] = capture
            if capture is None:
                continue
            name = gt.AUX_CAMERA_NAMES.get(int(index), f"aux{index}")
            stereo = ("aceita" if self.alignment.aux_usable(int(index))
                      else "ausente")
            print(f"[tracking] camera {name} (indice {index}): {obtained} "
                  f"| stereo: {stereo}", flush=True)
        self.auxiliary = self.auxiliaries.get(int(self.aux_indices[0])) \
            if self.aux_indices else None
        if not self.auxiliaries:
            print("[tracking] nenhuma webcam auxiliar: rodando com o Kinect "
                  "apenas (triangulacao stereo desligada, KT_* vem do "
                  "depth/esqueleto)", flush=True)
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 960, 540)

    @staticmethod
    def _open_auxiliary(cv2, index, size,
                        timeout=AUX_OPEN_TIMEOUT_SEC):
        """Abre a webcam auxiliar em thread propria, com timeout.

        Devolve (capture_ou_None, (largura, altura) obtidas). Se o timeout
        estourar, o worker continua rodando em daemon e a captura e liberada
        assim que ele terminar (sem vazar o dispositivo).
        """
        result = {"abandon": False}

        def worker():
            try:
                capture = cv2.VideoCapture(index)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
                obtained = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                result["capture"] = capture
                result["obtained"] = obtained
                if result.get("abandon"):
                    # O main thread desistiu enquanto abríamos: libera aqui
                    # (release() em captura já liberada é no-op, então o
                    # release compensatório do main thread é inofensivo).
                    capture.release()
                    result.pop("capture", None)
            except Exception as exc:          # backend ausente/ocupado
                result["error"] = exc

        thread = threading.Thread(target=worker, daemon=True,
                                  name="aux-camera-open")
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            # Marca para o worker liberar a captura quando (e se) abrir, e
            # reaproveita qualquer captura que tenha chegado na janela de
            # corrida entre o join e a marcacao.
            result["abandon"] = True
            leftover = result.pop("capture", None)
            if leftover is not None:
                leftover.release()
            print(f"[tracking] AVISO: camera auxiliar {index} nao abriu em "
                  f"{timeout:g} s (USB saturado?); seguindo SEM a segunda "
                  "vista", flush=True)
            return None, (0, 0)
        capture = result.get("capture")
        if capture is None:
            print(f"[tracking] AVISO: falha ao abrir a camera auxiliar "
                  f"{index} ({result.get('error', 'sem detalhe')})", flush=True)
            return None, (0, 0)
        if not capture.isOpened():
            capture.release()
            print(f"[tracking] AVISO: camera auxiliar {index} nao respondeu "
                  "(isOpened=False)", flush=True)
            return None, (0, 0)
        if result.get("abandon"):
            capture.release()                 # corrida: libera e segue sem ela
            return None, (0, 0)
        return capture, result.get("obtained", (0, 0))

    # -------------------------------------------------------------- main loop
    def run(self):
        if self.with_kinect:
            try:
                self._setup_tracking()
            except Exception as exc:  # Kinect ausente / SDK ausente
                print(f"[tracking] Kinect indisponivel ({exc}); "
                      "modo IMU-only (KT_* ficara NaN, valid=0)", flush=True)
                self.mode = "imu_only"
        try:
            if self.mode == "kinect":
                self._loop_kinect()
            else:
                self._loop_imu_only()
        finally:
            self._shutdown()

    def _loop_kinect(self):
        cv2 = self.cv2
        while self.running and not self.state.quit_flag:
            color, depth = self.tracker.frames()
            # Fase/cue do instante atual (antes das deteccoes): `arm_cue_side`
            # diz QUAL mao a trial pede. Com as duas maos visiveis, rastrear a
            # mao errada contaminaria o alvo do SAND em silencio.
            _motion, phase, _cue, message = self.state.snapshot()
            side_hint = self.state.arm_cue_side
            want_side = side_hint or None
            # ---- le TODAS as webcams configuradas ----
            aux_frames, aux_hands_all = {}, {}
            for index, capture in self.auxiliaries.items():
                if capture is None:
                    aux_hands_all[index] = []
                    continue
                ok, aux_frame = capture.read()
                if ok and aux_frame is not None:
                    aux_frame = self.gt.transform_auxiliary_frame(aux_frame)
                    self.alignment.set_active_aux(index)
                    self.alignment.set_auxiliary_size(aux_frame)
                    aux_frames[index] = aux_frame
                    aux_hands_all[index] = self.tracker.detect_auxiliary_hands(
                        aux_frame, aux_index=index, want_side=want_side)
                else:
                    aux_hands_all[index] = []
            display, palm_position, landmarks = self.tracker.detect_hand(
                color, depth, want_side=want_side)
            if not any(aux_hands_all.values()) \
                    and self.alignment._hand_depth_ema_initialized:
                self.alignment.reset_hand_depth_ema()

            wrist3d, source, aux_used, aux_hands = self._resolve_wrist3d(
                palm_position, landmarks, aux_hands_all, side_hint)
            joints_all = self.tracker.arm_joints_all()
            arm, arm_side = self._resolve_arm(wrist3d, side_hint, joints_all)
            palm = self._resolve_palm(landmarks)
            depths_for_projection = self._projection_depths(palm_position,
                                                            wrist3d)
            if display is not None and (
                self.alignment.stereo_usable
                or self.alignment.hand_landmark_ready
            ) and aux_hands:
                self.tracker.draw_auxiliary_hands_on_color(
                    display, aux_hands, self.alignment,
                    depths_for_projection, self._triangulated)
                self._draw_badge(display, phase, message)
                if not self.state.link_ok:
                    # Link de EEG caiu (#15): aviso visivel na janela de
                    # tracking, alem do console e do marcador 899 no CSV.
                    cv2.putText(display, "LINK EEG PERDIDO", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if display is not None:
                self._draw_arm(display, arm, arm_side)
                if self.extra_draw is not None:
                    try:
                        self.extra_draw(display, self.state)
                    except Exception as exc:
                        print(f"[tracking] erro no overlay extra: {exc}",
                              flush=True)

            self._collect_origin(wrist3d, joints_all)

            # ---- video do movimento (priming da pausa em 0,5x) ----
            self._buffer_frame(color)
            # ---- INICIO DO MOVIMENTO (onset): marca nos dados + ancora o clipe
            if phase in (PHASE_HOME, PHASE_ORIGIN) and self.onset_detector:
                self.onset_detector.reset()     # nova trial: zera a deteccao
                self.onset_time = None
            if self.onset_detector is not None and self.onset_detector.update(
                    time.monotonic(), wrist3d, phase == PHASE_ME):
                self.onset_time = self.onset_detector.onset_time
                self._onset_flag_frames = 2
                self._emit_onset_marker()
                print(f"[onset] INICIO DO MOVIMENTO detectado "
                      f"({self.onset_detector.onset_index} amostras apos a cue, "
                      f"limiar {MOVE_ONSET_SPEED_MPS:g} m/s)", flush=True)
            onset_flag = 0
            if self._onset_flag_frames > 0:
                onset_flag = 1
                self._onset_flag_frames -= 1
            if wrist3d is not None:
                self._traj_history.append(
                    (time.monotonic(), np.asarray(wrist3d, np.float64)))
            pedido = self.state.take_slowmo_request()
            if pedido is not None:
                self._publish_slowmo(pedido)

            imu = self.imu.get_latest()
            imu_fused, zero_lock = None, 0
            if self.fusion is not None:
                # zeragem: camera valida (medida 3D) entra na fusao
                self.fusion.update(imu, wrist3d if wrist3d is not None
                                   else None)
                imu_fused = self.fusion.absolute_position
                zero_lock = int(self.fusion.zero_lock)
            self.state.publish_motion(**self._motion_payload(
                wrist3d, source, imu, arm=arm, arm_side=arm_side, palm=palm,
                imu_fused=imu_fused, zero_lock=zero_lock, fusion=self.fusion,
                aux_used=aux_used,
                hand_side=self.tracker.last_hand_side,
                onset_flag=onset_flag))

            if display is not None:
                cv2.imshow(self.WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.state.request_quit()

    def _loop_imu_only(self):
        """Sem Kinect: publica apenas o ESP32 (KT_*/ARM_*/PALM_* = NaN)."""
        while self.running and not self.state.quit_flag:
            imu = self.imu.get_latest()
            payload = self._motion_payload(None, "imu_only", imu)
            payload["valid"] = 0
            self.state.publish_motion(**payload)
            time.sleep(0.005)

    # ------------------------------------------------------------- resolva 3D
    def _resolve_wrist3d(self, palm_position, landmarks, aux_hands_all,
                         prefer_side=None):
        """Cadeia de fallback com varias webcams: escolhe a melhor para stereo.

        Para cada webcam com deteccao e calibracao stereo, faz a triangulacao
        Kinect+webcam e guarda a de MENOR erro mediano de reprojecao; o
        alinhamento fica na camera escolhida (aux_used). Sem stereo, cai para
        depth do Kinect -> esqueleto SDK.
        """
        self._triangulated = None
        best = None
        best_err = float("inf")
        best_aux = None
        best_hands = []
        if landmarks is not None and len(landmarks) == 21:
            kin_px = [(lm.x * self.tracker.color_width,
                       lm.y * self.tracker.color_height) for lm in landmarks]
            for index, hands in (aux_hands_all or {}).items():
                if not hands or not self.alignment.aux_usable(int(index)):
                    continue
                self.alignment.set_active_aux(int(index))
                aux_px = [(lm.x * self.alignment.auxiliary_width,
                           lm.y * self.alignment.auxiliary_height)
                          for lm in hands[0]]
                tri3d, reproj = self.alignment.triangulate_hand_points(
                    aux_px, kin_px)
                if tri3d is None:
                    continue
                error = (float(np.nanmedian(reproj))
                         if reproj is not None and np.isfinite(reproj).any()
                         else float("inf"))
                if error < best_err:
                    best, best_err = tri3d, error
                    best_aux, best_hands = int(index), hands
        if best is not None:
            self._triangulated = best
            self.alignment.last_triangulated_3d = best
            self.alignment.last_triangulated_3d_time = time.monotonic()
            name = self.gt.AUX_CAMERA_NAMES.get(best_aux, f"aux{best_aux}")
            # Ponto rastreado = landmark 9 (centro da palma), conforme a
            # metodologia; se estiver invalido cai para o pulso (landmark 0).
            ponto = best[HAND_TRACK_LANDMARK]
            if not np.isfinite(ponto).all():
                ponto = best[0]
            if np.isfinite(ponto).all():
                return ponto.astype(float), f"triangulado_{name}", \
                    best_aux, best_hands
        else:
            self.alignment.last_triangulated_3d = None
            self.alignment.last_triangulated_3d_time = None
        if palm_position is not None:
            return palm_position.astype(float), "kinect_depth", None, []
        anchor = self.tracker.body_hand_anchor(prefer_side)
        if anchor is not None:
            return np.asarray(anchor[:3], dtype=float), "esqueletoSDK", \
                None, []
        return None, "", None, []

    def _projection_depths(self, palm_position, wrist3d):
        """Profundidade para projetar o overlay vermelho (prioridade: 3D)."""
        if self._triangulated is not None:
            palm_z = float(self._triangulated[9, 2])
            if not np.isfinite(palm_z):
                palm_z = float(np.nanmedian(self._triangulated[:, 2]))
            return [palm_z]
        if palm_position is not None:
            return [float(palm_position[2])]
        if wrist3d is not None:
            return [float(wrist3d[2])]
        return [self.gt.NOMINAL_HAND_DEPTH_M]

    # ------------------------------------------------- braco (IK) / palma
    def _select_arm(self, joints_all, wrist3d, prefer_side):
        """Escolhe o lado do braco a publicar.

        Preferencia: o lado do cue (esquerda/direita). Sem lado definido
        (circulo ou nada), usa a mao mais PROXIMA do punho medido; se a medida
        3D falhou, usa a mao que estiver se movendo mais rapido (e a mao que
        esta executando o movimento).
        """
        if not joints_all:
            return None
        if prefer_side and prefer_side in joints_all:
            return joints_all[prefer_side]
        if wrist3d is not None:
            target = np.asarray(wrist3d, np.float64)
            return min(joints_all.values(),
                       key=lambda item: float(np.linalg.norm(item["wrist"]
                                                            - target)))
        side = self._fastest_side(joints_all)
        if side is not None:
            return joints_all[side]
        return joints_all["right"] if "right" in joints_all else next(
            iter(joints_all.values()))

    def _fastest_side(self, joints_all, max_period=1.5):
        """Lado cujo pulso se moveu mais rapido desde o frame anterior.

        Usa apenas os pulsos do proprio frame (velocidade instantanea), sem
        historico longo: e um desempate, nao um filtro de tracking.
        """
        now = time.monotonic()
        speeds = {}
        for side, data in joints_all.items():
            wrist = np.asarray(data["wrist"], np.float64)
            previous = self._arm_prev.get(side)
            self._arm_prev[side] = (wrist, now)
            if previous is None:
                continue
            dt = now - previous[1]
            if dt <= 1e-6 or dt > max_period:
                continue
            speeds[side] = float(np.linalg.norm(wrist - previous[0]) / dt)
        if not speeds:
            return None
        side = max(speeds, key=speeds.get)
        return side if speeds[side] >= ARM_ACTIVE_SPEED_MPS else None

    def _resolve_arm(self, wrist3d, prefer_side, joints_all=None):
        """Geometria do braco: juntas do esqueleto + IK de elos rigidos.

        O alvo da IK e o PUNHO MEDIDO (triangulado stereo/MediaPipe/depth),
        que e a melhor estimativa da mao e a mesma grandeza gravada em KT_*;
        o ombro e o cotovelo vem do esqueleto do SDK. Se o punho medido
        estiver muito longe do pulso do esqueleto (deteccao trocou de mao),
        usa-se o pulso do SDK como alvo.
        """
        if self.tracker is None:
            return None, 0
        if joints_all is None:
            joints_all = self.tracker.arm_joints_all()
        if not joints_all:
            return None, 0
        selected = self._select_arm(joints_all, wrist3d, prefer_side)
        if selected is None:
            return None, 0
        side = selected["side"]
        target = np.asarray(selected["wrist"], np.float64)
        if wrist3d is not None:
            measured = np.asarray(wrist3d, np.float64)
            if np.isfinite(measured).all() and float(np.linalg.norm(
                    measured - target)) <= ARM_WRIST_TRUST_M:
                target = measured
        if self.arm_model is not None:
            arm = self.arm_model.solve_arm(selected["shoulder"],
                                           selected["elbow"], target)
        else:
            arm = self.gt.ArmLinkModel.solve_raw(selected["shoulder"],
                                                 selected["elbow"], target)
        if arm is None:
            return None, ARM_SIDE_CODE.get(side, 0)
        return arm, ARM_SIDE_CODE.get(side, 0)

    def _resolve_palm(self, landmarks):
        """Orientacao da palma (normal + direcao dos dedos) pelo Kinect.

        Prioridade: triangulacao stereo (nos 0, 5, 9, 17, sem ruido de depth
        por landmark) -> depth do Kinect por landmark. Devolve None se a mao
        nao estiver disponivel.
        """
        tri = self._triangulated
        if tri is not None and len(tri) == 21:
            nodes = []
            for index in (0, 5, 9, 17):
                point = np.asarray(tri[index], np.float64)
                nodes.append(point if np.isfinite(point).all() else None)
            if nodes[0] is not None and (nodes[1] is not None
                                         or nodes[3] is not None):
                normal, direction = self.gt.palm_frame(*nodes)
                if normal is not None:
                    return self.gt.palm_angles(normal, direction)
        if landmarks is not None and len(landmarks) == 21:
            normal, direction = self.tracker.palm_orientation_camera(landmarks)
            if normal is not None:
                return self.gt.palm_angles(normal, direction)
        return None

    def _arm_model_info(self):
        """Detalhes do modelo de elos para o JSON de eventos (ou {})."""
        model = self.arm_model
        if model is None or not model.ready:
            return {"ik_ativa": bool(self.use_ik), "modelo": "ausente"}
        return {
            "ik_ativa": bool(self.use_ik),
            "modelo": "calibrado",
            "arquivo": str(model.path),
            "l1_ombro_cotovelo_m": float(model.l1),
            "l2_cotovelo_punho_m": float(model.l2),
            "alcance_m": float(model.reach_m),
            "lado": model.side,
            "amostras": int(model.samples),
            "bend_ref": (None if model.bend_ref is None
                         else [float(v) for v in model.bend_ref]),
            "nota": ("Elos rigidos medidos na calibracao de origem. ARM_ik: "
                     "0=esqueleto cru, 1=IK com dobramento medido, 2=IK com "
                     "bend_ref, 3=IK sem referencia de dobramento."),
        }

    def _sample_arm_links(self, joints_all):
        """Acumula os elos do braco (ambos os lados) na calibracao de origem.

        Chamado apenas durante a janela de origem, com a mao parada na mesa.
        Os dois bracos da mesma pessoa tem elos praticamente iguais (diferenca
        tipica < 1 cm), entao a mediana sobre os dois lados deixa o modelo
        mais robusto; o lado de CADA frame continua sendo escolhido pelo cue.
        """
        if self.arm_model is None or not joints_all:
            return
        sampled = []
        for side, data in joints_all.items():
            if self.arm_model.add_sample(data["shoulder"], data["elbow"],
                                         data["wrist"], side=side):
                sampled.append(side)
        if len(sampled) > 1:
            self.arm_model.side = "both"

    def _hand_code(self, hand_side):
        """Codigo da lateralidade (0/1/2) usando a tabela do modulo do Kinect.

        Mantido local para a thread continuar funcionando com `--sem-kinect`
        (sem importar/instanciar o modulo de ground truth).
        """
        tabela = getattr(self.gt, "HAND_SIDE_CODE", None) if self.gt else None
        if not tabela:
            tabela = {"": 0, "right": 1, "left": 2}
        return int(tabela.get(hand_side, 0))

    def _motion_payload(self, wrist3d, source, imu, arm=None, arm_side=0,
                        palm=None, imu_fused=None, zero_lock=0,
                        fusion=None, aux_used=None, hand_side="", onset_flag=0):
        """Monta os campos publicados no SharedState (movimento + braco + palma)."""
        payload = {
            "x": np.nan if wrist3d is None else float(wrist3d[0]),
            "y": np.nan if wrist3d is None else float(wrist3d[1]),
            "z": np.nan if wrist3d is None else float(wrist3d[2]),
            "roll": np.nan if imu is None else imu.roll_deg,
            "pitch": np.nan if imu is None else imu.pitch_deg,
            "yaw": np.nan if imu is None else imu.yaw_deg,
            "valid": int(wrist3d is not None),
            "src": source,
            # Lateralidade da mao efetivamente rastreada e codigo numerico da
            # fonte de profundidade: sem isso, uma troca silenciosa de mao (com
            # as duas visiveis) contaminaria o alvo do SAND sem deixar rastro.
            "hand": self._hand_code(hand_side),
            "src_code": kt_src_code(source),
            # 1 na(s) amostra(s) em que o INICIO DO MOVIMENTO foi detectado.
            "onset": int(onset_flag),
            "esp_ts_us": np.nan if imu is None else float(imu.timestamp_us),
            "pc_time": time.monotonic(),
            "imu_pos": (np.asarray(imu_fused, np.float64)
                        if imu_fused is not None else _nan3()),
            "imu_vel": np.nan if fusion is None
            else float(np.linalg.norm(fusion.velocity)),
            "zero_lock": int(zero_lock),
            "aux_used": -1 if aux_used is None else int(aux_used),
        }
        if arm is None:
            payload.update(
                arm_shoulder=_nan3(), arm_elbow=_nan3(), arm_wrist=_nan3(),
                arm_elbow_angle=np.nan, arm_shoulder_elev=np.nan,
                arm_shoulder_azim=np.nan, arm_upper_len=np.nan,
                arm_fore_len=np.nan, arm_valid=0, arm_ik=0, arm_clamped=0,
                arm_side=0)
        else:
            payload.update(
                arm_shoulder=np.asarray(arm["shoulder"], float),
                arm_elbow=np.asarray(arm["elbow"], float),
                arm_wrist=np.asarray(arm["wrist"], float),
                arm_elbow_angle=arm["elbow_angle_deg"],
                arm_shoulder_elev=arm["shoulder_elev_deg"],
                arm_shoulder_azim=arm["shoulder_azim_deg"],
                arm_upper_len=arm["l1_m"], arm_fore_len=arm["l2_m"],
                arm_valid=1, arm_ik=int(arm["ik"]),
                arm_clamped=int(arm["clamped"]), arm_side=int(arm_side))
        if palm is None:
            payload.update(
                palm_normal=_nan3(), palm_dir=_nan3(), palm_elev=np.nan,
                palm_azim=np.nan, palm_pitch=np.nan, palm_yaw=np.nan,
                palm_valid=0)
        else:
            payload.update(
                palm_normal=np.asarray(palm["normal"], float),
                palm_dir=np.asarray(palm["dir"], float),
                palm_elev=palm["elev"], palm_azim=palm["azim"],
                palm_pitch=palm["pitch"], palm_yaw=palm["yaw"],
                palm_valid=int(palm["valid"]))
        return payload

    def _draw_arm(self, frame, arm, arm_side):
        """Desenha ombro-cotovelo-punho projetados no RGB do Kinect.

        Ciano = geometria publicada no CSV (esqueleto + IK). O cotovelo ganha
        um circulo: amarelo quando a IK esta ativa (1-3), branco quando o
        valor vem do esqueleto cru (0 = sem modelo de elos). As juntas do SDK
        estao no camera space do depth; projetar com a intrinseca do RGB deixa
        um deslocamento fixo de poucos cm — aceitavel num overlay de conferencia.
        """
        cv2 = self.cv2
        if arm is None or self.alignment is None:
            return
        points = np.vstack((np.asarray(arm["shoulder"], np.float64),
                            np.asarray(arm["elbow"], np.float64),
                            np.asarray(arm["wrist"], np.float64)))
        pixels = self.alignment.project_camera_points(
            points, (self.tracker.color_width, self.tracker.color_height))
        if not np.isfinite(pixels).all():
            return
        pixels = np.round(pixels).astype(int)
        cv2.line(frame, tuple(pixels[0]), tuple(pixels[1]), (255, 255, 0), 2)
        cv2.line(frame, tuple(pixels[1]), tuple(pixels[2]), (255, 255, 0), 2)
        elbow_color = (0, 255, 255) if int(arm["ik"]) else (255, 255, 255)
        for point in pixels:
            cv2.circle(frame, tuple(point), 5, elbow_color, -1)
        label = f"ik={int(arm['ik'])} lado={ARM_SIDE_CODE.get(arm_side, 0)}"
        if int(arm["clamped"]):
            label += " CLAMP"
        cv2.putText(frame, label, (int(pixels[1][0]) + 8, int(pixels[1][1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, elbow_color, 2)

    # -------------------------------------------------------------- origem
    def _collect_origin(self, wrist3d, joints_all=None):
        """Coleta a origem (10 s) e, no mesmo periodo, os ELOS do braco.

        A mao fica parada na mesa durante a calibracao: e o momento com menor
        ruido para medir ombro->cotovelo e cotovelo->punho com a mediana.
        """
        with self.state.lock:
            if self.state.origin_collect_until is not None:
                active = time.monotonic() < self.state.origin_collect_until
                measure_arm = self.state.measure_arm
                if active:
                    imu = self.imu.get_latest()
                    if wrist3d is not None and imu is not None:
                        self.state._origin_samples.append(
                            (float(wrist3d[0]), float(wrist3d[1]),
                             float(wrist3d[2]),
                             float(imu.roll_deg), float(imu.pitch_deg),
                             float(imu.yaw_deg)))
                    if measure_arm:
                        self._sample_arm_links(joints_all)
                    return
                # janela terminou: resume a origem coletada
                samples = self.state._origin_samples
                if len(samples) >= 10:
                    arr = np.asarray(samples, dtype=float)
                    self.state.origin_xyz = arr[:, :3].mean(axis=0)
                    self.state.origin_rpy = arr[:, 3:].mean(axis=0)
                    print(f"[origem] definida em "
                          f"{np.round(self.state.origin_xyz, 4)} m "
                          f"(amostras: {len(samples)})", flush=True)
                else:
                    print(f"[origem] AVISO: apenas {len(samples)} amostras "
                          "validas (Kinect/IMU falhando?); origem NAO definida",
                          flush=True)
                if measure_arm and self.arm_model is not None:
                    ok = self.arm_model.finish_calibration()
                    self.state.arm_model_ready = bool(ok)
                    self.state.arm_model_info = self._arm_model_info()
                self.state.arm_model_done = True
                self.state.origin_ready = True
                self.state.origin_collect_until = None

    # ------------------------------------------------- badge / encerramento
    # --------------------------------------------------- video (camera lenta)
    def _buffer_frame(self, color):
        """Guarda uma versao reduzida do RGB do Kinect no buffer circular."""
        if color is None or self.cv2 is None:
            return
        cv2 = self.cv2
        try:
            if color.ndim == 3 and color.shape[2] == 4:
                small = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            else:
                small = color
            small = cv2.resize(small, self._clip_size,
                               interpolation=cv2.INTER_AREA)
        except Exception:
            return
        self._frame_buffer.append(
            (time.monotonic(), np.ascontiguousarray(small)))

    def _emit_onset_marker(self):
        """Publica o marcador CODE_MOVE_ONSET (795) na fila do muxer.

        O muxer fixa o codigo no CSV de EEG na amostra exata do evento, o que
        permite recortar a janela de EEG [onset - 0,5 s, onset + 1,5 s] no
        treino (`--event-code 795 --window-start-sec -0.5`).
        """
        if self.marker_queue is None:
            return
        try:
            self.marker_queue.put({
                "code": CODE_MOVE_ONSET,
                "nome": CODE_NAMES.get(CODE_MOVE_ONSET, "inicio do movimento"),
                "hora": time.strftime("%H:%M:%S"),
                "monotonic_s": round(time.monotonic(), 6),
                "origem": "tracking",
            })
        except Exception as exc:            # fila cheia/indisponivel: nao trava
            print(f"[onset] falha ao publicar o marcador 795: {exc}", flush=True)

    def _clip_window(self, request_time):
        """Intervalo (t0, t1) do movimento mostrado no video de priming.

        ANCORADO NO ONSET (opcao A do protocolo): o clipe comeca no INICIO DO
        MOVIMENTO detectado pelo Kinect (nao num instante fixo pos-cue) e mostra
        SLOWMO_SPAN_SEC segundos a partir dele; em 0,5x isso ocupa
        ~2x o tempo real e cabe na pausa de 2 s. Se o onset nao foi detectado
        (Kinect perdeu a mao, movimento muito lento), cai no recuo fixo do fim
        da ME (comportamento antigo).
        """
        if self.onset_time is not None:
            return self.onset_time, self.onset_time + SLOWMO_SPAN_SEC
        t1 = request_time - SLOWMO_END_LAG_SEC
        return t1 - SLOWMO_SPAN_SEC, t1

    def _trajectory_pixels(self, t0, t1):
        """Trajetoria do punho no intervalo, em pixels do quadro reduzido."""
        if self.alignment is None or self.tracker is None:
            return None
        pontos = [posicao for quando, posicao in self._traj_history
                  if t0 <= quando <= t1]
        if len(pontos) < 2:
            return None
        try:
            full = self.alignment.project_camera_points(
                np.vstack(pontos), (self.tracker.color_width,
                                    self.tracker.color_height))
        except Exception:
            return None
        if not np.isfinite(full).all():
            return None
        escala = self._clip_size[0] / float(self.tracker.color_width)
        return np.round(full * escala).astype(np.int32)

    def _publish_slowmo(self, pedido):
        """Monta e publica o clipe em camera lenta (0,5x) do movimento."""
        if isinstance(pedido, (tuple, list)):
            request_time, destino = (pedido[0],
                                     pedido[1] if len(pedido) > 1 else None)
        else:                        # compatibilidade: apenas o instante
            request_time, destino = pedido, None
        t0, t1 = self._clip_window(request_time)
        frames = [(quando, quadro) for quando, quadro in self._frame_buffer
                  if t0 <= quando <= t1]
        pixels = self._trajectory_pixels(t0, t1)
        repeticoes = max(1, int(round(1.0 / SLOWMO_SPEED)))
        clipe = []
        for _quando, quadro in frames:
            copia = quadro.copy()
            if pixels is not None and len(pixels) > 1:
                self.cv2.polylines(copia,
                                   [pixels.reshape(-1, 1, 2).astype(np.int32)],
                                   False, (0, 255, 255), 3)
                self.cv2.circle(copia, tuple(pixels[-1]), 7, (0, 0, 255), -1)
            clipe.extend([copia] * repeticoes)
        self.state.publish_video(clipe)
        print(f"[video] clipe do movimento: {len(frames)} quadros reais -> "
              f"{len(clipe)} em {SLOWMO_SPEED:g}x "
              f"({len(clipe) / SLOWMO_CAPTURE_FPS:.1f} s de video)", flush=True)
        if destino:
            self._save_clip(clipe, destino)

    def _save_clip(self, clipe, destino):
        """Grava o clipe do priming em disco (#7).

        Custo medido: ~1 s de movimento em 640x360 a 30 fps em MJPG da'
        ~200-400 kB por trial (~60-100 MB por sessao; ~1,5-2,5 GB nas 25
        sessoes do estudo). Vale a pena porque permite (a) auditar o priming
        mostrado e (b) usar o video como ground truth VISUAL da ME, para
        rejeitar na analise as trials em que o movimento nao ocorreu como
        planejado.
        """
        if not clipe or self.cv2 is None:
            return
        try:
            pasta = os.path.dirname(destino)
            if pasta:
                os.makedirs(pasta, exist_ok=True)
            height, width = clipe[0].shape[:2]
            fourcc = self.cv2.VideoWriter_fourcc(*"MJPG")
            escritor = self.cv2.VideoWriter(destino, fourcc,
                                            SLOWMO_CAPTURE_FPS, (width, height))
            if not escritor.isOpened():
                print(f"[video] nao foi possivel gravar {destino} "
                      f"(codec MJPG indisponivel?)", flush=True)
                return
            for quadro in clipe:
                escritor.write(quadro)
            escritor.release()
            tamanho = os.path.getsize(destino)
            self._clips_saved += 1
            self._clips_bytes += tamanho
            print(f"[video] clipe salvo: {os.path.basename(destino)} "
                  f"({tamanho / 1024.0:.0f} kB)", flush=True)
        except Exception as exc:
            print(f"[video] falha ao salvar o clipe {destino}: {exc}",
                  flush=True)

    def _draw_badge(self, frame, phase, message):
        cv2 = self.cv2
        label = {"": "aguardando", PHASE_ORIGIN: "CALIBRACAO ORIGEM",
                 PHASE_HOME: "repouso (home)",
                 PHASE_ME: "fase ME (execucao)",
                 PHASE_VIDEO: "pausa + video 0,5x",
                 PHASE_MI: "fase MI (imaginacao)"}[phase]
        if self.state.cue:
            label += f" | {self.state.cue}"
        if message:
            label = f"{message} | {label}"
        if phase != PHASE_ORIGIN:
            # Fora da calibracao o participante NAO deve ver o movimento no
            # overlay do Kinect (na ME ele executa; na MI ele imagina, sem
            # feedback). O overlay segue desenhado para o experimentador.
            frame[:, :, 0] = (frame[:, :, 0] * 0.35).astype(frame.dtype)
            frame[:, :, 1] = (frame[:, :, 1] * 0.35).astype(frame.dtype)
            frame[:, :, 2] = (frame[:, :, 2] * 0.35).astype(frame.dtype)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)

    def _shutdown(self):
        if self.tracker is not None:
            try:
                self.tracker.close()
            except Exception:
                pass
        for capture in self.auxiliaries.values():
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
        self.auxiliaries = {}
        self.auxiliary = None
        if self.cv2 is not None:
            try:
                self.cv2.destroyAllWindows()
            except Exception:
                pass
        print("[tracking] encerrado", flush=True)


# =============================================================================
# Paradigma Qt (janela de estimulos + maquina de estados)
# =============================================================================
class _StimulusCanvas(QWidget):
    """Tela do participante: tudo desenhado em vetor (sem arquivos de imagem).

    Modos:
      home      -> ponto de fixacao + marca de "home" (mao na posicao inicial);
      cue       -> fixacao + OBJETO alvo (garrafa/bola/caneta) + seta do lado
                   da mao indicada (esquerda/direita);
      video     -> quadro do video do movimento em camera lenta (0,5x);
      message   -> texto grande (instrucoes, pausas, baseline);
      countdown -> numero da contagem regressiva.
    Teclas: ESPACO avisa o controlador (pular pausa) e ESC encerra a sessao.
    """

    WHITE = QColor(245, 245, 245)
    DIM = QColor(120, 120, 120)

    def __init__(self):
        super().__init__()
        self.mode = "message"
        self.text = ""
        self.object_name = None
        self.hand = None
        self.image = None
        self.on_skip = None
        self.on_abort = None
        self.show_home()

    # ---------------------------------------------------------------- modos
    def show_home(self):
        self.mode = "home"
        self.image = None
        self.update()

    def show_condition(self, object_name, hand):
        self.mode = "cue"
        self.object_name = object_name
        self.hand = hand
        self.image = None
        self.update()

    def show_message(self, text):
        self.mode = "message"
        self.text = text
        self.image = None
        self.update()

    def show_countdown(self, number):
        self.mode = "countdown"
        self.text = str(number)
        self.image = None
        self.update()

    def show_video_frame(self, image):
        self.mode = "video"
        self.image = image
        self.update()

    # -------------------------------------------------------------- pintura
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), Qt.black)
        width, height = self.width(), self.height()
        if self.mode == "video" and self.image is not None:
            self._paint_image(painter, width, height)
            return
        if self.mode == "cue":
            self._paint_object(painter, width, height)
            self._paint_arrow(painter, width, height)
        elif self.mode == "home":
            self._paint_home_hint(painter, width, height)
        elif self.mode == "countdown":
            self._paint_text(painter, width, height)
            return
        elif self.mode == "message":
            self._paint_text(painter, width, height)
        self._paint_fixation(painter, width, height)

    def _paint_fixation(self, painter, width, height):
        """Ponto de fixacao central (mantem o olhar no centro da tela)."""
        size = max(18, int(min(width, height) * 0.018))
        thickness = max(3, int(size * 0.22))
        painter.setPen(QPen(self.WHITE, thickness, Qt.SolidLine, Qt.RoundCap))
        cx, cy = width // 2, height // 2
        painter.drawLine(cx - size, cy, cx + size, cy)
        painter.drawLine(cx, cy - size, cx, cy + size)

    def _paint_home_hint(self, painter, width, height):
        """Marca discreta da posicao inicial da mao (home)."""
        radius = max(26, int(min(width, height) * 0.03))
        painter.setPen(QPen(self.DIM, 4))
        painter.setBrush(Qt.NoBrush)
        cx, cy = width // 2, int(height * 0.82)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)
        painter.setPen(QPen(self.DIM, 3))
        painter.setFont(QFont("Arial", max(12, int(radius * 0.42))))
        painter.drawText(QRectF(cx - radius * 2.2, cy + radius * 1.1,
                                radius * 4.4, radius * 1.4),
                         Qt.AlignHCenter, "home")

    def _paint_object(self, painter, width, height):
        """Objeto alvo, grande e centralizado (garrafa / bola / caneta)."""
        size = min(width, height) * 0.30
        cx, cy = width / 2, height * 0.40
        box = QRectF(cx - size * 0.5, cy - size * 0.5, size, size)
        painter.setPen(QPen(self.WHITE, max(5, int(size * 0.045))))
        painter.setBrush(Qt.NoBrush)
        if self.object_name == "garrafa":
            self._draw_bottle(painter, box)
        elif self.object_name == "bola":
            self._draw_ball(painter, box)
        else:
            self._draw_pen(painter, box)

    @staticmethod
    def _draw_bottle(painter, box):
        body = QRectF(box.left() + box.width() * 0.22,
                      box.top() + box.height() * 0.30,
                      box.width() * 0.56, box.height() * 0.68)
        neck = QRectF(box.left() + box.width() * 0.41,
                      box.top() + box.height() * 0.10,
                      box.width() * 0.18, box.height() * 0.22)
        cap = QRectF(box.left() + box.width() * 0.38,
                     box.top() + box.height() * 0.03,
                     box.width() * 0.24, box.height() * 0.09)
        painter.drawRoundedRect(body, box.width() * 0.12, box.width() * 0.12)
        painter.drawRect(neck)
        painter.drawRoundedRect(cap, 4, 4)

    @staticmethod
    def _draw_ball(painter, box):
        radius = box.width() * 0.46
        center = box.center()
        painter.drawEllipse(center, radius, radius)
        upper = QPainterPath()
        upper.moveTo(center.x() - radius * 0.95, center.y() - radius * 0.25)
        upper.arcTo(QRectF(center.x() - radius * 1.1,
                           center.y() - radius * 1.1,
                           radius * 2.2, radius * 2.2), 200, 140)
        painter.drawPath(upper)
        lower = QPainterPath()
        lower.moveTo(center.x() - radius * 0.95, center.y() + radius * 0.25)
        lower.arcTo(QRectF(center.x() - radius * 1.1,
                           center.y() - radius * 1.1,
                           radius * 2.2, radius * 2.2), 20, 140)
        painter.drawPath(lower)

    @staticmethod
    def _draw_pen(painter, box):
        body = QRectF(box.left() + box.width() * 0.16,
                      box.top() + box.height() * 0.40,
                      box.width() * 0.62, box.height() * 0.20)
        painter.drawRoundedRect(body, 8, 8)
        tip = QPolygonF([QPointF(body.right(), body.top()),
                         QPointF(box.right() - box.width() * 0.02,
                                 box.center().y()),
                         QPointF(body.right(), body.bottom())])
        painter.drawPolygon(tip)
        clip = QRectF(body.left() - box.width() * 0.06, body.top(),
                      box.width() * 0.06, body.height())
        painter.drawRect(clip)

    def _paint_arrow(self, painter, width, height):
        """Seta grande do lado da mao indicada (esquerda/direita)."""
        if self.hand not in ("esquerda", "direita"):
            return
        left_side = self.hand == "esquerda"
        length = min(width, height) * 0.30
        thickness = length * 0.16
        cy = height * 0.40
        margin = width * 0.06
        if left_side:
            tail, head = (QPointF(margin + length, cy), QPointF(margin, cy))
        else:
            tail = QPointF(width - margin - length, cy)
            head = QPointF(width - margin, cy)
        painter.setPen(QPen(self.WHITE, thickness, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(tail, head)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.WHITE))
        wing = length * 0.34
        if left_side:
            points = [head, QPointF(head.x() + wing, cy - wing * 0.8),
                      QPointF(head.x() + wing, cy + wing * 0.8)]
        else:
            points = [head, QPointF(head.x() - wing, cy - wing * 0.8),
                      QPointF(head.x() - wing, cy + wing * 0.8)]
        painter.drawPolygon(QPolygonF(points))

    def _paint_text(self, painter, width, height):
        painter.setPen(QPen(self.WHITE))
        size = 420 if self.mode == "countdown" else max(
            22, int(min(width, height) * 0.045))
        painter.setFont(QFont("Arial", size, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, self.text)

    def _paint_image(self, painter, width, height):
        """Quadro do video (camera lenta) ajustado a tela mantendo proporcao."""
        image = self.image
        scale = min(width / image.width(), height / image.height())
        target_w = int(image.width() * scale)
        target_h = int(image.height() * scale)
        painter.drawImage(
            QRectF((width - target_w) / 2, (height - target_h) / 2,
                   target_w, target_h), image)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.on_skip is not None:
                self.on_skip()
        elif event.key() == Qt.Key_Escape:
            if self.on_abort is not None:
                self.on_abort()
        else:
            super().keyPressEvent(event)


class StimulusWindow(QMainWindow):
    """Janela do participante (pode ir a tela cheia) com a tela de estimulos."""

    def __init__(self, full_screen=False, on_close=None, on_skip=None,
                 on_abort=None):
        super().__init__()
        self.setWindowTitle("Paradigma ME/MI - sessao de gravacao")
        self.canvas = _StimulusCanvas()
        self.canvas.on_skip = on_skip
        self.canvas.on_abort = on_abort
        self.setCentralWidget(self.canvas)
        self.resize(900, 900)
        if full_screen:
            self.showFullScreen()
        self._progress = QLabel("")
        self._progress.setStyleSheet("color: white; background: black;")
        self.statusBar().addPermanentWidget(self._progress)
        self._on_close = on_close
        self.show_home()

    # --- API usada pelo controlador do paradigma ---
    def show_home(self):
        self.canvas.show_home()

    def show_condition(self, object_name, hand):
        self.canvas.show_condition(object_name, hand)

    def show_message(self, text):
        self.canvas.show_message(text)

    def show_countdown(self, number):
        self.canvas.show_countdown(number)

    def show_video_frame(self, image):
        self.canvas.show_video_frame(image)

    def show_trial_progress(self, done, total):
        self._progress.setText(f"Trial {done}/{total}")

    def closeEvent(self, event):
        if self._on_close is not None:
            self._on_close()
        event.accept()


class _ParadigmController(QObject):
    """Maquina de estados do paradigma ME/MI (roda na thread principal do Qt).

    Sequencia por trial (total ~13,25 s + ITI aleatorio 1-2 s):
      fixacao 3 s (768) -> beep em -1 s -> cue 1.25 s (769/770/771)
      -> ME 4 s (778; seta permanece; execucao do braco + fechar a mao)
      -> pausa 1 s (779) -> MI 4 s (780; overlay do Kinect na tela de tracking)
      -> fim MI (781) -> ITI 1-2 s -> proximo trial.
    A calibracao de origem (500/501, 10 s, mao apoiada na mesa) acontece uma
    unica vez antes do primeiro trial.
    """

    def __init__(self, window, marker_queue, state, config, **kwargs):
        super().__init__(**kwargs)
        self._window = window
        self._queue = marker_queue
        self._state = state
        self._cfg = config
        self._generation = 0
        self._finished = False
        # pausas (preparacao, baseline e entre blocos): deadline + ESPACO
        self._pause_on_end = None
        self._pause_end = 0.0
        self._pause_skip = False
        self._pause_prefix = ""
        # navegacao
        self._baseline_steps = []
        self._trial_index = 0
        self._bloco = 0
        self._video_deadline = 0.0
        #: True durante a PREPARACAO (tela com o painel de impedancias).
        self._prep_active = False

    @staticmethod
    def build_trial_list(trials_por_bloco, blocos, rng):
        """Lista balanceada das 6 condicoes (objeto x mao) da sessao inteira.

        250 trials / 6 condicoes = 41 de cada + 4 extras (42 para as primeiras),
        embaralhadas; o controlador corta em blocos de `trials_por_bloco`.
        """
        total = int(trials_por_bloco) * int(blocos)
        condicoes = [(obj, mao) for obj in OBJECTS for mao in HANDS]
        base, sobra = divmod(total, len(condicoes))
        extras = list(condicoes)
        rng.shuffle(extras)          # quem recebe os extras e' sorteado
        sorteio = condicoes * base + extras[:sobra]
        rng.shuffle(sorteio)
        return [{"objeto": obj, "mao": mao, "condicao": f"{obj}_{mao}"}
                for obj, mao in sorteio]

    # ------------------------------------------------------------ controle
    def start(self, delay_ms=300):
        self._generation += 1
        self._finished = False
        QTimer.singleShot(int(delay_ms), self._begin_preparation)

    def skip(self):
        """ESPACO do experimentador: encerra a pausa/espera atual."""
        if self._pause_on_end is not None:
            self._pause_skip = True

    def abort(self):
        print("[paradigma] sessao ABORTADA pelo experimentador (ESC)",
              flush=True)
        self.stop()
        QApplication.quit()

    def stop(self):
        self._generation += 1
        self._pause_on_end = None

    def _emit(self, code, **extra):
        """Publica um evento (codigo + legenda + instante + metadados)."""
        evento = {
            "code": code,
            "nome": CODE_NAMES.get(code, "desconhecido"),
            "hora": time.strftime("%H:%M:%S"),
            "monotonic_s": round(time.monotonic(), 6),
        }
        evento.update(extra)
        self._queue.put(evento)

    # ------------------------------------------------------------- pausas
    def _start_pause_phase(self, seconds, prefix, on_end):
        """Pausa com contagem na tela (ESPACO pula). Usada na preparacao,
        no baseline e entre blocos."""
        self._pause_end = time.monotonic() + float(seconds)
        self._pause_skip = False
        self._pause_prefix = prefix
        self._pause_on_end = on_end
        self._pause_tick()

    def _pause_tick(self):
        if self._pause_on_end is None:
            return
        remaining = self._pause_end - time.monotonic()
        if remaining <= 0 or self._pause_skip:
            callback = self._pause_on_end
            self._pause_on_end = None
            callback()
            return
        minutos, segundos = divmod(int(remaining) + 1, 60)
        if self._pause_prefix:
            self._window.show_message(
                f"{self._pause_prefix}\n\n{minutos:02d}:{segundos:02d}\n\n"
                "(ESPACO pula)")
        self._schedule(250, self._pause_tick)

    # --- abertura da sessao ---
    def _begin_preparation(self):
        """Fase de PREPARACAO (~10 min): EEG, Kinect e sinal; ESPACO inicia.

        A tela mostra o PAINEL DE IMPEDANCIAS ao vivo (#23/#24): o
        experimentador acompanha canal a canal e so' inicia a sessao (ESPACO)
        quando todos os eletrodos estao dentro do limite.
        """
        if not self._cfg.get("preparacao", True):
            self._begin_session()
            return
        self._prep_active = True
        self._prep_tick()
        print("[paradigma] PREPARACAO: aguardando ESPACO para iniciar "
              "(painel de impedancias ativo)", flush=True)
        self._start_pause_phase(1e9, "", self._begin_session)

    def _priming_file(self, indice):
        """Caminho do arquivo do clipe de priming da trial (#7) ou None.

        Nome: <pasta_priming>/tNNN_<condicao>.avi -- o indice e' o numero da
        trial na sessao (1-based), o que permite cruzar o clipe com o JSON de
        eventos e com o CSV de EEG sem ambiguidade.
        """
        if not self._cfg.get("salvar_priming", True):
            return None
        pasta = self._cfg.get("priming_dir") or ""
        if not pasta:
            return None
        trial = self._cfg["trials"][indice]
        return os.path.join(pasta,
                            f"t{indice + 1:03d}_{trial['condicao']}.avi")

    def _prep_tick(self):
        """Refresca a tela de preparacao com a tabela de impedancias atual."""
        if not getattr(self, "_prep_active", False):
            return
        _valores, texto = self._state.impedance_snapshot()
        if not texto:
            texto = "(medindo impedancias...)"
        self._window.show_message(
            "PREPARACAO\n\n"
            "1) Eletrodos / impedancia (abaixo)\n"
            "2) Calibracao do Kinect\n"
            "3) Teste de sinal\n\n"
            f"{texto}\n\n"
            "ESPACO inicia a sessao")
        self._schedule(1000, self._prep_tick)

    def _begin_session(self):
        """Baseline (olhos abertos, fechados e repouso ativo) -> origem."""
        self._prep_active = False
        callback = self._cfg.get("on_session_start")
        if callback is not None:
            try:
                callback()
            except Exception as exc:
                print(f"[paradigma] aviso: on_session_start falhou ({exc})",
                      flush=True)
        cfg = self._cfg
        self._emit(CODE_SESSION)
        print(f"[paradigma] sessao iniciada | {len(cfg['trials'])} trials | "
              f"{cfg['blocos']} blocos de {cfg['trials_por_bloco']} | "
              f"trial {cfg['home_sec'] + cfg['me_sec'] + cfg['video_sec'] + cfg['mi_sec']:g} s",
              flush=True)
        self._baseline_steps = []
        if cfg["olhos_abertos"] > 0:
            self._baseline_steps.append(
                ("olhos ABERTOS (olhe o ponto de fixacao)",
                 CODE_EYES_OPEN, cfg["olhos_abertos"]))
        if cfg["olhos_fechados"] > 0:
            self._baseline_steps.append(
                ("olhos FECHADOS (relaxe, sem dormir)",
                 CODE_EYES_CLOSED, cfg["olhos_fechados"]))
        if cfg["repouso_ativo"] > 0:
            self._baseline_steps.append(
                ("repouso ATIVO (maos na posicao inicial)",
                 CODE_BASELINE_REST, cfg["repouso_ativo"]))
        self._run_next_baseline()

    def _run_next_baseline(self):
        if not self._baseline_steps:
            self._origin_calibration()
            return
        label, code, seconds = self._baseline_steps.pop(0)
        self._emit(code)
        self._state.set_phase(PHASE_IDLE, cue="", message=label)
        print(f"[paradigma] baseline: {label} por {seconds:g} s", flush=True)
        self._start_pause_phase(seconds, f"BASELINE\n{label}",
                                self._run_next_baseline)

    # --- calibracao de origem (mao apoiada na mesa = 0,0,0) ---
    def _origin_calibration(self):
        # Se os elos do braco ainda nao estao calibrados (primeira sessao ou
        # modelo apagado), aproveita os 10 s da origem para medi-los: e o
        # momento em que a mao fica mais parada na mesa.
        need_arm = self._state.needs_arm_calibration
        self._emit(CODE_ORIGIN_START)
        self._state.set_phase(PHASE_ORIGIN, message="Mao parada na mesa")
        self._state.start_origin_collection(ORIGIN_CAL_SEC, measure_arm=need_arm)
        mensagem = (f"Calibracao da origem ({ORIGIN_CAL_SEC:g} s)\n"
                    "Apóie a mao PARADA na mesa")
        if need_arm:
            mensagem += "\n(medindo tambem os elos do braco)"
        self._window.show_message(mensagem)
        print(f"[paradigma] calibracao de origem: {ORIGIN_CAL_SEC:g} s"
              + (" (medindo elos do braco)" if need_arm else ""), flush=True)
        self._schedule(ORIGIN_CAL_SEC * 1000, self._origin_done)

    def _origin_done(self):
        ok = self._state.finish_origin_collection()
        origin_xyz, origin_rpy = self._state.origin_snapshot()
        self._emit(CODE_ORIGIN_END)
        self._window.show_message("Origem calibrada. Relaxe a mao.")
        if origin_xyz is None:
            print("[paradigma] AVISO: origem nao definida; os valores KT_* "
                  "no CSV ficarao ABSOLUTOS (ver JSON de eventos)")
        else:
            print(f"[paradigma] origem: xyz={np.round(origin_xyz, 4)} "
                  f"rpy={np.round(origin_rpy, 2)} (ok={ok})")
        self._schedule(2000, self._countdown)

    def _countdown(self):
        n = self._cfg["contagem"]
        self._window.show_message("Prepare-se...")
        for k in range(n):
            self._schedule(k * 1000, lambda k=k: self._window.show_countdown(n - k))
        self._schedule(n * 1000, lambda: self._run_trial(0))

    # --- trial (12 s: home 2 + cue/ME 4 + video 2 + MI 4) ---
    def _run_trial(self, index):
        cfg = self._cfg
        total = len(cfg["trials"])
        if index >= total:
            self._finish()
            return
        self._trial_index = index
        trial = cfg["trials"][index]
        self._window.show_trial_progress(index + 1, total)
        self._window.show_home()
        self._state.set_phase(PHASE_HOME, cue="")
        self._emit(CODE_TRIAL, bloco=self._bloco + 1, trial=index + 1,
                   objeto=trial["objeto"], mao=trial["mao"],
                   condicao=trial["condicao"])
        print(f"[paradigma] trial {index + 1}/{total} ({trial['condicao']})",
              flush=True)
        # REPOUSO (mao no home) -> CUE + ME
        self._schedule(cfg["home_sec"] * 1000,
                       lambda: self._start_cue_me(trial))

    def _start_cue_me(self, trial):
        """CUE na tela + inicio da execucao motora (mesma imagem volta na MI)."""
        cfg = self._cfg
        objeto, mao = trial["objeto"], trial["mao"]
        self._window.show_condition(objeto, mao)
        self._state.set_phase(PHASE_ME, cue=trial["condicao"])
        # Codigos: mao no onset do cue; ME e condicao logo depois. Os offsets
        # evitam colisao na coluna Marker (um codigo por amostra).
        self._emit(CODE_HAND[mao], objeto=objeto, mao=mao,
                   condicao=trial["condicao"], bloco=self._bloco + 1,
                   trial=self._trial_index + 1)
        self._schedule(50, lambda: self._emit(CODE_ME_START))
        self._schedule(100, lambda: self._emit(
            CODE_CONDITION[(objeto, mao)], objeto=objeto, mao=mao,
            condicao=trial["condicao"], bloco=self._bloco + 1,
            trial=self._trial_index + 1))
        self._schedule(cfg["me_sec"] * 1000, self._start_video_pause)

    def _start_video_pause(self):
        """Fim da ME -> PAUSA com o video do movimento em camera lenta (0,5x)."""
        cfg = self._cfg
        self._emit(CODE_ME_END)
        self._emit(CODE_SLOWMO_START)
        self._state.set_phase(PHASE_VIDEO)
        if cfg["video_sec"] <= 0:
            self._start_mi()
            return
        self._state.request_slowmo(
            destino=self._priming_file(self._trial_index))
        print("[paradigma] pausa + video em camera lenta (0,5x)", flush=True)
        self._video_deadline = time.monotonic() + cfg["video_sec"]
        self._schedule(int(1000 / SLOWMO_CAPTURE_FPS), self._video_tick)

    def _video_tick(self):
        """Toca o clipe em 0,5x (o tracking ja entrega os quadros duplicados)."""
        frame = self._state.take_video_frame()
        if frame is not None:
            self._window.show_video_frame(self._to_qimage(frame))
        if time.monotonic() >= self._video_deadline:
            self._state.clear_video()
            self._start_mi()
            return
        self._schedule(int(1000 / SLOWMO_CAPTURE_FPS), self._video_tick)

    @staticmethod
    def _to_qimage(frame):
        """ndarray BGR (Kinect) -> QImage (copia: o buffer pode ser reusado)."""
        height, width = frame.shape[:2]
        if frame.ndim == 3 and frame.shape[2] == 3:
            image = QImage(frame.data, width, height, 3 * width,
                           QImage.Format_BGR888)
        else:
            image = QImage(frame.data, width, height, 4 * width,
                           QImage.Format_ARGB32)
        return image.copy()

    def _start_mi(self):
        """MI: o MESMO cue permanece na tela; sem video/overlay (imaginacao)."""
        self._emit(CODE_MI_START)
        condicao = self._cfg["trials"][self._trial_index]["condicao"]
        self._state.set_phase(PHASE_MI, cue=condicao)
        self._schedule(self._cfg["mi_sec"] * 1000, self._end_mi)

    def _end_mi(self):
        cfg = self._cfg
        self._emit(CODE_MI_END)
        self._state.set_phase(PHASE_IDLE, cue="")
        self._window.show_home()
        proximo = self._trial_index + 1
        if proximo < len(cfg["trials"]) \
                and proximo % cfg["trials_por_bloco"] == 0:
            self._end_block(proximo)
        else:
            self._run_trial(proximo)

    def _end_block(self, index):
        """Fim do bloco -> pausa de 2 min (ESPACO pula) -> proximo bloco."""
        self._emit(CODE_BLOCK_END)
        self._emit(CODE_PAUSE_START)
        self._window.show_message("PAUSA entre blocos")
        bloco = index // self._cfg["trials_por_bloco"]
        print(f"[paradigma] bloco {bloco}/{self._cfg['blocos']} concluido "
              f"({index} trials) | pausa de "
              f"{self._cfg['pausa_bloco_sec']:g} s", flush=True)
        self._start_pause_phase(
            self._cfg["pausa_bloco_sec"],
            f"PAUSA\n\nBloco {bloco} de {self._cfg['blocos']} concluido",
            lambda: self._finish_block_pause(index))

    def _finish_block_pause(self, index):
        self._emit(CODE_PAUSE_END)
        self._start_block(index // self._cfg["trials_por_bloco"])

    def _start_block(self, bloco):
        self._bloco = bloco
        self._emit(CODE_BLOCK_START, bloco=bloco + 1,
                   trials_por_bloco=self._cfg["trials_por_bloco"])
        print(f"[paradigma] bloco {bloco + 1}/{self._cfg['blocos']}",
              flush=True)
        self._run_trial(bloco * self._cfg["trials_por_bloco"])

    def _finish(self):
        if self._finished:
            return
        self._finished = True
        self._emit(CODE_SESSION_END)
        self._window.show_message("Sessao concluida.\nObrigado!")
        print("[paradigma] sessao concluida", flush=True)
        self._schedule(3000, QApplication.quit)


    def _schedule(self, delay_ms, fn):
        generation = self._generation

        def wrapped():
            if generation == self._generation:
                fn()

        QTimer.singleShot(int(delay_ms), wrapped)


# =============================================================================
# Nos da pipeline (muxer + writer)
# =============================================================================
class CountsToMicrovolts(IONode):
    """Converte os counts crus do ADC do g.Nautilus para microvolts (uV).

    O g.Nautilus entrega counts de um ADC de 24 bits e o no GNautilus do
    g.Pype e' pass-through (nao aplica escala). A conversao do fabricante e':

        uV = (count * Factor + Offset) * (sensibilidade / 2**24) / 1000

    onde a "sensibilidade" do aparelho esta em nV (fundo de escala do ADC),
    logo o LSB vale sensibilidade/2**24 nV, e Factor/Offset vem de
    get_scaling() (calibracao de ganho/offset do proprio aparelho). Com
    sensibilidade = 2250000 nV o LSB vale 0.134 nV/count, ou seja ~7.5
    counts/uV: as colunas EEG_ChNN do CSV passam a ficar em uV (faixa tipica
    de EEG ~1-100 uV) e podem ser lidas direto por qualquer ferramenta.
    """

    def __init__(self, sensitivity_nv=2250000.0, factor=1.0, offset=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._sensitivity_nv = float(sensitivity_nv)
        self._lsb_uv = self._sensitivity_nv / float(2 ** 24) / 1000.0
        self._factor = float(factor)
        self._offset = float(offset)

    def setup(self, data, port_context_in):
        port_context_out = super().setup(data, port_context_in)
        ctx = port_context_in[PORT_IN]
        canais = int(ctx[Constants.Keys.CHANNEL_COUNT])
        print(f"[escala] counts -> uV | sensibilidade={self._sensitivity_nv:.0f} "
              f"nV | LSB={self._lsb_uv * 1000:.4f} nV/count "
              f"({self._lsb_uv ** -1:.2f} counts/uV) | "
              f"Factor={self._factor:.6f} Offset={self._offset:.3f} | "
              f"{canais} canais", flush=True)
        return port_context_out

    def step(self, data):
        block = data.get(PORT_IN)
        if block is None or len(block) == 0:
            return None
        convertido = (np.asarray(block, dtype=np.float64) * self._factor
                      + self._offset) * self._lsb_uv
        return {PORT_OUT: convertido}


def _relative3(vector, origin):
    """Tripla (m) relativa a origem; NaN onde nao houver medida."""
    values = np.asarray(vector, np.float64).reshape(3)
    if origin is not None:
        values = values - np.asarray(origin, np.float64).reshape(3)
    return [float(v) for v in values]


def _imu_do_lado(motion, lado, orpy):
    """Le um lado (1 = direita, 2 = esquerda) do dicionario de movimento.

    Dois IMUs: `motion["imu"][lado] = {"rpy_deg": (r,p,y), "accel_g": (x,y,z),
    "pos_m": (x,y,z), "valid": 1, "zero_lock": 0}`; o `orpy_deg` do PROPRIO lado
    (opcional) e' usado na zeragem de orientacao -- cada sensor e' montado com um
    offset diferente.

    Formato antigo (um IMU): as chaves legadas (roll/pitch/yaw, imu_pos,
    zero_lock) sao atribuidas ao lado da MAO ATIVA (`motion["hand"]`); o outro
    lado sai com NaN e valid=0.
    """
    por_lado = motion.get("imu") or {}
    if lado in por_lado:
        dados = dict(por_lado[lado])
        rpy = tuple(float(v) for v in
                    np.asarray(dados.get("rpy_deg", _nan3()),
                               np.float64).reshape(3))
        referencia = dados.get("orpy_deg", orpy)
        if referencia is not None and np.isfinite(rpy).all():
            rpy = tuple(_wrap180(valor - base) for valor, base
                        in zip(rpy, np.asarray(referencia, np.float64)))
        return {"rpy": rpy,
                "accel_g": dados.get("accel_g", _nan3()),
                "pos": dados.get("pos_m", _nan3()),
                "valid": dados.get("valid", 0),
                "zero_lock": dados.get("zero_lock", 0)}
    ativo = int(motion.get("hand", 0) or 0)
    if ativo == 0:
        # Um so' IMU e mao desconhecida: usa o lado DECLARADO da sessao
        # (`imu_lado`, padrao 1 = direita). Nunca duplicar o mesmo sensor nos
        # dois blocos -- isso seria dado mal rotulado.
        ativo = int(motion.get("imu_lado", 1) or 1)
    if ativo == int(lado):            # legado: um IMU, atribuido a mao ativa
        rpy = (motion["roll"], motion["pitch"], motion["yaw"])
        if orpy is not None and np.isfinite(np.asarray(rpy, np.float64)).all():
            rpy = tuple(_wrap180(valor - base) for valor, base
                        in zip(rpy, np.asarray(orpy, np.float64)))
        return {"rpy": rpy,
                "accel_g": motion.get("imu_accel_g", _nan3()),
                "pos": motion.get("imu_pos", _nan3()),
                "valid": motion.get("valid", 0),
                "zero_lock": motion.get("zero_lock", 0)}
    return {"rpy": (np.nan,) * 3, "accel_g": _nan3(), "pos": _nan3(),
            "valid": 0, "zero_lock": 0}


def _bloco_imu(motion, sufixo, origin, orpy):
    """11 valores do lado `sufixo` ("L" | "R"), na ordem de _imu_block()."""
    lado = 1 if sufixo == "R" else 2
    dados = _imu_do_lado(motion, lado, orpy)
    return (list(dados["rpy"])
            + [float(valor) for valor in
               np.asarray(dados["accel_g"], np.float64).reshape(3)]
            + _relative3(dados["pos"], origin)
            + [float(dados["valid"]), float(dados["zero_lock"])])


def build_motion_row(motion, origin, orpy):
    """Linha de movimento na ORDEM de MOTION_COLUMNS.

    Compartilhada entre o RecordingMuxer (quando o movimento vai embutido no
    CSV de EEG) e o MotionRecorder (arquivo proprio a MOTION_HZ), garantindo
    que os dois caminhos gerem exatamente as mesmas colunas.

    KT_* e ARM_* (posicoes) sao relativos a origem; IMU_* e' relativo a origem
    de orientacao; angulos articulares do braco e versores da palma sao
    absolutos (transladar a origem nao muda angulo nem versor).
    """
    x, y, z = (motion["x"], motion["y"], motion["z"])
    if origin is not None:
        x, y, z = x - origin[0], y - origin[1], z - origin[2]
    # Ordem: KT (3) + bloco de CADA lado (11 + 11) + IMU_hand (1) + KTT_valid (1).
    # A zeragem de orientacao de cada lado e' feita em `_imu_do_lado` (cada
    # sensor tem o seu proprio offset de montagem).
    row = [x, y, z]
    row += _bloco_imu(motion, "L", origin, orpy)
    row += _bloco_imu(motion, "R", origin, orpy)
    row += [float(motion.get("hand", 0)), float(motion["valid"])]
    row += _relative3(motion["arm_shoulder"], origin)
    row += _relative3(motion["arm_elbow"], origin)
    row += _relative3(motion["arm_wrist"], origin)
    row += [
        motion["arm_elbow_angle"], motion["arm_shoulder_elev"],
        motion["arm_shoulder_azim"], motion["arm_upper_len"],
        motion["arm_fore_len"], float(motion["arm_valid"]),
        float(motion["arm_ik"]), float(motion["arm_clamped"]),
        float(motion["arm_side"]),
    ]
    row += [float(v) for v in np.asarray(motion["palm_normal"]).reshape(3)]
    row += [float(v) for v in np.asarray(motion["palm_dir"]).reshape(3)]
    row += [
        motion["palm_elev"], motion["palm_azim"], motion["palm_pitch"],
        motion["palm_yaw"], float(motion["palm_valid"]),
    ]
    # Lateralidade da mao usada na amostra (0/1/2) e a FONTE da posicao 3D
    # (1=triangulado, 2=depth do Kinect, 3=esqueleto SDK, 4=modelo, ...).
    row += [float(motion.get("hand", 0)), float(motion.get("src_code", 0))]
    # 1 na(s) amostra(s) em que o INICIO DO MOVIMENTO foi detectado (item do
    # protocolo: ancora da janela de planejamento motor e do clipe de priming).
    row += [float(motion.get("onset", 0))]
    return row


class RecordingMuxer(IONode):
    """Adiciona colunas de movimento (KT_*/IMU_*/ARM_*/PALM_*/valid) e Marker.

    Entrada unica de EEG (pipeline 100% sincrona). A cada amostra:
      - drena a fila de eventos (thread-safe, preenchida pelo controlador Qt)
        e fixa o codigo na coluna "Marker" por hold_s (precisao de 1 amostra
        com frame_size=1);
      - le o estado compartilhado da thread de tracking (zero-order hold: o
        ultimo valor do Kinect/ESP32 e repetido ate chegar medida nova) e
        acrescenta, na ordem de MOTION_COLUMNS:
          * KT_x/y/z  = punho 3D (m) RELATIVO a origem;
          * IMU_roll/pitch/yaw = ESP32, RELATIVOS a origem;
          * KTT_valid = 1 se a posicao 3D e do frame atual;
          * ARM_*  = braco (ombro/cotovelo/punho) com cinematica inversa de
            elos rigidos: posicoes RELATIVAS a origem, angulos articulares
            (invariantes a translacao, logo absolutos), elos efetivos em m e
            diagnostico da IK;
          * PALM_* = orientacao da palma pelo Kinect (normal + direcao dos
            dedos, vetores unitarios em camera space, ABSOLUTOS: a origem de
            translacao nao define um referencial de rotacao do Kinect);
      - em stop(), grava o JSON *_eventos.json com a amostra exata de cada
        evento e a origem medida.
    """

    def __init__(self, marker_queue, state, hold_s=1.0, fs=FS,
                 events_path=None, write_motion=MOTION_IN_EEG_CSV, meta=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._queue = marker_queue
        self._state = state
        # False (padrao): o CSV de EEG sai com EEG + Marker apenas; o
        # movimento vai para o MotionRecorder (arquivo a MOTION_HZ).
        self._write_motion = bool(write_motion)
        self._n_motion = N_MOTION_COLS if self._write_motion else 0
        self._hold = max(1, int(round(hold_s * fs)))
        self._fs = float(fs)
        self._events_path = events_path
        # Metadados da sessao (#2): identidade, unidades, versoes, hashes das
        # calibracoes e ordem dos trials. Vao para o JSON de eventos.
        self._meta = dict(meta or {})
        self._counter = 0
        self._marker = 0
        self._remaining = 0
        self._channel_count = None
        self._events = []
        self._pending = []               # eventos adiados por colisao na amostra
        self._last_status = 0.0

    def setup(self, data, port_context_in):
        ctx = port_context_in[PORT_IN]
        if int(ctx[Constants.Keys.FRAME_SIZE]) != 1:
            raise ValueError("RecordingMuxer exige frame_size=1 na fonte "
                             "(use --frame-size 1).")
        self._channel_count = int(ctx[Constants.Keys.CHANNEL_COUNT])
        self._counter = 0
        port_context_out = super().setup(data, port_context_in)
        # O muxer acrescenta as colunas de movimento (opcional) + 1 Marker: o
        # contexto de saida precisa anunciar channel_count + n_motion + 1.
        self._n_motion = N_MOTION_COLS if self._write_motion else 0
        for context in port_context_out.values():
            context[Constants.Keys.CHANNEL_COUNT] = (
                self._channel_count + self._n_motion + 1)
        if self._write_motion:
            print(f"[RecordingMuxer] {self._channel_count} canais de EEG + "
                  f"{N_MOTION_COLS} de movimento ({len(KT_COLUMNS)} KT + "
                  f"{len(IMU_COLUMNS)} IMU + 1 valid + {len(ARM_COLUMNS)} ARM + "
                  f"{len(PALM_COLUMNS)} PALM) + 1 Marker "
                  f"(codigo retido por {self._hold} amostras)", flush=True)
        else:
            print(f"[RecordingMuxer] {self._channel_count} canais de EEG + "
                  f"1 Marker (codigo retido por {self._hold} amostras) | "
                  f"movimento em arquivo separado a {MOTION_HZ:g} Hz",
                  flush=True)
        return port_context_out

    @property
    def samples_written(self):
        """Amostras de EEG ja processadas (usado pelo LinkWatchdog/#15)."""
        return self._counter

    @property
    def current_marker(self):
        """Codigo de marcador vigente (usado pelo MarkerFanout do escopo)."""
        return self._marker

    def _apply_event(self, event):
        """Aplica um evento: registra no JSON e fixa o codigo no Marker.

        A coluna Marker guarda UM codigo por amostra. Se dois eventos cairem
        na MESMA amostra, o segundo e adiado (self._pending) e aplicado na
        amostra seguinte: assim o codigo anterior nao e engolido em silencio
        (ex.: cue 769/770 perdido porque a ME 778 chegou no mesmo instante).
        """
        self._events.append(event)
        self._marker = int(event["code"])
        self._remaining = self._hold
        print(f"[marca] {event['code']} ({event['nome']}) "
              f"@ amostra {event['sample_index']}", flush=True)

    def step(self, data):
        block = data.get(PORT_IN)
        if block is None or len(block) == 0:
            return None
        n = block.shape[0]
        # --- eventos (drain) ---
        # 1) um evento adiado por colisao entra QUANDO o hold do anterior
        #    termina (self._remaining <= 0): cada codigo preserva a janela
        #    inteira no Marker, sem ser engolido pelo proximo.
        if self._pending and self._remaining <= 0:
            event = self._pending.pop(0)
            event["sample_index"] = self._counter
            self._apply_event(event)
        # 2) fila nova
        try:
            while True:
                event = self._queue.get_nowait()
                event["sample_index"] = self._counter
                if (self._events
                        and self._events[-1]["sample_index"] == self._counter):
                    # mesmo indice que o anterior: espera o hold terminar
                    self._pending.append(event)
                else:
                    self._apply_event(event)
        except queue.Empty:
            pass
        marker_column = np.zeros((n, 1), dtype=block.dtype)
        for i in range(n):
            if self._remaining > 0:
                marker_column[i, 0] = self._marker
                self._remaining -= 1
                if self._remaining == 0:
                    self._marker = 0
            else:
                self._marker = 0
        # --- movimento (zero-order hold + origem) ---
        motion, _phase, _cue, _msg = self._state.snapshot()
        origin = self._state.origin_xyz if self._state.origin_ready else None
        orpy = self._state.origin_rpy if self._state.origin_ready else None
        self._counter += n
        self._console_status(motion)
        if not self._write_motion:
            return {PORT_OUT: np.column_stack((block, marker_column))}
        motion_row = np.array(build_motion_row(motion, origin, orpy),
                              dtype=block.dtype)
        motion_columns = np.tile(motion_row, (n, 1))
        return {PORT_OUT: np.column_stack(
            (block, motion_columns, marker_column))}

    def _console_status(self, motion, period=2.0):
        """Status periodico no console (EEG/movimento/braco/palma/validade)."""
        now = time.monotonic()
        if now - self._last_status < period:
            return
        self._last_status = now
        print(f"[rec] amostra={self._counter} "
              f"KT=({motion['x']:+.3f},{motion['y']:+.3f},{motion['z']:+.3f}) "
              f"[{motion['src'] or '-'}] valid={motion['valid']} "
              f"IMU=({motion['roll']:.1f},{motion['pitch']:.1f},"
              f"{motion['yaw']:.1f}) "
              f"ARM(ik={motion['arm_ik']},lado={motion['arm_side']},"
              f"cot={motion['arm_elbow_angle']:.1f}deg,"
              f"clamp={motion['arm_clamped']}) "
              f"PALM(el={motion['palm_elev']:.0f},az={motion['palm_azim']:.0f},"
              f"pit={motion['palm_pitch']:.0f},yaw={motion['palm_yaw']:.0f})",
              flush=True)

    def stop(self):
        if self._events_path and self._events:
            origin_xyz, origin_rpy = self._state.origin_snapshot()
            payload = {
                "fs": self._fs,
                "canais_eeg": self._channel_count,
                "movimento_no_csv_de_eeg": bool(self._write_motion),
                "colunas_movimento": MOTION_COLUMNS,
                "meta": self._meta,
                "hold_samples": self._hold,
                "legenda_codigos": CODE_NAMES,
                "origem_xyz_m": (None if origin_xyz is None
                                 else origin_xyz.tolist()),
                "origem_rpy_deg": (None if origin_rpy is None
                                   else origin_rpy.tolist()),
                "nota_origem": ("KT_*/IMU_* sao RELATIVOS a esta origem "
                                "apos o evento 501; antes disso sao "
                                "absolutos. Origem = mao apoiada na mesa."),
                "modelo_braco": self._state.arm_model_info,
                "nota": ("sample_index = indice (0-based) da primeira linha "
                         "do CSV em que o codigo aparece na coluna Marker"),
                "numero_de_eventos": len(self._events),
                "eventos": self._events,
            }
            try:
                with open(self._events_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                print(f"[RecordingMuxer] {len(self._events)} eventos salvos "
                      f"em {self._events_path}", flush=True)
            except OSError as exc:
                print(f"[RecordingMuxer] falha ao salvar eventos: {exc}",
                      flush=True)
        self._events = []
        return super().stop()


def _wrap180(angle_deg):
    """Normaliza angulo para (-180, 180]."""
    value = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return value if value > -180.0 else 180.0


class MarkerFanout(IONode):
    """Acrescenta a coluna do marcador (valor atual do muxer) a um bloco EEG.

    O CSV grava o sinal CRU, mas o escopo de monitoramento deve mostrar a EEG
    FILTRADA com as linhas de evento. Este no copia o codigo corrente do
    RecordingMuxer para a ultima coluna, sem drenar a fila de eventos (quem
    drena e' o muxer, que e' tambem quem grava o CSV cru).
    """

    def __init__(self, muxer, **kwargs):
        super().__init__(**kwargs)
        self._muxer = muxer
        self._channels = None

    def setup(self, data, port_context_in):
        context = port_context_in[PORT_IN]
        self._channels = int(context[Constants.Keys.CHANNEL_COUNT])
        port_context_out = super().setup(data, port_context_in)
        for out_context in port_context_out.values():
            out_context[Constants.Keys.CHANNEL_COUNT] = self._channels + 1
        return port_context_out

    def step(self, data):
        block = data.get(PORT_IN)
        if block is None or len(block) == 0:
            return None
        n = block.shape[0]
        marker = np.full((n, 1), float(self._muxer.current_marker),
                         dtype=block.dtype)
        return {PORT_OUT: np.column_stack((block, marker))}


def _fmt_num(value):
    """Formata um numero para o CSV de movimento (compacto; 'nan' se invalido)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.6g}"


class MotionRecorder(threading.Thread):
    """Grava o movimento (KT_*/IMU_*/ARM_*/PALM_*) em CSV proprio a MOTION_HZ.

    O EEG roda a 500 Hz e as cameras entregam ~30 Hz: repetir as 40 colunas de
    movimento em cada amostra de EEG era o maior consumidor de disco do
    projeto sem ganho de informacao (a banda do movimento e' ~15 Hz, logo
    30 Hz ja satisfaz Nyquist). Aqui o movimento e' amostrado a 30 Hz
    (~7 kB/s = ~25 MB/h), ficando independente da taxa do EEG.

    Alinhamento com o EEG: o JSON de eventos traz, para cada evento, o
    (sample_index) no EEG e o (monotonic_s) do mesmo instante; dois eventos
    definem a reta amostra <-> relogio de parede. A coluna t_mono_s deste
    arquivo usa o mesmo relogio (time.monotonic).
    """

    def __init__(self, state, path, hz=MOTION_HZ, fs=FS):
        super().__init__(daemon=True)
        self._state = state
        self._path = str(path)
        self._dt = 1.0 / max(1e-3, float(hz))
        self._fs = float(fs)
        self._stop = threading.Event()
        self._rows = 0
        self._t0_mono = None
        self._t0_wall = None
        self.error = None

    @property
    def rows(self):
        """Linhas efetivamente gravadas (0 se nada foi escrito)."""
        return self._rows

    @property
    def path(self):
        return self._path

    def run(self):
        try:
            with open(self._path, "w", encoding="utf-8", newline="") as handle:
                # Tempos ABSOLUTOS, no mesmo relogio do JSON de eventos (que
                # grava monotonic_s em cada evento): e' isso que permite
                # alinhar o movimento de 30 Hz com a EEG de 500 Hz depois.
                handle.write("t_mono_s,t_epoch_s," + ",".join(MOTION_COLUMNS)
                             + "\n")
                print(f"[movimento] gravando {os.path.basename(self._path)} a "
                      f"{1.0 / self._dt:.0f} Hz ({N_MOTION_COLS} colunas)",
                      flush=True)
                next_t = time.monotonic()
                while not self._stop.is_set():
                    now = time.monotonic()
                    if now < next_t:
                        time.sleep(min(0.005, next_t - now))
                        continue
                    next_t += self._dt
                    if next_t < now - 1.0:      # atraso grande: realinha
                        next_t = now + self._dt
                    motion, _phase, _cue, _msg = self._state.snapshot()
                    origin, orpy = self._state.origin_snapshot()
                    row = build_motion_row(motion, origin, orpy)
                    handle.write(f"{now:.6f},{time.time():.6f},")
                    handle.write(",".join(_fmt_num(v) for v in row) + "\n")
                    self._rows += 1
        except Exception as exc:            # nunca derruba a sessao
            self.error = exc
            print(f"[movimento] FALHA ao gravar {self._path}: {exc}",
                  flush=True)

    def stop(self):
        self._stop.set()
        self.join(timeout=2.0)


class RecordingCsvWriter(gp.CsvWriter):
    """CsvWriter com cabecalho EEG_ChNN + movimento + coluna "Marker"."""

    def _write_block(self, block, timestamps):
        if self._file_handle is None:
            return
        header = ""
        # Colunas = Time + EEG_ChNN + [movimento, se embutido] + Marker.
        n_motion = N_MOTION_COLS if MOTION_IN_EEG_CSV else 0
        n_eeg = block.shape[1] - n_motion - 1
        if not self._header_written:
            eeg_names = [f"EEG_Ch{i + 1:02d}" for i in range(n_eeg)]
            nomes = eeg_names + (MOTION_COLUMNS if n_motion else []) + ["Marker"]
            header = "Time," + ",".join(nomes)
            self._header_written = True
        full_block = np.column_stack((timestamps, block))
        # Formato compacto: o formato antigo (%.17g em TODAS as colunas)
        # gastava ~1.4 kB por amostra (2.5 GB/hora). Com %.6g o EEG mantem 7
        # digitos significativos (de sobra para uV) e o arquivo cai ~2.5x;
        # %.4f no tempo preserva resolucao de 0.1 ms na coluna Time.
        fmt = ["%.4f"] + ["%.6g"] * n_eeg + ["%.6g"] * n_motion + ["%d"]
        np.savetxt(
            self._file_handle,
            full_block,
            fmt=fmt,
            delimiter=",",
            header=header,
            comments="",
        )


def _impedance_table(values, limite_kohm=IMPEDANCE_LIMIT_KOHM, per_line=8):
    """Tabela de impedancias (kOhm) em linhas curtas; '*' = canal a verificar."""
    if values is None:
        return "(impedancia ainda nao medida)"
    linhas = []
    for inicio in range(0, len(values), per_line):
        partes = []
        for offset, valor in enumerate(values[inicio:inicio + per_line]):
            canal = inicio + offset + 1
            if valor is None or valor < 0:
                partes.append(f"Ch{canal:02d}:  ---  ")
            else:
                marca = " " if valor <= limite_kohm else "*"
                partes.append(f"Ch{canal:02d}:{valor:5.1f}{marca}")
        linhas.append(" ".join(partes))
    bons = sum(1 for v in values if v is not None and 0 <= v <= limite_kohm)
    linhas.append(f"{bons}/{len(values)} canais OK (<= {limite_kohm:g} kOhm); "
                  "'*' = verificar eletrodo, '---' = nao medido")
    return "\n".join(linhas)


class LinkWatchdog(threading.Thread):
    """Vigia do link EEG <-> PC durante a gravacao (#15).

    Se o amplificador parar de entregar amostras (USB, link sem fio, bateria),
    o paradigma continuaria rodando com a EEG congelada e o problema so'
    apareceria no fim da sessao. Este vigia acompanha o contador de amostras do
    muxer: sem amostras novas por LINK_TIMEOUT_SEC ele avisa no console, marca
    o estado (a janela de tracking mostra "LINK EEG PERDIDO") e registra
    CODE_LINK_LOST / CODE_LINK_OK no JSON -- assim o trecho pode ser cortado na
    analise.

    Decisao consciente: NAO pausamos o paradigma automaticamente. Pausar no
    meio de uma trial quebraria o timing do protocolo (a trial ja se perdeu) e
    deixaria a tela em um estado que o participante nao sabe retomar; o
    experimentador e' quem decide abortar ou seguir.
    """

    def __init__(self, muxer, state, marker_queue, timeout=LINK_TIMEOUT_SEC,
                 period=0.2):
        super().__init__(daemon=True)
        self._muxer = muxer
        self._state = state
        self._queue = marker_queue
        self._timeout = float(timeout)
        self._period = float(period)
        self._stop = threading.Event()
        self.perdas = 0                # numero de vezes que o link caiu

    def stop(self):
        self._stop.set()
        self.join(timeout=1.5)

    def _event(self, code):
        return {"code": int(code), "nome": CODE_NAMES.get(code, "?"),
                "hora": time.strftime("%H:%M:%S"),
                "monotonic_s": round(time.monotonic(), 6),
                "origem": "vigia"}

    def run(self):
        ultimo = self._muxer.samples_written
        visto = time.monotonic()
        perdido = False
        while not self._stop.is_set():
            time.sleep(self._period)
            atual = self._muxer.samples_written
            agora = time.monotonic()
            if atual != ultimo:
                ultimo, visto = atual, agora
            if agora - visto > self._timeout:
                if not perdido:
                    perdido = True
                    self.perdas += 1
                    self._state.set_link_status(
                        0, f"sem amostras de EEG ha {agora - visto:.1f} s")
                    self._queue.put(self._event(CODE_LINK_LOST))
                    print(f"[vigia] LINK EEG PERDIDO: nenhuma amostra nova ha "
                          f"{agora - visto:.1f} s. Verifique o cabo/USB do "
                          f"amplificador, a bateria e o console do g.tec. A "
                          f"trial em curso esta perdida.", flush=True)
            elif perdido:
                perdido = False
                self._state.set_link_status(1, "")
                self._queue.put(self._event(CODE_LINK_OK))
                print("[vigia] link EEG recuperado (marcador 898 gravado)",
                      flush=True)


class ImpedanceMonitor(threading.Thread):
    """Painel de impedancias durante a PREPARACAO (#23/#24).

    Mede a impedancia de todos os canais a cada IMPEDANCE_PERIOD_S e publica uma
    tabela pronta para a TELA do paradigma (SharedState.impedance_text) e no
    console. O objetivo e' decidir eletrodo a eletrodo ANTES de comecar: o
    experimentador so' inicia a sessao (ESPACO) quando todos estiverem dentro do
    limite. No console a tabela so' e' impressa quando a LISTA de canais ruins
    muda, para nao inundar o terminal.

    E' parado quando a sessao comeca (callback `on_session_start`), para nao
    ficar conversando com o aparelho durante a gravacao.
    """

    def __init__(self, device, state, limite_kohm=IMPEDANCE_LIMIT_KOHM,
                 period=IMPEDANCE_PERIOD_S):
        super().__init__(daemon=True)
        self._device = device
        self._state = state
        self._limite = float(limite_kohm)
        self._period = float(period)
        self._stop = threading.Event()
        self._primeiro = True
        self._ultimo_ruins = None

    def stop(self):
        self._stop.set()
        self.join(timeout=2.5)

    def run(self):
        if self._device is None:
            self._state.set_impedance(
                None, "(medicao de impedancia indisponivel nesta fonte)")
            return
        while not self._stop.is_set():
            try:
                valores = np.asarray(self._device.get_impedance(self._primeiro),
                                     dtype=float)
                self._primeiro = False
            except Exception as exc:
                self._state.set_impedance(None, f"(falha na medicao: {exc})")
                time.sleep(self._period)
                continue
            texto = _impedance_table(valores, self._limite)
            self._state.set_impedance(valores, texto)
            ruins = tuple(indice + 1 for indice, valor in enumerate(valores)
                          if valor < 0 or valor > self._limite)
            if ruins != self._ultimo_ruins:      # so' fala quando MUDA
                self._ultimo_ruins = ruins
                print(texto, flush=True)
                if not ruins:
                    print("[impedancia] TODOS os canais OK -> pode iniciar a "
                          "sessao (ESPACO)", flush=True)
                else:
                    print(f"[impedancia] canais a verificar: {list(ruins)}",
                          flush=True)
            time.sleep(self._period)


def _ask_participant(args):
    """Questionario do participante no FIM da gravacao (#2).

    Perguntado ao final (e nao no inicio) para nao atrasar a montagem com o
    participante ja' sentado e com o EEG colocado. ENTER mantem o valor padrao;
    Ctrl+C/EOF encerra o questionario sem perder o que ja' foi respondido.
    """
    perguntas = [
        ("participante", "Codigo do participante", args.participante),
        ("sessao", "Sessao (ex.: S1)", args.sessao),
        ("idade", "Idade", ""),
        ("sexo", "Sexo (F/M/outro)", ""),
        ("mao_dominante", "Mao dominante (direita/esquerda)",
         args.mao_dominante),
        ("horas_sono", "Horas de sono na ultima noite", ""),
        ("cafeina", "Cafeina/nicotina nas ultimas 2 h (sim/nao)", ""),
        ("medicamentos", "Medicamentos relevantes", ""),
        ("experiencia_eeg", "Ja' participou de experimento com EEG? (sim/nao)",
         ""),
        ("observacoes", "Observacoes do experimentador", ""),
    ]
    respostas = {}
    print("\n--- Questionario do participante (ENTER mantem o padrao) ---",
          flush=True)
    for chave, texto, padrao in perguntas:
        sufixo = f" [{padrao}]" if padrao else ""
        try:
            resposta = input(f"{texto}{sufixo}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("(questionario interrompido pelo experimentador)", flush=True)
            break
        respostas[chave] = resposta or padrao
    respostas["respondido_em"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return respostas


def _write_participant_file(csv_name, meta, respostas):
    """Grava `<prefixo>_participante.json` (dados do participante + meta)."""
    caminho = os.path.splitext(str(csv_name))[0] + "_participante.json"
    try:
        with open(caminho, "w", encoding="utf-8") as handle:
            json.dump({"participante": respostas, "sessao_meta": meta},
                      handle, ensure_ascii=False, indent=2)
        print(f"  Participante -> {os.path.abspath(caminho)}", flush=True)
    except OSError as exc:
        print(f"[participante] falha ao salvar o questionario: {exc}",
              flush=True)


def _motion_file_name(csv_name):
    """Nome do CSV de movimento derivado do CSV de EEG."""
    return f"{os.path.splitext(str(csv_name))[0]}_movimento.csv"


def _file_sha1(path, chunk=1 << 20):
    """SHA1 curto (16 hex) de um arquivo, ou None se nao existir."""
    try:
        digest = hashlib.sha1()
        with open(path, "rb") as handle:
            while True:
                data = handle.read(chunk)
                if not data:
                    break
                digest.update(data)
        return digest.hexdigest()[:16]
    except OSError:
        return None


def _package_versions():
    """Versoes dos pacotes relevantes (reprodutibilidade da sessao)."""
    import importlib.metadata as metadata
    nomes = ["gpype", "gtec-gds", "mediapipe", "opencv-python", "numpy",
             "scipy", "torch", "PySide6", "pykinect2", "comtypes"]
    versoes = {}
    for nome in nomes:
        try:
            versoes[nome] = metadata.version(nome)
        except Exception:
            versoes[nome] = None
    return versoes


def _build_session_meta(args, trials, source, csv_name, events_name,
                        motion_name, priming_dir=None):
    """Metadados da sessao gravados no JSON de eventos (#2).

    Sem isto, juntar as ~25 sessoes do estudo vira arqueologia: nao se sabe a
    qual participante pertence, em que unidade o EEG esta, com que protocolo
    foi gravado nem quais calibracoes valiam naquele dia (o hash resolve isso).
    """
    import platform
    contagem = {}
    for trial in trials:
        chave = f"{trial['objeto']}_{trial['mao']}"
        contagem[chave] = contagem.get(chave, 0) + 1
    raiz = os.path.dirname(os.path.abspath(__file__))
    calibracoes = {}
    for base in ("camera_alignment.npz", "stereo_calibration.npz",
                 "hand_landmark_calibration.npz", "overlay_trim.json",
                 "arm_model.json"):
        caminho = os.path.join(raiz, base)
        if os.path.exists(caminho):
            calibracoes[base] = _file_sha1(caminho)
    return {
        "protocolo_versao": PROTOCOL_VERSION,
        "participante": args.participante,
        "sessao": args.sessao,
        "montagem": args.montagem,
        "mao_declarada": args.mao_dominante,
        "inicio_data_hora": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inicio_epoch_s": time.time(),
        "inicio_monotonic_s": time.monotonic(),
        "host": platform.node(),
        "fonte": str(source),
        "fs_eeg_hz": args.fs,
        "fs_movimento_hz": args.mov_hz,
        "movimento_no_csv_de_eeg": bool(MOTION_IN_EEG_CSV),
        "unidades": {
            "EEG_ChNN": "uV (counts do ADC convertidos; sinal CRU, sem filtro)",
            "Time": "s (relogio do pipeline g.Pype)",
            "t_mono_s (movimento)": "s (time.monotonic; alinhar pelo JSON)",
            "KT_*": "m (relativo a origem)",
            "IMU_roll/pitch/yaw": "graus (relativo a origem)",
            "ARM_* posicoes": "m (relativo a origem)",
            "ARM_* angulos": "graus (absolutos)",
            "PALM_n*/dir*": "versor unitario (absoluto)",
            "Marker": "codigo inteiro",
        },
        "pacotes": _package_versions(),
        "calibracoes_sha1": calibracoes,
        "arquivos": {"eeg_csv": str(csv_name), "movimento_csv": str(motion_name),
                     "eventos_json": str(events_name),
                     "priming_dir": (str(priming_dir) if priming_dir
                                     else None)},
        "tempos_fase_s": {"home": args.home_sec, "cue_me": args.me_sec,
                          "video": args.video_sec, "mi": args.mi_sec},
        "trials_por_bloco": args.trials_por_bloco,
        "blocos": args.blocos,
        "condicoes_contagem": contagem,
        "ordem_dos_trials": [t["condicao"] for t in trials],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=["generator", "gnautilus"],
                        default="gnautilus")
    parser.add_argument("--mov-hz", type=float, default=MOTION_HZ,
                        help="Taxa do CSV de movimento (arquivo separado).")
    parser.add_argument("--mov-no-eeg-csv", action="store_true",
                        help="Formato antigo: movimento embutido no CSV de EEG.")
    parser.add_argument("--participante", default="P01",
                        help="Identificador do participante (vai no JSON).")
    parser.add_argument("--sessao", default="S1",
                        help="Identificador da sessao (vai no JSON).")
    parser.add_argument("--montagem", default="",
                        help="Montagem de eletrodos (ex.: 'C3,Cz,C4,P3,P4').")
    parser.add_argument("--mao-dominante", default="",
                        help="Mao dominante declarada pelo participante.")
    parser.add_argument("--pasta-sessoes", default="gravacoes",
                        help="Onde gravar as sessoes (CSV de EEG, movimento "
                             "e clipes). Aponte para OUTRO DISCO se quiser "
                             "tirar os dados do OneDrive, ex.: "
                             "'D:\\Mestrado_Dados\\gravacoes'.")
    parser.add_argument("--sem-questionario", action="store_true",
                        help="Nao pergunta os dados do participante no fim.")
    parser.add_argument("--sem-onset", action="store_true",
                        help="Desativa a deteccao do INICIO DO MOVIMENTO "
                             "(marcador 795 + clipe ancorado no onset).")
    parser.add_argument("--sem-painel-impedancia", action="store_true",
                        help="Nao mede/mostra o painel de impedancias na "
                             "preparacao.")
    parser.add_argument("--sem-priming-arquivo", action="store_true",
                        help="Nao grava os clipes do priming em disco "
                             "(mostra na tela, mas nao salva).")
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument("--channel-count", type=int, default=CHANNEL_COUNT)
    parser.add_argument("--frame-size", type=int, default=FRAME_SIZE)
    parser.add_argument("--blocos", type=int, default=BLOCKS,
                        help="Blocos por sessao")
    parser.add_argument("--trials-por-bloco", type=int,
                        default=TRIALS_PER_BLOCK,
                        help="Trials por bloco (o total da sessao e' "
                             "blocos x trials-por-bloco)")
    parser.add_argument("--home-sec", type=float, default=HOME_SEC,
                        help="Repouso com a mao no home (s)")
    parser.add_argument("--me-sec", type=float, default=CUE_ME_SEC,
                        help="Cue + execucao motora (s)")
    parser.add_argument("--video-sec", type=float, default=PAUSE_VIDEO_SEC,
                        help="Pausa + video do movimento em 0,5x (s)")
    parser.add_argument("--mi-sec", type=float, default=MI_SEC,
                        help="Imaginacao motora (s)")
    parser.add_argument("--pausa-bloco-sec", type=float,
                        default=BLOCK_PAUSE_SEC,
                        help="Pausa entre blocos (s; ESPACO pula)")
    parser.add_argument("--olhos-abertos", type=float,
                        default=BASELINE_EYES_OPEN_SEC,
                        help="Baseline olhos abertos (s; 0 desativa)")
    parser.add_argument("--olhos-fechados", type=float,
                        default=BASELINE_EYES_CLOSED_SEC,
                        help="Baseline olhos fechados (s; 0 desativa)")
    parser.add_argument("--repouso-ativo", type=float, default=BASELINE_REST_SEC,
                        help="Baseline repouso ativo (s; 0 desativa)")
    parser.add_argument("--contagem", type=int, default=COUNTDOWN)
    parser.add_argument("--hold-sec", type=float, default=HOLD_SEC)
    parser.add_argument("--sem-video", action="store_true",
                        help="Desativa o video em camera lenta da pausa")
    parser.add_argument("--sem-preparacao", action="store_true",
                        help="Pula a espera da fase de preparacao "
                             "(usado em testes automatizados)")
    parser.add_argument("--tela-cheia", action="store_true")
    parser.add_argument("--sem-kinect", action="store_true",
                        help="Roda sem Kinect (apenas ESP32; KT_* = NaN)")
    parser.add_argument("--sem-ik", action="store_true",
                        help=("Desliga a cinematica inversa de elos rigidos: "
                              "ARM_* vem do esqueleto cru do SDK (sem "
                              "arm_model.json)"))
    parser.add_argument("--aux-cameras", default="0,1",
                        help=("Indices (OpenCV) das webcams auxiliares usadas "
                              "com o Kinect, separados por virgula. Padrao "
                              "'0,1' = webcam do laptop (0) + webcam USB (1)."))
    parser.add_argument("--sem-zeroing", action="store_true",
                        help=("Desativa a zeragem do IMU pelas cameras "
                              "(colunas IMU_*_pos_* ficam NaN e ZERO_lock_*=0)"))
    parser.add_argument("--imu-lado", type=int, default=1,
                        choices=(1, 2),
                        help=("Lado do UNICO IMU quando a sessao tem so' um "
                              "(1 = direita, 2 = esquerda). Com dois IMUs "
                              "(um por mao) este valor e' ignorado: cada lado "
                              "vai para o seu bloco IMU_L_*/IMU_R_*"))
    parser.add_argument("--impedancia", action="store_true",
                        help="Relatorio de impedancias antes da sessao")
    parser.add_argument("--impedancia-tentativas", type=int, default=3,
                        help="Tentativas da medida de impedancia (o link sem "
                             "fio pode falhar com CRC)")
    parser.add_argument("--impedancia-sec", type=float, default=6.0,
                        help="Duracao de cada medida de impedancia (s)")
    parser.add_argument("--sem-escala-uv", action="store_true",
                        help="Mantem os counts crus do ADC (sem converter "
                             "para uV)")
    parser.add_argument("--sensibilidade", type=float, default=None,
                        help="Sensibilidade do ADC em nV (padrao: a do "
                             "aparelho; g.Nautilus 500 Hz = 2250000)")
    parser.add_argument("--csv", default=None,
                        help="Nome do CSV (padrao: gravacao_MEMI_<data>.csv)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _check_gtec_environment():
    """Confere se o gtec_gds desta Python consegue falar com o GDS/g.Nautilus.

    Existem duas instalacoes de gtec_gds na maquina tipica deste projeto:
      * .venv\\Scripts\\python.exe -> gtec_gds 1.6.0 (cdefs COMPILADOS no
        pacote: nao precisa dos headers C do g.tec) -> funciona;
      * Python do PATH (C:\\Program Files\\Python) -> gtec_gds 1.4.0, que
        exige os headers em Documentos\\gtec\\gNEEDaccessClientAPI\\C
        (que podem nao estar instalados) -> falha com
        "Initialize failed to load either header or DLL" e depois
        NameError: name '_ffi' is not defined.
    Esta checagem transforma esse erro criptico numa instrucao clara.
    """
    try:
        import gtec_gds
    except ImportError as exc:
        raise RuntimeError(
            "gtec_gds nao esta instalado nesta Python. Instale o pacote "
            "g.tec (pip install gtec_gds) ou rode com o .venv do projeto."
        ) from exc
    version = getattr(gtec_gds, "__version__", "?")
    venv_python = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    try:
        from gtec_gds.lib.gtec_gds_wrapper import ConnectedDevices
        devices = [(name, dtype, inuse) for name, dtype, inuse in
                   ConnectedDevices()]
    except Exception as exc:
        raise RuntimeError(
            f"gtec_gds {version} desta Python nao inicializou o GDS "
            f"({type(exc).__name__}: {exc}).\n"
            "  Causa mais comum: esta e a Python do PATH, que usa gtec_gds "
            "1.4.0 e exige os headers C do g.tec em "
            "'Documentos\\gtec\\gNEEDaccessClientAPI\\C' (ausentes).\n"
            "  Solucao: rode com o .venv do projeto, que tem o gtec_gds "
            "1.6.0 (cdefs compilados):\n"
            f"      {venv_python} eeg_motor_paradigm.py ..."
        ) from exc
    if not devices:
        print(f"[gtec] gtec_gds {version} OK, mas nenhum dispositivo "
              "conectado encontrado (ligue o g.Nautilus)", flush=True)
    else:
        print(f"[gtec] gtec_gds {version} OK | dispositivos: {devices}",
              flush=True)


def build_source(args):
    if args.source == "gnautilus":
        _check_gtec_environment()
        print(f"g.Nautilus: {args.channel_count} canais, "
              f"frame_size={args.frame_size}")
        return gp.GNautilus(sampling_rate=args.fs,
                            channel_count=args.channel_count,
                            frame_size=args.frame_size)
    print(f"Generator sintetico: {args.channel_count} canais (teste)")
    return gp.Generator(sampling_rate=args.fs,
                        channel_count=args.channel_count,
                        signal_frequency=10.0, signal_amplitude=10.0,
                        noise_amplitude=5.0)


def _make_counts_to_uv(source_node, args):
    """Cria o no counts -> uV lendo sensibilidade e calibracao do aparelho."""
    device = getattr(source_node, "_device", None)
    sensitivity = args.sensibilidade
    if sensitivity is None:
        sensitivity = getattr(device, "sensitivity", None)
    if sensitivity is None:
        sensitivity = 2250000.0
        print("[escala] aviso: o aparelho nao informou a sensibilidade; "
              "usando 2250000 nV (g.Nautilus, 500 Hz, 32 canais)", flush=True)
    factor, offset = 1.0, 0.0
    if device is not None:
        try:
            scaling = device.get_scaling()
            if isinstance(scaling, (tuple, list)) and len(scaling) == 2:
                factor = float(np.mean(np.asarray(scaling[0], dtype=float)))
                offset = float(np.mean(np.asarray(scaling[1], dtype=float)))
        except Exception as exc:
            print(f"[escala] aviso: get_scaling() falhou "
                  f"({type(exc).__name__}: {exc}); usando Factor=1, Offset=0",
                  flush=True)
    return CountsToMicrovolts(sensitivity_nv=sensitivity, factor=factor,
                              offset=offset)


def main():
    # Console/pipe do Windows pode usar cp1252: o ioiocore imprime uma tabela
    # com caracteres Unicode no monitor da pipeline e morre se nao couber no
    # codec. Reconfigura para UTF-8 (sem efeito no console interativo).
    import sys

    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    args = parse_args()
    if args.sem_video:
        args.video_sec = 0.0
    if args.mov_no_eeg_csv:
        # Formato antigo (movimento embutido no CSV de EEG): util apenas para
        # comparar/compatibilidade. O padrao e' o movimento em arquivo proprio.
        global MOTION_IN_EEG_CSV
        MOTION_IN_EEG_CSV = True
    args.impedancia_painel = not args.sem_painel_impedancia
    rng = random.Random(args.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    # Sessoes vao para gravacoes/ (a mesma pasta que o treino le com
    # `--data gravacoes/`); essa pasta esta' no .gitignore.
    if args.csv:
        pasta_sessoes = os.path.dirname(str(args.csv))
        csv_name = args.csv
        if pasta_sessoes:
            os.makedirs(pasta_sessoes, exist_ok=True)
    else:
        pasta_sessoes = args.pasta_sessoes
        os.makedirs(pasta_sessoes, exist_ok=True)
        csv_name = os.path.join(pasta_sessoes, f"gravacao_MEMI_{stamp}.csv")
    events_name = os.path.splitext(csv_name)[0] + "_eventos.json"

    trials = _ParadigmController.build_trial_list(args.trials_por_bloco,
                                                  args.blocos, rng)
    trial_sec = (args.home_sec + args.me_sec + args.video_sec + args.mi_sec)
    n_pausas = max(0, args.blocos - 1)
    estimated = (args.olhos_abertos + args.olhos_fechados + args.repouso_ativo
                 + ORIGIN_CAL_SEC + args.contagem
                 + len(trials) * trial_sec + n_pausas * args.pausa_bloco_sec)
    print(f"Arquivo CSV : {csv_name}\nEventos JSON: {events_name}")
    print(f"Sessao      : {len(trials)} trials em {args.blocos} blocos de "
          f"{args.trials_por_bloco} | trial {trial_sec:g} s "
          f"(home {args.home_sec:g} + cue/ME {args.me_sec:g} + "
          f"video {args.video_sec:g} + MI {args.mi_sec:g})")
    contagem = {cond: 0 for cond in
                [(obj, mao) for obj in OBJECTS for mao in HANDS]}
    for trial in trials:
        contagem[(trial["objeto"], trial["mao"])] += 1
    detalhe = ", ".join(f"{obj}/{mao}={n}"
                        for (obj, mao), n in sorted(contagem.items()))
    print(f"Condicoes   : {detalhe}")
    print(f"Duracao estimada ~{estimated / 60:.1f} min "
          f"(+ preparacao e pausas)")

    marker_queue = queue.Queue()
    state = SharedState()
    #: Lado do UNICO IMU (com dois IMUs cada bloco ja' e' rotulado; este valor
    #: resolve so' as amostras em que a mao ativa e' desconhecida, como as
    #: linhas de base do inicio da sessao).
    state.motion["imu_lado"] = int(getattr(args, "imu_lado", 1) or 1)
    imu = ImuReceiver(4210)
    imu.start()

    app = gp.MainApp(caption="Gravacao ME/MI - Execucao e Imaginacao Motora")
    pipeline = gp.Pipeline()
    source = build_source(args)
    # Counts do ADC do g.Nautilus -> uV (o no GNautilus do g.Pype e' pass-through).
    converter = None
    if args.source == "gnautilus" and not args.sem_escala_uv:
        converter = _make_counts_to_uv(source, args)
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)
    notch50 = gp.Bandstop(f_lo=48, f_hi=52)
    notch60 = gp.Bandstop(f_lo=58, f_hi=62)
    # ---- movimento em arquivo proprio (#1) + metadados da sessao (#2) ----
    motion_name = _motion_file_name(csv_name)
    # Pasta dos clipes do priming (#7): ao lado do CSV da sessao.
    priming_dir = (None if args.sem_priming_arquivo
                   else os.path.splitext(str(csv_name))[0] + "_priming")
    session_meta = _build_session_meta(args, trials, args.source, csv_name,
                                       events_name, motion_name,
                                       priming_dir=priming_dir)
    muxer = RecordingMuxer(marker_queue, state, hold_s=args.hold_sec,
                           fs=args.fs, events_path=events_name,
                           write_motion=MOTION_IN_EEG_CSV,
                           meta=session_meta)
    writer = RecordingCsvWriter(file_name=csv_name)
    # O escopo mostra a EEG FILTRADA (monitoramento) com as linhas de evento;
    # o CSV guarda o sinal cru. O MarkerFanout copia o codigo vigente do muxer.
    fanout = MarkerFanout(muxer)
    marker_channel = args.channel_count  # 0-based: EEG + Marker (fanout)
    # ---- vigilancia da aquisicao (#15) e painel de impedancias (#23/#24) ----
    watchdog = LinkWatchdog(muxer, state, marker_queue)
    impedance_device = getattr(source, "_device", None)
    impedance_monitor = None
    if args.impedancia_painel and impedance_device is not None:
        impedance_monitor = ImpedanceMonitor(impedance_device, state)
    elif args.impedancia_painel:
        print("[impedancia] painel desativado: a fonte atual nao expoe o "
              "aparelho (use --source gnautilus)", flush=True)
    mk = gp.TimeSeriesScope.Markers
    scope = gp.TimeSeriesScope(
        amplitude_limit=50, time_window=10,
        markers=[mk(color="r", label="mao esq", channel=marker_channel, value=769),
                 mk(color="b", label="mao dir", channel=marker_channel,
                    value=770),
                 mk(color="g", label="trial", channel=marker_channel,
                    value=768),
                 mk(color="k", label="ME", channel=marker_channel, value=778),
                 mk(color="c", label="MI", channel=marker_channel, value=780),
                 mk(color="y", label="bloco", channel=marker_channel,
                    value=790),
                 mk(color="m", label="pausa", channel=marker_channel,
                    value=792)])

    # O CSV guarda o sinal CRU (apenas counts -> uV). Os filtros (1-30 Hz +
    # notches) vao SO para o escopo de monitoramento: gravar filtrado impedia
    # refazer a analise com outra banda, forcava dupla filtragem no treino
    # (que aplica sosfiltfilt) e criava mismatch de fase entre treino
    # (zero-phase) e tempo real (causal).
    upstream = converter if converter is not None else source
    if converter is not None:
        pipeline.connect(source, converter)
    pipeline.connect(upstream, bandpass)
    pipeline.connect(upstream, muxer)
    pipeline.connect(bandpass, notch50)
    pipeline.connect(notch50, notch60)
    pipeline.connect(notch60, fanout)
    pipeline.connect(fanout, scope)
    pipeline.connect(muxer, writer)

    tracking = TrackingThread(state, imu, with_kinect=not args.sem_kinect,
                              use_ik=not args.sem_ik,
                              aux_indices=tuple(
                                  int(index) for index in
                                  str(args.aux_cameras).split(",")
                                  if index.strip().lstrip("-").isdigit()),
                              use_zeroing=not args.sem_zeroing,
                              marker_queue=marker_queue,
                              use_onset=not args.sem_onset)
    window = StimulusWindow(full_screen=args.tela_cheia,
                            on_close=lambda: controller.stop())
    controller = _ParadigmController(
        window=window, marker_queue=marker_queue, state=state,
        config={
            "trials": trials,
            "home_sec": args.home_sec,
            "me_sec": args.me_sec,
            "video_sec": args.video_sec,
            "mi_sec": args.mi_sec,
            "trials_por_bloco": args.trials_por_bloco,
            "blocos": args.blocos,
            "pausa_bloco_sec": args.pausa_bloco_sec,
            "contagem": args.contagem,
            "olhos_abertos": args.olhos_abertos,
            "olhos_fechados": args.olhos_fechados,
            "repouso_ativo": args.repouso_ativo,
            "preparacao": not args.sem_preparacao,
            # Clipes do priming (#7): pasta ao lado do CSV da sessao.
            "priming_dir": priming_dir,
            "salvar_priming": not args.sem_priming_arquivo,
            # Para o painel de impedancias quando a sessao comeca (nao fica
            # conversando com o aparelho durante a gravacao).
            "on_session_start": (None if impedance_monitor is None
                                 else impedance_monitor.stop),
        })
    # ESPACO (experimentador) pula a espera/pausa atual; ESC aborta a sessao.
    window.canvas.on_skip = controller.skip
    window.canvas.on_abort = controller.abort
    app.add_widget(scope)

    tracking.start()
    window.show()
    pipeline.start()
    watchdog.start()
    if impedance_monitor is not None:
        impedance_monitor.start()
    motion_recorder = MotionRecorder(state, motion_name, hz=args.mov_hz,
                                     fs=args.fs)
    motion_recorder.start()
    if args.source == "gnautilus" and args.impedancia:
        _impedance_report(source, args.impedancia_tentativas,
                          args.impedancia_sec)
        print("Impedancia medida. Corrija os eletrodos marcados e use ESPACO "
              "na tela de PREPARACAO para iniciar a sessao.", flush=True)
    controller.start(300)
    print("Gravando... ESPACO inicia/pula pausas | ESC aborta | "
          "feche a janela de estimulos para interromper.")
    try:
        app.run()
    finally:
        pipeline.stop()
        controller.stop()
        motion_recorder.stop()
        watchdog.stop()
        if impedance_monitor is not None:
            impedance_monitor.stop()
        tracking.running = False
        tracking.join(timeout=3.0)
        imu.stop()
        _align_event_json(writer, events_name, csv_name)
        actual_csv = getattr(writer, "_file_path", None)
        print(f"\nGravacao finalizada:\n"
              f"  EEG        -> {os.path.abspath(actual_csv or csv_name)}\n"
              f"  Movimento  -> {os.path.abspath(motion_name)} "
              f"({motion_recorder.rows} linhas)\n"
              f"  Eventos    -> {os.path.abspath(events_name)}\n"
              f"  Priming    -> {priming_dir or '(nao salvo)'} "
              f"({tracking.clips_saved} clipes, "
              f"{tracking.clips_bytes / (1024.0 * 1024.0):.1f} MB)\n"
              f"  Vigia EEG  -> {watchdog.perdas} queda(s) de link detectada(s)")
        if not args.sem_questionario:
            respostas = _ask_participant(args)
            _write_participant_file(actual_csv or csv_name, session_meta,
                                    respostas)


def _read_impedance(device, seconds):
    """Le a impedancia direto do aparelho por `seconds` ('first' no 1o read)."""
    values = None
    first = True
    deadline = time.monotonic() + float(seconds)
    while time.monotonic() < deadline:
        values = np.asarray(device.get_impedance(first), dtype=float)
        first = False
        time.sleep(0.1)
    return values


def _print_impedance(values, limite_kohm=50.0):
    print("[impedancia] valores (kOhm; -10 = desconhecido/nao medido):",
          flush=True)
    bons = 0
    for indice, valor in enumerate(values, 1):
        ruim = valor < 0 or valor > limite_kohm
        bons += 0 if ruim else 1
        aviso = "  <-- verificar eletrodo" if ruim else ""
        print(f"  Ch{indice:02d}: {valor:7.1f}{aviso}")
    print(f"[impedancia] {bons}/{len(values)} canais com contato bom "
          f"(<= {limite_kohm:g} kOhm)", flush=True)


def _impedance_report(source_node, attempts=3, seconds=6.0):
    """Mede e imprime as impedancias do g.Nautilus com retentativas.

    O g.Nautilus mede a impedancia ENQUANTO ESTA TRANSMITINDO e no link sem
    fio o comando pode falhar com 'Invalid CRC checksum'. Em vez de usar a
    thread do g.Pype (que morre em silencio no primeiro erro, deixando todos
    os canais em -10), lemos o aparelho direto em um loop com retentativas.
    """
    device = getattr(source_node, "_device", None)
    values = None
    if device is None or not hasattr(device, "get_impedance"):
        print("[impedancia] aparelho sem get_impedance; tentando a API do "
              "g.Pype (start_impedance_check)", flush=True)
        try:
            source_node.start_impedance_check()
            time.sleep(seconds)
            values, _fresh = source_node.get_impedance()
            source_node.stop_impedance_check()
        except Exception as exc:
            print(f"[impedancia] falhou ({type(exc).__name__}: {exc})",
                  flush=True)
            return None
    else:
        for tentativa in range(1, int(attempts) + 1):
            try:
                values = _read_impedance(device, seconds)
                if values is not None and np.any(values > 0):
                    break
                print(f"[impedancia] tentativa {tentativa}/{int(attempts)} "
                      "sem valores validos; repetindo", flush=True)
            except Exception as exc:
                print(f"[impedancia] tentativa {tentativa}/{int(attempts)} "
                      f"falhou ({type(exc).__name__}: {exc}); repetindo",
                      flush=True)
            time.sleep(0.5)
    if values is None:
        print("[impedancia] NAO foi possivel medir (link do g.Nautilus). "
              "Sem eletrodos no escalpo e' normal todos ficarem em -10.",
              flush=True)
        return None
    _print_impedance(values)
    return values


def _align_event_json(writer, events_name, csv_name):
    """Alinha o JSON ao nome real do CSV (gp.FileWriter anexa timestamp)."""
    actual_csv = getattr(writer, "_file_path", None)
    if not actual_csv:
        return
    new_json = os.path.splitext(actual_csv)[0] + "_eventos.json"
    if os.path.exists(events_name) and new_json != events_name:
        os.replace(events_name, new_json)
        events_name = new_json
    try:
        with open(events_name, "r+", encoding="utf-8") as handle:
            payload = json.load(handle)
            payload["arquivo_csv"] = os.path.basename(actual_csv)
            handle.seek(0)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except (OSError, json.JSONDecodeError):
        pass


if __name__ == "__main__":
    main()

