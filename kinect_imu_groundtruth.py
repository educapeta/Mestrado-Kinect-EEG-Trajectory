"""Biblioteca do ground truth 3D da mao: Kinect + MediaPipe Hands + MPU6050.

Este modulo e' uma BIBLIOTECA (sem programa proprio). Oferece:

  - CameraAlignment   : calibracao stereo, triangulacao, projecao, trim;
  - KinectHandTracker : MediaPipe (RGB + webcams), profundidade do Kinect,
                        esqueleto do SDK e lateralidade das maos;
  - ArmLinkModel      : cinematica inversa de 2 elos (ombro/cotovelo/punho);
  - palm_frame/palm_angles : orientacao da palma a partir dos landmarks.

O IMU (UDP do ESP32) e a fusao IMU x cameras vivem em `imu.py` (fonte unica:
ImuReceiver, ImuSample, PositionFusion, rotation_matrix).

Quem usa esta biblioteca: `eeg_motor_paradigm.py` (aquisicao offline),
`sand_traj_tempo_real.py` (sistema online) e os testes (`test_*.py`).

O programa autonomo de calibracao/diagnostico (que antes era o `main()` deste
arquivo) vive em `tools/kinect_groundtruth_tool.py`. Requisitos de hardware:
Kinect v2 com Kinect for Windows SDK 2.0 e ESP32 enviando
`timestamp_us,roll,pitch,yaw,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps` em UDP 4210.

Instalacao: `python -m pip install opencv-python mediapipe numpy pykinect2`
(ver `requirements.txt`).
"""

from __future__ import annotations

import os

# Silencia logs INFO/WARNING do TensorFlow Lite/MediaPipe (nao sao erros).
# Precisa ser definido ANTES de importar mediapipe/cv2.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import csv
import json
import math
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    from pykinect2 import PyKinectV2
    from pykinect2.PyKinectRuntime import PyKinectRuntime
    from _ctypes import COMError
except ImportError as exc:
    raise RuntimeError(
        "pykinect2 nao esta instalado. Instale pykinect2 e o Kinect for Windows SDK 2.0."
    ) from exc

UDP_PORT = 4210
AUX_CAMERA_INDEX = 0
# Cameras auxiliares usadas em conjunto com o Kinect (indices do OpenCV).
# O padrao liga as DUAS webcams: 0 = webcam do laptop, 1 = webcam USB.
# A calibracao por indice e armazenada em stereo_calibration_aux{idx}.npz /
# hand_landmark_calibration_aux{idx}.npz (o nome legado = indice 0).
AUX_CAMERA_INDICES = (0, 1)
AUX_CAMERA_NAMES = {0: "laptop", 1: "usb", 2: "usb2"}
# Keep the same physical pixel convention used by stereo_calibration.py.
AUXILIARY_MIRROR_HORIZONTAL = True
AUXILIARY_DISPLAY_MIRROR = False
DEPTH_MIN_MM = 400
DEPTH_MAX_MM = 3000
DEPTH_PATCH_RADIUS = 3
# Profundidade nominal (m) usada apenas quando nunca houve medida valida;
# a sobreposicao aparece com erro de paralaxe proporcional ao desvio.
NOMINAL_HAND_DEPTH_M = 1.0
# Ganhos da fusao IMU x camera, o eixo "para cima" e as constantes da zeragem
# vivem em `imu.py` (FONTE UNICA, item #9 da revisao -- antes havia uma copia
# aqui e outra em camera_mpu_fusion.py). Reexportados para quem usa `gt.UP_AXIS`
# e para as anotacoes abaixo.
from imu import (  # noqa: E402
    IMU_ZERO_BIAS_EMA,
    IMU_ZERO_MAX_BIAS_G,
    IMU_ZERO_STD_M,
    IMU_ZERO_WINDOW,
    POSITION_CAMERA_GAIN,
    UP_AXIS,
    VELOCITY_DAMPING,
    ImuReceiver,
    ImuSample,
    PositionFusion,
    rotation_matrix,
)
CALIBRATION_SECONDS = 3.0
PALM_VECTOR_LENGTH_M = 0.20
#: Lateralidade da mao detectada (colunas KT_hand / PALM_hand do CSV):
#: 0 = desconhecida, 1 = direita, 2 = esquerda.
HAND_SIDE_CODE = {"": 0, "right": 1, "left": 2}
#: O MediaPipe define a lateralidade assumindo imagem ESPELHADA (selfie). A
#: webcam auxiliar e' espelhada no nosso pipeline (rotulo ja correto) e o RGB
#: do Kinect NAO e' espelhado -> o rotulo do Kinect precisa ser invertido.
#: Sem isso, uma trial de mao ESQUERDA poderia triangular a mao DIREITA.
MEDIAPIPE_KINECT_HANDEDNESS_FLIP = True
CHECKERBOARD_SIZE = (8, 6)
SQUARE_SIZE_M = 0.025
CALIBRATION_SAMPLES_REQUIRED = 15
CALIBRATION_VERSION = 7
MAX_STEREO_RMS_PX = 2.0
# RMS stereo acima de 2 px: o stereo pode ser usado com precisao
# reduzida (sobreposicao aproximada) ate este limite mais folgado.
STEREO_OVERLAY_MAX_RMS_PX = 5.0
# Margen (px) para considerar una proyeccion "dentro" del frame y valida para
# desvio/dibujo. Proyecciones mas lejos indican profundidad invalida o mano
# fuera del volumen calibrado -> la mano se descarta por completo.
OVERLAY_FRAME_MARGIN_PX = 80
CALIBRATION_FILE = Path(__file__).with_name("camera_alignment.npz")
STEREO_CALIBRATION_FILE = Path(__file__).with_name("stereo_calibration.npz")


def aux_stereo_file(aux_index):
    """Arquivo de calibracao stereo de uma camera auxiliar.

    Indice 0 mantem o nome legado (stereo_calibration.npz) para compatibilidade
    com as sessoes ja calibradas; demais indices usam _aux{idx}.npz."""
    if int(aux_index) == 0 and STEREO_CALIBRATION_FILE.exists():
        return STEREO_CALIBRATION_FILE
    candidate = Path(__file__).with_name(f"stereo_calibration_aux{aux_index}.npz")
    return candidate if candidate.exists() else None
# Calibracao do esqueleto (MediaPipe): correspondencia no-a-no entre a mano
# auxiliar e a mano do Kinect, aprendida de fotos da propia mano. Resuelve
# los dedos "desgobernados" (el depth por landmark era ruidoso en los dedos).
HAND_CALIBRATION_FILE = Path(__file__).with_name("hand_landmark_calibration.npz")


def aux_hand_file(aux_index):
    """Arquivo de calibracao de landmarks da mao de uma camera auxiliar.

    Indice 0 mantem o nome legado (hand_landmark_calibration.npz)."""
    if int(aux_index) == 0 and HAND_CALIBRATION_FILE.exists():
        return HAND_CALIBRATION_FILE
    candidate = Path(__file__).with_name(
        f"hand_landmark_calibration_aux{aux_index}.npz")
    return candidate if candidate.exists() else None
HAND_CALIBRATION_VERSION = 2
HAND_CALIBRATION_SAMPLES_REQUIRED = 18
HAND_CALIBRATION_MIN_SPAN = 0.35  # varredura minima pedida na captura (norm)
# Variação mínima de profundidade (m) durante a captura. Sem variar pelo
# menos isso, o modelo 3D (escala -> depth) não aprende direito.
HAND_CALIBRATION_MIN_DEPTH_SPAN_M = 0.40
# Kinect landmarks usados como referencia de escala da mao (punho -> meio)
# para estimar profundidade aparente na camera auxiliar.
HAND_SCALE_LANDMARK_A = 0   # punho (wrist)
HAND_SCALE_LANDMARK_B = 9   # MCP do dedo medio (middle finger MCP)
# Fator de redimensionamento das janelas de video (0.5 = metade do tamanho).
# Para ver o terminal enquanto o programa roda, janelas menores ajudam.
WINDOW_SCALE = 0.6
POSE_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CSV_PATH = Path(__file__).with_name("kinect_imu_groundtruth.csv")
OVERLAY_LOG_PATH = Path(__file__).with_name("overlay_deviation_log.csv")
# Trim fixo do overlay: traducao constante (px Kinect) medida UMA vez com a
# mao PARADA (tecla K) e guardada aqui. Faz parte da cadeia de calibracao
# (corrige o desregistro fixo residual da stereo+intrinsecas); NAO e um
# seguidor: nunca usa a mao verde em tempo real, e uma constante armazenada.
OVERLAY_TRIM_PATH = Path(__file__).with_name("overlay_trim.json")
# ---------------------------------------------------------------- braco (IK)
# Modelo de elos rigidos do braco. Os comprimentos sao CONSTANTES da pessoa
# (ombro->cotovelo e cotovelo->punho nunca mudam), entao medem-se UMA vez
# (durante a calibracao de origem, com o braco em repouso) e guardam-se aqui.
# Depois disso, a cinematica inversa (IK) de 2 elos reconstroi o cotovelo:
# o punho medido (triangulado/MediaPipe) diz ONDE a mao esta; os comprimentos
# fixos dizem ONDE o cotovelo TEM de estar para aquela posicao ser alcancavel.
ARM_MODEL_PATH = Path(__file__).with_name("arm_model.json")
ARM_MODEL_VERSION = 1
# Faixa plausivel (m) para os elos de um braco humano adulto. Fora disso a
# medicao e descartada (esqueleto mal rastreado/oclusao).
ARM_LINK_MIN_M = 0.12
ARM_LINK_MAX_M = 0.45
# Numero minimo de amostras validas para aceitar a calibracao dos elos.
ARM_MODEL_MIN_SAMPLES = 30
# Tolerancia relativa entre o comprimento medido frame-a-frame e o do modelo:
# acima disso o cotovelo medido e considerado ruido e o IK manda no resultado.
ARM_LINK_TOLERANCE = 0.20
# Folga (m) aceita antes de considerar o punho fora do alcance dos elos. Sem
# folga, ruido de 1-2 cm no punho marcaria quase todo frame como "clamped".
ARM_IK_REACH_TOL_M = 0.03
# Parametros intrinsecos nominais do RGB do Kinect v2 @1920x1080 (fallback para
# desenhar o overlay do braco quando a calibracao stereo nao existe).
KINECT_V2_INTRINSICS_1080P = np.array([[1081.37, 0.0, 959.5],
                                       [0.0, 1081.37, 539.5],
                                       [0.0, 0.0, 1.0]])
KINECT_V2_COLOR_SIZE = (1920, 1080)


def unit_vector(vector, eps=1e-9):
    """Vetor normalizado (float64) ou None se o comprimento for degenerado."""
    vector = np.asarray(vector, np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return None
    return vector / norm


def palm_frame(wrist, index_mcp, pinky_mcp, middle_mcp):
    """Frame ortonormal do plano da palma a partir de 4 pontos 3D (m).

    Devolve (normal, direcao_dedos), ambas unitarias, ou (None, None):
      - normal: perpendicular ao plano punho/indicador/minimo, com sinal
        fixado apontando para a CAMERA (nz < 0) -> comparavel entre frames;
      - direcao_dedos: punho -> MCP do dedo medio, ortogonalizada em relacao
        a normal (fica exatamente no plano da palma).
    """
    normal = None
    if index_mcp is not None and pinky_mcp is not None and wrist is not None:
        normal = KinectHandTracker.palm_normal_from_points(
            wrist, index_mcp, pinky_mcp)
    direction = None
    if normal is not None and middle_mcp is not None and wrist is not None:
        fingers = unit_vector(np.asarray(middle_mcp, np.float64)
                              - np.asarray(wrist, np.float64))
        if fingers is not None:
            # ortogonaliza em relacao a normal: fica no plano da palma
            direction = unit_vector(fingers - normal * float(np.dot(fingers,
                                                                   normal)))
            if direction is None:
                direction = fingers      # dedos alinhados com a normal
    return normal, direction


def palm_angles(normal, direction):
    """Angulos (graus) de orientacao da palma no frame do Kinect.

    Convencao (Y para cima, Z para frente/camera):
      elev  = asin(ny da normal) -> +90 palma virada para cima (teto),
              -90 palma virada para baixo (mesa);
      azim  = atan2(nx, nz)      -> +90 normal virada para a direita;
      pitch = asin(dir_y)        -> +90 dedos apontando para cima;
      yaw   = atan2(dir_x, dir_z)-> +90 dedos apontando para a direita,
              0 dedos apontando para frente, +-180 dedos para tras.

    Valores ausentes viram NaN (o CSV nunca grava 0.0 como se fosse medido).
    """
    nan = float("nan")
    result = {"normal": None if normal is None else np.asarray(normal, np.float64),
              "dir": None if direction is None else np.asarray(direction, np.float64),
              "elev": nan, "azim": nan, "pitch": nan, "yaw": nan, "valid": 0}
    if normal is not None:
        result["elev"] = float(math.degrees(
            math.asin(float(np.clip(normal[1], -1.0, 1.0)))))
        result["azim"] = float(math.degrees(
            math.atan2(float(normal[0]), float(normal[2]))))
    if direction is not None:
        result["pitch"] = float(math.degrees(
            math.asin(float(np.clip(direction[1], -1.0, 1.0)))))
        result["yaw"] = float(math.degrees(
            math.atan2(float(direction[0]), float(direction[2]))))
    if normal is not None and direction is not None:
        result["valid"] = 1
    return result

BODY_CONNECTIONS = [
    (PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck),
    (PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder),
    (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft),
    (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight),
    (PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft),
    (PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft),
    (PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft),
    (PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight),
    (PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight),
    (PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight),
    (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid),
    (PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase),
    (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft),
    (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight),
    (PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft),
    (PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft),
    (PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft),
    (PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight),
    (PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight),
    (PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight),
]

# Juntas do esqueleto do SDK (camera space, metros) usadas pelo paradigma:
# ancora de profundidade do punho e cadeia do braco por lado.
ANCHOR_JOINTS = {
    "right": ((PyKinectV2.JointType_WristRight, "wrist_right"),
              (PyKinectV2.JointType_HandTipRight, "handtip_right")),
    "left": ((PyKinectV2.JointType_WristLeft, "wrist_left"),
             (PyKinectV2.JointType_HandTipLeft, "handtip_left")),
}
ARM_JOINT_TYPES = {
    "right": (PyKinectV2.JointType_ShoulderRight,
              PyKinectV2.JointType_ElbowRight,
              PyKinectV2.JointType_WristRight),
    "left": (PyKinectV2.JointType_ShoulderLeft,
             PyKinectV2.JointType_ElbowLeft,
             PyKinectV2.JointType_WristLeft),
}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# (ImuSample, ImuReceiver e rotation_matrix foram movidos para imu.py.)


def ensure_hand_model() -> Path:
    if not POSE_MODEL_PATH.exists():
        print("Baixando modelo MediaPipe Hands...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    return POSE_MODEL_PATH


class CameraAlignment:
    def __init__(self, aux_indices=None):
        # Cameras auxiliares registradas (por indice OpenCV). O padrao vem da
        # constante do modulo, mas o programa pode passar a lista que o
        # usuario escolheu (ex.: (1, 2) quando a webcam do laptop fica no
        # mesmo lugar do Kinect e nao serve de par estereo).
        self.aux_indices_requested = tuple(
            int(index) for index in (aux_indices
                                     if aux_indices is not None
                                     else AUX_CAMERA_INDICES))
        self.stereo_ready = False
        self.kinect_matrix = None
        self.kinect_dist = None
        self.auxiliary_matrix = None
        self.auxiliary_dist = None
        self.stereo_rotation = None
        self.stereo_translation = None
        self.stereo_rms = None
        self.calibrated_auxiliary_size = None
        self.calibrated_kinect_size = None
        # Calibração do esqueleto (correspondência nó-a-nó auxiliar -> Kinect).
        self.hand_landmark_homography = None
        self.hand_landmark_residuals = None
        self.hand_landmark_rms = None
        # Modelo de profundidade 3D: escala aparente da mao na auxiliar -> depth,
        # e offsets de Z por nó (relativos ao centro da palma).
        self.hand_scale_depth_params = None  # C = depth_ref * scale_ref (modelo multiplicativo)
        self.hand_reference_scale = None     # escala de referencia (media das amostras)
        self.hand_node_z_offset = None       # (21,) desvio mediano de Z por nó
        # Esqueleto 3D medido por triangulacao estereo (espaco vetorial), com o
        # instante da medicao (time.monotonic). Invalidado quando a mao deixa
        # de ser vista pelas DUAS cameras (sem medicao, nao ha dado novo).
        self.last_triangulated_3d = None
        self.last_triangulated_3d_time = None
        # Trim fixo do overlay (calibracao, ver OVERLAY_TRIM_PATH acima).
        self.overlay_trim = np.zeros(2, np.float64)
        try:
            with open(OVERLAY_TRIM_PATH, "r", encoding="utf-8") as fh:
                trim_data = json.load(fh)
            self.overlay_trim = np.array(
                [float(trim_data["dx"]), float(trim_data["dy"])], np.float64
            )
        except (OSError, ValueError, KeyError, TypeError):
            self.overlay_trim = np.zeros(2, np.float64)
        # EMA da profundidade da mão para suavizar o jitter frame-a-frame.
        self._hand_depth_ema = None
        self._hand_depth_ema_initialized = False
        self._hand_depth_ema_alpha = 0.15
        # ---- calibracao POR CAMERA AUXILIAR (multicamera) ----------------
        # Cada indice (0=laptop, 1=usb, ...) tem a propria calibracao stereo e
        # de landmarks. Os campos vivos (self.auxiliary_matrix, self.stereo_*,
        # self.hand_landmark_*) referem-se SEMPRE a camera ativa; trocar de
        # camera e so set_active_aux(idx) (a matematica nao muda).
        self.aux_models = {}
        self.active_aux = None
        for aux_index in self.aux_indices_requested:
            self.aux_models[int(aux_index)] = self._parse_aux_calibration(
                int(aux_index))
        if self.aux_indices_requested:
            self.set_active_aux(int(self.aux_indices_requested[0]))
        self.homography = None
        self.auxiliary_width = 0
        self.auxiliary_height = 0
        self.auxiliary_points = []
        self.kinect_points = []
        self.auxiliary_matrix_runtime = None
        self.auxiliary_scale_warned = False
        self.refinement_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if CALIBRATION_FILE.exists():
            data = np.load(CALIBRATION_FILE)
            if "version" in data.files and int(data["version"]) == CALIBRATION_VERSION:
                self.homography = data["homography"]
                if "refinement_affine" in data.files:
                    self.refinement_affine = data["refinement_affine"]

    # ------------------------------------------------------------------ multica
    def _parse_aux_calibration(self, aux_index):
        """Le stereo + landmarks de UMA camera auxiliar (arquivo por indice)."""
        model = {
            "kinect_matrix": None, "kinect_dist": None,
            "matrix": None, "dist": None,
            "rotation": None, "translation": None, "rms": None,
            "stereo_ready": False, "stereo_usable": False,
            "auxiliary_size": None, "kinect_size": None,
            "hand_homography": None, "hand_residuals": None, "hand_rms": None,
            "scale_depth_params": None, "reference_scale": None,
            "node_z_offset": None, "hand_ready": False, "runtime_matrix": None,
        }
        stereo_path = aux_stereo_file(aux_index)
        if stereo_path is not None:
            stereo_data = np.load(stereo_path)
            required = {
                "kinect_matrix", "kinect_dist", "auxiliary_matrix",
                "auxiliary_dist", "rotation", "translation",
            }
            if required.issubset(stereo_data.files):
                model["kinect_matrix"] = stereo_data["kinect_matrix"]
                model["kinect_dist"] = stereo_data["kinect_dist"]
                model["matrix"] = stereo_data["auxiliary_matrix"]
                model["dist"] = stereo_data["auxiliary_dist"]
                model["rotation"] = stereo_data["rotation"]
                model["translation"] = stereo_data["translation"]
                saved_board = (tuple(stereo_data["checkerboard_size"].astype(int))
                               if "checkerboard_size" in stereo_data.files else None)
                saved_square = (float(stereo_data["square_size_m"])
                                if "square_size_m" in stereo_data.files else None)
                model["rms"] = (float(stereo_data["rms"])
                                if "rms" in stereo_data.files else None)
                model["stereo_ready"] = (
                    saved_board == CHECKERBOARD_SIZE
                    and saved_square is not None
                    and abs(saved_square - SQUARE_SIZE_M) < 1e-6
                    and model["rms"] is not None
                    and model["rms"] <= MAX_STEREO_RMS_PX)
                model["stereo_usable"] = (
                    model["rms"] is not None
                    and model["rms"] <= STEREO_OVERLAY_MAX_RMS_PX)
                model["auxiliary_size"] = (
                    tuple(stereo_data["auxiliary_size"].astype(int))
                    if "auxiliary_size" in stereo_data.files else None)
                model["kinect_size"] = (
                    tuple(stereo_data["kinect_size"].astype(int))
                    if "kinect_size" in stereo_data.files else None)
        return self._parse_aux_hand_calibration(aux_index, model)

    def _parse_aux_hand_calibration(self, aux_index, model):
        """Le a calibracao de landmarks (homografia + modelo 3D) da camera."""
        hand_path = aux_hand_file(aux_index)
        if hand_path is None:
            return model
        hand_data = np.load(hand_path)
        if (
            "version" in hand_data.files
            and int(hand_data["version"]) == HAND_CALIBRATION_VERSION
            and {"homography", "residuals", "rms"}.issubset(hand_data.files)
        ):
            model["hand_homography"] = hand_data["homography"]
            model["hand_residuals"] = hand_data["residuals"]
            model["hand_rms"] = float(hand_data["rms"])
            if "scale_depth_params" in hand_data.files:
                raw = np.asarray(hand_data["scale_depth_params"],
                                 np.float64).reshape(-1)
                model["scale_depth_params"] = (
                    float(raw[0]) if raw.size >= 1 else None)
            if "reference_scale" in hand_data.files:
                model["reference_scale"] = float(hand_data["reference_scale"])
            if "node_z_offset" in hand_data.files:
                model["node_z_offset"] = np.asarray(
                    hand_data["node_z_offset"], np.float64)
            model["hand_ready"] = True
        return model

    def set_active_aux(self, aux_index):
        """Torna `aux_index` a camera auxiliar ativa (copia nos campos vivos)."""
        if int(aux_index) not in self.aux_models:
            raise KeyError(f"Sem registro para a camera auxiliar {aux_index}.")
        model = self.aux_models[int(aux_index)]
        self.active_aux = int(aux_index)
        self.kinect_matrix = model["kinect_matrix"]
        self.kinect_dist = model["kinect_dist"]
        self.auxiliary_matrix = model["matrix"]
        self.auxiliary_dist = model["dist"]
        self.stereo_rotation = model["rotation"]
        self.stereo_translation = model["translation"]
        self.stereo_rms = model["rms"]
        self.stereo_ready = bool(model["stereo_ready"])
        self.calibrated_auxiliary_size = model["auxiliary_size"]
        self.calibrated_kinect_size = model["kinect_size"]
        self.hand_landmark_homography = model["hand_homography"]
        self.hand_landmark_residuals = model["hand_residuals"]
        self.hand_landmark_rms = model["hand_rms"]
        self.hand_scale_depth_params = model["scale_depth_params"]
        self.hand_reference_scale = model["reference_scale"]
        self.hand_node_z_offset = model["node_z_offset"]
        self.auxiliary_matrix_runtime = model["runtime_matrix"]
        self.auxiliary_scale_warned = False
        return self.active_aux

    def aux_ready(self, aux_index):
        """True se a camera tem calibracao utilizavel (stereo ou landmarks)."""
        model = self.aux_models.get(int(aux_index))
        return bool(model and (model["stereo_ready"] or model["hand_ready"]))

    def aux_usable(self, aux_index):
        """True se o STEREO da camera e utilizavel para triangulacao."""
        model = self.aux_models.get(int(aux_index))
        return bool(model and model["stereo_usable"])

    def aux_indices(self):
        """Indices das cameras auxiliares registradas (ordem preferida)."""
        return list(self.aux_models)

    @property
    def ready(self):
        return self.stereo_ready

    @property
    def stereo_usable(self):
        """Stereo carregado com RMS aceitavel para sobreposicao aproximada."""
        return (
            self.stereo_rotation is not None
            and self.stereo_translation is not None
            and self.stereo_rms is not None
            and self.stereo_rms <= STEREO_OVERLAY_MAX_RMS_PX
        )

    @property
    def hand_landmark_ready(self):
        """Calibração do esqueleto cargada (mapeo nó-a-nó aux->Kinect). Inclui
        o modelo 3D (profundidade estimada) quando disponível."""
        return (
            self.hand_landmark_homography is not None
            and self.hand_landmark_residuals is not None
        )

    @property
    def hand_3d_ready(self):
        """Modelo 3D (profundidade) carregado: permite estimar Z da mao aux."""
        return (
            self.hand_scale_depth_params is not None
            and self.hand_node_z_offset is not None
        )

    @staticmethod
    def _apparent_hand_scale(normalized_landmarks):
        """Escala aparente da mao (distancia normalizada entre dois pontos
        anatomicos fixos). Maior = mao mais perto; menor = mao mais longe.
        Usa landmarks 0 (punho) e 9 (MCP do dedo medio)."""
        pts = np.asarray(normalized_landmarks, np.float64).reshape(-1, 2)
        if pts.shape[0] <= max(HAND_SCALE_LANDMARK_A, HAND_SCALE_LANDMARK_B):
            return None
        diff = pts[HAND_SCALE_LANDMARK_A] - pts[HAND_SCALE_LANDMARK_B]
        return float(np.hypot(diff[0], diff[1]))

    def estimate_hand_depth(self, normalized_landmarks, reference_scale=None):
        """Estima a profundidade (m) da mao auxiliar usando o modelo 3D.
        Modelo fisico: depth = C * ref_scale / escala_atual
        (quanto menor a escala aparente, maior a profundidade).
        Retorna None se o modelo nao estiver carregado."""
        if not self.hand_3d_ready:
            return None
        scale = self._apparent_hand_scale(normalized_landmarks)
        if scale is None or scale <= 1e-6:
            return None
        C = self.hand_scale_depth_params  # multiplicative model: depth = C * ref_scale / scale
        ref = reference_scale if reference_scale is not None else self.hand_reference_scale
        if ref is None or ref <= 1e-6:
            return None
        return float(C * ref / scale)

    def estimate_node_depths(self, normalized_landmarks):
        """Estima a profundidade (m) de cada um dos 21 nós da mao aux.
        depth[nó] = profundidade_da_palma + offset_do_nó (modelado do Kinect).
        Retorna (21,) float ou None se o modelo 3D nao estiver carregado."""
        if not self.hand_3d_ready:
            return None
        palm_depth = self.estimate_hand_depth(normalized_landmarks)
        if palm_depth is None:
            return None
        offsets = self.hand_node_z_offset
        if offsets is None or len(offsets) < 21:
            return np.full(21, palm_depth, dtype=np.float64)
        return palm_depth + offsets

    def get_smoothed_hand_depth(self, normalized_landmarks):
        """Estima a profundidade da mão (m) com suavização temporal (EMA).

        Usa o modelo 3D (escala aparente) e aplica um filtro exponencial para
        eliminar o jitter frame-a-frame que faz os dedos "saltarem".
        Retorna None se o modelo 3D não estiver carregado.
        """
        depth = self.estimate_hand_depth(normalized_landmarks)
        if depth is None:
            return None
        if self._hand_depth_ema is None or not self._hand_depth_ema_initialized:
            self._hand_depth_ema = depth
            self._hand_depth_ema_initialized = True
        else:
            # Rejeita saltos br (>30 cm) como outlier (mão entrou/saiu do frame):
            # mantém o EMA, não atualiza.
            if abs(depth - self._hand_depth_ema) < 0.30:
                self._hand_depth_ema = (
                    self._hand_depth_ema_alpha * depth
                    + (1.0 - self._hand_depth_ema_alpha) * self._hand_depth_ema
                )
        return self._hand_depth_ema

    def reset_hand_depth_ema(self):
        """Reseta a profundidade suavizada (após P ou ao iniciar captura)."""
        self._hand_depth_ema = None
        self._hand_depth_ema_initialized = False

    def solve_hand_landmark_calibration(
        self, samples_aux, samples_kin, samples_depths=None, verbose=True
    ):
        """Aprende a correspondencia nó-a-nó entre a mao aux e o Kinect.

        samples_aux/_kin: listas do mesmo tamanho; cada elemento e (21,2) em
        coordenadas normalizadas [0,1].
        samples_depths: lista opcional do mesmo tamanho; cada elemento e (21,)
        com a profundidade (m) de cada landmark medida pelo Kinect. Se
        fornecido, aprende tambem o modelo 3D (profundidade por escala).

        Modelo: homografia global (RANSAC) + residual mediano por no (0..20).
        Guarda hand_landmark_calibration.npz.
        Devolve True se a calibracao for aceitavel.
        """
        if not samples_aux or len(samples_aux) != len(samples_kin):
            return False
        count = len(samples_aux)
        all_aux = np.concatenate(samples_aux).reshape(-1, 1, 2)
        all_kin = np.concatenate(samples_kin).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(all_aux, all_kin, cv2.RANSAC, 2e-3)
        if homography is None:
            return False
        residuals = np.zeros((21, 2), np.float64)
        for node in range(21):
            aux = np.asarray([s[node] for s in samples_aux], np.float64).reshape(-1, 1, 2)
            kin = np.asarray([s[node] for s in samples_kin], np.float64).reshape(-1, 2)
            projected = cv2.perspectiveTransform(aux, homography).reshape(-1, 2)
            residuals[node] = np.median(kin - projected, axis=0)
        projected = cv2.perspectiveTransform(all_aux, homography).reshape(-1, 2)
        repeats = np.repeat(np.arange(count), 21)
        corrected = projected + residuals[repeats]
        rms = float(
            np.sqrt(np.mean(np.linalg.norm(all_kin.reshape(-1, 2) - corrected, axis=1) ** 2))
        )

        # --- Model 3D: apparent scale -> depth (multiplicative) ----------------
        # Fisica: depth = (focal * tamanho_real) / escala_aparente = C / escala
        # Mais robusto: depth = C * escala_referencia / escala_atual
        # Onde C = profundidade_media_das_amostras (na distancia de referencia)
        scale_depth_params = None
        node_z_offset = None
        good_3d_count = 0
        depth_range = (0.0, 0.0)
        if samples_depths is not None and len(samples_depths) == count:
            scales = []
            palm_depths = []
            per_node_z = [[] for _ in range(21)]
            for i in range(count):
                scale = self._apparent_hand_scale(samples_aux[i])
                if scale is None or scale <= 1e-6:
                    continue
                depths_i = np.asarray(samples_depths[i], np.float64).reshape(-1)
                if depths_i.shape[0] < 21:
                    continue
                valid_mask = np.isfinite(depths_i) & (depths_i > 0.1)
                if valid_mask.sum() < 14:
                    continue
                scales.append(scale)
                ref_depth = float(np.median(depths_i[valid_mask]))
                palm_depths.append(ref_depth)
                for n in range(21):
                    if valid_mask[n]:
                        per_node_z[n].append(depths_i[n] - ref_depth)
                good_3d_count += 1
            if len(scales) >= 4:
                scales = np.asarray(scales)
                palm_depths = np.asarray(palm_depths)
                depth_range = (float(palm_depths.min()), float(palm_depths.max()))
                # Modelo multiplicativo: depth = C * escala_media / escala_atual
                # C = produto medio (depth_i * escala_i) / escala_media
                # Isso e mais estavel que ajuste linear a/scale + b
                ref_scale = float(np.mean(scales))
                C = float(np.mean(palm_depths * scales) / ref_scale)
                # Valida o modelo
                predicted = C * ref_scale / scales
                residual = float(np.sqrt(np.mean((palm_depths - predicted) ** 2)))
                scale_depth_params = C
                self.hand_reference_scale = ref_scale
                # Per-node offsets: median + MAD-based outlier rejection
                node_z_offset = np.zeros(21, np.float64)
                for n in range(21):
                    vals = np.asarray(per_node_z[n])
                    if vals.size < 3:
                        continue
                    med = float(np.median(vals))
                    mad = float(np.median(np.abs(vals - med)))
                    filtered = vals[np.abs(vals - med) <= max(3.0 * mad, 0.02)]
                    node_z_offset[n] = float(np.median(filtered)) if filtered.size > 0 else med
                if verbose:
                    depth_span = depth_range[1] - depth_range[0]
                    print(
                        f"  3D model: {good_3d_count}/{count} good depth samples | "
                        f"depth range {depth_range[0]:.2f}-{depth_range[1]:.2f} m "
                        f"(span {depth_span:.2f} m) | C={C:.4f} | fit residual {residual*100:.1f} cm"
                    )
                    if good_3d_count < 6 or depth_span < 0.15:
                        print(
                            "  AVISO: poucos dados de depth ou pouca variacao de "
                            "profundidade. Durante H, varie mais a distancia "
                            "(0.8-1.5 m) e mantenha mao centrada no Kinect."
                        )
                    if depth_span < 0.10:
                        scale_depth_params = None
                        node_z_offset = None

        if verbose:
            extra = ""
            if scale_depth_params is not None:
                C = float(scale_depth_params)
                extra = (
                    f" | depth model: {C:.4f}*ref_scale/scale "
                    f"(z-offset range {node_z_offset.min()*100:.1f}.."
                    f"{node_z_offset.max()*100:.1f} cm)"
                )
            else:
                extra = " | 3D model NOT learned (insufficient depth data)"
            print(
                f"Calibracao de esqueleto: {count} fotos | RMS no-a-no {rms:.5f} "
                f"(~px Kinect {rms * 1920:.1f}){extra}"
            )
        save_dict = {
            "homography": homography,
            "residuals": residuals,
            "rms": rms,
            "version": HAND_CALIBRATION_VERSION,
            "samples": count,
        }
        if scale_depth_params is not None:
            save_dict["scale_depth_params"] = scale_depth_params
            save_dict["node_z_offset"] = node_z_offset
        np.savez(HAND_CALIBRATION_FILE, **save_dict)
        self.hand_landmark_homography = homography
        self.hand_landmark_residuals = residuals
        self.hand_landmark_rms = rms
        self.hand_scale_depth_params = scale_depth_params
        self.hand_node_z_offset = node_z_offset
        return True

    def auxiliary_to_kinect_landmark(self, normalized_points):
        """Mapeia (N,2) normalizado da aux -> (N,2) normalizado do Kinect.

        Usa a homografia global + residual por nó (significativo para os 21
        nós da mão, na ordem de MediaPipe).
        """
        points = np.asarray(normalized_points, np.float64).reshape(-1, 2)
        if not self.hand_landmark_ready:
            return np.full((points.shape[0], 2), np.nan, dtype=np.float64)
        projected = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2), self.hand_landmark_homography
        ).reshape(-1, 2)
        count = min(points.shape[0], 21)
        projected[:count] += self.hand_landmark_residuals[:count]
        return projected

    def find_board(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray,
                CHECKERBOARD_SIZE,
                cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )
        else:
            found, corners = cv2.findChessboardCorners(
                gray,
                CHECKERBOARD_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
        if not found:
            return None
        if hasattr(cv2, "findChessboardCornersSB"):
            return corners.reshape(-1, 2)
        return cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        ).reshape(-1, 2)

    def draw_board_detection(self, frame):
        corners = self.find_board(frame)
        if corners is not None:
            cv2.drawChessboardCorners(
                frame,
                CHECKERBOARD_SIZE,
                corners.reshape(-1, 1, 2),
                True,
            )
        return corners is not None

    def add_sample(self, auxiliary_frame, kinect_frame):
        auxiliary_corners = self.find_board(auxiliary_frame)
        kinect_corners = self.find_board(kinect_frame)
        if auxiliary_corners is None or kinect_corners is None:
            return False
        self.auxiliary_points.append(auxiliary_corners)
        self.kinect_points.append(kinect_corners)
        return True

    def solve(self):
        auxiliary_points = np.concatenate(self.auxiliary_points)
        kinect_points = np.concatenate(self.kinect_points)
        self.homography, _ = cv2.findHomography(auxiliary_points, kinect_points, cv2.RANSAC)
        np.savez(
            CALIBRATION_FILE,
            homography=self.homography,
            square_size_m=SQUARE_SIZE_M,
            version=CALIBRATION_VERSION,
            refinement_affine=self.refinement_affine,
        )

    def auxiliary_to_kinect(self, points):
        if self.homography is None:
            return np.full((len(points), 2), np.nan, dtype=np.float32)
        transformed = cv2.perspectiveTransform(
            np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), self.homography
        )
        transformed = transformed.reshape(-1, 2)
        affine_transformed = cv2.transform(
            np.asarray(transformed, dtype=np.float32).reshape(-1, 1, 2),
            self.refinement_affine
        )
        return affine_transformed.reshape(-1, 2)

    def auxiliary_to_kinect_stereo(self, points, kinect_depth_m):
        if not self.stereo_usable:
            return self.auxiliary_to_kinect(points)
        auxiliary_points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        matrix = (
            self.auxiliary_matrix_runtime
            if self.auxiliary_matrix_runtime is not None
            else self.auxiliary_matrix
        )
        normalized = cv2.undistortPoints(
            auxiliary_points, matrix, self.auxiliary_dist
        ).reshape(-1, 2)
        depths = np.asarray(kinect_depth_m, dtype=np.float32).reshape(-1)
        if len(depths) == 1:
            depths = np.full(len(normalized), depths[0], dtype=np.float32)
        if len(depths) != len(normalized):
            return self.auxiliary_to_kinect(points)
        valid_depths = depths[np.isfinite(depths) & (depths > 0)]
        if valid_depths.size == 0:
            return self.auxiliary_to_kinect(points)
        depths = np.where(
            np.isfinite(depths) & (depths > 0),
            depths,
            np.median(valid_depths),
        )
        # Profundidades inválidas (NaN, 0, fora do sensor) são substituídas
        # pela mediana das válidas. Depois, um clip físico (0.4–4.0 m) evita
        # valores absurdos. NÃO fazemos suavização por índice de landmark aqui:
        # os índices 0..20 do MediaPipe não são espacialmente contíguos
        # (0=punho, 8=ponta do indicador, 12=ponta do medio), e a suavização
        # por vizinhança de índice destruiria os offsets 3D aprendidos pelo
        # modelo hand_3d (dedos ficariam sempre no mesmo Z da palma).
        depths = np.clip(np.asarray(depths, np.float64), 0.4, 4.0)
        # Se metade ou mais das profundidades originais não são válidas,
        # usa a mediana única para toda a mão (fallback rígido).
        original = np.asarray(kinect_depth_m, dtype=np.float32).reshape(-1)
        original_finite = np.isfinite(original) & (original > 0)
        if original_finite.sum() < 0.5 * original.size:
            depths = np.full(original.size, float(np.median(depths)), dtype=np.float64)
        kinect_3d = []
        rotation_inverse = np.asarray(self.stereo_rotation, np.float64).T
        translation = np.asarray(self.stereo_translation, np.float64).reshape(3)
        offset = -rotation_inverse @ translation
        for ray_xy, target_z in zip(normalized, depths):
            ray = np.array([ray_xy[0], ray_xy[1], 1.0])
            direction = rotation_inverse @ ray
            denominator = direction[2]
            if target_z <= 0 or abs(denominator) < 1e-6:
                kinect_3d.append([np.nan, np.nan, np.nan])
                continue
            scale = (target_z - offset[2]) / denominator
            kinect_3d.append(rotation_inverse @ (scale * ray - translation))
        kinect_3d = np.asarray(kinect_3d, dtype=np.float32)
        projected, _ = cv2.projectPoints(
            kinect_3d,
            np.zeros(3),
            np.zeros(3),
            self.kinect_matrix,
            self.kinect_dist,
        )
        return projected.reshape(-1, 2)

    def triangulate_hand_points(self, aux_pixels, kinect_pixels):
        """Triangula nos 3D (m, frame do Kinect) a partir das deteccoes 2D
        SIMULTANEAS das duas cameras (MediaPipe em cada uma; correspondencia
        trivial pelo indice do no MediaPipe). Usa a calibracao stereo
        (R, T, K, dist) ja carregada.

        Este e o caminho do 'espaco vetorial 3D': a profundidade deixa de ser
        ESTIMADA (modelo de escala aparente / EMA) e passa a ser MEDIDA pela
        interseccao dos raios das duas vistas. Imune a inclinacao da mao.

        Convencao (identica a auxiliary_to_kinect_stereo): X_aux = R X_kin + T.
        Camera 1 = Kinect (na origem, P1=[I|0]); camera 2 = auxiliar (P2=[R|T]).
        Com coordenadas normalizadas e T em metros, a saida vem em metros.

        Retorna (points_3d (N,3) no frame do Kinect, reproj_err_px (N,)) ou
        (None, None) se a triangulacao nao for possivel/confiavel.
        """
        if not (self.stereo_usable and self.kinect_matrix is not None):
            return None, None
        aux_px = np.asarray(aux_pixels, np.float64).reshape(-1, 1, 2)
        kin_px = np.asarray(kinect_pixels, np.float64).reshape(-1, 1, 2)
        if aux_px.shape[0] == 0 or aux_px.shape[0] != kin_px.shape[0]:
            return None, None
        aux_matrix = (
            self.auxiliary_matrix_runtime
            if self.auxiliary_matrix_runtime is not None
            else self.auxiliary_matrix
        )
        rays_kin = cv2.undistortPoints(
            kin_px, self.kinect_matrix, self.kinect_dist
        ).reshape(-1, 2)
        rays_aux = cv2.undistortPoints(
            aux_px, aux_matrix, self.auxiliary_dist
        ).reshape(-1, 2)
        if not (np.isfinite(rays_kin).all() and np.isfinite(rays_aux).all()):
            return None, None
        R = np.asarray(self.stereo_rotation, np.float64).reshape(3, 3)
        T = np.asarray(self.stereo_translation, np.float64).reshape(3, 1)
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, T])
        points4 = cv2.triangulatePoints(
            P1,
            P2,
            rays_kin.reshape(-1, 1, 2).astype(np.float64),
            rays_aux.reshape(-1, 1, 2).astype(np.float64),
        )
        w = points4[3]
        if not np.isfinite(w).all() or (np.abs(w) < 1e-9).any():
            return None, None
        points3d = (points4[:3] / w).T  # (N,3) em metros, frame do Kinect
        z = points3d[:, 2]
        # Plausibilidade fisica (volume de trabalho da mao). Nos improvaveis
        # (oclusao, deteccao trocada entre maos) viram NaN e nao sao desenhados.
        plausible = np.isfinite(points3d).all(axis=1) & (z > 0.3) & (z < 4.0)
        if plausible.sum() < max(5, int(0.5 * len(points3d))):
            return None, None
        # Erro de reprojecao por no (px Kinect) = qualidade da triangulacao
        projected, _ = cv2.projectPoints(
            points3d.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            self.kinect_matrix,
            self.kinect_dist,
        )
        reproj = np.linalg.norm(
            projected.reshape(-1, 2) - kin_px.reshape(-1, 2), axis=1
        )
        points3d[~plausible] = np.nan
        return points3d, reproj

    def project_camera_points(self, points_camera, size=None):
        """Projeta pontos 3D (m, camera space) em pixels do RGB do Kinect.

        Usa a matriz intrinseca da calibracao stereo; sem calibracao cai nos
        parametros nominais do Kinect v2 (@1920x1080, escalados para `size`)
        — suficiente para o overlay do braco, que e ilustrativo.
        """
        import cv2

        points = np.asarray(points_camera, np.float64).reshape(-1, 1, 3)
        matrix, dist = self.kinect_matrix, self.kinect_dist
        if matrix is None or dist is None:
            matrix = KINECT_V2_INTRINSICS_1080P.copy()
            dist = np.zeros(5, np.float64)
            if size is not None:
                width, height = float(size[0]), float(size[1])
                matrix[0, 0] *= width / KINECT_V2_COLOR_SIZE[0]
                matrix[0, 2] *= width / KINECT_V2_COLOR_SIZE[0]
                matrix[1, 1] *= height / KINECT_V2_COLOR_SIZE[1]
                matrix[1, 2] *= height / KINECT_V2_COLOR_SIZE[1]
        projected, _ = cv2.projectPoints(
            points, np.zeros(3), np.zeros(3),
            np.asarray(matrix, np.float64), np.asarray(dist, np.float64),
        )
        return projected.reshape(-1, 2)

    def set_auxiliary_size(self, frame):
        height, width = frame.shape[:2]
        if (width, height) == (self.auxiliary_width, self.auxiliary_height):
            return
        self.auxiliary_height, self.auxiliary_width = height, width
        self.auxiliary_matrix_runtime = None
        runtime = None
        if self.calibrated_auxiliary_size is not None \
                and self.auxiliary_matrix is not None:
            cal_width, cal_height = self.calibrated_auxiliary_size
            if (width, height) != (cal_width, cal_height):
                scale_x = width / cal_width
                scale_y = height / cal_height
                runtime = np.asarray(self.auxiliary_matrix, np.float64).copy()
                runtime[0, 0] *= scale_x
                runtime[0, 2] *= scale_x
                runtime[1, 1] *= scale_y
                runtime[1, 2] *= scale_y
                self.auxiliary_matrix_runtime = runtime
                if not self.auxiliary_scale_warned:
                    self.auxiliary_scale_warned = True
                    print(
                        f"AVISO: camera auxiliar {self.active_aux} em "
                        f"{width}x{height}, mas a calibracao foi feita em "
                        f"{cal_width}x{cal_height}; usando matriz intrinseca "
                        "escalada. Recalibre na mesma resolucao para melhor "
                        "precisao.", flush=True
                    )
        # persistencia da matriz em escala por camera (troca de aux mantem)
        if self.active_aux in self.aux_models:
            self.aux_models[self.active_aux]["runtime_matrix"] = runtime


class ArmLinkModel:
    """Cinematica inversa (IK) de 2 elos para o braco, com elos rigidos.

    Anatomia: o comprimento ombro->cotovelo (L1) e cotovelo->punho (L2) sao
    CONSTANTES da pessoa (nao mudam com o movimento). O Kinect mede as juntas
    com ruido de ~cm, e o MediaPipe/triangulacao mede o PUNHO bem melhor do que
    o cotovelo. Logo:

      1. mede-se L1 e L2 UMA vez (mediana de N amostras com o braco em
         repouso, durante a calibracao de origem) e guarda-se em
         arm_model.json;
      2. a cada frame, dados o OMBRO e o PUNHO, o cotovelo e reconduzido ao
         circulo de interseccao das duas esferas (raio L1 em volta do ombro,
         raio L2 em volta do punho), preservando a DIRECAO DE DOBRAMENTO
         observada (plano do braco).

    Resultado: a geometria fica consistente (o braco nunca "estica" nem
    "encolhe" quando o esqueleto erra) e o cotovelo deixa de tremer, porque
    passa a depender so de ombro + punho + 2 constantes.

    Convencao de eixos: camera space do Kinect v2 -> X direita, Y para CIMA,
    Z para frente (afastando-se da camera). Angulos em graus.
    """

    def __init__(self, path=ARM_MODEL_PATH):
        self.path = Path(path)
        self.l1 = None                 # ombro -> cotovelo (m)
        self.l2 = None                 # cotovelo -> punho (m)
        self.bend_ref = None           # direcao de dobramento de referencia
        self.side = None               # "right" / "left"
        self.samples = 0
        self._l1_samples = []
        self._l2_samples = []
        self._bend_samples = []
        self.load()

    # ----------------------------------------------------------- persistencia
    def load(self):
        """Carrega arm_model.json. Devolve True se o modelo ficou pronto."""
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError, KeyError):
            return False
        if int(data.get("version", 0)) != ARM_MODEL_VERSION:
            return False
        l1, l2 = data.get("l1_m"), data.get("l2_m")
        if not l1 or not l2:
            return False
        self.l1, self.l2 = float(l1), float(l2)
        bend = data.get("bend_ref")
        self.bend_ref = None if bend is None else np.asarray(bend, np.float64)
        self.side = data.get("side")
        self.samples = int(data.get("samples", 0))
        return True

    def save(self):
        if not self.ready:
            return False
        payload = {
            "version": ARM_MODEL_VERSION,
            "l1_m": self.l1,
            "l2_m": self.l2,
            "bend_ref": (None if self.bend_ref is None
                         else [float(v) for v in self.bend_ref]),
            "side": self.side,
            "samples": self.samples,
            "nota": ("Elos rigidos do braco (m) no camera space do Kinect, "
                     "usados pela cinematica inversa de 2 elos. bend_ref = "
                     "direcao unitaria de dobramento do cotovelo no repouso."),
        }
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"[braco] falha ao salvar {self.path}: {exc}", flush=True)
            return False
        return True

    @property
    def ready(self):
        """True quando L1/L2 sao conhecidos (IK disponivel)."""
        return self.l1 is not None and self.l2 is not None

    @property
    def reach_m(self):
        """Alcance maximo ombro->punho com estes elos (m)."""
        return None if not self.ready else float(self.l1 + self.l2)


    # ------------------------------------------------------------ calibracao
    def add_sample(self, shoulder, elbow, wrist, side=None):
        """Acumula juntas para calibrar L1/L2 e a direcao de dobramento.

        Devolve True se a amostra passou pelo filtro de plausibilidade.
        """
        if shoulder is None or elbow is None or wrist is None:
            return False
        s = np.asarray(shoulder, np.float64)
        e = np.asarray(elbow, np.float64)
        w = np.asarray(wrist, np.float64)
        if not np.all(np.isfinite(np.concatenate((s, e, w)))):
            return False
        l1 = float(np.linalg.norm(e - s))
        l2 = float(np.linalg.norm(w - e))
        if not ARM_LINK_MIN_M < l1 < ARM_LINK_MAX_M:
            return False
        if not ARM_LINK_MIN_M < l2 < ARM_LINK_MAX_M:
            return False
        self._l1_samples.append(l1)
        self._l2_samples.append(l2)
        bend = self.bend_direction(s, e, w)
        if bend is not None:
            self._bend_samples.append(bend)
        if side:
            self.side = side
        return True

    def finish_calibration(self, min_samples=ARM_MODEL_MIN_SAMPLES):
        """Fecha a calibracao com a MEDIANA das amostras (robusta a outliers).

        Devolve True se o modelo ficou pronto e foi salvo em arm_model.json.
        """
        n = len(self._l1_samples)
        if n < min_samples:
            print(f"[braco] AVISO: {n} amostras de elos (minimo "
                  f"{min_samples}); modelo NAO calibrado -> sem IK nesta "
                  "sessao (colunas ARM_* vem do esqueleto cru)", flush=True)
            self.reset_samples()
            return False
        self.l1 = float(np.median(self._l1_samples))
        self.l2 = float(np.median(self._l2_samples))
        if self._bend_samples:
            mean = np.asarray(self._bend_samples, np.float64).mean(axis=0)
            norm = float(np.linalg.norm(mean))
            self.bend_ref = (mean / norm) if norm > 1e-6 else None
        self.samples = n
        self.reset_samples()
        ok = self.save()
        lado = f", lado {self.side}" if self.side else ""
        print(f"[braco] elos calibrados: ombro->cotovelo={self.l1:.3f} m, "
              f"cotovelo->punho={self.l2:.3f} m, alcance maximo="
              f"{self.reach_m:.3f} m (amostras: {self.samples}{lado})",
              flush=True)
        return ok

    def reset_samples(self):
        self._l1_samples = []
        self._l2_samples = []
        self._bend_samples = []

    # -------------------------------------------------------------- geometria
    @staticmethod
    def bend_direction(shoulder, elbow, wrist):
        """Direcao unitaria de dobramento do cotovelo: componente de
        (cotovelo-ombro) perpendicular ao eixo ombro->punho, normalizada.

        E o vetor que define o PLANO do braco. Devolve None quando degenerado
        (cotovelo colinear com o eixo: braco totalmente esticado).
        """
        axis = np.asarray(wrist, np.float64) - np.asarray(shoulder, np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return None
        unit = axis / norm
        offset = np.asarray(elbow, np.float64) - np.asarray(shoulder, np.float64)
        bend = offset - unit * float(np.dot(offset, unit))
        norm = float(np.linalg.norm(bend))
        if norm < 1e-6:
            return None
        return bend / norm

    @staticmethod
    def interior_angle_deg(shoulder, elbow, wrist, eps=1e-9):
        """Angulo INTERNO no cotovelo (graus): 180 = braco esticado,
        ~30-60 = dobrado. NaN se algum elo tiver comprimento nulo."""
        a = np.asarray(shoulder, np.float64) - np.asarray(elbow, np.float64)
        b = np.asarray(wrist, np.float64) - np.asarray(elbow, np.float64)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < eps or nb < eps:
            return float("nan")
        cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        return float(math.degrees(math.acos(cosine)))

    @staticmethod
    def _unit(vector, eps=1e-9):
        """Atalho para unit_vector (mantido por compatibilidade interna)."""
        return unit_vector(vector, eps)


    @staticmethod
    def _perpendicular(unit, reference):
        """Componente de `reference` perpendicular a `unit`, normalizada."""
        offset = np.asarray(reference, np.float64) - unit * float(
            np.dot(reference, unit))
        norm = float(np.linalg.norm(offset))
        if norm < 1e-9:
            return None
        return offset / norm

    def solve_elbow(self, shoulder, wrist, bend_ref=None):
        """Posiciona o cotovelo com os elos FIXOS (IK de 2 esferas).

        O cotovelo pertence ao circulo de interseccao da esfera de raio L1 em
        volta do ombro com a esfera de raio L2 em volta do punho; dentro desse
        circulo, a direcao e dada pela direcao de dobramento.

        Devolve (elbow, clamped): `clamped=1` quando o punho medido esta a uma
        distancia maior que L1+L2 (cotovelo satura com o braco esticado) ou
        menor que |L1-L2| -> geometria fisicamente inalcancavel.
        """
        if not self.ready:
            return None, 0
        s = np.asarray(shoulder, np.float64)
        w = np.asarray(wrist, np.float64)
        axis = w - s
        d = float(np.linalg.norm(axis))
        if d < 1e-6:
            return None, 0
        unit = axis / d
        reach = self.l1 + self.l2
        folded = abs(self.l1 - self.l2)
        clamped = int(d > reach + ARM_IK_REACH_TOL_M or
                      d < folded - ARM_IK_REACH_TOL_M)
        dd = float(np.clip(d, folded + 1e-9, reach))
        # Sem epsilon no limite SUPERIOR: com dd = reach o raio de solucao e
        # exatamente zero (braco esticado, angulo interno = 180.0 exato).
        # Com epsilon, o acos perto de 180 amplifica o erro para ~0.005 graus.
        # a = distancia (m) do ombro ao centro do circulo de solucoes
        a = (self.l1 ** 2 - self.l2 ** 2 + dd ** 2) / (2.0 * dd)
        radius = math.sqrt(max(self.l1 ** 2 - a ** 2, 0.0))
        bend = None
        if bend_ref is not None:
            bend = self._perpendicular(unit, bend_ref)
        if bend is None and self.bend_ref is not None:
            bend = self._perpendicular(unit, self.bend_ref)
        if bend is None:
            # Sem referencia de dobramento: qualquer perpendicular serve, mas
            # a direcao do cotovelo fica indeterminada (marcado como ik=3).
            helper = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(unit, helper))) > 0.9:
                helper = np.array([1.0, 0.0, 0.0])
            bend = self._unit(np.cross(unit, helper))
            if bend is None:
                return None, clamped
        elbow = s + unit * a + bend * radius
        return elbow, clamped

    def solve_arm(self, shoulder, elbow_measured, wrist):
        """Geometria completa do braco a partir das juntas disponiveis.

        Combina a medida (esqueleto + punho triangulado) com os elos fixos:
          - sem modelo calibrado -> devolve as juntas medidas (ik=0);
          - com modelo -> cotovelo reconstruido pela IK (ik=1/2/3).

        Devolve dict (posicoes em m, angulos em graus) ou None se faltar o
        ombro ou o punho (sem esses dois nao ha geometria possivel).
        """
        if shoulder is None or wrist is None:
            return None
        s = np.asarray(shoulder, np.float64)
        w = np.asarray(wrist, np.float64)
        elbow_measured = (None if elbow_measured is None
                          else np.asarray(elbow_measured, np.float64))
        bend = None
        if elbow_measured is not None:
            bend = self.bend_direction(s, elbow_measured, w)
        elbow, clamped = self.solve_elbow(s, w, bend_ref=bend)
        ik = 0
        if elbow is not None:
            if bend is not None:
                ik = 1
            elif self.bend_ref is not None:
                ik = 2
            else:
                ik = 3
        elif elbow_measured is not None:
            elbow = elbow_measured          # sem IK: esqueleto cru
        else:
            return None
        return self._assemble(s, np.asarray(elbow, np.float64), w, ik, clamped)

    @staticmethod
    def solve_raw(shoulder, elbow, wrist):
        """Geometria do braco SEM modelo de elos (esqueleto cru, ik=0).

        Usado quando a IK esta desligada (--sem-ik) ou quando nao houve
        calibracao dos elos. Exige as 3 juntas medidas: sem cotovelo nao ha
        angulo articular a calcular.
        """
        if shoulder is None or elbow is None or wrist is None:
            return None
        return ArmLinkModel._assemble(
            np.asarray(shoulder, np.float64), np.asarray(elbow, np.float64),
            np.asarray(wrist, np.float64), 0, 0)

    @staticmethod
    def _assemble(s, e, w, ik, clamped):
        """Monta o dicionario de geometria (juntas + angulos) do braco."""
        upper = unit_vector(e - s)
        result = {
            "shoulder": s,
            "elbow": e,
            "wrist": w,
            "ik": int(ik),
            "clamped": int(clamped),
            "l1_m": float(np.linalg.norm(e - s)),
            "l2_m": float(np.linalg.norm(w - e)),
            "elbow_angle_deg": ArmLinkModel.interior_angle_deg(s, e, w),
        }
        if upper is None:
            result["shoulder_elev_deg"] = float("nan")
            result["shoulder_azim_deg"] = float("nan")
        else:
            # Elevacao do braco (ombro->cotovelo): +90 = apontando para cima,
            # 0 = horizontal, -90 = para baixo. Azimute no plano horizontal
            # medido a partir do eixo Z (frente): +90 = para a direita.
            result["shoulder_elev_deg"] = float(math.degrees(
                math.asin(float(np.clip(upper[1], -1.0, 1.0)))))
            result["shoulder_azim_deg"] = float(math.degrees(
                math.atan2(float(upper[0]), float(upper[2]))))
        return result


class KinectHandTracker:
    """Wrapper do Kinect v2: frames de cor/depth/body + MediaPipe por camera."""

    def __init__(self, aux_indices=AUX_CAMERA_INDICES):
        self.aux_indices = tuple(int(index) for index in aux_indices)
        self.kinect = PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color
            | PyKinectV2.FrameSourceTypes_Depth
            | PyKinectV2.FrameSourceTypes_Body
        )
        self.mapper = self.kinect._mapper
        self._depth_error_reported = set()
        self._map_error_reported = False
        self.last_hand_reason = ""
        #: Lateralidade da mao utilizada no ultimo frame ('left'/'right'/'').
        self.last_hand_side = ""
        self._last_depth_capture_time = 0
        self._patch_depth_handler()
        self.last_color = None
        self.last_depth = None
        self.last_bodies = None
        self.last_kinect_hands = []
        self.last_hand_depth_m: Optional[float] = None
        self.depth_frame_index = 0
        self._color_to_depth_cache = None
        self._color_to_depth_cache_index = -1
        self._color_to_depth_output = None
        self.color_width = self.kinect.color_frame_desc.Width
        self.color_height = self.kinect.color_frame_desc.Height
        self.depth_width = self.kinect.depth_frame_desc.Width
        self.depth_height = self.kinect.depth_frame_desc.Height
        self.hand_options_kinect = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_hand_model())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            # Kinect 1920x1080: mao fica pequena longe -> limiar baixo p/ detectar.
            # (Depth enxerga no escuro via IR, mas o RGB precisa de LUZ boa.)
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
        )
        self.hand_options_aux = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_hand_model())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(self.hand_options_kinect)
        self.auxiliary_hands = mp.tasks.vision.HandLandmarker.create_from_options(self.hand_options_aux)
        # Detector por camera auxiliar (0=laptop, 1=usb, ...): cada stream usa o
        # SEU detector/timestamp (o VideoMode do MediaPipe nao admite intercalar
        # dois fluxos no mesmo landmarker sem derrubar o tracking por aparicao).
        self.auxiliary_detectors = {}
        self.auxiliary_timestamps_ms = {}
        for index in self.aux_indices:
            self.auxiliary_detectors[int(index)] = (
                mp.tasks.vision.HandLandmarker.create_from_options(
                    self.hand_options_aux))
            self.auxiliary_timestamps_ms[int(index)] = 0
        self.kinect_frame_timestamp_ms = 0
        self.auxiliary_frame_timestamp_ms = 0

    def frames(self):
        if self.kinect.has_new_color_frame():
            data = self.kinect.get_last_color_frame()
            self.last_color = data.reshape((self.color_height, self.color_width, 4))
        # Novos frames de depth sao detectados pelo timestamp real de captura
        # (o has_new_depth_frame do pykinect2 nao e confiavel: o acesso nunca
        # e registrado e o buffer pre-alocado com zeros mascara falhas).
        capture_time = getattr(self.kinect, "_last_depth_frame_time", 0)
        if capture_time != self._last_depth_capture_time:
            data = self.kinect.get_last_depth_frame()
            if data is not None:
                self.last_depth = data.reshape((self.depth_height, self.depth_width))
                if self._last_depth_capture_time == 0:
                    print("Primeiro frame de profundidade recebido")
                self._last_depth_capture_time = capture_time
                self.depth_frame_index += 1
        if self.kinect.has_new_body_frame():
            self.last_bodies = self.kinect.get_last_body_frame()
        return self.last_color, self.last_depth

    def _log_depth_error(self, step, exc):
        """Reporta uma vez cada etapa que falhou no fluxo de depth.

        O handler original do pykinect2 engole excecoes com 'except: pass',
        o que deixa o depth morto em silencio (buffer pre-alocado com zeros).
        """
        if step not in self._depth_error_reported:
            self._depth_error_reported.add(step)
            print(
                f"ERRO no fluxo de profundidade ({step}): "
                f"{type(exc).__name__}: {exc}"
            )

    def _patch_depth_handler(self):
        """Substitui o handler de depth por uma versao que reporta erros."""
        kinect = self.kinect

        def robust_depth_handler(handle_index):
            try:
                event_data = kinect._depth_frame_reader.GetFrameArrivedEventData(
                    kinect._handles[handle_index]
                )
            except BaseException as exc:
                self._log_depth_error("GetFrameArrivedEventData", exc)
                return
            frame_ref = event_data.FrameReference
            try:
                frame = frame_ref.AcquireFrame()
            except BaseException as exc:
                self._log_depth_error("AcquireFrame", exc)
                frame_ref = None
                return
            try:
                with kinect._depth_frame_lock:
                    frame.CopyFrameDataToArray(
                        kinect._depth_frame_data_capacity,
                        kinect._depth_frame_data,
                    )
                    kinect._last_depth_frame_time = time.perf_counter()
            except BaseException as exc:
                self._log_depth_error("CopyFrameDataToArray", exc)
            frame = None
            frame_ref = None
            event_data = None

        kinect.handle_depth_arrived = robust_depth_handler

    def _color_to_depth_pixel(self, color_x: int, color_y: int) -> Optional[tuple[int, int]]:
        # MapColorFrameToDepthSpace (COM) exige o ponteiro ctypes do buffer
        # interno de depth e um buffer de saida _DepthSpacePoint pre-alocado;
        # passar o array numpy falha sempre (o wrapper do pykinect2 nao o
        # aceita) e o except engolia o erro -> "0/48"/"Kinect sem mao".
        # O mapeamento roda uma vez por frame de depth (cache por indice).
        try:
            if self._color_to_depth_output is None:
                count = self.color_width * self.color_height
                self._color_to_depth_output = (PyKinectV2._DepthSpacePoint * count)()
            if self._color_to_depth_cache_index != self.depth_frame_index:
                self.kinect._mapper.MapColorFrameToDepthSpace(
                    self.depth_width * self.depth_height,
                    self.kinect._depth_frame_data,
                    self.color_width * self.color_height,
                    self._color_to_depth_output,
                )
                self._color_to_depth_cache_index = self.depth_frame_index
            point = self._color_to_depth_output[color_y * self.color_width + color_x]
            if not math.isfinite(float(point.x)) or not math.isfinite(float(point.y)):
                return None
            depth_x, depth_y = int(round(point.x)), int(round(point.y))
        except BaseException as exc:
            if not self._map_error_reported:
                self._map_error_reported = True
                print(
                    f"ERRO mapeo color->depth ({type(exc).__name__}): {exc}\n"
                    "  (relativo ao caminho MapColorFrameToDepthSpace; pode ser "
                    "rigido de FOV/posicion do depth ou del proprio mapper)"
                )
            return None
        if 0 <= depth_x < self.depth_width and 0 <= depth_y < self.depth_height:
            return depth_x, depth_y
        return None

    def _median_depth(self, depth_x: int, depth_y: int, radius=None) -> Optional[float]:
        if radius is None:
            radius = DEPTH_PATCH_RADIUS
        patch = self.last_depth[
            max(0, depth_y - radius):min(self.depth_height, depth_y + radius + 1),
            max(0, depth_x - radius):min(self.depth_width, depth_x + radius + 1),
        ].astype(np.float32)
        valid = patch[(patch >= DEPTH_MIN_MM) & (patch <= DEPTH_MAX_MM)]
        if valid.size < 3:
            return None
        value = float(np.median(valid))
        return value if math.isfinite(value) else None

    def depth_at_projected(self, alignment, auxiliary_landmark, guess_depth_m):
        """Profundidade da cena onde a palma auxiliar projeta.

        Projeta o landmark da auxiliar com um chute de profundidade, le a
        profundidade do Kinect nesse pixel e devolve a medicao (m). Serve
        de prior de profundidade quando a mao nao esta detectada no
        proprio Kinect. Retorna o chute se qualquer passo falhar.
        """
        if alignment.auxiliary_width == 0 or self.last_depth is None:
            return guess_depth_m
        palm_pixel = (
            auxiliary_landmark.x * alignment.auxiliary_width,
            auxiliary_landmark.y * alignment.auxiliary_height,
        )
        estimate = guess_depth_m
        try:
            for _ in range(2):
                projected = np.asarray(
                    alignment.auxiliary_to_kinect_stereo([palm_pixel], [estimate]),
                    np.float64,
                ).reshape(-1, 2)
                if not np.isfinite(projected).all():
                    return guess_depth_m
                x, y = int(round(projected[0, 0])), int(round(projected[0, 1]))
                if not (0 <= x < self.color_width and 0 <= y < self.color_height):
                    return guess_depth_m
                depth_pixel = self._color_to_depth_pixel(x, y)
                if depth_pixel is None:
                    return guess_depth_m
                depth_mm = self._median_depth(*depth_pixel)
                if depth_mm is None:
                    return guess_depth_m
                sampled = depth_mm / 1000.0
                tolerance = max(0.5, 0.5 * estimate)
                if abs(sampled - estimate) > tolerance:
                    # amostra provavelmente pegou o fundo (o chute estava
                    # longe); mantem o chute para nao entrar em loop
                    return estimate
                estimate = sampled
            return estimate
        except (ValueError, cv2.error):
            return guess_depth_m

    def depth_view(self):
        if self.last_depth is None:
            return None
        scale = 255.0 / (DEPTH_MAX_MM - DEPTH_MIN_MM)
        depth_view = cv2.convertScaleAbs(
            self.last_depth, alpha=scale, beta=-DEPTH_MIN_MM * scale
        )
        invalid = (self.last_depth < DEPTH_MIN_MM) | (self.last_depth > DEPTH_MAX_MM)
        depth_view[invalid] = 0
        return cv2.applyColorMap(depth_view, cv2.COLORMAP_JET)

    def _draw_skeleton_depth(self, frame):
        if self.last_bodies is None:
            return
        for body in self.last_bodies.bodies:
            if not body.is_tracked:
                continue
            points = {}
            for joint_type in range(PyKinectV2.JointType_Count):
                joint = body.joints[joint_type]
                if joint.TrackingState == PyKinectV2.TrackingState_NotTracked:
                    continue
                point = self.mapper.MapCameraPointToDepthSpace(joint.Position)
                if not np.isfinite((point.x, point.y)).all():
                    continue
                pixel = (int(round(point.x)), int(round(point.y)))
                if 0 <= pixel[0] < self.depth_width and 0 <= pixel[1] < self.depth_height:
                    points[joint_type] = pixel
            for start, end in BODY_CONNECTIONS:
                if start in points and end in points:
                    cv2.line(frame, points[start], points[end], (255, 80, 0), 3)
            for pixel in points.values():
                cv2.circle(frame, pixel, 5, (255, 180, 0), -1)

    def depth_stats(self):
        """(pixels_validos, min_mm, max_mm, fracao_valida) do frame atual."""
        if self.last_depth is None:
            return None
        depth = self.last_depth
        valid = (depth >= DEPTH_MIN_MM) & (depth <= DEPTH_MAX_MM)
        return (
            int(valid.sum()),
            float(depth.min()),
            float(depth.max()),
            float(valid.sum()) / depth.size,
        )

    def _color_to_depth_pixels(self, color_pixels):
        pixels = []
        for color_x, color_y in color_pixels:
            depth_pixel = self._color_to_depth_pixel(int(color_x), int(color_y))
            if depth_pixel is not None:
                pixels.append(depth_pixel)
        return pixels

    def _landmark_camera_point(self, landmark):
        color_x = int(np.clip(landmark.x * self.color_width, 0, self.color_width - 1))
        color_y = int(np.clip(landmark.y * self.color_height, 0, self.color_height - 1))
        depth_pixel = self._color_to_depth_pixel(color_x, color_y)
        if depth_pixel is None:
            return None
        # Parche mas grande que el default (11x11): los dedos son contornos
        # finos y el depth del Kinect es ruidoso ahi; un parche mayor estabiliza
        # la profundidad por landmark (menos huecos/fondo).
        depth_mm = self._median_depth(*depth_pixel, radius=5)
        if depth_mm is None:
            return None
        depth_point = PyKinectV2._DepthSpacePoint()
        depth_point.x, depth_point.y = map(float, depth_pixel)
        # MapDepthPointToCameraSpace espera depth como c_ushort (inteiros,
        # milimetros); medir as medias em float causa ctypes.ArgumentError.
        camera_point = self.mapper.MapDepthPointToCameraSpace(
            depth_point, int(round(depth_mm))
        )
        return np.array([camera_point.x, camera_point.y, camera_point.z], dtype=float)

    @staticmethod
    def palm_normal_from_points(wrist, index_mcp, pinky_mcp, eps=1e-9):
        """Normal unitaria do plano da palma a partir de 3 pontos 3D (m).

        Usa punho (0), MCP do indicador (5) e MCP do dedo minimo (17). O sinal
        e fixado apontando para a CAMERA (nz < 0), o que torna a orientacao
        comparavel entre frames: palma vs. dorso virado para a camera.
        """
        normal = np.cross(
            np.asarray(index_mcp, np.float64) - np.asarray(wrist, np.float64),
            np.asarray(pinky_mcp, np.float64) - np.asarray(wrist, np.float64),
        )
        length = float(np.linalg.norm(normal))
        if length < eps:
            return None
        normal = normal / length
        if normal[2] > 0:
            normal = -normal
        return normal

    def palm_normal_camera(self, landmarks):
        """Normal da palma medida pelo Kinect (um lookup de profundidade por
        landmark; mais caro que a versao sobre a triangulacao)."""
        wrist = self._landmark_camera_point(landmarks[0])
        index_mcp = self._landmark_camera_point(landmarks[5])
        pinky_mcp = self._landmark_camera_point(landmarks[17])
        if wrist is None or index_mcp is None or pinky_mcp is None:
            return None
        return self.palm_normal_from_points(wrist, index_mcp, pinky_mcp)

    def palm_orientation_camera(self, landmarks):
        """(normal, direcao_dedos) da palma pelo depth do Kinect.

        Quatro lookups de profundidade (punho 0, MCP medio 9, MCP indicador 5,
        MCP minimo 17). Devolve (None, None) se os tres pontos do plano
        falharem; a direcao dos dedos pode vir None isoladamente.
        """
        wrist = self._landmark_camera_point(landmarks[0])
        if wrist is None:
            return None, None
        index_mcp = self._landmark_camera_point(landmarks[5])
        pinky_mcp = self._landmark_camera_point(landmarks[17])
        middle_mcp = self._landmark_camera_point(landmarks[9])
        return palm_frame(wrist, index_mcp, pinky_mcp, middle_mcp)

    @staticmethod
    def _joint_camera_position(joints, jtype, z_range=(0.3, 4.0)):
        """Posicao (m) de uma junta do SDK: devolve (vetor, tracked) ou
        (None, False). Juntas 'Inferred' (extrapoladas pelo SDK numa oclusao
        parcial) SAO aceitas; o flag `tracked` diz se a junta e medida."""
        try:
            joint = joints[jtype]
        except (IndexError, KeyError, TypeError):
            return None, False
        not_tracked = getattr(PyKinectV2, "TrackingState_NotTracked", 0)
        state = getattr(joint, "TrackingState", not_tracked)
        if state == not_tracked:
            return None, False
        pos = joint.Position
        x, y, z = float(pos.x), float(pos.y), float(pos.z)
        if not np.all(np.isfinite([x, y, z])):
            return None, False
        if not z_range[0] < z < z_range[1]:
            return None, False
        tracked = state == PyKinectV2.TrackingState_Tracked
        return np.array([x, y, z], np.float64), bool(tracked)

    @staticmethod
    def _side_order(prefer_side):
        """Ordem de lados a consultar ("left" preferido vem primeiro)."""
        return ("left", "right") if prefer_side == "left" else ("right", "left")

    @staticmethod
    def _anchor_joint_order(prefer_side):
        """Ordem (junta, nome) da ancora do punho.

        Sem preferencia mantem a ordem historica (pulso direito, pulso
        esquerdo, handtip direito, handtip esquerdo). Com preferencia = o
        lado do cue, o pulso daquele lado vem primeiro: numa trial de mao
        esquerda, ancorar no pulso direito (parado na mesa) seria errado.
        """
        right, left = ANCHOR_JOINTS["right"], ANCHOR_JOINTS["left"]
        if prefer_side == "left":
            return (left[0], right[0], left[1], right[1])
        return (right[0], left[0], right[1], left[1])

    @staticmethod
    def _arm_joints_from(joints, side):
        """Juntas ombro/cotovelo/punho de um lado (m, camera space) ou None."""
        shoulder_type, elbow_type, wrist_type = ARM_JOINT_TYPES[side]
        shoulder, s_tracked = KinectHandTracker._joint_camera_position(
            joints, shoulder_type)
        wrist, w_tracked = KinectHandTracker._joint_camera_position(
            joints, wrist_type)
        if shoulder is None or wrist is None:
            return None
        elbow, e_tracked = KinectHandTracker._joint_camera_position(
            joints, elbow_type)
        return {
            "side": side,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "tracked": {"shoulder": s_tracked, "elbow": e_tracked,
                        "wrist": w_tracked},
        }

    def arm_joints_all(self):
        """Juntas do braco dos DOIS lados do primeiro corpo rastreado.

        Devolve {"left": {...}, "right": {...}} (só os lados com ombro+punho
        utilizaveis) ou {} . Necessario para escolher a mao ativa quando nao
        ha lado no cue (trial de circulo = ambas as maos).
        """
        if self.last_bodies is None:
            return {}
        for body in self.last_bodies.bodies:
            if not body.is_tracked:
                continue
            joints = getattr(body, "joints", None)
            if joints is None:
                continue
            found = {}
            for side in ("right", "left"):
                data = self._arm_joints_from(joints, side)
                if data is not None:
                    found[side] = data
            if found:
                return found
        return {}

    def arm_joints(self, prefer_side=None):
        """Juntas ombro/cotovelo/punho de UM lado do esqueleto (m).

        Devolve dict {"side", "shoulder", "elbow", "wrist", "tracked": {...}}
        ou None se nenhum corpo rastreado tiver ombro+punho utilizaveis.
        O cotovelo pode vir None (oclusao/fora do quadro): a cinematica
        inversa (ArmLinkModel.solve_arm) reconstroi-lo a partir do ombro e do
        punho, usando os comprimentos de elo fixos.
        """
        found = self.arm_joints_all()
        for side in self._side_order(prefer_side):
            if side in found:
                return found[side]
        return None

    def body_hand_anchor(self, prefer_side=None):
        """Ancora 3D (x, y, z, nome) do pulso/mao vinda do esqueleto do SDK.

        As juntas do body frame (PyKinectV2) sao MEDIDAS pelo depth/body
        index do Kinect — nao sao estimativa do MediaPipe. Serve de
        ancora de profundidade quando o MediaPipe do Kinect perde a mao
        (movimento rapido): o Z medido mantem a projecao calibrada no
        lugar, sem seguir tracking nenhum (regra de desenho preservada).

        Retorna (x, y, z, nome) em metros (camera space) ou None.
        """
        if self.last_bodies is None:
            return None
        for body in self.last_bodies.bodies:
            if not body.is_tracked:
                continue
            joints = getattr(body, "joints", None)
            if joints is None:
                continue
            # Pulso primeiro (coincide com o no 0 do MediaPipe); HandTip
            # como reserva. Qualquer das duas maos serve.
            for jtype, name in self._anchor_joint_order(prefer_side):
                try:
                    joint = joints[jtype]
                except (IndexError, KeyError):
                    continue  # frame parcial; SDK real sempre traz as 25
                if joint.TrackingState != PyKinectV2.TrackingState_Tracked:
                    continue
                pos = joint.Position
                x, y, z = float(pos.x), float(pos.y), float(pos.z)
                if not np.all(np.isfinite([x, y, z])) or not 0.3 < z < 4.0:
                    continue
                return (x, y, z, name)
        return None

    def landmark_depths(self, landmarks):
        depths = []
        for landmark in landmarks:
            point = self._landmark_camera_point(landmark)
            depths.append(np.nan if point is None else point[2])
        return np.asarray(depths, dtype=np.float32)

    def draw_kinect_hands_on_depth(self, frame):
        for hand_landmarks in self.last_kinect_hands:
            color_pixels = [
                (
                    landmark.x * self.color_width,
                    landmark.y * self.color_height,
                )
                for landmark in hand_landmarks
            ]
            depth_pixels = self._color_to_depth_pixels(color_pixels)
            for pixel in depth_pixels:
                cv2.circle(frame, pixel, 4, (0, 255, 0), -1)

    # ------------------------------------------------- lateralidade (maos)
    @staticmethod
    def _handedness_of(result, index, flip=False):
        """Rotulo de lateralidade ('left'/'right'/'') da mao de indice `index`."""
        try:
            groups = result.handedness[index]
            label = (groups[0].category_name or "").strip().lower()
        except (AttributeError, IndexError, TypeError):
            return ""
        if label not in ("left", "right"):
            return ""
        if flip:
            label = "left" if label == "right" else "right"
        return label

    @staticmethod
    def _pick_hand_index(result, want_side=None, flip=False):
        """Indice da mao desejada (None se o lado pedido nao estiver visivel).

        Sem `want_side`, escolhe a mao mais confiante (o MediaPipe devolve a
        lista ordenada por escore), que era o comportamento anterior.
        """
        total = len(getattr(result, "hand_landmarks", None) or [])
        if total == 0:
            return None
        if want_side in ("left", "right"):
            for index in range(total):
                if KinectHandTracker._handedness_of(result, index,
                                                    flip) == want_side:
                    return index
            return None
        return 0

    def _order_hands(self, result, want_side, flip=False):
        """Maos detectadas com a desejada em PRIMEIRO lugar (usada em [0]).

        Devolve [] quando o lado pedido existe no cue mas o detector so viu a
        outra mao: melhor nao triangular nada do que triangular a mao errada.
        """
        hands = list(getattr(result, "hand_landmarks", None) or [])
        index = self._pick_hand_index(result, want_side, flip)
        if index is None:
            return []
        if index:
            hands.insert(0, hands.pop(index))
        return hands

    def detect_auxiliary_hands(self, auxiliary_frame, aux_index=None,
                               want_side=None):
        """Detecta maos na webcam auxiliar.

        `aux_index` None usa a camera primaria (compatibilidade com chamadas
        legadas); com indice usa o detector/timestamp daquela camera.
        `want_side` ('left'/'right') coloca a mao daquele lado em [0] e
        devolve [] se o detector viu somente a outra mao."""
        if auxiliary_frame is None:
            return []
        if aux_index is not None:
            index = int(aux_index)
            detector = self.auxiliary_detectors.get(index)
            if detector is None:
                raise KeyError(f"Sem detector para a camera auxiliar "
                               f"{index}.")
            rgb = cv2.cvtColor(auxiliary_frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            stamp = max(self.auxiliary_timestamps_ms[index] + 1,
                        int(time.monotonic() * 1000))
            self.auxiliary_timestamps_ms[index] = stamp
            result = detector.detect_for_video(image, stamp)
            # A auxiliar e' espelhada no pipeline -> rotulo do MediaPipe ja
            # corresponde a mao fisica (sem flip).
            return self._order_hands(result, want_side)
        rgb = cv2.cvtColor(auxiliary_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.auxiliary_frame_timestamp_ms = max(
            self.auxiliary_frame_timestamp_ms + 1, int(time.monotonic() * 1000)
        )
        result = self.auxiliary_hands.detect_for_video(
            image, self.auxiliary_frame_timestamp_ms
        )
        return self._order_hands(result, want_side)

    def detect_auxiliary_hands_all(self, frames):
        """Detecta em TODAS as webcams configuradas.

        `frames` = {indice: frame_ou_None}; devolve {indice: landmarks}."""
        results = {}
        for index, frame in (frames or {}).items():
            try:
                results[int(index)] = self.detect_auxiliary_hands(
                    frame, aux_index=int(index))
            except Exception as exc:
                print(f"[tracking] camera aux {index}: deteccao falhou "
                      f"({exc}); ignorada neste frame", flush=True)
                results[int(index)] = []
        return results

    def draw_auxiliary_hands_on_depth(self, frame, auxiliary_hands, alignment):
        for hand_landmarks in auxiliary_hands:
            auxiliary_pixels = [
                (
                    landmark.x * alignment.auxiliary_width,
                    landmark.y * alignment.auxiliary_height,
                )
                for landmark in hand_landmarks
            ]
            kinect_color_pixels = alignment.auxiliary_to_kinect(auxiliary_pixels)
            for depth_pixel in self._color_to_depth_pixels(kinect_color_pixels):
                cv2.circle(frame, depth_pixel, 5, (0, 0, 255), -1)
            for start, end in HAND_CONNECTIONS:
                points = self._color_to_depth_pixels(
                    [auxiliary_pixels[start], auxiliary_pixels[end]]
                )
                if len(points) == 2:
                    kinect_start, kinect_end = alignment.auxiliary_to_kinect(
                        [auxiliary_pixels[start], auxiliary_pixels[end]]
                    )
                    depth_points = self._color_to_depth_pixels([kinect_start, kinect_end])
                    if len(depth_points) == 2:
                        cv2.line(frame, depth_points[0], depth_points[1], (0, 0, 220), 2)

    def draw_auxiliary_hands_on_color(self, frame, auxiliary_hands, alignment, kinect_depth_m=None, triangulated_3d=None):
        """Desenha as maos auxiliares projetadas no RGB do Kinect.

        Estrategia (por ordem de preferencia):
        1) Stereo 3D com profundidade por no (modelo de escala aparente).
        2) Stereo 3D com profundidade unica (palma/EMA).
        3) Homografia 2D do tabuleiro (fallback impreciso).

        A homografia 2D do esqueleto (hand_landmark_homography) e usada apenas
        para o diagnostico, nunca para o overlay: RMS ~400 px a torna inutil
        para projecao (mao e um objeto 3D visto de angulos diferentes).

        Retorna (desvio_medio_px|None, dx|None, dy|None, red_pixels,
        green_pixels|None) entre a projecao vermelha e o esqueleto verde
        do proprio Kinect (primeira mao), ou None se nada foi desenhado.
        red/green sao (21,2) em px Kinect (podem conter NaN fora do frame);
        green e None quando o Kinect nao detectou a mao — o vermelho segue
        desenhado e registrado, pois NAO depende do tracking verde.
        """
        if not (alignment.stereo_usable or alignment.hand_landmark_ready):
            return None
        first_deviation = None
        first_red = None
        first_green = None
        for hand_index, hand_landmarks in enumerate(auxiliary_hands):
            normalized = np.array(
                [(lm.x, lm.y) for lm in hand_landmarks], dtype=np.float64
            )
            auxiliary_pixels = [
                (
                    landmark.x * alignment.auxiliary_width,
                    landmark.y * alignment.auxiliary_height,
                )
                for landmark in hand_landmarks
            ]
            kinect_pixels = None

            # --- 0) Triangulacao estereo (espaco vetorial 3D) ----------------
            # Prioridade maxima: com as DUAS cameras detectando, o no 3D e
            # MEDIDO pela interseccao de raios (calibracao stereo) e projetado
            # de volta no Kinect. Nenhuma estimativa de profundidade entra
            # aqui; imune a inclinacao da mao e independente do tracking.
            if triangulated_3d is not None and hand_index == 0:
                tri = np.asarray(triangulated_3d, np.float64).reshape(-1, 3)
                finite3 = np.isfinite(tri).all(axis=1)
                if finite3.sum() >= 5:
                    projected, _ = cv2.projectPoints(
                        tri.reshape(-1, 1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        alignment.kinect_matrix,
                        alignment.kinect_dist,
                    )
                    kinect_pixels = projected.reshape(-1, 2)

            # --- 1) Stereo 3D with SINGLE rigid-hand depth -----------------------
            # Per-node depth (from apparent-scale model) scatters fingers because
            # the model has ~12 cm residual and z-offsets are learned from noisy
            # Kinect depth at fingertips. Single-depth rigid mode is stable.
            # (Fallback: so quando a triangulacao nao esta disponivel.)
            if kinect_pixels is None and alignment.stereo_usable and self.last_depth is not None:
                init_depth = NOMINAL_HAND_DEPTH_M
                if kinect_depth_m is not None:
                    arr = np.asarray(kinect_depth_m, np.float64).reshape(-1)
                    valid = arr[np.isfinite(arr) & (arr > 0.1)]
                    if valid.size > 0:
                        init_depth = float(np.nanmedian(valid))
                    else:
                        smoothed = alignment.get_smoothed_hand_depth(normalized)
                        if smoothed is not None:
                            init_depth = smoothed
                node_depths = [init_depth]
                kinect_pixels = alignment.auxiliary_to_kinect_stereo(
                    auxiliary_pixels, node_depths
                )
                kinect_pixels = np.asarray(kinect_pixels, np.float64).reshape(-1, 2)

            if kinect_pixels is None:
                continue
            kinect_pixels = np.asarray(kinect_pixels, dtype=np.float64).reshape(-1, 2)
            # Trim fixo de calibracao (medido com tecla K, salvo em
            # overlay_trim.json): traducao constante que remove o desregistro
            # fixo residual da cadeia stereo+intrinsecas. Constante armazenada,
            # nao depende do tracking do Kinect.
            if np.any(alignment.overlay_trim):
                kinect_pixels = kinect_pixels + alignment.overlay_trim[None, :]
            finite = np.isfinite(kinect_pixels).all(axis=1)
            # Porta de validez: si la proyeccion cae lejos del frame (profundidad
            # invalida o mano fuera del volumen calibrado), se descarta la mano
            # por completo. Evita dibujar un esqueleto rojo basura y que el
            # desvio explote a miles de px en el log de diagnostico.
            in_frame = (
                finite
                & (kinect_pixels[:, 0] >= -OVERLAY_FRAME_MARGIN_PX)
                & (kinect_pixels[:, 0] < self.color_width + OVERLAY_FRAME_MARGIN_PX)
                & (kinect_pixels[:, 1] >= -OVERLAY_FRAME_MARGIN_PX)
                & (kinect_pixels[:, 1] < self.color_height + OVERLAY_FRAME_MARGIN_PX)
            )
            if in_frame.sum() < 5:
                continue
            if (
                first_deviation is None
                and self.last_kinect_hands
                and hand_index < len(self.last_kinect_hands)
            ):
                green = np.array(
                    [
                        (lm.x * self.color_width, lm.y * self.color_height)
                        for lm in self.last_kinect_hands[hand_index]
                    ],
                    dtype=np.float64,
                )
                if green.shape == kinect_pixels.shape:
                    # Desvio calculado SOLO con nodos dentro de la imagen y con
                    # correspondencia verde finita: medicion honesta.
                    green_finite = np.isfinite(green).all(axis=1)
                    valid = finite & in_frame & green_finite
                    if valid.any():
                        diffs = kinect_pixels - green
                        first_deviation = (
                            float(np.mean(np.linalg.norm(diffs[valid], axis=1))),
                            float(np.mean(diffs[valid, 0])),
                            float(np.mean(diffs[valid, 1])),
                        )
                        first_green = green.copy()
                # A projecao vermelha e registrada MESMO sem verde: quando o
                # Kinect perde a mao, o log continua mostrando onde o overlay
                # achou que a mao estava (via triangulacao/calibracao/fallback),
                # em vez de sumir do diagnostico.
                if first_red is None:
                    red_copy = kinect_pixels.copy()
                    red_copy[~in_frame] = np.nan
                    first_red = red_copy
            for pixel, ok in zip(kinect_pixels, in_frame):
                if not ok:
                    continue
                x, y = int(round(pixel[0])), int(round(pixel[1]))
                if 0 <= x < self.color_width and 0 <= y < self.color_height:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            for start, end in HAND_CONNECTIONS:
                if not (in_frame[start] and in_frame[end]):
                    continue
                start_x, start_y = map(int, np.round(kinect_pixels[start]))
                end_x, end_y = map(int, np.round(kinect_pixels[end]))
                if all(
                    (
                        0 <= start_x < self.color_width,
                        0 <= start_y < self.color_height,
                        0 <= end_x < self.color_width,
                        0 <= end_y < self.color_height,
                    )
                ):
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 0, 220), 2)
        if first_red is None:
            return None
        if first_deviation is None:
            # Sem mao verde (Kinect cego): desvio indefinido, mas a trajetoria
            # vermelha segue disponivel para o log de diagnostico.
            return (None, None, None, first_red, None)
        return (*first_deviation, first_red, first_green)

    def draw_auxiliary_hands(self, frame, auxiliary_hands):
        height, width = frame.shape[:2]
        for hand_index, hand_landmarks in enumerate(auxiliary_hands, start=1):
            points = []
            for landmark in hand_landmarks:
                pixel = (
                    int(np.clip(landmark.x * width, 0, width - 1)),
                    int(np.clip(landmark.y * height, 0, height - 1)),
                )
                points.append(pixel)
                cv2.circle(frame, pixel, 4, (0, 255, 0), -1)
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, points[start], points[end], (0, 180, 0), 2)
            cv2.putText(
                frame,
                f"mao auxiliar {hand_index}",
                points[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

    def _draw_skeleton(self, frame):
        if self.last_bodies is None:
            return
        for body in self.last_bodies.bodies:
            if not body.is_tracked:
                continue
            points = {}
            for joint_type in range(PyKinectV2.JointType_Count):
                joint = body.joints[joint_type]
                if joint.TrackingState == PyKinectV2.TrackingState_NotTracked:
                    continue
                point = self.mapper.MapCameraPointToColorSpace(joint.Position)
                if not np.isfinite((point.x, point.y)).all():
                    continue
                pixel = (int(round(point.x)), int(round(point.y)))
                if 0 <= pixel[0] < self.color_width and 0 <= pixel[1] < self.color_height:
                    points[joint_type] = pixel
            for start, end in BODY_CONNECTIONS:
                if start in points and end in points:
                    cv2.line(frame, points[start], points[end], (255, 80, 0), 3)
            for pixel in points.values():
                cv2.circle(frame, pixel, 5, (255, 180, 0), -1)

    def _draw_hand(self, frame, landmarks, destaque=True):
        """Desenha o esqueleto da mao no RGB do Kinect.

        `destaque` marca a mao ESCOLHIDA pela trial (lateralidade do cue); a
        outra mao, quando visivel, fica em tom apagado para nao confundir o
        experimentador.
        """
        pontos_cor = (0, 255, 0) if destaque else (90, 90, 90)
        linha_cor = (0, 180, 0) if destaque else (70, 70, 70)
        points = []
        for landmark in landmarks:
            pixel = (
                int(np.clip(landmark.x * self.color_width, 0, self.color_width - 1)),
                int(np.clip(landmark.y * self.color_height, 0, self.color_height - 1)),
            )
            points.append(pixel)
            cv2.circle(frame, pixel, 4 if destaque else 3, pontos_cor, -1)
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], linha_cor,
                     2 if destaque else 1)

    def detect_hand(self, color_frame, depth_frame, want_side=None):
        """Detecta a mao no RGB do Kinect.

        `want_side` ('left'/'right') seleciona a mao DAQUELE lado (necessario
        com as duas maos visiveis: a trial define a mao do movimento). Sem
        `want_side` escolhe a mais confiante. A lateralidade e' corrigida pela
        convencao de espelho do MediaPipe (o RGB do Kinect nao e' espelhado).
        """
        if color_frame is None or depth_frame is None:
            self.last_hand_reason = "sem frame Kinect"
            return None, None, None
        self.last_kinect_hands = []
        self.last_hand_side = ""
        raw_color_frame = color_frame.copy()
        self._draw_skeleton(color_frame)
        rgb = cv2.cvtColor(raw_color_frame, cv2.COLOR_BGRA2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.kinect_frame_timestamp_ms = max(
            self.kinect_frame_timestamp_ms + 1, int(time.monotonic() * 1000)
        )
        result = self.hands.detect_for_video(image, self.kinect_frame_timestamp_ms)
        index = self._pick_hand_index(result, want_side,
                                      MEDIAPIPE_KINECT_HANDEDNESS_FLIP)
        if index is None:
            self.last_hand_reason = (
                f"lado pedido ({want_side}) nao visivel no Kinect"
                if want_side in ("left", "right")
                else "MediaPipe sem deteccion")
            return color_frame, None, None
        self.last_kinect_hands = result.hand_landmarks
        self.last_hand_side = self._handedness_of(
            result, index, MEDIAPIPE_KINECT_HANDEDNESS_FLIP)
        for position, hand_landmarks in enumerate(result.hand_landmarks):
            self._draw_hand(color_frame, hand_landmarks,
                            destaque=(position == index))
        landmarks = result.hand_landmarks[index]
        palm = landmarks[9]
        color_x = int(np.clip(palm.x * self.color_width, 0, self.color_width - 1))
        color_y = int(np.clip(palm.y * self.color_height, 0, self.color_height - 1))
        depth_pixel = self._color_to_depth_pixel(color_x, color_y)
        if depth_pixel is None:
            self.last_hand_reason = "mapeo color->depth falhou"
            self._log_map_failure(color_x, color_y)
            return color_frame, None, landmarks
        depth_mm = self._median_depth(*depth_pixel)
        if depth_mm is None:
            self.last_hand_reason = "profundidade invalida na palma"
            return color_frame, None, landmarks
        depth_point = PyKinectV2._DepthSpacePoint()
        depth_point.x = float(depth_pixel[0])
        depth_point.y = float(depth_pixel[1])
        # MapDepthPointToCameraSpace espera depth como c_ushort (inteiros,
        # milimetros); medir as medias em float causa ctypes.ArgumentError.
        camera_point = self.mapper.MapDepthPointToCameraSpace(
            depth_point, int(round(depth_mm))
        )
        position = np.array([camera_point.x, -camera_point.y, camera_point.z], dtype=float)
        palm_depth_m = float(position[2])
        self.last_hand_reason = "OK"
        if self.last_hand_depth_m is None:
            self.last_hand_depth_m = palm_depth_m
        else:
            # media movel: memoria suave da profundidade da mao para os
            # momentos em que o depth falha (bordas do FOV, mao muito perto)
            self.last_hand_depth_m = 0.8 * self.last_hand_depth_m + 0.2 * palm_depth_m
        cv2.circle(color_frame, (color_x, color_y), 10, (0, 255, 255), -1)
        return color_frame, position, landmarks

    def _log_map_failure(self, color_x, color_y):
        """Report of the color->depth mapping failing for the palm."""
        if not self._map_error_reported:
            self._map_error_reported = True
            print(
                f"mapeo color->depth falhou para a palma ({color_x}, {color_y}): "
                "use 'T' com o tabuleiro para diagnosticar (debe ver 'N/48 cantos')"
            )

    def draw_palm_vector(self, frame, landmarks, direction_camera):
        palm = self._landmark_camera_point(landmarks[9])
        endpoint = None if palm is None else palm + direction_camera * PALM_VECTOR_LENGTH_M
        if endpoint is None:
            return
        start_point = PyKinectV2._CameraSpacePoint(*palm)
        end_point = PyKinectV2._CameraSpacePoint(*endpoint)
        start = self.mapper.MapCameraPointToColorSpace(start_point)
        end = self.mapper.MapCameraPointToColorSpace(end_point)
        start_pixel = (int(round(start.x)), int(round(start.y)))
        end_pixel = (int(round(end.x)), int(round(end.y)))
        if all(np.isfinite((start.x, start.y, end.x, end.y))):
            cv2.arrowedLine(frame, start_pixel, end_pixel, (0, 0, 255), 5, tipLength=0.2)
            cv2.putText(frame, "direcao da palma", end_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    def close(self):
        self.hands.close()
        self.auxiliary_hands.close()
        self.kinect.close()


def transform_auxiliary_frame(frame):
    if frame is None:
        return None
    if AUXILIARY_MIRROR_HORIZONTAL:
        return cv2.flip(frame, 1)
    return frame


# PositionFusion vive em imu.py (fonte unica).


# O programa autonomo de calibracao/diagnostico (que antes era o
# `main()` deste arquivo) vive em tools/kinect_groundtruth_tool.py. Este
# modulo e' uma BIBLIOTECA: e' importada pelos programas de aquisicao e
# pelo sistema online, e nao deve ter entrada propria.
