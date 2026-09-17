"""IMU do ESP32 (MPU6050) por UDP + fusao IMU x cameras.

FONTE UNICA (item #9 da revisao): antes existiam DUAS copias de `ImuSample`,
`ImuReceiver`, `rotation_matrix` e `PositionFusion` -- uma em
`kinect_imu_groundtruth.py` e outra em `camera_mpu_fusion.py` ("espelho de
kinect_imu_groundtruth"), o que significa que corrigir um bug em uma deixava a
outra silenciosamente errada. Agora os dois modulos importam daqui.

Este modulo e' DELIBERADAMENTE leve (math, socket, threading, time, dataclass e
numpy): ler o IMU nao pode exigir cv2/mediapipe/pykinect2. Era o problema de se
importar `kinect_imu_groundtruth` (que exige o SDK do Kinect) apenas para
receber um datagrama UDP.

Protocolo esperado do ESP32 (uma linha ASCII por datagrama, porta 4210):

    timestamp_us, roll, pitch, yaw, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps

Os quatro primeiros campos sao obrigatorios; acc/gyro sao opcionais.
"""
from __future__ import annotations

import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---- ganhos da fusao --------------------------------------------------------
#: Ganho do puxao da posicao para a medida da camera (mao em movimento).
POSITION_CAMERA_GAIN = 0.90
#: Amortecimento da velocidade a cada frame (sem camera estavel).
VELOCITY_DAMPING = 0.80
#: Eixo "para cima" no camera space do Kinect (y cresce para baixo).
UP_AXIS = 2
# ---- zeragem do IMU com as cameras -----------------------------------------
# Janela (frames) e desvio (m) usados para decidir que a mao esta PARADA na
# medida da camera. Com a mao parada, o residuo do acelerometro e' composto so'
# de bias + ruido, entao a janela permite estimar o bias com precisao.
IMU_ZERO_WINDOW = 15
IMU_ZERO_STD_M = 0.008
#: Forca do EMA na correcao continua do bias enquanto a camera estiver valida.
IMU_ZERO_BIAS_EMA = 0.05
#: Limite (g) do bias estimado: acima disso a estimativa e' descartada (o EMA
#: da janela pode pegar um trecho de movimento residual).
IMU_ZERO_MAX_BIAS_G = 0.5
#: Gravidade (m/s^2) usada na integracao.
GRAVITY_MPS2 = 9.80665


@dataclass
class ImuSample:
    """Uma amostra do ESP32 (orientacao em graus + acc/gyro)."""

    timestamp_us: int
    received_at: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    accel_g: np.ndarray
    gyro_dps: np.ndarray


class ImuReceiver:
    """Recebe as amostras do ESP32 por UDP em uma thread propria.

    Guarda sempre a ULTIMA amostra valida (`get_latest`); amostras vazias,
    truncadas ou com valores nao finitos sao descartadas em silencio (a thread
    nunca morre por causa de um datagrama ruim).
    """

    def __init__(self, port: int = 4210):
        self.port = int(port)
        self.running = False
        self.latest: Optional[ImuSample] = None
        self.lock = threading.Lock()
        self.socket: Optional[socket.socket] = None

    def start(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", self.port))
        self.socket.settimeout(0.2)
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        assert self.socket is not None
        while self.running:
            try:
                payload, _ = self.socket.recvfrom(512)
                values = np.fromstring(payload.decode("ascii").strip(), sep=",")
                received_at = time.monotonic()
                if values.size < 4 or not np.isfinite(values[:4]).all():
                    continue
                accel = values[4:7] if values.size >= 7 else np.zeros(3)
                gyro = values[7:10] if values.size >= 10 else np.zeros(3)
                if not np.isfinite(accel).all() or not np.isfinite(gyro).all():
                    continue
                sample = ImuSample(
                    timestamp_us=int(values[0]),
                    received_at=received_at,
                    roll_deg=float(values[1]),
                    pitch_deg=float(values[2]),
                    yaw_deg=float(values[3]),
                    accel_g=accel.astype(np.float64),
                    gyro_dps=gyro.astype(np.float64),
                )
                with self.lock:
                    self.latest = sample
            except (OSError, UnicodeDecodeError, ValueError):
                continue

    def get_latest(self) -> Optional[ImuSample]:
        with self.lock:
            return self.latest

    def stop(self) -> None:
        self.running = False
        if self.socket is not None:
            self.socket.close()
        self.socket = None


def rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Matriz de rotacao (corpo -> mundo) a partir de roll/pitch/yaw em graus."""
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


class PositionFusion:
    """Fusao IMU + camera com ZERAGEM quando ha medida de camera valida.

    Enquanto a camera mede o punho de forma valida e ESTAVEL (mao parada:
    desvio da janela < IMU_ZERO_STD_M):
      - estima e corrige o bias do acelerometro (EMA sobre o residuo de
        repouso, limitado a IMU_ZERO_MAX_BIAS_G);
      - zera a velocidade e ancora a posicao na medida da camera
        ("zeragem" espacial: o IMU nunca acumula deriva enquanto a camera
        estiver presente).

    Quando a camera some, a integracao continua com o bias JA CORRIGIDO, o que
    segura o drift por segundos. `absolute_position` e' a posicao no camera
    space do Kinect (ancora = primeira medida valida).
    """

    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.origin: Optional[np.ndarray] = None
        self.last_timestamp_us: Optional[int] = None
        self.accel_bias = np.zeros(3)
        self.bias_samples = []
        self.imu_reference_rotation = None
        self.palm_reference_direction = None
        # zeragem
        self.zero_lock = False          # camera valida + mao parada agora
        self.bias_estimated = False
        self._camera_window = []
        self._camera_lost_frames = 0

    def reset(self):
        self.position[:] = 0.0
        self.velocity[:] = 0.0
        self.origin = None
        self.last_timestamp_us = None
        self.imu_reference_rotation = None
        self.palm_reference_direction = None
        self.zero_lock = False
        self._camera_window = []
        self._camera_lost_frames = 0

    @property
    def absolute_position(self):
        """Posicao fusionada no camera space (absoluta) ou None."""
        if self.origin is None:
            return None
        return self.origin + self.position

    # ------------------------------------------------- orientacao da palma
    def calibrate_palm_direction(self, sample, palm_direction):
        """Ancora a direcao da palma (medida pelo Kinect) no referencial do IMU."""
        if sample is None or palm_direction is None:
            return False
        if not np.isfinite(np.asarray(palm_direction, np.float64)).all():
            return False
        self.imu_reference_rotation = rotation_matrix(
            sample.roll_deg, sample.pitch_deg, sample.yaw_deg
        )
        self.palm_reference_direction = np.asarray(palm_direction,
                                                   np.float64).copy()
        return True

    def palm_direction(self, sample):
        """Direcao da palma prevista pelo IMU (None sem referencia)."""
        if sample is None or self.imu_reference_rotation is None:
            return None
        current_rotation = rotation_matrix(
            sample.roll_deg, sample.pitch_deg, sample.yaw_deg
        )
        direction = (current_rotation @ self.imu_reference_rotation.T
                     @ self.palm_reference_direction)
        length = np.linalg.norm(direction)
        return direction / length if length > 1e-6 else None

    def add_bias_sample(self, sample):
        """Coleta amostras de repouso para o bias (media das ultimas 500)."""
        if sample is not None:
            self.bias_samples.append(sample.accel_g.copy())
            if len(self.bias_samples) > 500:
                self.bias_samples.pop(0)
            self.accel_bias = np.mean(self.bias_samples, axis=0)

    # ---------------------------------------------------------------- zeragem
    def _camera_is_stable(self, camera_position):
        """True se a janela de posicoes da camera tem desvio pequeno (parada)."""
        self._camera_window.append(camera_position.copy())
        if len(self._camera_window) > IMU_ZERO_WINDOW:
            self._camera_window.pop(0)
        if len(self._camera_window) < IMU_ZERO_WINDOW:
            return False
        window = np.asarray(self._camera_window, np.float64)
        return bool(np.all(window.std(axis=0) < IMU_ZERO_STD_M))

    def _update_accel_bias(self, sample):
        """Bias estimado com a mao PARADA: no repouso o residuo em mundo e' o
        bias projetado; corrige por EMA limitado por frame."""
        rot = rotation_matrix(sample.roll_deg, sample.pitch_deg,
                              sample.yaw_deg)
        residual_world = rot @ (sample.accel_g * GRAVITY_MPS2)
        residual_world[UP_AXIS] -= GRAVITY_MPS2
        hint = (rot.T @ residual_world) / GRAVITY_MPS2
        hint = np.clip(hint, -IMU_ZERO_MAX_BIAS_G, IMU_ZERO_MAX_BIAS_G)
        if np.isfinite(hint).all():
            self.accel_bias += IMU_ZERO_BIAS_EMA * (hint - self.accel_bias)
            self.bias_estimated = True

    def update(self, sample, camera_position):
        """Fusao com ZERAGEM; devolve a posicao relativa a ancora da camera.

        - camera valida + mao parada  -> zero espacial + zero de velocidade +
          correcao do bias (self.zero_lock = True);
        - camera valida + mao em movimento -> puxa com ganho (sem zerar bias);
        - camera ausente -> integracao continua com o bias ja corrigido.
        """
        if sample is not None and self.last_timestamp_us is not None:
            dt = (sample.timestamp_us - self.last_timestamp_us) * 1e-6
            if 0.0 < dt < 0.1:
                acceleration = rotation_matrix(
                    sample.roll_deg, sample.pitch_deg, sample.yaw_deg
                ) @ ((sample.accel_g - self.accel_bias) * GRAVITY_MPS2)
                acceleration[UP_AXIS] -= GRAVITY_MPS2
                self.velocity += acceleration * dt
                self.position += (self.velocity * dt
                                  + 0.5 * acceleration * dt * dt)
        if sample is not None:
            self.last_timestamp_us = sample.timestamp_us

        self.zero_lock = False
        if camera_position is not None:
            if self.origin is None:
                self.origin = camera_position.copy()
            camera_relative = camera_position - self.origin
            if self._camera_is_stable(camera_position):
                # ---- ZERAGEM: mao parada e vista pela camera ----
                self.zero_lock = True
                self._camera_lost_frames = 0
                self.position[:] = camera_relative      # ancora espacial
                self.velocity[:] = 0.0                  # zera velocidade
                if sample is not None:
                    self._update_accel_bias(sample)     # zera bias
            else:
                self._camera_lost_frames = 0
                self.position += POSITION_CAMERA_GAIN * (
                    camera_relative - self.position)
                self.velocity *= VELOCITY_DAMPING
        else:
            self._camera_lost_frames += 1
            if self._camera_lost_frames > 3:
                self._camera_window = []        # janela velha nao vale
        return self.position.copy()