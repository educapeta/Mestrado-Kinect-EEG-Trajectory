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


# =============================================================================
# Orientacao da palma: alinhamento IMU <-> camera (por luva)
# =============================================================================
#: Amostras MINIMAS para resolver o alinhamento (o modelo tem 5 GDL: 3 da
#: rotacao de alinhamento + 2 da direcao da palma no corpo do sensor).
PALM_ALIGNMENT_MIN_SAMPLES = 6
#: Residuo medio (graus) acima do qual o alinhamento e' considerado ruim: com a
#: palma sempre virada para a camera e as duas luvas certas, o normal medido pelo
#: Kinect tem ~1-2 graus de ruido -> residuo de 2-5 graus. Residuo alto quer
#: dizer captura ruim (palma virou para o dorso no meio, mao fora do quadro) ou
#: dado de OUTRA luva (porta/IP trocados).
PALM_ALIGNMENT_MAX_RESIDUAL_DEG = 15.0
#: Limite de iteracoes do Kabsch alternado (converge em <10 na pratica).
PALM_ALIGNMENT_ITERATIONS = 200


def kabsch_rotation(pairs, eps=1e-12):
    """Rotacao R que melhor leva os vetores `a_i` nos `b_i` (Kabsch/SVD).

    Resolve `min R sum_i |R a_i - b_i|^2` com det(R) = +1 (rotacao propria, sem
    espelhamento -- um sensor "montado ao contrario" e' uma ROTACAO, nao um
    espelho, entao isto cobre a montagem da luva).

    Args:
        pairs: sequencia de (a, b), vetores 3D equivalentes em dois referenciais.
        eps: norma minima de cada vetor para entrar no ajuste.

    Returns:
        ndarray (3, 3) ou None se nenhum par valido (comprimento < eps).
    """
    matriz = np.zeros((3, 3))
    usados = 0
    for origem, destino in pairs:
        a = np.asarray(origem, np.float64).reshape(3)
        b = np.asarray(destino, np.float64).reshape(3)
        if np.linalg.norm(a) < eps or np.linalg.norm(b) < eps:
            continue
        matriz += np.outer(b, a)
        usados += 1
    if usados == 0:
        return None
    u, _s, vt = np.linalg.svd(matriz)
    # reflexao (det = -1) nao e' rotacao: corrige o eixo menos significativo
    sinal = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    return u @ np.diag([1.0, 1.0, sinal]) @ vt


def angle_between(a, b):
    """Angulo (graus) entre dois vetores, robusto a ruido (clip do cosseno)."""
    va = np.asarray(a, np.float64).reshape(3)
    vb = np.asarray(b, np.float64).reshape(3)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return float(math.degrees(math.acos(cos)))


def solve_palm_alignment(rotations, palm_directions,
                         iterations=PALM_ALIGNMENT_ITERATIONS):
    """Resolve (T, f_body, residuo_graus) do modelo `v_i = T R_i f`.

    Por que este modelo (e nao o antigo `R_novo R_ref^T v_ref`): a rotacao que o
    IMU entrega (`R_i`, corpo -> mundo do IMU) e a direcao da palma medida pelo
    Kinect (`v_i`, no referencial da CAMERA) vivem em referenciais DIFERENTES. O
    referencial do IMU tem a vertical do acelerometro, mas o "norte" dele e'
    arbitrario (sem magnetometro) e ainda muda com a MONTAGEM de cada luva --
    cada lado tem o seu. Compor as duas rotacoes sem converter da' um vetor certo
    APENAS na pose de calibracao e erra cada vez mais conforme a mao gira, e o
    erro e' diferente em cada luva (uma pode ate' parecer certa). Aqui o
    referencial e' RESOLVIDO a partir das amostras: `T` leva o mundo do IMU para
    o da camera e `f` e' a direcao da palma no corpo do sensor.

    Solucao: Kabsch alternado. Com `f` fixo, `T` e' o Kabsch dos pares
    (`R_i f` -> `v_i`); com `T` fixo, `f` e' a media normalizada de
    `R_i^T T^T v_i`. Comeca em `f = R_0^T v_0` e `T = I` (o comportamento antigo)
    e converge em poucas iteracoes.

    Args:
        rotations: lista de (3, 3) corpo -> mundo do IMU, uma por amostra.
        palm_directions: lista de (3,) direcao da palma no referencial da camera.
        iterations: teto de iteracoes.

    Returns:
        (T, f_body, residuo_graus) ou None se houver menos de
        PALM_ALIGNMENT_MIN_SAMPLES pares validos.
    """
    pares = []
    for rotacao, direcao in zip(rotations or [], palm_directions or []):
        r = np.asarray(rotacao, np.float64)
        v = np.asarray(direcao, np.float64).reshape(3)
        if r.shape != (3, 3) or not np.isfinite(r).all():
            continue
        norma = np.linalg.norm(v)
        if not np.isfinite(v).all() or norma < 1e-9:
            continue
        pares.append((r, v / norma))
    if len(pares) < PALM_ALIGNMENT_MIN_SAMPLES:
        return None
    rotacoes = [r for r, _v in pares]
    direcoes = [v for _r, v in pares]
    f_body = rotacoes[0].T @ direcoes[0]
    norma = np.linalg.norm(f_body)
    if norma < 1e-9:
        return None
    f_body = f_body / norma
    alinhamento = np.eye(3)
    for _ in range(max(1, int(iterations))):
        alinhamento = kabsch_rotation([(r @ f_body, v)
                                       for r, v in pares])
        if alinhamento is None:
            return None
        media = np.mean([r.T @ alinhamento.T @ v for r, v in pares], axis=0)
        norma = np.linalg.norm(media)
        if norma < 1e-9:
            return None
        novo = media / norma
        convergiu = np.linalg.norm(novo - f_body) < 1e-12
        f_body = novo
        if convergiu:
            break
    residuos = [angle_between(alinhamento @ r @ f_body, v)
                for r, v in pares]
    residuos = [valor for valor in residuos if np.isfinite(valor)]
    if not residuos:
        return None
    return alinhamento, f_body, float(np.mean(residuos))


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
        # --- orientacao da palma ------------------------------------------
        # `palm_reference_body`: direcao da palma NO CORPO do sensor (unitaria);
        # `imu_to_camera`: rotacao mundo-do-IMU -> referencial da camera, por
        # LUVA (ver `solve_palm_alignment`). Identidade = comportamento antigo
        # (so' vale se a luva estiver com o referencial alinhado ao da camera,
        # o que NAO e' verdade em geral -- calibre com `apply_palm_alignment`).
        self.palm_reference_body: Optional[np.ndarray] = None
        self.imu_to_camera = np.eye(3)
        #: Residuo (graus) do ultimo alinhamento resolvido (None = identidade).
        self.palm_alignment_residual_deg: Optional[float] = None
        # Compatibilidade: referencia antiga (pose + direcao no referencial da
        # camera) mantida para quem le estes campos.
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
        self.palm_reference_body = None
        self.palm_alignment_residual_deg = None
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
        """Ancora a direcao da palma (medida pelo Kinect) no referencial do IMU.

        Guarda a direcao NO CORPO do sensor (`f = R^T T^T v`, ver
        `solve_palm_alignment`); `palm_direction` devolve `T R f`. Com
        `imu_to_camera` = identidade isto reproduz o comportamento antigo
        (`R_novo R_ref^T v_ref`) -- que so' e' correto se o referencial do IMU
        coincidir com o da camera.
        """
        if sample is None or palm_direction is None:
            return False
        valor = np.asarray(palm_direction, np.float64).reshape(3)
        if not np.isfinite(valor).all() or np.linalg.norm(valor) < 1e-9:
            return False
        referencia = rotation_matrix(sample.roll_deg, sample.pitch_deg,
                                     sample.yaw_deg)
        corpo = referencia.T @ self.imu_to_camera.T @ (valor / np.linalg.norm(valor))
        norma = np.linalg.norm(corpo)
        if norma < 1e-9:
            return False
        self.palm_reference_body = corpo / norma
        self.imu_reference_rotation = referencia
        self.palm_reference_direction = valor.copy()
        return True

    def apply_palm_alignment(self, rotations, palm_directions):
        """Resolve o alinhamento IMU <-> camera DESTA luva e o aplica.

        Chamada com as amostras de uma captura em que a mao gira com a palma
        virada para a camera (e o Kinect enxergando a mao): resolve `T` (mundo do
        IMU -> camera) e `f` (palma no corpo) por `solve_palm_alignment` e passa a
        usa-los em `palm_direction`. Sem isso a direcao prevista acerta na pose de
        calibracao e erra conforme a mao gira -- e o erro e' DIFERENTE em cada
        luva (montagem diferente), o que faz uma das maos parecer certa.

        Returns:
            (T, f_body, residuo_deg) ou None (amostras insuficientes/invalidas).
        """
        resolvido = solve_palm_alignment(rotations, palm_directions)
        if resolvido is None:
            return None
        alinhamento, corpo, residuo = resolvido
        self.imu_to_camera = np.asarray(alinhamento, np.float64)
        self.palm_reference_body = np.asarray(corpo, np.float64)
        self.palm_alignment_residual_deg = float(residuo)
        return alinhamento, self.palm_reference_body, float(residuo)

    def set_palm_alignment(self, alignment, body_direction=None,
                           residual_deg=None):
        """Aplica um alinhamento JA' resolvido (ex.: carregado de arquivo).

        Args:
            alignment: (3, 3) mundo do IMU -> referencial da camera.
            body_direction: (3,) direcao da palma no corpo; None = mantem a atual.
            residual_deg: residuo do ajuste, para diagnostico.

        Returns:
            True se aplicou (matriz valida e direcao disponivel).
        """
        matriz = np.asarray(alignment, np.float64).reshape(3, 3)
        if not np.isfinite(matriz).all():
            return False
        if abs(np.linalg.det(matriz)) < 1e-6:
            return False
        corpo = (self.palm_reference_body if body_direction is None
                 else np.asarray(body_direction, np.float64).reshape(3))
        if corpo is None:
            return False
        norma = np.linalg.norm(corpo)
        if not np.isfinite(corpo).all() or norma < 1e-9:
            return False
        self.imu_to_camera = matriz
        self.palm_reference_body = corpo / norma
        self.palm_alignment_residual_deg = (None if residual_deg is None
                                            else float(residual_deg))
        return True

    def palm_alignment_state(self):
        """Alinhamento atual em tipos serializaveis (JSON) ou None."""
        if self.palm_reference_body is None:
            return None
        return {
            "rotation": np.asarray(self.imu_to_camera,
                                   np.float64).tolist(),
            "body": np.asarray(self.palm_reference_body, np.float64).tolist(),
            "residual_deg": self.palm_alignment_residual_deg,
        }

    def palm_direction(self, sample):
        """Direcao da palma prevista pelo IMU (None sem referencia).

        `T R_novo f`, com `T` = `imu_to_camera` (mundo do IMU -> camera) e
        `f` = direcao da palma no corpo do sensor.
        """
        if sample is None or self.palm_reference_body is None:
            return None
        current_rotation = rotation_matrix(
            sample.roll_deg, sample.pitch_deg, sample.yaw_deg
        )
        direction = (self.imu_to_camera @ current_rotation
                     @ self.palm_reference_body)
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


# =============================================================================
# Fundir a VELOCIDADE decodificada da EEG com a ACELERACAO do MPU6050
# =============================================================================
def accel_no_referencial(sample, bias_g=None, up_axis=UP_AXIS,
                         gravity=GRAVITY_MPS2):
    """Aceleracao do IMU em m/s^2, rodada para o referencial da ORIGEM.

    Mesma convencao do `PositionFusion`: subtrai o bias (em g), converte para
    m/s^2, roda pela atitude medida (roll/pitch/yaw) e remove a gravidade no eixo
    vertical da origem (`up_axis`). Sem amostra devolve `None`.
    """
    if sample is None:
        return None
    bias = np.zeros(3) if bias_g is None else np.asarray(bias_g, np.float64)
    accel = rotation_matrix(sample.roll_deg, sample.pitch_deg, sample.yaw_deg) \
        @ ((np.asarray(sample.accel_g, np.float64) - bias) * gravity)
    accel[up_axis] -= gravity
    return accel


def integra_velocidade(velocidade, dt_s, posicao_inicial=None):
    """Posicoes (T, 3) por integracao TRAPEZOIDAL de uma velocidade (T, 3).

    Usada para desenhar a trajetoria prevista quando o modelo devolve m/s
    (`--alvo velocidade`): a posicao vem da integral da velocidade, e o filtro de
    Kalman (`KalmanTrajectory`) e' quem ancora essa integracao.
    """
    velocidade = np.asarray(velocidade, np.float64)
    if velocidade.ndim != 2 or velocidade.shape[1] != 3:
        raise ValueError(f"velocidade precisa ser (T, 3); veio "
                         f"{velocidade.shape}")
    if not np.isfinite(dt_s) or float(dt_s) <= 0:
        raise ValueError(f"intervalo entre amostras invalido: {dt_s}")
    inicio = (np.zeros(3, np.float64) if posicao_inicial is None
              else np.asarray(posicao_inicial, np.float64).reshape(3))
    if velocidade.shape[0] == 1:
        return inicio.reshape(1, 3).copy()
    passos = 0.5 * (velocidade[1:] + velocidade[:-1]) * float(dt_s)
    return np.vstack([inicio.reshape(1, 3),
                      inicio.reshape(1, 3) + np.cumsum(passos, axis=0)])


class KalmanTrajectory:
    """Filtro de Kalman (3 estados por eixo) para a posicao do punho.

    Estado por eixo: ``[posicao (m), velocidade (m/s), bias de aceleracao]``.

    - **predicao**: modelo de aceleracao constante, com a aceleracao **medida**
      pelo MPU6050 como entrada de controle (descontado o bias estimado);
    - **medidas**: a velocidade **decodificada da EEG** (a cada inferencia) e,
      opcionalmente, a posicao do Kinect/cameras como ancora lenta.

    Por que nao integrar a aceleracao direto (o argumento do erro quadratico):
    integrar duas vezes faz o erro de posicao crescer com ``t^2`` e explodir com
    o bias do acelerometro -- 1 cm/s^2 de bias da' 2 cm em 2 s e 50 cm em 10 s.
    Com o filtro o bias entra no ESTADO (e' estimado) e a velocidade da EEG
    ancora a integracao, deixando o erro **limitado**.
    """

    def __init__(self, sigma_accel=0.5, sigma_bias=0.02, sigma_vel_eeg=0.25,
                 sigma_pos_camera=0.02, up_axis=UP_AXIS):
        self.sigma_accel = float(sigma_accel)
        self.sigma_bias = float(sigma_bias)
        self.sigma_vel_eeg = float(sigma_vel_eeg)
        self.sigma_pos_camera = float(sigma_pos_camera)
        self.up_axis = int(up_axis)
        self._x = None
        self._P = None
        self._ultima_accel = None
        self._ultimo = None
        self.reset()

    def reset(self, position=None, velocity=None, bias=None):
        """Reinicia o estado (zeragem do IMU, novo trial, perda de medida)."""
        inicio = np.zeros(3, np.float64) if position is None \
            else np.asarray(position, np.float64).reshape(3)
        velocidade = np.zeros(3, np.float64) if velocity is None \
            else np.asarray(velocity, np.float64).reshape(3)
        bias = np.zeros(3, np.float64) if bias is None \
            else np.asarray(bias, np.float64).reshape(3)
        self._x = [np.array([inicio[eixo], velocidade[eixo], bias[eixo]])
                   for eixo in range(3)]
        #: Covariancia inicial: posicao incerta (quem mostra e' o modelo),
        #: velocidade e bias razoavelmente confiaveis.
        self._P = [np.diag([1.0, 0.5 ** 2, 0.2 ** 2]) for _ in range(3)]
        self._ultimo = None
        return self.position

    # ------------------------------------------------------------- matrizes
    def _transicao(self, dt):
        return np.array([[1.0, dt, -0.5 * dt * dt],
                         [0.0, 1.0, -dt],
                         [0.0, 0.0, 1.0]])

    def _ruido(self, dt):
        """Q = B*sigma_accel^2*B^T + diag(0, 0, sigma_bias^2).

        O primeiro termo vem da incerteza da aceleracao medida (entra pelo vetor
        de controle B); o segundo permite que o bias estime lentamente.
        """
        vetor = np.array([0.5 * dt * dt, dt, 0.0]) * self.sigma_accel
        return np.outer(vetor, vetor) \
            + np.diag([0.0, 0.0, self.sigma_bias ** 2])

    def _corrige(self, eixo, linha, medida, variancia):
        x = self._x[eixo]
        P = self._P[eixo]
        S = float(linha @ P @ linha) + float(variancia)
        if not np.isfinite(S) or S <= 0:
            return
        ganho = (P @ linha) / S
        self._x[eixo] = x + ganho * (float(medida) - float(linha @ x))
        self._P[eixo] = (np.eye(3) - np.outer(ganho, linha)) @ P


    # ---------------------------------------------------------------- passo
    def step(self, dt_s=None, accel_m_s2=None, velocidade_m_s=None,
             posicao_m=None):
        """Avanca o filtro. `dt_s=None` usa o relogio entre chamadas.

        `accel_m_s2` (3,) aceleracao medida (m/s^2, referencial da origem);
        `velocidade_m_s` (3,) medicao decodificada da EEG; `posicao_m` (3,)
        ancora lenta (Kinect/cameras). Devolve a posicao filtrada (3,).
        """
        agora = time.monotonic()
        if dt_s is None:
            dt_s = 0.0 if self._ultimo is None else (agora - self._ultimo)
        self._ultimo = agora
        dt = float(np.clip(dt_s, 1e-3, 0.5))
        controle = (np.zeros(3, np.float64) if accel_m_s2 is None
                    else np.asarray(accel_m_s2, np.float64).reshape(3))
        self._ultima_accel = controle
        F = self._transicao(dt)
        B = np.array([0.5 * dt * dt, dt, 0.0])
        Q = self._ruido(dt)
        for eixo in range(3):
            self._x[eixo] = F @ self._x[eixo] + B * controle[eixo]
            self._P[eixo] = F @ self._P[eixo] @ F.T + Q
        if velocidade_m_s is not None:
            velocidade = np.asarray(velocidade_m_s, np.float64).reshape(3)
            linha = np.array([0.0, 1.0, 0.0])
            for eixo in range(3):
                if np.isfinite(velocidade[eixo]):
                    self._corrige(eixo, linha, velocidade[eixo],
                                  self.sigma_vel_eeg ** 2)
        if posicao_m is not None:
            posicao = np.asarray(posicao_m, np.float64).reshape(3)
            linha = np.array([1.0, 0.0, 0.0])
            for eixo in range(3):
                if np.isfinite(posicao[eixo]):
                    self._corrige(eixo, linha, posicao[eixo],
                                  self.sigma_pos_camera ** 2)
        return self.position

    # ------------------------------------------------------------ consultas
    def _estado(self, indice):
        return np.array([self._x[eixo][indice] for eixo in range(3)])

    @property
    def position(self):
        return self._estado(0)

    @property
    def velocity(self):
        return self._estado(1)

    @property
    def bias(self):
        return self._estado(2)

    @property
    def acceleration_est(self):
        """Aceleracao estimada atual: ultimo controle medido MENOS o bias."""
        if self._ultima_accel is None:
            return -self._estado(2)
        return self._ultima_accel - self._estado(2)

    def predizer(self, horizonte_s, passos=10):
        """Trajetoria futura (passos, 3) com a aceleracao estimada atual.

        Serve para desenhar a trajetoria prevista a frente do instante atual no
        overlay. Com aceleracao estimada nula e' uma reta (velocidade constante).
        """
        passos = max(2, int(passos))
        horizonte = float(horizonte_s)
        if horizonte <= 0:
            raise ValueError(f"horizonte invalido: {horizonte_s}")
        aceleracao = self.acceleration_est
        posicao, velocidade = self.position, self.velocity
        return np.array([posicao + velocidade * t + 0.5 * aceleracao * t * t
                         for t in np.linspace(0.0, horizonte, passos)])


#: Portas UDP padrao por lado (convencao do projeto: 1 = direita, 2 = esquerda,
#: igual a ARM_SIDE_CODE em eeg_motor_paradigm.py).
IMU_PORTAS_PADRAO = {1: 4210, 2: 4211}


def parse_portas_imu(texto):
    """'4210' -> [(1, 4210)]; '4210,4211' -> [(1,4210), (2,4211)].

    Aceita tambem o formato explicito 'lado:porta' (ex.: '1:4210,2:4311').
    """
    itens = [item.strip() for item in str(texto).split(",") if item.strip()]
    if not itens:
        raise ValueError("nenhuma porta de IMU informada")
    saida = []
    for indice, item in enumerate(itens):
        if ":" in item:
            lado, porta = item.split(":", 1)
            lado, porta = int(lado), int(porta)
        else:
            lado = list(IMU_PORTAS_PADRAO)[indice] if len(itens) > 1 else 1
            porta = int(item)
        if lado not in IMU_PORTAS_PADRAO:
            raise ValueError(f"lado {lado} desconhecido (use 1=direita, "
                             "2=esquerda)")
        saida.append((lado, porta))
    return saida


class ImuBank:
    """Conjunto de IMUs (um por mao), cada um com receptor e fusao PROPRIOS.

    Motivo (hardware de 18/09): havera dois MPU6050, um em cada mao, cada um com
    o seu ESP32. Cada dispositivo tem o **seu relogio** (`timestamp_us`), entao
    amostras de lados diferentes NUNCA entram no mesmo filtro: cada lado tem o
    seu `ImuReceiver` (porta propria) e o seu `PositionFusion` (bias e zeragem
    independentes). Assim o protocolo, que alterna as maos a cada trial, passa a
    ter o IMU da mao CERTA -- e o lado que nao esta executando vira referencia de
    repouso para a taxa de falso movimento.
    """

    def __init__(self, portas=None):
        self.portas = parse_portas_imu(portas or "4210,4211")
        self.receivers = {lado: ImuReceiver(porta)
                          for lado, porta in self.portas}
        self.fusoes = {lado: PositionFusion() for lado, _ in self.portas}
        self._iniciado = False
        self.contagem = {lado: 0 for lado, _ in self.portas}

    # ---------------------------------------------------------------- ciclo
    def start(self):
        if self._iniciado:
            return
        for receiver in self.receivers.values():
            receiver.start()
        self._iniciado = True

    def stop(self):
        for receiver in self.receivers.values():
            try:
                receiver.stop()
            except Exception:
                pass
        self._iniciado = False

    # -------------------------------------------------------------- consulta
    @property
    def lados(self):
        return list(self.receivers)

    def amostra(self, lado):
        """Ultima amostra do lado (None se ainda nao chegou nada)."""
        receiver = self.receivers.get(int(lado))
        return None if receiver is None else receiver.get_latest()

    def idade_s(self, lado):
        """Idade da ultima amostra do lado (s), ou None sem amostra."""
        amostra = self.amostra(lado)
        return None if amostra is None else (time.monotonic()
                                             - amostra.received_at)

    def accel_no_referencial(self, lado, bias_g=None):
        """Aceleracao do lado (m/s^2, referencial da origem) ou None."""
        return accel_no_referencial(self.amostra(lado), bias_g)

    def atualiza_fusao(self, lado, posicao_camera=None):
        """Roda a fusao (zeragem) do lado; devolve (posicao, zero_lock)."""
        lado = int(lado)
        amostra = self.amostra(lado)
        fusao = self.fusoes[lado]
        posicao = fusao.update(amostra, posicao_camera)
        if amostra is not None:
            self.contagem[lado] += 1
        return posicao, int(fusao.zero_lock)

    def amostra_ativa(self, lado=None):
        """Amostra do lado pedido; sem lado, a MAIS RECENTE entre os lados.

        Usado pelos campos "legados" (um so' IMU) quando ha' mais de um sensor:
        o bloco por lado e' quem carrega o dado completo.
        """
        if lado is not None:
            return self.amostra(lado)
        recente, marca = None, -1.0
        for lado_disponivel in self.lados:
            amostra = self.amostra(lado_disponivel)
            if amostra is not None and amostra.received_at > marca:
                recente, marca = amostra, amostra.received_at
        return recente

    def get_latest(self):
        """Alias de `amostra_ativa()`: compatibilidade com quem chamava o
        `ImuReceiver` diretamente (devolve a amostra mais recente)."""
        return self.amostra_ativa()

    def blocos_motion(self, lado_ativo=0, posicao_camera=None):
        """Bloco por lado pronto para `motion["imu"]` (colunas IMU_L_*/IMU_R_*).

        Roda a fusao (zeragem) de CADA lado -- a ancora de camera so' vai para o
        `lado_ativo`, porque a medida do Kinect e' da mao que esta' sendo
        rastreada. Devolve `{lado: {"rpy_deg", "accel_g", "pos_m", "valid",
        "zero_lock"}}`; lados sem amostra saem com valid=0 e NaN.
        """
        blocos = {}
        for lado in self.lados:
            amostra = self.amostra(lado)
            ancora = posicao_camera if int(lado) == int(lado_ativo) else None
            posicao, zero_lock = self.atualiza_fusao(lado, ancora)
            if amostra is None:
                blocos[int(lado)] = {"rpy_deg": (np.nan,) * 3,
                                     "accel_g": np.full(3, np.nan),
                                     "pos_m": np.full(3, np.nan),
                                     "valid": 0, "zero_lock": 0}
                continue
            blocos[int(lado)] = {
                "rpy_deg": (float(amostra.roll_deg), float(amostra.pitch_deg),
                            float(amostra.yaw_deg)),
                "accel_g": np.asarray(amostra.accel_g, np.float64).reshape(3),
                "pos_m": np.asarray(posicao, np.float64).reshape(3),
                "valid": int(np.isfinite(amostra.accel_g).all()
                             and self.idade_s(lado) is not None
                             and self.idade_s(lado) < 0.5),
                "zero_lock": int(zero_lock),
            }
        return blocos

    def resumo(self):
        """Uma linha por lado: porta, contagem e idade da ultima amostra."""
        partes = []
        for lado, porta in self.portas:
            idade = self.idade_s(lado)
            nome = "direita" if lado == 1 else "esquerda"
            partes.append(f"{nome}(udp {porta}): {self.contagem[lado]} amostras"
                          + ("" if idade is None else f", ultima ha "
                             f"{idade:.2f} s"))
        return " | ".join(partes)