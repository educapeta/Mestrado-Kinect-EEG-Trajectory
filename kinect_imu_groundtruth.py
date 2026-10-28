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

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as exc:          # dependencia do PROJETO, nao do Python global
    # Aconteceu de verdade (05/10/2026): rodar este arquivo com o Python do
    # Windows Store (`WindowsApps\python3.11.exe`) morria em
    # "ModuleNotFoundError: No module named 'cv2'" sem dizer o que fazer -- e o
    # que fazer e' usar o `.venv` do projeto, onde as dependencias vivem.
    raise SystemExit(
        f"Faltam as dependencias do projeto: {exc}\n"
        "\n"
        "Elas vivem no `.venv` DESTE projeto. Use:\n"
        "    .venv\\Scripts\\python.exe tools\\kinect_groundtruth_tool.py   "
        "(a ferramenta)\n"
        "    .venv\\Scripts\\python.exe tools\\probe_lateralidade.py      "
        "(probe de lateralidade)\n"
        "\n"
        "E note: ESTE arquivo e' a biblioteca do projeto (nao abre nada sozinho)."
    ) from None

# pykinect2 64 BITS: o pacote do PyPI foi gerado para Python 32 bits e o import
# morre no assert do `tagSTATSTG` (issues #37/#115 do upstream). `kinect_compat`
# instala o import hook que aceita as duas ABIs -- e TEM de vir ANTES do import
# do pykinect2 (ver o docstring de `kinect_compat.py`).
import kinect_compat

kinect_compat.aplicar()

try:
    from pykinect2 import PyKinectV2
    from pykinect2.PyKinectRuntime import PyKinectRuntime
    from _ctypes import COMError
except ImportError as exc:
    raise RuntimeError(
        "pykinect2 nao esta instalado. Instale pykinect2 e o Kinect for Windows "
        f"SDK 2.0 (compatibilidade: {kinect_compat.motivo()})."
    ) from exc

UDP_PORT = 4210
# Camera auxiliar aberta AO VIVO (a que forma o par estereo com o Kinect).
# ATENCAO (medido em 23/09/2026): o indice do OpenCV NAO e' estavel -- ele e' a
# POSICAO da camera na enumeracao do DirectShow, e essa posicao muda quando uma
# camera entra/sai do USB. No MESMO dia, com as duas externas ligadas, o mapa era
#   0 = externa (vista ampla) | 1 = webcam do LAPTOP | 2 = externa (bancada)
# e, quando as externas cairam fora do USB, o 0 passou a ser a webcam do LAPTOP
# (foi assim que "a camera auxiliar" virou o rosto de quem esta' no laptop).
# Por isso o indice abaixo e' apenas o PONTO DE PARTIDA: quem manda e' o NOME.
# `camera_names.py` diz qual camera esta' em cada indice (nome do Windows) e
# `indices_auxiliares_automaticos()` monta a lista SEM a webcam do laptop.
AUX_CAMERA_INDEX = 2
# Cameras auxiliares usadas em conjunto com o Kinect (indices do OpenCV).
# Valor MEDIDO com as duas externas ligadas (2 = bancada, 0 = ampla). O PRIMEIRO
# da tupla e' a camera ativa quando ninguem passa `aux_indices` (ver
# CameraAlignment/PositionFusion): por isso a 2 vem antes.
# NUNCA incluir o indice da webcam do LAPTOP (1 no dia da medicao): ela fica no
# lugar do Kinect, sem baseline para a triangulacao.
# A calibracao por indice e armazenada em stereo_calibration_aux{idx}.npz /
# hand_landmark_calibration_aux{idx}.npz (o nome legado = indice 0).
AUX_CAMERA_INDICES = (2, 0)
# Apelido por indice, usado SO' em mensagens curtas. E' um mapa ESTATICO e ja'
# ficou desatualizado uma vez: para saber quem e' quem use camera_names.resumo().
AUX_CAMERA_NAMES = {
    0: "externa-ampla",
    1: "laptop",
    2: "externa-bancada",
}
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
# Nome REAL de cada webcam (Windows/DirectShow). Modulo leve e OPCIONAL: sem
# comtypes ele nao importa e tudo que depende de nome degrada para o apelido
# estatico (`AUX_CAMERA_NAMES`). Ver `camera_names.py`.
try:
    import camera_names
except ImportError:                   # pragma: no cover - sem comtypes
    camera_names = None

from imu import (  # noqa: E402
    IMU_ZERO_BIAS_EMA,
    IMU_ZERO_MAX_BIAS_G,
    IMU_ZERO_STD_M,
    IMU_ZERO_WINDOW,
    PALM_ALIGNMENT_MAX_RESIDUAL_DEG,
    PALM_ALIGNMENT_MIN_SAMPLES,
    POSITION_CAMERA_GAIN,
    UP_AXIS,
    VELOCITY_DAMPING,
    ImuReceiver,
    ImuSample,
    PositionFusion,
    angle_between,
    parse_portas_imu,
    rotation_matrix,
    solve_palm_alignment,
    solve_palm_alignment_pesos,
)
CALIBRATION_SECONDS = 3.0
PALM_VECTOR_LENGTH_M = 0.20

# ============================================================================
# CALIBRACAO GEOMETRICA DA PALMA (item 127, tecla X)
# ============================================================================
#: Tempo PARADO (s) com o alvo geometrico alcancado para gravar um take.
CALIB_GEO_TAKE_SEC = 2.0
#: Angulo maximo (graus) entre a normal medida pelo Kinect e o alvo para o
#: take valer. E' um controle de EXECUCAO (a verdade e' o alvo exato): o plano
#: do Kinect serve de QA, nao de verdade.
CALIB_GEO_QA_MAX_DEG = 8.0
#: Variacao maxima de rotacao (graus) e posicao (m) dentro da janela de 2 s.
CALIB_GEO_ESTAB_ROT_DEG = 4.0
CALIB_GEO_ESTAB_POS_M = 0.03
#: Movimento minimo desde o ultimo take antes de armar o proximo (evita take
#: duplicado da mesma pose).
CALIB_GEO_MOV_MIN_DEG = 8.0
CALIB_GEO_MOV_MIN_M = 0.10
#: Minimo de takes para resolver/salvar e limite de residuo medio (graus).
#: O GATE de gravacao oficial exige ainda a pose de MIRA e a validacao
#: independente dela (sem isso o ajuste pode nao generalizar para a direcao
#: absoluta) e um limite para o pior take geometrico.
CALIB_GEO_MIN_TAKES = 6
CALIB_GEO_MAX_RESIDUO_DEG = 12.0
CALIB_GEO_MAX_POSE_DEG = 20.0
CALIB_GEO_MAX_VALIDACAO_MIRA_DEG = 20.0
CALIB_GEO_MAX_LOO_MEDIANA_DEG = 15.0
#: QA EFETIVO das poses de POSICAO (palma/dorso/mira): o angulo medido e'
#: dobrado para min(ang, 180-ang) porque o Kinect pode inverter o sentido da
#: normal; assim o gate confere o ALINHAMENTO do plano a linha de mira sem
#: depender de palma x dorso (item 136). Geometricas NAO usam isso.
CALIB_GEO_QA_MAX_POSICAO_DEG = 10.0
#: Limite (graus) para o operador FORCAR um take quando o QA do Kinect esta'
#: torto mas ele confirma visualmente que a mao esta' alinhada (tecla F).
#: A verdade continua sendo o alvo exato/posicao; o take fica marcado
#: `forcado: true` no JSON piloto (item 137).
CALIB_GEO_QA_FORCAR_MAX_DEG = 35.0
#: Separacao minima (graus) entre DIRECOES de mira de dois takes de
#: posicao (item 139): garante espalhamento no plano de posicoes.
CALIB_GEO_SEPARACAO_MIN_DEG = 15.0
#: Espalhamento minimo do conjunto: menor par >= 15, maior par >= 30.
CALIB_GEO_SPREAD_MIN_DEG = 15.0
CALIB_GEO_SPREAD_MAX_MIN_DEG = 30.0
#: Pesos no ajuste hibrido: a pose de mira (verdade por posicao, independente
#: do plano do Kinect) ancora a direcao absoluta; as poses geometricas tem
#: verdade EXATA (alvo do referencial da camera) mas erro de execucao maior.
CALIB_GEO_PESO_GEOMETRICA = 1.0
CALIB_GEO_PESO_POSICAO = 1.5
#: Poses por mao. `alvo` e' a direcao EXATA da normal da palma no referencial
#: da CAMERA (x=direita, y=CIMA na imagem, z=para DENTRO da cena -- a lente
#: fica em -Z vista de quem esta' na frente dela). `dedos` e' a direcao de
#: referencia da mao no MESMO referencial (guia a "rolagem" do punho, nao entra
#: na verdade). O tipo "posicao" usa a verdade `-normalize(posicao da mao)`.
_R2 = 0.7071067811865476
POSES_CALIBRACAO_GEOMETRICA = (
    {"nome": "CENTRO (~45 cm)", "tipo": "posicao", "sinal": 1.0,
     "alvo": None, "dedos": None,
     "instrucao": "Mao a ~45 cm, no CENTRO do quadro; PALMA MIRANDO A LENTE."},
    {"nome": "ESQUERDA (~25 cm)", "tipo": "posicao", "sinal": 1.0,
     "alvo": None, "dedos": None,
     "instrucao": "Mao deslocada ~25 cm para a ESQUERDA da imagem (~45 cm "
                  "de distancia); PALMA MIRANDO A LENTE."},
    {"nome": "DIREITA (~25 cm)", "tipo": "posicao", "sinal": 1.0,
     "alvo": None, "dedos": None,
     "instrucao": "Mao deslocada ~25 cm para a DIREITA da imagem (~45 cm); "
                  "PALMA MIRANDO A LENTE."},
    {"nome": "CIMA (~20 cm)", "tipo": "posicao", "sinal": 1.0,
     "alvo": None, "dedos": None,
     "instrucao": "Mao deslocada ~20 cm para CIMA na imagem (~45 cm); PALMA "
                  "MIRANDO A LENTE."},
    {"nome": "BAIXO (~20 cm)", "tipo": "posicao", "sinal": 1.0,
     "alvo": None, "dedos": None,
     "instrucao": "Mao deslocada ~20 cm para BAIXO na imagem (~45 cm); PALMA "
                  "MIRANDO A LENTE."},
    {"nome": "DIAG. ESQ+CIMA", "tipo": "posicao", "sinal": 1.0,
     "alvo": None, "dedos": None,
     "instrucao": "Mao na diagonal ESQUERDA+CIMA (~20 cm em cada eixo); "
                  "PALMA MIRANDO A LENTE."},
    {"nome": "DIAG. DIR+BAIXO", "tipo": "posicao", "sinal": 1.0,
     "alvo": None, "dedos": None,
     "instrucao": "Mao na diagonal DIREITA+BAIXO (~20 cm em cada eixo); "
                  "PALMA MIRANDO A LENTE."},
    {"nome": "DIAG. ESQ+BAIXO", "tipo": "posicao", "sinal": 1.0,
     "alvo": None, "dedos": None,
     "instrucao": "Mao na diagonal ESQUERDA+BAIXO (~20 cm em cada eixo); "
                  "PALMA MIRANDO A LENTE."},
    {"nome": "DIAG. DIR+CIMA", "tipo": "posicao", "sinal": 1.0,
     "alvo": None, "dedos": None,
     "instrucao": "Mao na diagonal DIREITA+CIMA (~20 cm em cada eixo); "
                  "PALMA MIRANDO A LENTE."},
)


def _rotacao_a_para_b(a, b):
    """Rotacao minima que leva o vetor unitario `a` no vetor unitario `b`."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)          # entrada degenerada: rotacao neutra
    a = a / na
    b = b / nb
    eixo = np.cross(a, b)
    cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(eixo) < 1e-12:
        if cos > 0.0:
            return np.eye(3)
        # 180 graus: qualquer eixo perpendicular
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-9:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return _matriz_rotacao_eixo(perp, np.pi)
    eixo /= np.linalg.norm(eixo)
    return _matriz_rotacao_eixo(eixo, float(np.arccos(cos)))


def _matriz_rotacao_eixo(eixo, angulo):
    x, y, z = eixo
    c, s = np.cos(angulo), np.sin(angulo)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], np.float64)


def _angulo_rotacoes(rot_a, rot_b):
    """Angulo (graus) entre duas rotacoes."""
    a = np.asarray(rot_a, np.float64)
    b = np.asarray(rot_b, np.float64)
    c = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class CalibracaoGeometrica:
    """Calibracao da palma por ALVOS GEOMETRICOS da camera (item 127).

    Diferenca para o J (verdade = direcao mao->lente): aqui as primeiras 8
    poses tem verdade EXATA no referencial da camera (ex.: normal da palma
    paralela a X, Y ou Z da camera), com o usuario alinhando o plano da mao a
    um plano-guia desenhado no video. A ultima pose e' de MIRA (verdade por
    posicao) para ancorar a direcao absoluta. O ajuste e' hibrido e ponderado:
    as direcoes geometricas nao tem erro de MEDIDA (so de execucao); a mira
    nao depende do plano do Kinect.

    Uma mao por vez: o tool cria uma instancia para "right" e, no fim, outra
    para "left". Cada `amostra` recebe rotacao do IMU, posicao 3D da palma e
    normal da palma medida pelo Kinect (usada apenas como QA de execucao).

    Regras de take (mesmas dos takes do J):
      - QA = angulo entre normal medida e o alvo <= CALIB_GEO_QA_MAX_DEG;
      - 2 s PARADO (rotacao/posicao estaveis) -> grava;
      - exige movimento minimo desde o ultimo take antes do proximo;
      - pose de mira: alvo/verdade = -normalize(posicao da mao).
    """

    def __init__(self, lado, poses=None, avisar=print):
        self.lado = str(lado)
        self.poses = [dict(pose) for pose in (poses or
                                              POSES_CALIBRACAO_GEOMETRICA)]
        self.avisar = avisar
        self.indice = 0
        self.capturas = []
        self.ativa = True
        self.qa_atual = None
        self.qa_efetivo = None
        self.status = "aguardando a mao"
        self._inicio_ok = None
        self._rot_ref = None
        self._pos_ref = None
        self._ultima_rot = None
        self._ultima_pos = None
        #: ultima amostra VALIDA (rotacao/posicao/normal/QA) para `forcar_take`
        self._ultimo = None

    # -- estado ---------------------------------------------------------
    @property
    def completa(self):
        return self.indice >= len(self.poses)

    @property
    def pose_atual(self):
        return None if self.completa else self.poses[self.indice]

    def parar(self):
        self.ativa = False

    def progresso(self, now):
        if self.completa:
            return {"indice": len(self.poses), "total": len(self.poses),
                    "pose": None, "qa_deg": self.qa_atual,
                    "tempo_ok": 0.0, "status": "COMPLETA"}
        pose = self.pose_atual
        tempo = (0.0 if self._inicio_ok is None
                 else max(0.0, float(now) - self._inicio_ok))
        return {"indice": self.indice, "total": len(self.poses),
                "pose": pose, "qa_deg": self.qa_atual,
                "qa_efetivo_deg": self.qa_efetivo,
                "tempo_ok": tempo, "status": self.status}

    # -- verdade --------------------------------------------------------
    def _verdade(self, pose, posicao):
        if pose.get("tipo") == "posicao":
            if posicao is None:
                return None
            # sinal=+1: PALMA mirando a lente (normal = -normalize(pos));
            # sinal=-1: DORSO mirando a lente (normal = +normalize(pos)).
            vetor = -np.asarray(posicao, np.float64).reshape(3)
            norma = np.linalg.norm(vetor)
            if norma < 1e-6:
                return None
            vetor = vetor / norma
            return vetor * float(pose.get("sinal", 1.0))
        alvo = np.asarray(pose["alvo"], np.float64).reshape(3)
        norma = np.linalg.norm(alvo)
        return None if norma < 1e-9 else alvo / norma

    # -- amostra --------------------------------------------------------
    def amostra(self, now, rotacao, posicao_palma, normal_medida):
        """Recebe uma amostra; devolve evento {'tipo': 'take'...} ou None.

        `normal_medida` e' usada como QA apenas nas poses GEOMETRICAS. Nas
        poses de POSICAO (palma/dorso mirando a lente, mira) a verdade vem da
        posicao 3D da mao: o Kinect pode inverter o sentido da normal
        (palma x dorso) e isso NAO pode bloquear o take -- o QA fica apenas
        informativo (item 135).
        """
        if not self.ativa or self.completa:
            return None
        pose = self.pose_atual
        tipo_pose = pose.get("tipo", "geometrica")
        if rotacao is None or posicao_palma is None:
            self.qa_atual = None
            self.qa_efetivo = None
            self._inicio_ok = None
            self.status = "SEM MAO -- mostre a mao para o Kinect"
            return None
        rot = np.asarray(rotacao, np.float64)
        pos = np.asarray(posicao_palma, np.float64).reshape(3)
        if (rot.shape != (3, 3) or not np.isfinite(rot).all()
                or not np.isfinite(pos).all()):
            self.qa_atual = None
            self.qa_efetivo = None
            self._inicio_ok = None
            return None
        normal = None
        if normal_medida is not None:
            try:
                candidato = np.asarray(normal_medida, np.float64).reshape(3)
                norma_normal = float(np.linalg.norm(candidato))
                if np.isfinite(candidato).all() and norma_normal > 1e-9:
                    normal = candidato / norma_normal
            except (TypeError, ValueError):
                normal = None
        verdade = self._verdade(pose, pos)
        if verdade is None:
            self.qa_atual = None
            self.qa_efetivo = None
            self._inicio_ok = None
            return None
        self.qa_atual = (None if normal is None
                         else angle_between(normal, verdade))
        # QA "dobrado": min(ang, 180-ang). Para POSICAO ele aceita o Kinect
        # invertendo palma/dorso (item 136); para GEOMETRICA nao e' usado no
        # gate (la' o sentido importa: +X != -X).
        self.qa_efetivo = (
            None if self.qa_atual is None or not np.isfinite(self.qa_atual)
            else float(min(self.qa_atual, 180.0 - self.qa_atual)))
        self._ultimo = {
            "rotacao": rot,
            "posicao": pos,
            "normal": normal,
            "qa_deg": self.qa_atual,
            "qa_efetivo_deg": self.qa_efetivo,
        }
        if tipo_pose == "geometrica":
            if (normal is None or self.qa_atual is None
                    or not np.isfinite(self.qa_atual)
                    or self.qa_atual > CALIB_GEO_QA_MAX_DEG):
                self._inicio_ok = None
                if normal is None:
                    self.status = "SEM PLANO do Kinect -- mostre o dorso da mao"
                else:
                    self.status = (
                        f"ALINHE o plano ao alvo (desvio "
                        f"{self.qa_atual:.1f} > {CALIB_GEO_QA_MAX_DEG:g})")
                return None
        else:
            # Pose de POSICAO: gate pelo angulo DOBRADO (alinhamento do plano
            # a linha de mira, sem depender do sentido palma x dorso).
            if (self.qa_efetivo is not None
                    and self.qa_efetivo > CALIB_GEO_QA_MAX_POSICAO_DEG):
                self._inicio_ok = None
                self.status = (
                    f"ALINHE o plano a linha de mira (desvio efetivo "
                    f"{self.qa_efetivo:.1f} > "
                    f"{CALIB_GEO_QA_MAX_POSICAO_DEG:g}; bruto "
                    f"{self.qa_atual:.1f})")
                return None
            if self.qa_atual is not None and np.isfinite(self.qa_atual):
                self.status = (
                    f"MIRANDO (QA bruto {self.qa_atual:.1f}, efetivo "
                    f"{self.qa_efetivo:.1f}; sentido palma/dorso NAO bloqueia)")
            else:
                self.status = "MIRANDO (posicao; normal do Kinect indisponivel)"

        if self._ultima_rot is not None:
            mudou_rot = _angulo_rotacoes(rot, self._ultima_rot)
            mudou_pos = float(np.linalg.norm(pos - self._ultima_pos))
            if mudou_rot < CALIB_GEO_MOV_MIN_DEG and                     mudou_pos < CALIB_GEO_MOV_MIN_M:
                self._inicio_ok = None
                self.status = "MOVA a mao para a proxima pose"
                return None
        if self._inicio_ok is None:
            self._inicio_ok = float(now)
            self._rot_ref = rot
            self._pos_ref = pos
            self.status = "PARADO 0.0/2.0 s"
            return None
        if _angulo_rotacoes(rot, self._rot_ref) > CALIB_GEO_ESTAB_ROT_DEG or                 float(np.linalg.norm(pos - self._pos_ref)) >                 CALIB_GEO_ESTAB_POS_M:
            self._inicio_ok = float(now)
            self._rot_ref = rot
            self._pos_ref = pos
            self.status = "PARADO 0.0/2.0 s"
            return None
        decorrido = float(now) - self._inicio_ok
        if decorrido < CALIB_GEO_TAKE_SEC:
            self.status = f"PARADO {decorrido:.1f}/{CALIB_GEO_TAKE_SEC:.1f} s"
            return None
        if tipo_pose == "posicao":
            for _cap_ant in self.capturas:
                if _cap_ant.get("tipo") != "posicao":
                    continue
                _ang_sep = angle_between(verdade, _cap_ant["verdade"])
                if np.isfinite(_ang_sep) and \
                        _ang_sep < CALIB_GEO_SEPARACAO_MIN_DEG:
                    self._inicio_ok = None
                    self.status = (
                        f"MOVA A MAO: direcao {_ang_sep:.0f} graus < "
                        f"{CALIB_GEO_SEPARACAO_MIN_DEG:g} da pose anterior "
                        f"({_cap_ant['pose']})")
                    return None
        captura = {
            "indice": self.indice,
            "pose": pose["nome"],
            "tipo": pose.get("tipo", "geometrica"),
            "sinal": float(pose.get("sinal", 1.0)),
            "alvo": (None if pose.get("alvo") is None
                     else np.asarray(pose["alvo"], np.float64).tolist()),
            "verdade": verdade.tolist(),
            "qa_deg": (None if self.qa_atual is None
                       or not np.isfinite(self.qa_atual)
                       else float(self.qa_atual)),
            "qa_efetivo_deg": self.qa_efetivo,
            "rotacao": rot.tolist(),
            "posicao": pos.tolist(),
            "normal_medida": (None if normal is None else normal.tolist()),
            "forcado": False,
        }
        self.capturas.append(captura)
        self.indice += 1
        self._ultima_rot = rot
        self._ultima_pos = pos
        self._inicio_ok = None
        self.qa_atual = None
        self.qa_efetivo = None
        self.status = "TAKE GRAVADO"
        return {"tipo": "take", "captura": captura,
                "total": len(self.poses)}

    def pular_pose(self):
        """Pula a pose atual (sem gravar) e devolve evento."""
        if not self.ativa or self.completa:
            return None
        pose = self.pose_atual
        self.indice += 1
        self._inicio_ok = None
        self.qa_atual = None
        self.qa_efetivo = None
        self.status = "POSE PULADA"
        return {"tipo": "pulada", "pose": pose["nome"],
                "total": len(self.poses)}

    def forcar_take(self, now=None):
        """Grava a pose ATUAL mesmo com QA torto (tecla F do tool).

        A verdade e' sempre o alvo exato/posicao -- o que muda e' que o
        operador confirma visualmente o alinhamento quando o Kinect mede um
        QA entre o limite normal e `CALIB_GEO_QA_FORCAR_MAX_DEG`. O take fica
        marcado `forcado: True` no piloto e o QA bruto/efetivo e' registrado.
        Devolve {"tipo": "take"...} ou {"tipo": "recusado", "motivo": ...}.
        """
        if not self.ativa or self.completa:
            return {"tipo": "recusado", "motivo": "modo inativo/completo"}
        if self._ultimo is None:
            return {"tipo": "recusado",
                    "motivo": "sem amostra valida ainda (mostre a mao)"}
        pose = self.pose_atual
        tipo_pose = pose.get("tipo", "geometrica")
        rot = np.asarray(self._ultimo["rotacao"], np.float64)
        pos = np.asarray(self._ultimo["posicao"], np.float64)
        normal = self._ultimo["normal"]
        verdade = self._verdade(pose, pos)
        if verdade is None:
            return {"tipo": "recusado", "motivo": "verdade indisponivel"}
        qa = (None if normal is None else angle_between(normal, verdade))
        qa_efetivo = (None if qa is None or not np.isfinite(qa)
                      else float(min(qa, 180.0 - qa)))
        if tipo_pose == "geometrica":
            qa_gate = qa
        else:
            qa_gate = qa_efetivo
        if qa_gate is not None and qa_gate > CALIB_GEO_QA_FORCAR_MAX_DEG:
            return {"tipo": "recusado",
                    "motivo": (f"QA {qa_gate:.1f} > "
                               f"{CALIB_GEO_QA_FORCAR_MAX_DEG:g}; ajuste a mao"),
                    "qa_deg": qa, "qa_efetivo_deg": qa_efetivo}
        if tipo_pose == "geometrica" and normal is None:
            return {"tipo": "recusado",
                    "motivo": "pose geometrica sem normal do Kinect"}
        captura = {
            "indice": self.indice,
            "pose": pose["nome"],
            "tipo": tipo_pose,
            "sinal": float(pose.get("sinal", 1.0)),
            "alvo": (None if pose.get("alvo") is None
                     else np.asarray(pose["alvo"], np.float64).tolist()),
            "verdade": verdade.tolist(),
            "qa_deg": (None if qa is None or not np.isfinite(qa)
                       else float(qa)),
            "qa_efetivo_deg": qa_efetivo,
            "rotacao": rot.tolist(),
            "posicao": pos.tolist(),
            "normal_medida": (None if normal is None else normal.tolist()),
            "forcado": True,
        }
        self.capturas.append(captura)
        self.indice += 1
        self._inicio_ok = None
        self._ultimo = None
        self.qa_atual = None
        self.qa_efetivo = None
        self.status = "TAKE FORCADO (QA registrado)"
        return {"tipo": "take", "captura": captura,
                "total": len(self.poses)}

    # -- resultado ------------------------------------------------------
    def resultado(self):
        """Resolve T/f hibrido + estatisticas e validacao (dict)."""
        if len(self.capturas) < CALIB_GEO_MIN_TAKES:
            return {"erro": (f"takes insuficientes ({len(self.capturas)} < "
                             f"{CALIB_GEO_MIN_TAKES})")}
        rotacoes = [np.asarray(c["rotacao"], np.float64)
                    for c in self.capturas]
        verdades = [np.asarray(c["verdade"], np.float64)
                    for c in self.capturas]
        pesos = [CALIB_GEO_PESO_POSICAO if c["tipo"] == "posicao"
                 else CALIB_GEO_PESO_GEOMETRICA for c in self.capturas]
        solucao = solve_palm_alignment_pesos(rotacoes, verdades, pesos=pesos)
        if solucao is None:
            return {"erro": "o ajuste T/f nao convergiu"}
        T, f_body, residuo = solucao
        por_pose = []
        for c, rot, verdade in zip(self.capturas, rotacoes, verdades):
            erro = angle_between(T @ rot @ f_body, verdade)
            por_pose.append({"pose": c["pose"], "tipo": c["tipo"],
                             "qa_deg": c["qa_deg"], "erro_deg": float(erro)})
        erros = [p["erro_deg"] for p in por_pose
                 if np.isfinite(p["erro_deg"])]
        # validacao POR POSICAO: resolve sem a(s) pose(s) de mira e mede nelas
        validacao_posicao = None
        indices_pos = [i for i, c in enumerate(self.capturas)
                       if c["tipo"] == "posicao"]
        if indices_pos and len(self.capturas) - len(indices_pos) >=                 CALIB_GEO_MIN_TAKES:
            rot_sem = [rot for i, rot in enumerate(rotacoes)
                       if i not in indices_pos]
            ver_sem = [ver for i, ver in enumerate(verdades)
                       if i not in indices_pos]
            pe_sem = [pe for i, pe in enumerate(pesos) if i not in indices_pos]
            sol_sem = solve_palm_alignment_pesos(rot_sem, ver_sem,
                                                 pesos=pe_sem)
            if sol_sem is not None:
                erros_mira = []
                for i in indices_pos:
                    erros_mira.append(float(angle_between(
                        sol_sem[0] @ rotacoes[i] @ sol_sem[1], verdades[i])))
                validacao_posicao = {
                    "n": len(indices_pos),
                    "mediana_deg": float(np.median(erros_mira)),
                    "max_deg": float(np.max(erros_mira)),
                }
        # validacao CRUZADA: deixa cada take de fora e mede nele
        loo = []
        if len(self.capturas) >= CALIB_GEO_MIN_TAKES + 1:
            for i in range(len(self.capturas)):
                rot_sem = [r for j, r in enumerate(rotacoes) if j != i]
                ver_sem = [v for j, v in enumerate(verdades) if j != i]
                pe_sem = [p for j, p in enumerate(pesos) if j != i]
                sol = solve_palm_alignment_pesos(rot_sem, ver_sem,
                                                 pesos=pe_sem)
                if sol is not None:
                    loo.append(float(angle_between(
                        sol[0] @ rotacoes[i] @ sol[1], verdades[i])))
        erros_geo = [p["erro_deg"] for p in por_pose
                     if p["tipo"] == "geometrica" and
                     np.isfinite(p["erro_deg"])]
        erros_pos = [p["erro_deg"] for p in por_pose
                     if p["tipo"] == "posicao" and np.isfinite(p["erro_deg"])]
        espalhamento = [float(angle_between(verdades[i], verdades[j]))
                        for i in range(len(verdades))
                        for j in range(i + 1, len(verdades))]
        espalhamento = [valor for valor in espalhamento
                        if np.isfinite(valor)]
        return {
            "lado": self.lado,
            "rotation": np.asarray(T, np.float64).tolist(),
            "body": np.asarray(f_body, np.float64).tolist(),
            "residual_deg": float(residuo),
            "n_takes": len(self.capturas),
            "por_pose": por_pose,
            "erro_geometricas_mediana": (float(np.median(erros_geo))
                                         if erros_geo else None),
            "erro_geometricas_max": (float(np.max(erros_geo))
                                     if erros_geo else None),
            "erro_mira_mediana": (float(np.median(erros_pos))
                                  if erros_pos else None),
            "validacao_mira": validacao_posicao,
            "loo_mediana_deg": (float(np.median(loo)) if loo else None),
            "loo_max_deg": (float(np.max(loo)) if loo else None),
            "erros_globais_mediana": (float(np.median(erros)) if erros
                                      else None),
            "espalhamento_min_deg": (float(min(espalhamento))
                                     if espalhamento else None),
            "espalhamento_max_deg": (float(max(espalhamento))
                                     if espalhamento else None),
        }
#: Lateralidade da mao detectada (colunas KT_hand / PALM_hand do CSV):
#: 0 = desconhecida, 1 = direita, 2 = esquerda.
HAND_SIDE_CODE = {"": 0, "right": 1, "left": 2}
#: O MediaPipe define a lateralidade assumindo imagem ESPELHADA (selfie). A
#: webcam auxiliar e' espelhada no nosso pipeline (rotulo ja correto) e o RGB
#: do Kinect NAO e' espelhado -> o rotulo do Kinect precisa ser invertido.
#: Sem isso, uma trial de mao ESQUERDA poderia triangular a mao DIREITA.
MEDIAPIPE_KINECT_HANDEDNESS_FLIP = True
# --- LATERALIDADE POR CAMERA AUXILIAR (o que estava trocando as maos) ---------
#: Inversao do rotulo do MediaPipe POR CAMERA auxiliar (indice OpenCV -> bool).
#: True = o rotulo DESTA camera sai trocado em relacao ao Kinect e tem de ser
#: invertido antes de casar as maos.
#:
#: Por que o flip global nao bastava (bug de bancada, 05/10/2026): o pipeline
#: espelha TODAS as auxiliares com o MESMO `AUXILIARY_MIRROR_HORIZONTAL` e o
#: rotulo do MediaPipe depende de o frame estar espelhado (convencao "selfie").
#: Com duas webcams de MODELOS DIFERENTES (nesta maquina: 0 = SIGMA-W780M e
#: 1 = Trust USB Camera) o driver de uma delas pode JA' entregar o fluxo
#: espelhado: o nosso flip entao DESespelha, a premissa do MediaPipe cai e SO'
#: aquela camera fica com o rotulo invertido. Sintoma exato do relato: com as
#: duas maos no quadro uma auxiliar concorda com o Kinect e a outra troca as
#: maos -- e o casamento por lado (que e' o que liga cada mao ao IMU certo)
#: passa a triangular/desenhar a mao errada naquela camera.
#:
#: Quem descobre o valor NAO e' o usuario: `ArbitroDeLateralidade` compara o
#: rotulo com o lado medido pela GEOMETRIA (o Kinect tem o punho em 3D e cada
#: auxiliar tem a homografia no-a-no da mao) e grava aqui quando ha' consenso --
#: `tools\probe_lateralidade.py` faz o mesmo ao vivo, por camera, e o tool aceita
#: `--aux-flip-lateralidade N` para forcar/limpar na mao.
AUX_HANDEDNESS_FLIP_FILE = Path(__file__).with_name("aux_handedness_flip.json")
_AUX_HANDEDNESS_FLIP = {}


def aux_handedness_flip(aux_index):
    """True quando o rotulo do MediaPipe DESTA auxiliar tem de ser invertido."""
    return bool(_AUX_HANDEDNESS_FLIP.get(int(aux_index), False))


def carrega_aux_handedness_flip(caminho=None):
    """Le o json das inversoes APRENDIDAS (arquivo ausente ou corrompido = nada
    aprendido -- nunca chuta: sem o arquivo vale o flip global do pipeline)."""
    global _AUX_HANDEDNESS_FLIP
    arquivo = Path(caminho or AUX_HANDEDNESS_FLIP_FILE)
    if not arquivo.exists():
        return dict(_AUX_HANDEDNESS_FLIP)
    try:
        with open(arquivo, "r", encoding="utf-8") as aberto:
            conteudo = json.load(aberto)
        _AUX_HANDEDNESS_FLIP = {
            int(indice): bool(invertido)
            for indice, invertido in (conteudo.get("flip") or {}).items()
        }
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        print(f"AVISO: {arquivo.name} foi ignorado ({exc}); nenhuma inversao de "
              "lateralidade sera' aplicada ate' ser medida de novo", flush=True)
    return dict(_AUX_HANDEDNESS_FLIP)


def set_aux_handedness_flip(aux_index, invertido, motivo="", persistir=True,
                            silencioso=False):
    """Grava a inversao de UMA camera auxiliar (e persiste no json).

    Persiste SEMPRE, mesmo quando o valor nao mudou: o json e' o unico canal
    pelo qual o PROXIMO programa (protocolo, sand, demo) recebe a descoberta --
    um `True` que ficasse so' na memoria desta sessao faria a sessao seguinte
    voltar a trocar as maos.
    """
    global _AUX_HANDEDNESS_FLIP
    indice = int(aux_index)
    valor = bool(invertido)
    mudou = _AUX_HANDEDNESS_FLIP.get(indice) != valor
    _AUX_HANDEDNESS_FLIP[indice] = valor
    if persistir:
        arquivo = Path(AUX_HANDEDNESS_FLIP_FILE)
        try:
            with open(arquivo, "w", encoding="utf-8") as saida:
                json.dump(
                    {"flip": {str(chave): estado for chave, estado
                              in sorted(_AUX_HANDEDNESS_FLIP.items())},
                     "motivo": motivo or "",
                     "escrito_em": time.strftime("%d/%m/%Y %H:%M:%S")},
                    saida, indent=2, sort_keys=True)
        except OSError as exc:                     # disco cheio / somente leitura
            if not silencioso:
                print(f"AVISO: nao consegui gravar {arquivo.name} ({exc}); a "
                      "inversao vale SO' nesta sessao", flush=True)
    if (mudou or valor) and not silencioso:
        estado = "INVERTIDO (corrigido)" if valor else "normal"
        print(f"Lateralidade da camera auxiliar {indice}: {estado}"
              + (f" -- {motivo}" if motivo else ""), flush=True)
    return valor


def limpa_aux_handedness_flip(persistir=True):
    """Esquece as inversoes aprendidas (memoria E arquivo).

    Devolve o que estava gravado (para a mensagem poder dizer o que foi
    esquecido). Usado por `--aux-flip-lateralidade limpar`.
    """
    global _AUX_HANDEDNESS_FLIP
    antes = dict(_AUX_HANDEDNESS_FLIP)
    _AUX_HANDEDNESS_FLIP = {}
    if persistir:
        try:
            Path(AUX_HANDEDNESS_FLIP_FILE).unlink(missing_ok=True)
        except OSError as exc:                     # arquivo em uso/sem permissao
            print(f"AVISO: nao consegui apagar {AUX_HANDEDNESS_FLIP_FILE.name} "
                  f"({exc}); o arquivo volta a valer na proxima abertura",
                  flush=True)
    return antes


carrega_aux_handedness_flip()
# Precisa casar EXATAMENTE com stereo_calibration.py: o stereo salvo e' validado
# contra estes dois valores (se mudarem, a calibracao antiga fica INVALIDA e o
# arquivo tem de ser regerado com a placa nova; `CameraAlignment.stereo_status`
# diz o MOTIVO da recusa -- tabuleiro/quadrado diferente x RMS alto).
# PLACA ATUAL (impressa em 01/10/2026): 8 x 6 QUADRADOS de 45 mm -> 7 x 5 CANTOS
# internos (o cv2 conta cantos internos = quadrados - 1). A aresta de cantos mede
# 270 x 180 mm (diagonal 324 mm = 1,27x a da placa de 57 mm): e' o que faz o
# tabuleiro chegar a 35% da diagonal do Kinect a ~0,44 m (antes 0,35 m) e, com
# isso, a webcam auxiliar tambem enxerga-lo na mesma pose.
# Historico: 4x3/52 mm -> 5x3/52 mm -> 5x3/57 mm (23/09) -> 7x5/45 mm (01/10).
# MEDIDO em 01/10/2026: as duas constantes ficaram desalinhadas por uma sessao (o
# stereo ja' gravava 7x5/45 mm e aqui ainda estava 5x3/57 mm) e o sintoma NAO era
# "RMS alto": os DOIS npz recem-gravados voltavam
# 'rejeitada -- tabuleiro gravado 7x5 != 5x3 cantos da placa atual -> RECALIBRE',
# e o `kinect_intrinsics.npz` (5x3/57 mm) deixou de ser aceito por
# `carrega_intrinsecas_kinect` -- ou seja, a sessao voltou a RE-ESTIMAR o Kinect,
# que e' exatamente o bug que o modelo congelado existe para matar. Quem trava a
# divergencia daqui para a frente e' `test_intrinsecas_kinect.py`.
CHECKERBOARD_SIZE = (7, 5)
SQUARE_SIZE_M = 0.045
# --- Deteccao do tabuleiro no CAMINHO DE DIAGNOSTICO (CameraAlignment.find_board)
# Por que estes tres valores existem (medido na bancada, 28/09/2026, tecla T):
# o diagnostico respondia "tabuleiro nao visto nas duas cameras" com o tabuleiro
# NA MESA e as tres cameras abertas, porque `find_board` rodava SO' na resolucao
# cheia, sem CLAHE e com CALIB_CB_ACCURACY -- exatamente as tres coisas que o
# caminho da CALIBRACAO (`stereo_calibration.find_corners`) evita desde 23/09
# (ver o bloco DETECTION_WIDTHS de la: o SB fica na FRONTEIRA e a mesma
# configuracao acerta numa rodada e falha na outra; ACCURACY so' pesa). Aqui o
# diagnostico passou a usar o MESMO tipo de caminho, para ele nao negar o que a
# calibracao ve.
#: Larguras-alvo tentadas em cascata (0 = resolucao cheia, por ultimo).
BOARD_DETECTION_WIDTHS = (960, 640, 0)
#: CLAHE no cinza antes de detectar (0 desliga): o que mais ajuda com impressao
#: de baixo contraste, papel brilhante ou luz desigual.
BOARD_DETECTION_CLAHE_CLIP = 2.0
BOARD_DETECTION_CLAHE_GRID = 8
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


def aux_hand_output_file(aux_index):
    """Arquivo ONDE SALVAR a calibracao de landmarks de uma camera auxiliar.

    Espelha a regra do stereo (`output_file` em stereo_calibration.py): indice 0
    usa o nome LEGADO (hand_landmark_calibration.npz) e os outros _aux{N}.npz.

    Existe porque `solve_hand_landmark_calibration` salvava SEMPRE no nome legado:
    com a camera ativa em qualquer indice != 0 a calibracao da mao era gravada no
    arquivo ERRADO -- `aux_hand_file(2)` procura `_aux2.npz`, nao acha, e a
    captura da tecla H se perdia (o sintoma seria "calibrei a mao e o programa
    continua usando a homografia antiga").
    """
    if int(aux_index) == 0:
        return HAND_CALIBRATION_FILE
    return Path(__file__).with_name(
        f"hand_landmark_calibration_aux{int(aux_index)}.npz")


def indices_auxiliares_automaticos(recarregar=False):
    """Indices das webcams auxiliares desta maquina, pelo NOME (sem chutar).

    Motivo: o indice do OpenCV e' a POSICAO na enumeracao do DirectShow, entao
    ele muda quando uma camera entra/sai do USB -- foi assim que "a webcam
    auxiliar" virou a webcam do LAPTOP (ver AUX_CAMERA_INDEX). Aqui o filtro e'
    por NOME: entram todas as cameras MENOS a webcam do laptop, que fica no lugar
    do Kinect (sem baseline).

    Ordem: primeiro as que TEM calibracao stereo (`stereo_calibration_aux{N}.npz`)
    -- a primeira da lista e' a camera ATIVA (overlay/triangulacao), e usar a que
    tem calibracao evita abrir o programa com a sobreposicao desligada.

    Tupla vazia = nenhuma camera externa enumerada (ou comtypes/COM fora).
    """
    if camera_names is None:
        return ()
    externas = [indice for indice, nome in
                camera_names.mapa_de_cameras(recarregar).items()
                if not camera_names.e_do_laptop(nome)]
    com_calibracao = [i for i in externas if aux_stereo_file(i) is not None]
    sem_calibracao = [i for i in externas if i not in com_calibracao]
    return tuple(com_calibracao + sem_calibracao)


def indice_auxiliar_automatico(preferidas=None, reserva=None):
    """Indice da auxiliar ATIVA quando ninguem passa `--aux-index`.

    O default HISTORICO era a constante AUX_CAMERA_INDEX (2), que so' valia na
    maquina de 23/09/2026. Medido em 28/09/2026 nesta maquina: existem as DUAS
    externas nos indices 0 (SIGMA-W780M) e 1 (Trust USB Camera) e NENHUM indice
    2 -- a ferramenta abortava com "a webcam auxiliar ATIVA (indice 2) nao
    abriu" mesmo com as duas externas funcionando e ABRINDO nos indices 0 e 1.

    Agora o padrao sai do NOME (`indices_auxiliares_automaticos()`: todas as
    cameras menos a do laptop, as que TEM calibracao stereo primeiro) e a
    constante fica como ULTIMO recurso, quando a enumeracao nao devolve nada.

    `preferidas` existe para o teste injetar a lista sem tocar no Windows.
    """
    lista = (indices_auxiliares_automaticos() if preferidas is None
             else list(preferidas))
    if lista:
        return int(lista[0])
    return int(AUX_CAMERA_INDEX if reserva is None else reserva)


def nome_auxiliar(indice):
    """Nome real (Windows) da camera do indice, ou o apelido estatico, ou '?'.

    Fonte unica para mensagens: quem imprime "camera X (indice N)" usa isto, para
    nunca mais anunciar 'usb'/'laptop' por um mapa estatico ja' errado.
    """
    real = None if camera_names is None else camera_names.nome_da_camera(indice)
    return real or AUX_CAMERA_NAMES.get(int(indice), "?")


def nome_da_camera_do_indice(indice):
    """Nome (Windows) da camera daquele indice -- None se nao der para saber.

    Serve para CONFERIR A QUAL CAMERA um npz pertence (o indice do OpenCV e' a
    posicao na enumeracao do DirectShow: muda quando uma webcam entra/sai do
    USB). Nunca levanta: sem comtypes, com COM falhando ou indice inexistente o
    resultado e' None, e quem chama simplesmente nao faz a checagem -- nome e'
    conveniencia, nunca pre-requisito para o programa rodar.
    """
    if camera_names is None:
        return None
    try:
        return camera_names.nome_da_camera(indice)
    except Exception:                     # pragma: no cover - COM/ctypes
        return None


def nome_de_camera_confere(gravado, indice):
    """True se der para afirmar que o npz e' DESTA camera (ou se nao da' para saber).

    Confere o nome do Windows gravado no npz com o nome da camera do indice hoje,
    ignorando caixa e espacos. Devolve True (nada a declarar) quando um dos dois
    nomes e' desconhecido -- npz anterior a 29/09/2026, sem comtypes, indice fora
    da enumeracao: nome e' prova, nunca pre-requisito.
    """
    if not gravado:
        return True
    atual = nome_da_camera_do_indice(indice)
    if not atual:
        return True
    return gravado.strip().lower() == atual.strip().lower()


def aviso_camera_do_laptop(indice, papel="webcam auxiliar"):
    """Texto de aviso se o indice for a webcam do laptop (senao None).

    A frase vive em `camera_names.AVISO_LAPTOP` (fonte unica).
    """
    if camera_names is None:
        return None
    return camera_names.aviso_laptop(indice, papel)


def mapa_de_cameras_resumo():
    """Linhas 'idx N: nome [<- webcam do LAPTOP]' para imprimir no inicio."""
    if camera_names is None:
        return ["(sem nomes de camera: comtypes/COM indisponivel)"]
    return camera_names.resumo()
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
# Fator de redimensionamento das JANELAS de video (0.5 = metade do tamanho).
# 0.5 para as QUATRO janelas caberem: Kinect RGB (1920x1080) + Kinect depth +
# cada webcam auxiliar -- no tamanho real elas tomam a tela inteira e o
# experimentador nao consegue ver as duas auxiliares ao mesmo tempo.
# E' reducao de EXIBICAO apenas: deteccao, profundidade e calibracao continuam
# na resolucao cheia (a excecao e' o desenho do esqueleto, que sempre foi no
# frame do Kinect e cai junto com ele). stereo_calibration.py tem a sua propria
# DISPLAY_SCALE com o MESMO valor -- mantenha os dois iguais.
WINDOW_SCALE = 0.5
#: Numero MAXIMO de colunas da grade de janelas de video (ver `grade_de_janelas`).
#: 3 colunas cobrem as 4-6 janelas do dia a dia (RGB e depth do Kinect + uma por
#: webcam auxiliar) em 1080p, mas 1 ou 2 bastam como sao escolhidas pelo menor
#: numero que couber.
WINDOW_GRID_MAX_COLUMNS = 3
#: Recuo (px) entre as janelas e a borda da tela na grade de janelas.
WINDOW_GRID_MARGIN = 4
#: Deslocamento (px) de cada janela na CASCATA (usada quando a grade nao cabe).
#: Garante que nenhuma janela fique exatamente atras de outra: a barra de titulo
#: de todas continua clicavel.
WINDOW_CASCADE_STEP = 34
#: Menor escala de exibicao aceita pelo ajuste automatico (ver
#: `escala_para_grade`). Abaixo disso a mao no video fica pequena demais para
#: conferir o overlay -- melhor avisar que use `--escala-janelas` ou arraste.
WINDOW_SCALE_MIN = 0.3


def tamanho_da_tela():
    """(largura, altura) em pixels do monitor principal; (0, 0) se nao souber.

    Existe para POSICIONAR as janelas de video: o Windows abre todas na mesma
    posicao, empilhadas, e so' a ultima fica visivel -- foi o que fez "o video
    de uma camera nao aparecer" na bancada de 25/09/2026 (ele existia, estava
    coberto). Degrada em silencio (0, 0) fora do Windows.
    """
    if os.name != "nt":
        return 0, 0
    try:
        import ctypes

        user32 = ctypes.windll.user32
        # SM_CXSCREEN/SM_CYSCREEN = resolucao do monitor principal.
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:                          # noqa: BLE001 - sem user32
        return 0, 0


def _grade_em_colunas(caixas, colunas, largura_tela, margem):
    """(posicoes, altura_total) da grade de `colunas` colunas (greedy).

    Cada janela entra na COLUNA MENOS CHEIA, na ordem recebida (a da esquerda em
    caso de empate), empilhada abaixo do que ja' esta' naquela coluna. O x vem da
    tela dividida em partes iguais e o y do quanto ja' foi usado na coluna.
    """
    alturas = [0] * colunas
    posicoes = []
    for _largura, altura in caixas:
        coluna = alturas.index(min(alturas))
        posicoes.append((coluna * largura_tela // colunas,
                         margem + alturas[coluna]))
        alturas[coluna] += int(altura) + margem
    return posicoes, max(alturas) + margem


def grade_de_janelas(caixas, largura_tela, altura_tela,
                     colunas_max=WINDOW_GRID_MAX_COLUMNS,
                     margem=WINDOW_GRID_MARGIN):
    """Posicoes (x, y) das janelas de video, SEM uma tapar a outra.

    Por que existe: com o Kinect e as DUAS webcams externas ligadas sao QUATRO
    fluxos de video (RGB e depth do Kinect + uma janela por webcam); o Windows
    abre todos na mesma posicao e o experimentador conclui que "o video da
    terceira camera nao aparece" -- na verdade ele esta' atras das outras.

    Heuristica (determinista): cada janela entra na COLUNA MENOS CHEIA, na ordem
    recebida; o numero de colunas e' o MENOR entre 1 e `colunas_max` em que a
    coluna mais alta caiba na tela. Devolve None quando nao ha' tela conhecida
    (largura/altura 0) ou quando nem com `colunas_max` a grade cabe -- aí o
    chamador deixa o Windows posicionar (melhor janela fora de lugar do que
    janela escondida atras das outras).

    Args:
        caixas: [(largura, altura)] na ordem de criacao das janelas.
        largura_tela, altura_tela: pixels da tela (0/0 = nao posicionar).
        colunas_max: teto de colunas.
        margem: recuo (px) na borda e entre janelas empilhadas.

    Returns:
        [(x, y)] na MESMA ordem de `caixas`, ou None.
    """
    if not caixas or largura_tela <= 0 or altura_tela <= 0:
        return None
    largura_janela = max(int(largura) for largura, _altura in caixas)
    if largura_janela <= 0:
        return None
    limite = int(largura_tela // largura_janela)
    if limite < 1:
        return None
    for colunas in range(1, max(1, min(int(colunas_max), limite)) + 1):
        posicoes, altura_total = _grade_em_colunas(caixas, colunas,
                                                   largura_tela, margem)
        if altura_total <= altura_tela:
            return posicoes
    return None


def escala_para_grade(caixas_base, largura_tela, altura_tela,
                      escala_max=WINDOW_SCALE, escala_min=WINDOW_SCALE_MIN,
                      passo=0.05):
    """Maior escala de EXIBICAO (<= `escala_max`) em que TODAS as janelas cabem.

    Motivo: o WINDOW_SCALE de 0.5 foi escolhido para as QUATRO janelas caberem,
    mas em telas com DPI > 100% a area util que o Windows entrega e' menor
    (medido nesta maquina: 1536x960 "virtuais" num monitor 1920x1080 a 125%) e a
    grade nao cabe -- as janelas voltam a se sobrepor. Aqui a escala e' reduzida
    (ate' `escala_min`) apenas o necessario para a grade de TODAS caber, em
    passos de `passo`; assim as tres cameras + o depth ficam visiveis juntos sem
    o usuario editar codigo.

    Args:
        caixas_base: [(largura, altura)] dos quadros em tamanho CHEIO (o
            redimensionamento e' so' de exibicao).
        largura_tela, altura_tela: pixels da tela (0/0 = sem tela conhecida).
        escala_max/min: limites da escala procurada.
        passo: granularidade da busca (0.05 = 5%).

    Returns:
        float -- `escala_max` se ja' cabe (ou se nao ha' tela conhecida),
        `escala_min` se nem no minimo couber.
    """
    if not caixas_base or largura_tela <= 0 or altura_tela <= 0:
        return float(escala_max)
    valores = []
    escala = float(escala_max)
    while escala >= float(escala_min) - 1e-9:
        valores.append(round(escala, 3))
        escala -= float(passo)
    for candidata in valores:
        caixas = [(max(1, int(round(largura * candidata))),
                   max(1, int(round(altura * candidata))))
                  for largura, altura in caixas_base]
        if grade_de_janelas(caixas, largura_tela, altura_tela) is not None:
            return candidata
    return float(escala_min)


def posicoes_das_janelas(caixas, largura_tela, altura_tela,
                         colunas_max=WINDOW_GRID_MAX_COLUMNS,
                         margem=WINDOW_GRID_MARGIN,
                         passo_cascata=WINDOW_CASCADE_STEP):
    """Posicoes das janelas + COMO elas foram posicionadas.

    Devolve `(posicoes, modo)`, com `modo` em:
      * "grade"     -- `grade_de_janelas` coube: janelas lado a lado, sem
                       sobreposicao (o caso bom);
      * "cascata"   -- nao coube: cada janela deslocada em degrau
                       (`passo_cascata`), para NENHUMA ficar exatamente atras de
                       outra (a barra de titulo de todas continua clicavel);
      * "sem-tela"  -- tela desconhecida: `posicoes` e' None e quem chamou deixa
                       o Windows posicionar.

    O `modo` existe para o chamador poder AVISAR o usuario uma unica vez (ex.:
    "a grade nao coube em 1536x960: arraste as janelas ou use 0.4 de escala").
    """
    if not caixas:
        return None, "sem-tela"
    if largura_tela <= 0 or altura_tela <= 0:
        return None, "sem-tela"
    posicoes = grade_de_janelas(caixas, largura_tela, altura_tela, colunas_max,
                                margem)
    if posicoes is not None:
        return posicoes, "grade"
    # Cascata: degrau COMPRIMIDO para caber na tela mesmo quando as janelas sao
    # maiores que ela (o caso do DPI alto). O passo nunca fica abaixo de 1 px e
    # o total e' < largura/altura da tela, entao TODAS as posicoes sao distintas
    # -- nenhuma janela fica exatamente atras de outra e a barra de titulo de
    # cada uma continua alcancavel para arrastar.
    total = max(1, len(caixas))
    passo_x = max(1, min(int(passo_cascata),
                         (int(largura_tela) - 1) // total))
    passo_y = max(1, min(int(passo_cascata),
                         (int(altura_tela) - 1) // total))
    return [(indice * passo_x, indice * passo_y)
            for indice in range(len(caixas))], "cascata"
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


def palm_frame(wrist, index_mcp, pinky_mcp, middle_mcp, lado=None):
    """Frame ortonormal do plano da palma a partir de 4 pontos 3D (m).

    Devolve (normal, direcao_dedos), ambas unitarias, ou (None, None):
      - normal: perpendicular ao plano punho/indicador/minimo. COM `lado`
        ('right'/'left') o sinal e' ANATOMICO (aponta para fora da PALMA em
        qualquer pose -- ver `palm_normal_from_points`); SEM `lado` mantem o
        comportamento antigo (sinal forcado para a CAMERA, nz < 0);
      - direcao_dedos: punho -> MCP do dedo medio, ortogonalizada em relacao
        a normal (fica exatamente no plano da palma).
    """
    normal = None
    if index_mcp is not None and pinky_mcp is not None and wrist is not None:
        normal = KinectHandTracker.palm_normal_from_points(
            wrist, index_mcp, pinky_mcp, lado=lado)
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


#: Landmarks do DORSO da mao usados no PLANO MEDIO: punho (0) + os MCPs
#: indicador (5), medio (9), anelar (13) e minimo (17). O ajuste de plano por
#: MINIMOS QUADRADOS nesses pontos e' mais estavel que o produto vetorial de 3
#: pontos: o ruido de profundidade de um unico no' rotaciona a normal inteira,
#: enquanto no plano medio ele entra dividido entre 5 pontos (item 102).
PONTOS_DORSO_PALMA = {"wrist": 0, "index": 5, "middle": 9, "ring": 13,
                      "pinky": 17}


def palm_normal_from_plane(pontos, lado=None, normal_anterior=None,
                           eps=1e-9, rms_max_m=0.030, dist_max_m=0.12):
    """(normal, rms_m) do PLANO MEDIO do dorso da mao, ou (None, None).

    Por que existe (item 102): a normal da palma era o produto vetorial de 3
    pontos (punho, MCP do indicador, MCP do minimo). Com o ruido de
    profundidade do Kinect, UM ponto errado gira a normal inteira -- e o vetor
    da palma sai "apontando para lugares ruins" mesmo com T/f bons. Aqui
    ajusta-se um PLANO MEDIO por minimos quadrados (SVD) nos nos do DORSO
    (punho + 4 MCPs) e a normal e' o vetor singular minimo, bem mais estavel.

    `pontos`: dict {"wrist","index","middle","ring","pinky": (3,)} em metros no
    camera space (valores None/invalidos sao ignorados). Precisa de >=4 pontos
    validos e dos tres do sinal anatomico (wrist, index, pinky).

    SINAL ANATOMICO ROBUSTO (item 105): a referencia antiga
    `(index-punho) x (pinky-punho)` DEGENERA quando a mao fica plana (medido no
    CSV canonico: 37/296 quadros da direita com magnitude ~0 -> sinal virava
    moeda e a normal aparecia INVERTIDA, erro de 170-180 graus, era o que
    reprovava o certificado V). A referencia nova e' o produto vetorial de dois
    vetores GRANDES e quase ortogonais do dorso:
        transversal = index - pinky     (~12 cm, entre os MCPs)
        longitudinal = middle - punho   (~9 cm, do punho ao MCP medio)
    Se eles nao forem confiaveis (quase paralelos/curtos), cai para a referencia
    antiga; e se nem ela servir, usa `normal_anterior` para manter a
    CONTINUIDADE (o sinal nao pode pular entre quadros consecutivos com a mao
    parada). Sem nenhuma das tres -> (None, None): melhor nao medir.

    GATES DE QUALIDADE:
      - `rms_max_m`: RMS (m) dos pontos ao plano (borda/oclusao/ruido estoura);
      - `dist_max_m`: distancia maxima de um ponto ao centro (a mao tem ~7 cm
        de raio; ponto a 30 cm e' profundidade de FUNDO mapeada errado, nao mao).

    Orientacao final: 'right' = referencia como calculada; 'left' = espelhada
    (mesma convencao anatomica dos itens 79/102); lado None = forcado nz < 0.
    """
    validos = []
    for nome in ("wrist", "index", "middle", "ring", "pinky"):
        valor = pontos.get(nome) if isinstance(pontos, dict) else None
        if valor is None:
            continue
        valor = np.asarray(valor, np.float64).reshape(3)
        if np.isfinite(valor).all():
            validos.append((nome, valor))
    if len(validos) < 4:
        return None, None
    nomes = {nome for nome, _valor in validos}
    if not {"wrist", "index", "pinky"} <= nomes:
        return None, None
    matriz = np.array([valor for _nome, valor in validos], dtype=np.float64)
    centro = matriz.mean(axis=0)
    centrado = matriz - centro
    if float(np.max(np.linalg.norm(centrado, axis=1))) > float(dist_max_m):
        return None, None                         # ponto de fundo, nao da mao
    try:
        _u, _s, vt = np.linalg.svd(centrado, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    normal = np.asarray(vt[-1], np.float64).reshape(3)
    norma = float(np.linalg.norm(normal))
    if norma < eps:
        return None, None
    normal = normal / norma
    distancias = centrado @ normal
    rms = float(np.sqrt(np.mean(distancias ** 2)))
    if rms_max_m is not None and rms > float(rms_max_m):
        return None, None

    por_nome = {nome: valor for nome, valor in validos}
    referencia = None
    # 1) referencia NOVA: transversal x longitudinal (robusta com a mao plana).
    #    `middle` e' OPCIONAL: se o lookup de depth do MCP medio falhar naquele
    #    quadro (borda/oclusao), NAO pode estourar KeyError -- cai na
    #    continuidade/antiga (item 108).
    transversal = por_nome["index"] - por_nome["pinky"]
    norma_t = float(np.linalg.norm(transversal))
    if "middle" in por_nome:
        longitudinal = por_nome["middle"] - por_nome["wrist"]
        norma_l = float(np.linalg.norm(longitudinal))
        if norma_t > 0.030 and norma_l > 0.030:
            cruzado = np.cross(transversal, longitudinal)
            norma_c = float(np.linalg.norm(cruzado))
            seno = norma_c / (norma_t * norma_l)
            if seno > 0.45:                       # vetores nao quase paralelos
                referencia = cruzado / norma_c
    # 2) continuidade temporal: sinal do quadro anterior
    if referencia is None and normal_anterior is not None:
        anterior = np.asarray(normal_anterior, np.float64).reshape(3)
        norma_a = float(np.linalg.norm(anterior))
        if np.isfinite(anterior).all() and norma_a > eps:
            anterior = anterior / norma_a
            if float(np.dot(normal, anterior)) < 0.0:
                normal = -normal
    # 3) fallback: referencia ANTIGA (3 pontos), so' se ainda nao decidiu
    if referencia is None:
        antiga = np.cross(por_nome["index"] - por_nome["wrist"],
                          por_nome["pinky"] - por_nome["wrist"])
        norma_antiga = float(np.linalg.norm(antiga))
        if normal_anterior is not None:
            pass                                  # continuidade ja' decidiu
        elif norma_antiga > 0.003:
            antiga = antiga / norma_antiga
            if float(np.dot(normal, antiga)) < 0.0:
                normal = -normal
        else:
            return None, None                     # sem sinal confiavel
    if referencia is not None:
        if float(np.dot(normal, referencia)) < 0.0:
            normal = -normal
    if lado == "left":
        normal = -normal
    elif lado is None and normal[2] > 0.0:
        normal = -normal                          # legado sem `lado` (nz < 0)
    return normal, rms

def media_estavel_normais(historico, limite_spread_deg=8.0):
    """(normal_media, spread) das normais recentes, ou (None, None).

    O vetor do plano do dorso CAMBALEIA quando a profundidade do Kinect
    tem ruido/oclusao (item 105): a normal instantanea oscila alguns graus
    com a mao parada. Aqui a historico (lista de versores) e' alinhada em
    sinal (a referencia anatomica pode inverter pontualmente) e a media e'
    devolvida junto com o SPREAD maximo (graus). O chamador decide: spread
    baixo (<= limite) = pose parada e confiavel; alto = ainda em movimento,
    nao usar como verdade de calibracao.
    """
    vetores = []
    for valor in historico:
        if valor is None:
            continue
        vetor = np.asarray(valor, np.float64).reshape(3)
        norma = float(np.linalg.norm(vetor))
        if not np.isfinite(vetor).all() or norma < 1e-9:
            continue
        vetores.append(vetor / norma)
    if len(vetores) < 2:
        return None, None
    referencia = vetores[-1]
    alinhados = [v if float(np.dot(v, referencia)) >= 0.0 else -v
                 for v in vetores]
    media = np.mean(alinhados, axis=0)
    norma = float(np.linalg.norm(media))
    if norma < 1e-9:
        return None, None
    media = media / norma
    spread = 0.0
    for vetor in alinhados:
        cosseno = float(np.dot(vetor, media))
        spread = max(spread, float(np.degrees(np.arccos(
            np.clip(cosseno, -1.0, 1.0)))))
    return media, spread


def normal_estavel(historico, limite_spread_deg=4.0):
    """(normal_media, spread) so' se o historico estiver PARADO.

    Diferenca para `media_estavel_normais`: aqui o spread e' um GATE -- se
    as normais recentes variam mais que `limite_spread_deg` a mao ainda esta
    se movendo e a media NAO serve como verdade de calibracao (o atraso da
    media em relacao ao IMU viraria erro de alinhamento). Devolve
    (None, spread) nesse caso, para o chamador saber que houve movimento.
    """
    media, spread = media_estavel_normais(historico)
    if media is None or spread is None:
        return None, spread
    if float(spread) > float(limite_spread_deg):
        return None, spread
    return media, spread


def lado_mais_proximo(posicao, pontos_corpo, limite_m=0.30):
    """'right'/'left'/None: lado cujo ponto do esqueleto esta mais perto.

    Por que existe (item 113): o rotulo de lado do MediaPipe PISCA (o mesmo
    acontece nas auxiliares, que tem arbitro). Como o sinal anatomico da
    normal e ESPELHADO por lado, quando o lado troca a seta verde inverte 180
    graus -- exatamente o "as vezes aponta para dentro, as vezes para fora da
    palma" relatado. A junta Wrist/Hand do esqueleto 3D do Kinect e' estavel,
    entao ela manda: se a posicao da palma esta a <= `limite_m` de uma mao do
    esqueleto, esse e' o lado.
    """
    if posicao is None or not pontos_corpo:
        return None
    posicao = np.asarray(posicao, np.float64).reshape(3)
    if not np.isfinite(posicao).all():
        return None
    melhor_lado = None
    melhor_distancia = float("inf")
    for lado, ponto in pontos_corpo.items():
        if ponto is None:
            continue
        ponto = np.asarray(ponto, np.float64).reshape(3)
        if not np.isfinite(ponto).all():
            continue
        distancia = float(np.linalg.norm(posicao - ponto))
        if distancia < melhor_distancia:
            melhor_distancia = distancia
            melhor_lado = lado
    if melhor_lado is None or melhor_distancia > float(limite_m):
        return None
    return melhor_lado


def resgatar_profundidade(depth_mm, z_priors_m, tolerancia_m=0.30,
                         referencia_m=None):
    """Profundidade (mm) resgatada pelo prior do ESQUELETO, ou None.

    Quando o depth do landmark cai no fundo (buraco/oclusao: |depth - z_mao|
    > tolerancia), usa o z da mao do esqueleto como profundidade plausivel --
    melhor um ponto na profundidade da mao do que um ponto no fundo, que gira
    a normal inteira. `referencia_m` (z da mao do lado, ou EMA da ultima
    profundidade) escolhe qual prior usar quando o depth e' invalido.
    Devolve None se o depth ja' e' plausivel ou se nao ha prior.
    """
    if not z_priors_m:
        return None
    priors = [float(z) for z in z_priors_m
              if z is not None and np.isfinite(float(z)) and float(z) > 0.2]
    if not priors:
        return None
    try:
        depth = float(depth_mm) / 1000.0 if depth_mm is not None else None
    except (TypeError, ValueError):
        depth = None
    if depth is not None and np.isfinite(depth):
        if min(abs(depth - z) for z in priors) <= float(tolerancia_m):
            return None                      # depth ja' e plausivel
        alvo = min(priors, key=lambda z: abs(depth - z))
        return alvo * 1000.0
    referencia = None
    try:
        referencia = float(referencia_m) if referencia_m is not None else None
    except (TypeError, ValueError):
        referencia = None
    if referencia is not None and np.isfinite(referencia):
        alvo = min(priors, key=lambda z: abs(referencia - z))
    else:
        alvo = float(np.median(priors))
    return alvo * 1000.0


def normal_robusta(historico, normal_anterior=None, tolerancia_deg=10.0,
                   minimo_fracao=0.70, minimo_amostras=4,
                   referencia_sinal=None):
    """(normal_mediana, fracao_inliers) tolerante a OUTLIERS de 1-2 quadros.

    Por que existe (item 112, medido no log do usuario): no CSV de 22/10 o
    MediaPipe detectava a mao em 99% dos frames e a normal instantanea tinha
    salto mediano de 2,9 graus, mas 5-8% dos quadros saltavam >45 graus
    (depth hole/oclusao/sinal). A MEDIA de janela (media_estavel_normais) era
    contaminada por esses poucos quadros: o spread estourava e o gate zerava
    o J/V mesmo com a mao parada (havia RUNS longos de spread <=4 no log).
    Solucao: MEDIANA vetorial com alinhamento de sinal e fracao de INLIERS
    (angulo <= `tolerancia_deg` da mediana). Se `minimo_fracao` dos quadros
    concorda, a mediana e valida -- um ou dois outliers nao derrubam.
    O sinal e alinhado pela `normal_anterior` (continuidade), se dada; senao
    pelo primeiro vetor da janela.

    ATENCAO (item 131): alinhar o sinal pela continuidade IMPEDE uma virada
    real de ~180 graus da mao (ex.: palma -> dorso entre poses): o novo conjunto
    e' invertido para combinar com o anterior e a mediana fica com o sinal
    ANTIGO. Na calibracao geometrica (tecla X), que alterna alvos ±Z/±X/±Y, isso
    travava o QA em ~170 graus e nenhum take era gravado. Para esses casos,
    `referencia_sinal` (ex.: a normal ANATOMICA instantanea, ainda no mesmo
    referencial) define o sinal da mediana; `normal_anterior` continua valendo
    so' quando `referencia_sinal` nao e' dada.
    """
    vetores = []
    for valor in historico:
        if valor is None:
            continue
        vetor = np.asarray(valor, np.float64).reshape(3)
        norma = float(np.linalg.norm(vetor))
        if not np.isfinite(vetor).all() or norma < 1e-9:
            continue
        vetores.append(vetor / norma)
    if len(vetores) < int(minimo_amostras):
        return None, 0.0
    referencia = None
    # 1) referencia EXPLICITA de sinal (ex.: normal anatomica instantanea do
    #    modo X) -- permite a virada real de 180 graus entre poses.
    if referencia_sinal is not None:
        sinal = np.asarray(referencia_sinal, np.float64).reshape(3)
        norma_sinal = float(np.linalg.norm(sinal))
        if np.isfinite(sinal).all() and norma_sinal > 1e-9:
            referencia = sinal / norma_sinal
    # 2) continuidade com a normal anterior (comportamento antigo de J/V)
    if referencia is None and normal_anterior is not None:
        anterior = np.asarray(normal_anterior, np.float64).reshape(3)
        norma = float(np.linalg.norm(anterior))
        if np.isfinite(anterior).all() and norma > 1e-9:
            referencia = anterior / norma
    if referencia is None:
        referencia = vetores[0]
    alinhados = np.array([v if float(np.dot(v, referencia)) >= 0.0 else -v
                          for v in vetores])
    mediana = np.median(alinhados, axis=0)
    norma = float(np.linalg.norm(mediana))
    if norma < 1e-9:
        return None, 0.0
    mediana = mediana / norma
    angulos = [float(np.degrees(np.arccos(np.clip(
        float(np.dot(v, mediana)), -1.0, 1.0)))) for v in alinhados]
    fracao = float(np.mean([angulo <= float(tolerancia_deg)
                            for angulo in angulos]))
    if fracao < float(minimo_fracao):
        return None, fracao
    return mediana, fracao


#: POSES FISICAS DE CALIBRACAO (item 114). A verdade NAO vem do Kinect: vem
#: de tres orientacoes que o usuario consegue fazer com precisao FISICA, com a
#: normal da palma CONHECIDA no referencial da camera (Kinect nivelado, eixo Y
#: para cima, Z para dentro da cena):
#:   - palma de frente para o Kinect (na linha do sensor, dedos para cima)
#:       -> normal aponta para a camera = (0,0,-1)
#:   - palma para baixo apoiada na mesa (dedos para frente)
#:       -> normal aponta para o chao = (0,-1,0)
#:   - palma para cima com o DORSO apoiado na mesa
#:       -> normal aponta para o teto = (0,+1,0)
#: Com as rotacoes do IMU nessas 3 poses, `solve_palm_alignment` resolve T e f
#: SEM depender de landmark/depth/sinal do Kinect (que oscilavam).
#: POSES FISICAS (item 117): as 3 poses sao "palma MIRANDO A LENTE do
#: Kinect" em 3 POSICOES diferentes do quadro. A verdade de cada pose nao
#: depende da planura da mao: e a direcao GEOMETRICA do centro da palma ate
#: a lente (origem do camera space), medida pelo proprio Kinect (1 cm a 1 m
#: = ~0.6 grau). E a habilidade em que o usuario e mais preciso (a pose
#: "palma p/ Kinect" teve 0.5-5.5 graus em todas as rodadas). As poses de
#: MESA foram aposentadas como calibracao: a luva impede a mao de ficar
#: plana (~10-20 graus de execucao medidos). Os vetores sao NOMINAIS (so
#: exibicao/fallback); o J mede a posicao real e recalcula a verdade.
POSES_FISICAS_CALIBRACAO = (
    ("MAO PERTO DA LENTE (~50 cm), NO CENTRO DO QUADRO, PALMA MIRANDO A "
     "LENTE", (0.0, 0.0, -1.0)),
    ("MAO PERTO DA LENTE, BEM A ESQUERDA DO QUADRO (ainda no quadro), "
     "PALMA MIRANDO A LENTE", (0.45, 0.0, -0.89)),
    ("MAO PERTO DA LENTE, BEM A DIREITA DO QUADRO (ainda no quadro), "
     "PALMA MIRANDO A LENTE", (-0.45, 0.0, -0.89)),
    ("MAO PERTO DA LENTE, BEM PARA BAIXO NO QUADRO (mesma coluna), "
     "PALMA MIRANDO A LENTE", (0.0, 0.45, -0.89)),
)
#: Angulo minimo (graus) entre as verdades das poses para a calibracao ser
#: considerada BEM CONDICIONADA. Se as 3-4 poses apontam quase para a mesma
#: direcao, o T/f acerta aquelas poses mas erra feio em outras orientacoes --
#: medido no alinhamento salvo do usuario (poses a 1-13 graus): erro mediano de
#: 12 (R) e 39 (L) graus em direcoes arbitrarias. Com ~30-60 graus o erro cai
#: para ~2 graus. Poses muito parecidas NAO fecham o certificado.
POSES_FISICAS_SPREAD_MIN_DEG = 15.0
#: Espalhamento MAXIMO minimo (graus): as poses precisam cobrir um leque
#: real de direcoes. Simulacao com min 15/max 30-60: erro mediano de
#: rastreamento ~1-7 graus; com tudo quase igual (1-13 graus) era 12-39.
POSES_FISICAS_SPREAD_MAX_MIN_DEG = 30.0
#: Separacao minima (graus) exigida JA' NO TAKE: se a pose nova ficar
#: parecida com uma anterior, o take e' RECUSADO na hora e o usuario move
#: mais a mao, em vez de descobrir so' no fim que a calibracao era ruim.
POSES_FISICAS_TAKE_SEPARACAO_MIN_DEG = 15.0
#: MODO TAKES (item 120): sem tempo fixo por pose. A mao fica parada e o
#: take e' gravado sozinho; o usuario leva o tempo que quiser para ir a
#: cada posicao (como na calibracao com tabuleiro).
POSES_FISICAS_TAKE_SEC = 2.0
POSES_FISICAS_TAKE_MAX_DESVIO_DEG = 4.0
POSES_FISICAS_TAKE_MOVIMENTO_MIN_DEG = 8.0
POSES_FISICAS_TAKE_POS_MAX_M = 0.05
POSES_FISICAS_TAKE_MOVIMENTO_POS_M = 0.10
#: Duracao de cada pose fisica (s) e tempo para chegar na pose.
POSES_FISICAS_POSE_SEC = 3.0
POSES_FISICAS_SETTLE_SEC = 0.8
#: Limite (graus) do certificado das poses fisicas.
POSES_FISICAS_LIMITE_DEG = 12.0
#: Limites do certificado por pose: a pose 1 (palma p/ a camera) e' a mais
#: precisa (medida 1-3 graus na bancada); as poses de mesa carregam o erro de
#: execucao da mao/luva (a palma nao fica exatamente paralela a mesa).
POSES_FISICAS_LIMITE_POSE1_DEG = 10.0
POSES_FISICAS_LIMITE_MESA_DEG = 16.0


def ajustar_plano_mesa(pontos, limiar_m=0.02, iteracoes=120, semente=1):
    """Plano dominante (a MESA) da nuvem 3D do depth, normal apontando p/ CIMA.

    Mede o plano da mesa com RANSAC + minimos quadrados nos inliers e devolve a
    normal no referencial da camera (item 115). O codigo antigo do eixo Y do
    camera space e' ambiguo entre trechos; por isso quem usa a normal testa os
    dois sentidos de "para cima" e escolhe pelo residuo.
    """
    nuvem = np.asarray(pontos, np.float64).reshape(-1, 3)
    nuvem = nuvem[np.isfinite(nuvem).all(axis=1)]
    if len(nuvem) < 120:
        return None
    gerador = np.random.default_rng(int(semente))
    melhor = None
    for _ in range(max(1, int(iteracoes))):
        indices = gerador.choice(len(nuvem), 3, replace=False)
        p0, p1, p2 = nuvem[indices]
        normal = np.cross(p1 - p0, p2 - p0)
        norma = float(np.linalg.norm(normal))
        if norma < 1e-9:
            continue
        normal = normal / norma
        distancias = np.abs((nuvem - p0) @ normal)
        inliers = distancias < float(limiar_m)
        contagem = int(inliers.sum())
        if melhor is None or contagem > melhor[0]:
            melhor = (contagem, inliers)
    if melhor is None:
        return None
    contagem, inliers = melhor
    if contagem < max(200, int(0.12 * len(nuvem))):
        return None
    ajustados = nuvem[inliers]
    centroide = ajustados.mean(axis=0)
    _u, _s, vt = np.linalg.svd(ajustados - centroide)
    normal = vt[-1]
    normal = normal / max(1e-12, float(np.linalg.norm(normal)))
    if normal[1] < 0.0:
        normal = -normal
    tilt_deg = float(np.degrees(np.arccos(np.clip(float(normal[1]), -1.0, 1.0))))
    erro = np.abs((ajustados - centroide) @ normal)
    return {
        "normal": normal,
        "inliers": contagem,
        "fracao": float(contagem) / float(len(nuvem)),
        "tilt_deg": tilt_deg,
        "erro_m": float(np.mean(erro)),
    }


def poses_fisicas_com_mesa(normal_mesa, sinal=1.0):
    """Poses fisicas (item 114) com as normais 2/3 tiradas do PLANO DA MESA.

    Pose 1 = "palma de frente para o Kinect" = eixo da camera (0,0,-1).
    Poses 2/3 usam +-normal da mesa (medida pelo depth). `sinal` resolve o
    sentido ambiguo do eixo Y do camera space (testado pelo calibrador).
    """
    normal = np.asarray(normal_mesa, np.float64).reshape(3)
    normal = normal / max(1e-12, float(np.linalg.norm(normal)))
    if normal[1] < 0.0:
        normal = -normal
    orientada = float(sinal) * normal
    return (
        (POSES_FISICAS_CALIBRACAO[0][0], (0.0, 0.0, -1.0)),
        (POSES_FISICAS_CALIBRACAO[1][0], tuple(float(v) for v in -orientada)),
        (POSES_FISICAS_CALIBRACAO[2][0], tuple(float(v) for v in orientada)),
    )


def _normalizar_vetor(vetor):
    vetor = np.asarray(vetor, np.float64).reshape(3)
    norma = float(np.linalg.norm(vetor))
    if not np.isfinite(vetor).all() or norma < 1e-9:
        return None
    return vetor / norma


def _mediana_vetores(vetores, referencia=None):
    """Mediana vetorial (com alinhamento de sinal) ou None."""
    lista = [_normalizar_vetor(v) for v in vetores]
    lista = [v for v in lista if v is not None]
    if not lista:
        return None
    ref = _normalizar_vetor(referencia) if referencia is not None else lista[0]
    if ref is None:
        ref = lista[0]
    alinhados = np.array([v if float(np.dot(v, ref)) >= 0.0 else -v
                          for v in lista])
    mediana = _normalizar_vetor(np.median(alinhados, axis=0))
    return mediana


def _angulo_entre(a, b):
    a = _normalizar_vetor(a)
    b = _normalizar_vetor(b)
    if a is None or b is None:
        return None
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)),
                                             -1.0, 1.0))))


def _angulo_rotacoes(r1, r2):
    """Angulo (graus) entre duas rotacoes (3,3)."""
    a = np.asarray(r1, np.float64).reshape(3, 3)
    b = np.asarray(r2, np.float64).reshape(3, 3)
    traco = float(np.trace(a @ b.T))
    cosseno = float(np.clip((traco - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosseno)))


def verdades_por_angulos(v1, normal_mesa, theta12_deg, theta13_deg,
                         theta23_deg):
    """Constroi as 3 direcoes de verdade a partir dos ANGULOS relativos.

    Por que (item 116): nas poses de mesa a mao NAO fica exatamente paralela a
    mesa (luva/encaixe), o que gera erro de execucao de 10-20 graus. O Kinect
    mede a orientacao absoluta com viés (offset de calibracao/geometria), mas o
    ANGULO RELATIVO entre duas poses sadias do mesmo sensor NAO tem esse viés
    (o viés e' comum e cancela). Entao: pose 1 = direcao absoluta geometrica
    (palma p/ a camera, a mais precisa), e poses 2/3 sao construidas a partir
    dos angulos relativos medidos pelo Kinect -> a verdade passa a refletir a
    mao REAL do usuario, sem herdar o viés absoluto do Kinect.
    """
    v1 = _normalizar_vetor(v1)
    if v1 is None:
        return None
    mesa = _normalizar_vetor(normal_mesa)
    if mesa is None:
        return None
    theta12 = float(np.radians(theta12_deg))
    theta13 = float(np.radians(theta13_deg))
    theta23 = float(np.radians(theta23_deg))
    #: eixo perpendicular a v1 na direcao aproximada de -mesa (palma p/ baixo)
    alvo2 = -mesa
    perpendicular = alvo2 - float(np.dot(alvo2, v1)) * v1
    eixo2 = _normalizar_vetor(perpendicular)
    if eixo2 is None:
        alternativa = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(alternativa, v1))) > 0.9:
            alternativa = np.array([0.0, 1.0, 0.0])
        eixo2 = _normalizar_vetor(
            alternativa - float(np.dot(alternativa, v1)) * v1)
        if eixo2 is None:
            return None
    v2 = np.cos(theta12) * v1 + np.sin(theta12) * eixo2
    v2 = _normalizar_vetor(v2)
    if v2 is None:
        return None
    seno12 = np.sin(theta12)
    if abs(seno12) < 1e-6:
        return None
    #: decomposicao de v3 na base NAO ortogonal (v1, v2, n): resolve o
    #: sistema da matriz de Gram [[1,cos12],[cos12,1]] (o erro que a versao
    #: ingenua de dot-products cometia era ~20 graus).
    cos12 = np.cos(theta12)
    cos13 = np.cos(theta13)
    cos23 = np.cos(theta23)
    seno2 = seno12 * seno12
    alfa = (cos13 - cos12 * cos23) / seno2
    beta = (cos23 - cos12 * cos13) / seno2
    gamma2 = 1.0 - alfa * alfa - beta * beta - 2.0 * alfa * beta * cos12
    if gamma2 < 0.0:
        gamma2 = 0.0
    normal_plano = np.cross(v1, v2)
    normal_plano = _normalizar_vetor(normal_plano)
    if normal_plano is None:
        return None
    candidatos = []
    for sinal in (1.0, -1.0):
        v3 = (alfa * v1 + beta * v2
              + sinal * float(np.sqrt(gamma2)) * normal_plano)
        v3 = _normalizar_vetor(v3)
        if v3 is not None:
            candidatos.append(v3)
    if not candidatos:
        return None
    #: a pose 3 e' "palma p/ cima": escolhe o candidato mais alinhado com +mesa
    v3 = max(candidatos, key=lambda v: float(np.dot(v, mesa)))
    return (v1, v2, v3)


class ColetorPosesFisicas:
    """Coleta rotacoes do IMU nas poses fisicas e resolve/verifica T/f.

    MODOS:
      - "calibrar": resolve T/f; usa a verdade GEOMETRICA (pose 1) + os angulos
        RELATIVOS medidos pelo Kinect (imunes ao viés absoluto) para refletir a
        mao real nas poses de mesa; se os angulos nao estiverem disponiveis, cai
        para as normais do plano da mesa (item 115);
      - "verificar": avalia o alinhamento carregado contra essas mesmas verdades.
    O Kinect nao precisa ver a mao para COLETAR rotacoes (so' para os angulos).
    """

    def __init__(self, poses=POSES_FISICAS_CALIBRACAO,
                 seg_pose=POSES_FISICAS_POSE_SEC,
                 settle=POSES_FISICAS_SETTLE_SEC,
                 limite=POSES_FISICAS_LIMITE_DEG):
        self.poses = tuple(poses)
        self.seg_pose = float(seg_pose)
        self.settle = float(settle)
        self.limite_graus = float(limite)
        self.modo = "calibrar"
        self._ativa = False
        self._t0 = 0.0
        self._amostras = {"right": [], "left": []}
        self._normais = {"right": {}, "left": {}}   # lado -> {idx: [n]}
        #: Posicao 3D do centro da palma por pose (verdade = direcao p/ a
        #: lente, item 117): lado -> {idx: [posicao (3,)]}.
        self._posicoes = {"right": {}, "left": {}}
        #: MODO TAKES (item 120).
        self._modo_takes = False
        self._take_indice = 0
        self._take_buffer = {"right": [], "left": []}
        self._take_movimento_ok = {"right": True, "left": True}
        self._take_ref_rot = {"right": None, "left": None}
        self._take_ref_pos = {"right": None, "left": None}

    def iniciar(self, agora, modo="calibrar"):
        self.modo = str(modo)
        self._ativa = True
        self._modo_takes = False
        self._t0 = float(agora)
        self._amostras = {"right": [], "left": []}
        self._normais = {"right": {}, "left": {}}
        self._posicoes = {"right": {}, "left": {}}

    def parar(self):
        self._ativa = False

    @property
    def ativa(self):
        return self._ativa

    @property
    def duracao_s(self):
        return len(self.poses) * self.seg_pose

    def fase_atual(self, agora):
        if not self._ativa:
            return None
        decorrido = float(agora) - self._t0
        if decorrido < 0.0 or decorrido >= self.duracao_s:
            return None
        indice = int(decorrido // self.seg_pose)
        indice = min(max(indice, 0), len(self.poses) - 1)
        restante = (indice + 1) * self.seg_pose - decorrido
        return indice, self.poses[indice][0], max(0.0, restante)

    def coletar(self, lado, agora, rotacao):
        if not self._ativa or rotacao is None:
            return False
        fase = self.fase_atual(agora)
        if fase is None:
            return False
        indice = fase[0]
        dentro = (float(agora) - self._t0) - indice * self.seg_pose
        if dentro < self.settle:
            return False
        rot = np.asarray(rotacao, np.float64).reshape(3, 3)
        if not np.isfinite(rot).all():
            return False
        self._amostras[lado].append((indice, rot))
        return True

    def coletar_normal_kinect(self, lado, indice, normal):
        """Guarda a normal do plano da mao medida pelo Kinect (p/ angulos)."""
        if not self._ativa or normal is None:
            return False
        vetor = _normalizar_vetor(normal)
        if vetor is None:
            return False
        try:
            indice = int(indice)
        except (TypeError, ValueError):
            return False
        self._normais.setdefault(lado, {}).setdefault(indice, []).append(vetor)
        return True

    def contar(self, lado, indice_pose=None):
        if indice_pose is None:
            return len(self._amostras.get(lado, ()))
        return sum(1 for pose, _rot in self._amostras.get(lado, ())
                   if pose == indice_pose)

    def contar_normais(self, lado):
        return sum(len(v) for v in self._normais.get(lado, {}).values())

    def coletar_posicao_palma(self, lado, indice, posicao):
        """Guarda a POSICAO 3D do centro da palma (camera space) na pose."""
        if not self._ativa or posicao is None:
            return False
        vetor = np.asarray(posicao, np.float64).reshape(3)
        if not np.isfinite(vetor).all() or float(np.linalg.norm(vetor)) < 0.2:
            return False
        try:
            indice = int(indice)
        except (TypeError, ValueError):
            return False
        self._posicoes.setdefault(lado, {}).setdefault(indice, []).append(
            vetor)
        return True

    def _representante_posicao(self, lado, indice):
        lista = self._posicoes.get(lado, {}).get(indice, [])
        if len(lista) < 3:
            return None
        return np.median(np.asarray(lista, np.float64), axis=0)

    def calibrar_por_posicoes(self):
        """Calibra usando a DIRECAO mao->lente (verdade geometrica)."""
        resultado = {"modo": "calibrar", "lados": {}, "fechado": False,
                     "modo_verdade": "posicao_da_mao",
                     "faltam_posicoes": []}
        for lado in ("right", "left"):
            posicoes = [self._representante_posicao(lado, indice)
                        for indice in range(len(self.poses))]
            if any(posicao is None for posicao in posicoes):
                resultado["faltam_posicoes"].append(lado)
                resultado["lados"][lado] = {
                    "sem_amostras": True,
                    "poses_insuficientes": [indice + 1
                                            for indice, posicao
                                            in enumerate(posicoes)
                                            if posicao is None]}
                continue
            poses_lado = []
            for indice, posicao in enumerate(posicoes):
                direcao = _normalizar_vetor(-np.asarray(posicao, np.float64))
                if direcao is None:
                    resultado["faltam_posicoes"].append(lado)
                    break
                poses_lado.append((self.poses[indice][0],
                                   tuple(float(v) for v in direcao)))
            if len(poses_lado) != len(self.poses):
                resultado["lados"][lado] = {"sem_amostras": True,
                                            "poses_insuficientes": []}
                continue
            direcoes = [np.asarray(vetor, np.float64)
                        for _nome, vetor in poses_lado]
            espalhamento = []
            for i in range(len(direcoes)):
                for j in range(i + 1, len(direcoes)):
                    angulo = _angulo_entre(direcoes[i], direcoes[j])
                    if angulo is not None:
                        espalhamento.append(angulo)
            dados = self._calibrar_lado(lado, tuple(poses_lado))
            dados["poses_usadas"] = tuple(poses_lado)
            dados["posicoes"] = [list(map(float, p)) for p in posicoes]
            dados["espalhamento_min_deg"] = (
                float(min(espalhamento)) if espalhamento else None)
            dados["espalhamento_max_deg"] = (
                float(max(espalhamento)) if espalhamento else None)
            resultado["lados"][lado] = dados
        resultado["fechado"] = self._fechado(resultado["lados"])
        resultado["poses_verdade"] = {
            lado: list(dados.get("poses_usadas") or ())
            for lado, dados in resultado["lados"].items()}
        return resultado

    def iniciar_takes(self, agora, modo="calibrar"):
        """Modo TAKES (item 120): 2 s parado = take; usuario no ritmo dele."""
        self.modo = str(modo)
        self._ativa = True
        self._modo_takes = True
        self._t0 = float(agora)
        self._take_indice = 0
        self._amostras = {"right": [], "left": []}
        self._normais = {"right": {}, "left": {}}
        self._posicoes = {"right": {}, "left": {}}
        self._take_buffer = {"right": [], "left": []}
        self._take_movimento_ok = {"right": True, "left": True}
        self._take_ref_rot = {"right": None, "left": None}
        self._take_ref_pos = {"right": None, "left": None}

    @property
    def modo_takes(self):
        return bool(getattr(self, "_modo_takes", False))

    def take_indice(self):
        return int(getattr(self, "_take_indice", 0))

    def takes_completos(self):
        return self.take_indice() >= len(self.poses)

    def estado_lado(self, lado, agora):
        info = {"movimento_ok": bool(self._take_movimento_ok.get(lado)),
                "janela_s": 0.0, "desvio_deg": None,
                "estavel": False, "posicao_ok": False,
                "take": self.take_indice()}
        if not self._ativa or not self.modo_takes:
            return info
        janela = [(t, r, p) for (t, r, p)
                  in self._take_buffer.get(lado, ())
                  if float(agora) - t <= POSES_FISICAS_TAKE_SEC]
        if not janela:
            return info
        info["janela_s"] = max(0.0, float(agora) - janela[0][0])
        info["posicao_ok"] = any(p is not None for _t, _r, p in janela)
        if not info["movimento_ok"]:
            return info
        rotacoes = [r for _t, r, _p in janela]
        referencia = rotacoes[len(rotacoes) // 2]
        info["desvio_deg"] = max(
            _angulo_rotacoes(referencia, r) for r in rotacoes)
        estavel = (info["janela_s"] >= POSES_FISICAS_TAKE_SEC * 0.85
                   and len(janela) >= 8
                   and info["desvio_deg"]
                   <= POSES_FISICAS_TAKE_MAX_DESVIO_DEG)
        if estavel and self.modo == "calibrar":
            posicoes = [p for _t, _r, p in janela]
            if any(p is None for p in posicoes):
                estavel = False
            else:
                nuvem = np.asarray(posicoes, np.float64)
                mediana = np.median(nuvem, axis=0)
                espalhamento = float(np.max(
                    np.linalg.norm(nuvem - mediana, axis=1)))
                estavel = espalhamento <= POSES_FISICAS_TAKE_POS_MAX_M
        info["estavel"] = bool(estavel)
        return info

    def atualizar_lado(self, lado, agora, rotacao, posicao=None):
        if not self._ativa or not self.modo_takes or rotacao is None:
            return
        rot = np.asarray(rotacao, np.float64).reshape(3, 3)
        if not np.isfinite(rot).all():
            return
        posicao_vetor = None
        if posicao is not None:
            candidato = np.asarray(posicao, np.float64).reshape(3)
            if np.isfinite(candidato).all():
                posicao_vetor = candidato
        self._take_buffer.setdefault(lado, []).append(
            (float(agora), rot, posicao_vetor))
        limite = float(agora) - POSES_FISICAS_TAKE_SEC * 1.6
        self._take_buffer[lado] = [item for item in self._take_buffer[lado]
                                   if item[0] >= limite]
        if not self._take_movimento_ok.get(lado):
            referencia = self._take_ref_rot.get(lado)
            if (referencia is not None
                    and _angulo_rotacoes(rot, referencia)
                    >= POSES_FISICAS_TAKE_MOVIMENTO_MIN_DEG):
                self._take_movimento_ok[lado] = True
            referencia_pos = self._take_ref_pos.get(lado)
            if (not self._take_movimento_ok.get(lado)
                    and referencia_pos is not None
                    and posicao_vetor is not None
                    and float(np.linalg.norm(posicao_vetor
                                             - referencia_pos))
                    >= POSES_FISICAS_TAKE_MOVIMENTO_POS_M):
                self._take_movimento_ok[lado] = True

    def processar_takes(self, agora):
        if not self._ativa or not self.modo_takes:
            return None
        if self.takes_completos():
            return {"pronto": True}
        if not all(self._take_movimento_ok.values()):
            return None
        infos = {lado: self.estado_lado(lado, agora)
                 for lado in ("right", "left")}
        if not all(info.get("estavel") for info in infos.values()):
            return None
        indice = self.take_indice()
        #: Rejeicao ANTECIPADA por pose parecida (item 122): a direcao do
        #: take atual e' comparada com as direcoes ja' gravadas; se ficar
        #: perto demais, o take e' recusado e o usuario move mais a mao.
        if self.modo == "calibrar" and indice > 0:
            novas_direcoes = {}
            for lado in ("right", "left"):
                janela = [(t, r, p) for (t, r, p)
                          in self._take_buffer.get(lado, ())
                          if float(agora) - t <= POSES_FISICAS_TAKE_SEC]
                posicoes = [p for _t, _r, p in janela if p is not None]
                if not posicoes:
                    novas_direcoes[lado] = None
                    continue
                mediana = np.median(np.asarray(posicoes, np.float64),
                                    axis=0)
                novas_direcoes[lado] = _normalizar_vetor(-mediana)
            pior_angulo = None
            for lado in ("right", "left"):
                nova = novas_direcoes.get(lado)
                if nova is None:
                    continue
                for anterior in range(indice):
                    posicoes_antigas = self._posicoes.get(
                        lado, {}).get(anterior, [])
                    if not posicoes_antigas:
                        continue
                    direcao_antiga = _normalizar_vetor(
                        -np.median(np.asarray(posicoes_antigas,
                                             np.float64), axis=0))
                    if direcao_antiga is None:
                        continue
                    angulo = _angulo_entre(nova, direcao_antiga)
                    if angulo is None:
                        continue
                    if pior_angulo is None or angulo < pior_angulo:
                        pior_angulo = angulo
            if (pior_angulo is not None
                    and pior_angulo < POSES_FISICAS_TAKE_SEPARACAO_MIN_DEG):
                self._take_buffer = {"right": [], "left": []}
                self._take_movimento_ok = {"right": False,
                                           "left": False}
                return {"rejeitado": indice + 1,
                        "angulo": float(pior_angulo),
                        "minimo": POSES_FISICAS_TAKE_SEPARACAO_MIN_DEG}
        for lado in ("right", "left"):
            janela = [(t, r, p) for (t, r, p)
                      in self._take_buffer.get(lado, ())
                      if float(agora) - t <= POSES_FISICAS_TAKE_SEC]
            posicoes = []
            for _t, rot, posicao in janela:
                self._amostras[lado].append((indice, rot))
                if posicao is not None:
                    self._posicoes[lado].setdefault(
                        indice, []).append(posicao)
                    posicoes.append(posicao)
            rotacoes = [rot for _t, rot, _p in janela]
            if rotacoes:
                self._take_ref_rot[lado] = rotacoes[len(rotacoes) // 2]
            if posicoes:
                self._take_ref_pos[lado] = np.median(
                    np.asarray(posicoes, np.float64), axis=0)
        self._take_indice = indice + 1
        self._take_buffer = {"right": [], "left": []}
        self._take_movimento_ok = {"right": False, "left": False}
        return {"take": indice + 1, "total": len(self.poses),
                "pronto": self.takes_completos()}

    def _por_pose(self, lado):
        grupos = {}
        for indice, rot in self._amostras.get(lado, ()):
            grupos.setdefault(indice, []).append(rot)
        return grupos

    def _representante_normal(self, lado, indice):
        lista = self._normais.get(lado, {}).get(indice, [])
        if len(lista) < 3:
            return None
        return _mediana_vetores(lista)

    def _angulos_poses(self, lado):
        """(theta12, theta13, theta23) em graus medidos pelo Kinect, ou None."""
        representantes = [self._representante_normal(lado, indice)
                          for indice in range(len(self.poses))]
        if any(rep is None for rep in representantes):
            return None
        n1, n2, n3 = representantes
        return (_angulo_entre(n1, n2), _angulo_entre(n1, n3),
                _angulo_entre(n2, n3))

    def _calibrar_lado(self, lado, poses):
        grupos = self._por_pose(lado)
        faltam = [indice for indice in range(len(poses))
                  if len(grupos.get(indice, [])) < 3]
        if faltam:
            return {"sem_amostras": True,
                    "poses_insuficientes": [indice + 1 for indice in faltam]}
        rotacoes = []
        vetores = []
        for indice in range(len(poses)):
            vetor = np.asarray(poses[indice][1], np.float64)
            vetor = vetor / max(1e-12, float(np.linalg.norm(vetor)))
            for rot in grupos[indice]:
                rotacoes.append(rot)
                vetores.append(vetor)
        resolvido = solve_palm_alignment(rotacoes, vetores)
        if resolvido is None:
            return {"sem_amostras": True, "poses_insuficientes": []}
        t_matriz, f_corpo, residuo = resolvido
        erros = {}
        for indice in range(len(poses)):
            vetor = np.asarray(poses[indice][1], np.float64)
            vetor = vetor / max(1e-12, float(np.linalg.norm(vetor)))
            preditos = [t_matriz @ rot @ f_corpo for rot in grupos[indice]]
            valores = [_angulo_entre(v, vetor) for v in preditos]
            valores = [v for v in valores if v is not None]
            erros[indice] = float(np.median(valores)) if valores else None
        return {"T": t_matriz, "f": f_corpo, "residuo": float(residuo),
                "poses": erros, "amostras": len(rotacoes),
                "poses_usadas": tuple(poses)}

    def _calibrar_com_poses(self, poses):
        self.poses = tuple(poses)
        resultado = {"modo": "calibrar", "lados": {}, "fechado": False,
                     "modo_verdade": "poses_fixas"}
        for lado in ("right", "left"):
            resultado["lados"][lado] = self._calibrar_lado(lado, self.poses)
        resultado["fechado"] = self._fechado(resultado["lados"])
        return resultado

    def _fechado(self, lados):
        for dados in lados.values():
            if dados.get("sem_amostras"):
                return False
            erros = [erro for erro in dados.get("poses", {}).values()
                     if erro is not None]
            if len(erros) != len(self.poses):
                return False
            for indice, erro in dados["poses"].items():
                limite = (POSES_FISICAS_LIMITE_POSE1_DEG if indice == 0
                          else POSES_FISICAS_LIMITE_MESA_DEG)
                if erro is None or erro > limite:
                    return False
            espalhamento = dados.get("espalhamento_min_deg")
            if (espalhamento is not None
                    and espalhamento < POSES_FISICAS_SPREAD_MIN_DEG):
                return False
            espalhamento_max = dados.get("espalhamento_max_deg")
            if (espalhamento_max is not None
                    and espalhamento_max < POSES_FISICAS_SPREAD_MAX_MIN_DEG):
                return False
        return True

    def calibrar(self):
        """{lado: {T, f, residuo, poses:{idx: erro}}} com as poses atuais."""
        return self._calibrar_com_poses(self.poses)

    def calibrar_mesa(self, normal_mesa):
        """Calibra com a verdade fisica; usa angulos relativos do Kinect.

        Retorna tambem `poses_verdade` por lado (para o V usar exatamente as
        mesmas verdade do J) e `angulos` medidos.
        """
        resultado = {"modo": "calibrar", "lados": {}, "fechado": False,
                     "angulos": {}, "poses_verdade": {},
                     "modo_verdade": "plano_da_mesa"}
        for lado in ("right", "left"):
            angulos = self._angulos_poses(lado)
            poses_lado = poses_fisicas_com_mesa(normal_mesa)
            if angulos is not None:
                verdades = verdades_por_angulos(
                    (0.0, 0.0, -1.0), normal_mesa, *angulos)
            else:
                verdades = None
            if verdades is not None:
                poses_lado = tuple(
                    (POSES_FISICAS_CALIBRACAO[indice][0], tuple(verdades[indice]))
                    for indice in range(3))
                resultado["angulos"][lado] = angulos
            dados = self._calibrar_lado(lado, poses_lado)
            dados["poses_usadas"] = poses_lado
            resultado["lados"][lado] = dados
        #: escolhe o sinal da normal da mesa que FECHA/da' menor residuo
        melhor = None
        for sinal in (1.0, -1.0):
            resultado_sinal = {"modo": "calibrar", "lados": {},
                               "angulos": resultado["angulos"],
                               "modo_verdade": resultado["modo_verdade"]}
            for lado in ("right", "left"):
                angulos = resultado["angulos"].get(lado)
                if angulos is not None:
                    verdades = verdades_por_angulos(
                        (0.0, 0.0, -1.0), normal_mesa, *angulos)
                else:
                    verdades = None
                if verdades is not None:
                    poses_lado = tuple(
                        (POSES_FISICAS_CALIBRACAO[indice][0],
                         tuple(verdades[indice])) for indice in range(3))
                else:
                    poses_lado = poses_fisicas_com_mesa(normal_mesa,
                                                        sinal=sinal)
                resultado_sinal["lados"][lado] = self._calibrar_lado(
                    lado, poses_lado)
            resultado_sinal["fechado"] = self._fechado(resultado_sinal["lados"])
            residuos = [d.get("residuo") for d in resultado_sinal["lados"].values()
                        if not d.get("sem_amostras")]
            media = float(np.mean(residuos)) if residuos else float("inf")
            if melhor is None or media < melhor[0]:
                melhor = (media, sinal, resultado_sinal)
        if melhor is None:
            return resultado
        _media, sinal, vencedor = melhor
        vencedor["sinal_mesa"] = float(sinal)
        vencedor["modo_verdade"] = ("angulos_kinect"
                                    if vencedor["angulos"]
                                    else "plano_da_mesa")
        vencedor["fechado"] = self._fechado(vencedor["lados"])
        vencedor["poses_verdade"] = {
            lado: list(dados.get("poses_usadas") or ())
            for lado, dados in vencedor["lados"].items()}
        return vencedor

    def verificar(self, fusions, poses_por_lado=None):
        """Certificado contra as verdades (por lado, se fornecidas)."""
        resultado = {"modo": "verificar", "poses": [p[0] for p in self.poses],
                     "limite_graus": self.limite_graus, "lados": {},
                     "fechado": False, "poses_ruins": []}
        for lado in ("right", "left"):
            poses_lado = None
            if poses_por_lado is not None:
                poses_lado = poses_por_lado.get(lado)
            if not poses_lado:
                poses_lado = self.poses
            fusao = (fusions.get(lado) if hasattr(fusions, "get")
                     else fusions[lado])
            t_matriz = getattr(fusao, "imu_to_camera", None)
            f_corpo = getattr(fusao, "palm_reference_body", None)
            dados = {"poses": {}, "sem_alinhamento": False}
            if t_matriz is None or f_corpo is None:
                dados["sem_alinhamento"] = True
                resultado["lados"][lado] = dados
                continue
            t_matriz = np.asarray(t_matriz, np.float64).reshape(3, 3)
            f_corpo = np.asarray(f_corpo, np.float64).reshape(3)
            f_corpo = f_corpo / max(1e-12, float(np.linalg.norm(f_corpo)))
            grupos = self._por_pose(lado)
            ok_lado = True
            for indice in range(len(poses_lado)):
                vetor = _normalizar_vetor(poses_lado[indice][1])
                valores = []
                for rot in grupos.get(indice, ()):
                    erro = _angulo_entre(t_matriz @ rot @ f_corpo, vetor)
                    if erro is not None:
                        valores.append(erro)
                if len(valores) >= 3:
                    stats = {"n": len(valores),
                             "media": float(np.mean(valores)),
                             "p90": float(np.percentile(valores, 90)),
                             "max": float(np.max(valores))}
                    limite = (POSES_FISICAS_LIMITE_POSE1_DEG if indice == 0
                              else POSES_FISICAS_LIMITE_MESA_DEG)
                    stats["passou"] = stats["media"] <= limite
                    stats["limite_graus"] = float(limite)
                else:
                    stats = {"n": len(valores), "media": None, "p90": None,
                             "max": None, "passou": False}
                dados["poses"][indice] = stats
                if not stats["passou"]:
                    ok_lado = False
                    resultado["poses_ruins"].append(
                        (lado, indice + 1, poses_lado[indice][0],
                         stats.get("media")))
            dados["fechado"] = ok_lado
            resultado["lados"][lado] = dados
        resultado["fechado"] = all(
            dados.get("fechado") for dados in resultado["lados"].values())
        return resultado


def formatar_certificado(relatorio, amostras_min=3):
    """Linhas de texto do CERTIFICADO V, sem quebrar com valor None.

    Por que existe (item 110): o tool imprimia o certificado montando a
    string direto; uma pose sem amostras tem `media=None` e o formato
    `%.1f` estourava TypeError -- justamente DEPOIS de uma verificacao longa.
    Aqui a formatacao e' pura e testavel: pose sem amostras sai como
    "SEM ESTABILIDADE/AMOSTRAS", nunca excecao.
    """
    linhas = []
    relatorio = relatorio or {}
    for lado in ("right", "left"):
        dados = (relatorio.get("lados") or {}).get(lado) or {}
        if dados.get("sem_alinhamento"):
            linhas.append(
                f"  {lado}: SEM ALINHAMENTO carregado -- aperte O ou J "
                "antes de verificar")
            continue
        linhas.append(f"  {lado}:")
        for indice, nome in enumerate(relatorio.get("poses") or ()):
            stats = (dados.get("poses") or {}).get(indice) or {}
            n = int(stats.get("n") or 0)
            media = stats.get("media")
            p90 = stats.get("p90")
            maximo = stats.get("max")
            if media is None or n < int(amostras_min):
                linhas.append(
                    f"    {indice + 1}. {nome}: SEM ESTABILIDADE/AMOSTRAS "
                    f"(n={n}) -- FAIL (mantenha a mao visivel e PARADA "
                    "durante a pose)")
                continue
            veredito = "PASS" if stats.get("passou") else "FAIL"
            linhas.append(
                f"    {indice + 1}. {nome}: n={n} | media {media:.1f} | "
                f"p90 {p90:.1f} | max {maximo:.1f} graus -- {veredito}")
    if relatorio.get("fechado"):
        linhas.append(
            "  CERTIFICADO: FECHADO -- o vetor do IMU bate com a direcao "
            "anatomica da palma em TODAS as poses (<= limite).")
    else:
        textos = []
        for item in relatorio.get("poses_ruins") or ():
            lado, indice, nome, media = item
            detalhe = ("sem amostras" if media is None
                       else f"{float(media):.1f} graus")
            textos.append(f"{lado} {indice} ({nome}, {detalhe})")
        linhas.append(
            "  CERTIFICADO: NAO FECHADO -- poses ruins: "
            + "; ".join(textos))
        linhas.append(
            "  -> rode J (refit pelo plano do dorso) e repita V; se "
            "persistir, confira montagem/luvas trocadas (--imu-portas)")
    return linhas


#: Colunas do log continuo do vetor da palma (item 111: "--log-palma").
CABECALHO_LOG_PALMA = (
    "t_mono",
    "lado",
    "frame",
    "mao_detectada",
    "mapeamento_falhas",
    "depth_falhas",
    "fallbacks",
    "resgatados",
    "pontos_3d_ok",
    "depth_palma_mm",
    "depth_min_mm",
    "depth_max_mm",
    "normal_plano_x",
    "normal_plano_y",
    "normal_plano_z",
    "normal_estavel_x",
    "normal_estavel_y",
    "normal_estavel_z",
    "spread_deg",
    "rms_mm",
    "estavel_contagem",
    "robust_inliers",
    "palm_dir_x",
    "palm_dir_y",
    "palm_dir_z",
    "erro_plano_deg",
    "erro_palma_deg",
    "sync_color_depth_ms",
    "seta_visivel",
    "log_ligado",
)


def _fmt_log(valor, casas=4):
    """Valor para CSV: None vira vazio (nunca 0 disfarcado de medida)."""
    if valor is None:
        return ""
    if isinstance(valor, (bool, np.bool_)):
        return "1" if valor else "0"
    if isinstance(valor, (int, np.integer)):
        return str(int(valor))
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if not np.isfinite(numero):
        return ""
    return f"{numero:.{casas}f}"


class LogVetorPalma:
    """Grava 1 linha/frame/mao do VETOR DA PALMA (pedido do usuario, item 111).

    Para que serve: quando a seta fica "ruim" ou some, este CSV mostra POR QUE:
    quantos nos falharam o mapeamento color->depth, as profundidades, a normal
    instantanea x a estavel, o spread da janela, o erro contra o IMU e a
    sincronia color-depth -- sem precisar de print no terminal no meio da
    sessao. Togle: `--log-palma` na linha de comando OU tecla G no video.
    """

    def __init__(self, caminho):
        self.caminho = Path(caminho)
        self.arquivo = open(self.caminho, "w", encoding="utf-8", newline="")
        self.arquivo.write(",".join(CABECALHO_LOG_PALMA) + "\n")
        self.arquivo.flush()
        self.linhas = 0

    def registrar(self, **campos):
        """Grava uma linha; aceita None e arrays (x,y,z separados)."""
        linha = []
        for coluna in CABECALHO_LOG_PALMA:
            linha.append(_fmt_log(campos.get(coluna)))
        self.arquivo.write(",".join(linha) + "\n")
        self.linhas += 1
        if self.linhas % 10 == 0:
            self.arquivo.flush()

    def fechar(self):
        try:
            self.arquivo.flush()
            self.arquivo.close()
        except OSError:
            pass


def atualizar_estabilidade(contagem, falhas_consecutivas, plano_estavel,
                           minimo_quadros=5, faltas_toleradas=1):
    """(contagem, falhas) tolerando UM quadro sem plano (mapeamento falhou).

    Sem tolerancia, um unico frame com depth furo/oclusao zerava a contagem e
    a captura J/V quase nunca completava (visto no log de 22/10, item 111).
    Com 1 falta tolerada a janela continua se o plano voltar no quadro seguinte;
    2+ faltas consecutivas zeram (a mao realmente sumiu).
    """
    if plano_estavel is not None:
        return int(contagem) + 1, 0
    falhas = int(falhas_consecutivas) + 1
    if falhas > int(faltas_toleradas):
        return 0, falhas
    return int(contagem), falhas


#: Poses da VERIFICACAO GUIADA da palma (tecla V do groundtruth tool, item 103).
#: As 5 primeiras cobrem o espaco de orientacoes (frente/cima/baixo/lados); a
#: sexta exercita o punho girando. Cada linha e' a instrucao que aparece no video
#: e no console: o experimentador faz a pose e segura ate' a troca.
POSES_VERIFICACAO_PALMA = (
    "PALMA PARA O KINECT (de frente para o sensor)",
    "PALMA PARA CIMA (para o teto)",
    "PALMA PARA BAIXO (para a mesa)",
    "PALMA PARA A ESQUERDA",
    "PALMA PARA A DIREITA",
    "GIRE O PUNHO DEVAGAR (360 graus, sem sair do quadro)",
)
#: Duracao de cada pose da verificacao guiada (s). 3 s para dar tempo de
#: chegar, PARAR e acumular >=0,5 s de verdade estavel (item 110).
VERIFICACAO_POSE_SEC = 3.0
#: Tempo (s) descartado no inicio de cada pose (tempo de chegar na pose).
VERIFICACAO_SETTLE_SEC = 0.8
#: Limite (graus) do certificado "FECHADO" -- o MESMO espirito do limite por
#: fase do diagnostico (15 graus), mas aqui e' a prova pose a pose.
VERIFICACAO_LIMITE_DEG = 15.0


class VerificacaoPosesPalma:
    """Coleta pose a pose e emite o CERTIFICADO: o vetor do IMU bate com a verdade?

    A verdade e' a normal do PLANO MEDIO DO DORSO (`palm_normal_from_plane`,
    item 102). A previsao e' `T R f` do alinhamento carregado -- o MESMO da seta
    do video. Para cada pose (direita e esquerda) calcula n/media/p90/max do
    erro e classifica PASS/FAIL; no fim da' o veredito FECHADO (todas as poses
    <= `limite` graus) ou NAO FECHADO (lista as poses ruins). E' como se fecha a
    questao "o IMU aponta para onde a palma aponta?" com numeros, nao com
    impressao -- se nao fechar, o proximo passo e' refitar (tecla J) e repetir.

    Nao depende de Kinect/cv2: o tool injeta `(rotacao, normal_plano)` por frame
    e chama `relatorio(fusions)` no fim (testavel com fusions falsas).
    """

    def __init__(self, poses=POSES_VERIFICACAO_PALMA,
                 seg_pose=VERIFICACAO_POSE_SEC,
                 settle=VERIFICACAO_SETTLE_SEC,
                 limite=VERIFICACAO_LIMITE_DEG):
        self.poses = tuple(poses)
        self.seg_pose = float(seg_pose)
        self.settle = float(settle)
        self.limite_graus = float(limite)
        self._ativa = False
        self._t0 = 0.0
        self._amostras = {"right": [], "left": []}   # [(pose_idx, R, normal)]

    def iniciar(self, agora):
        self._ativa = True
        self._t0 = float(agora)
        self._amostras = {"right": [], "left": []}

    def parar(self):
        self._ativa = False

    @property
    def ativa(self):
        return self._ativa

    @property
    def duracao_s(self):
        return len(self.poses) * self.seg_pose

    def fase_atual(self, agora):
        """(indice, nome, restante_s) da pose agora, ou None se terminou."""
        if not self._ativa:
            return None
        decorrido = float(agora) - self._t0
        if decorrido < 0.0:
            return None
        total = self.duracao_s
        if decorrido >= total:
            return None
        indice = int(decorrido // self.seg_pose)
        indice = min(max(indice, 0), len(self.poses) - 1)
        restante = (indice + 1) * self.seg_pose - decorrido
        return indice, self.poses[indice], max(0.0, restante)

    def coletar(self, lado, agora, rotacao, normal_camera):
        """Guarda uma amostra (pose atual) se passou o settle da pose."""
        if not self._ativa:
            return False
        fase = self.fase_atual(agora)
        if fase is None:
            return False
        indice = fase[0]
        dentro = (float(agora) - self._t0) - indice * self.seg_pose
        if dentro < self.settle:
            return False
        if rotacao is None or normal_camera is None:
            return False
        rot = np.asarray(rotacao, np.float64).reshape(3, 3)
        normal = np.asarray(normal_camera, np.float64).reshape(3)
        norma = float(np.linalg.norm(normal))
        if not (np.isfinite(rot).all() and np.isfinite(normal).all()):
            return False
        if norma < 1e-9:
            return False
        self._amostras[lado].append((indice, rot, normal / norma))
        return True

    def contar(self, lado, indice_pose=None):
        if indice_pose is None:
            return len(self._amostras.get(lado, ()))
        return sum(1 for pose, _r, _n in self._amostras.get(lado, ())
                   if pose == indice_pose)

    def relatorio(self, fusions, limite=None):
        """CERTIFICADO: {lados: {lado: {poses: {...}}}, fechado, poses_ruins}.

        `fusions` = {"right": PositionFusion, "left": PositionFusion} (ou um
        objeto com `.get`/intexacao). Usa `imu_to_camera` e
        `palm_reference_body` -- se faltar, marca o lado sem alinhamento.
        """
        limite = self.limite_graus if limite is None else float(limite)

        def erro_graus(previsto, verdade):
            norma = float(np.linalg.norm(previsto))
            if norma < 1e-9:
                return float("nan")
            cosseno = float(np.dot(previsto / norma, verdade))
            return float(np.degrees(np.arccos(np.clip(cosseno, -1.0, 1.0))))

        resultado = {"limite_graus": limite, "poses": self.poses,
                     "lados": {}, "fechado": False, "poses_ruins": []}
        for lado in ("right", "left"):
            fusao = (fusions.get(lado) if hasattr(fusions, "get")
                     else fusions[lado])
            t_matriz = getattr(fusao, "imu_to_camera", None)
            f_corpo = getattr(fusao, "palm_reference_body", None)
            dados = {"poses": {}, "sem_alinhamento": False}
            if t_matriz is None or f_corpo is None:
                dados["sem_alinhamento"] = True
                resultado["lados"][lado] = dados
                continue
            t_matriz = np.asarray(t_matriz, np.float64).reshape(3, 3)
            f_corpo = np.asarray(f_corpo, np.float64).reshape(3)
            f_corpo = f_corpo / max(1e-12, float(np.linalg.norm(f_corpo)))
            erros_pose = {}
            for indice, rotacao, normal in self._amostras[lado]:
                previsto = t_matriz @ rotacao @ f_corpo
                erros_pose.setdefault(indice, []).append(
                    erro_graus(previsto, normal))
            ok_lado = True
            for indice, nome in enumerate(self.poses):
                valores = np.asarray([v for v in erros_pose.get(indice, [])
                                      if np.isfinite(v)])
                if valores.size >= 3:
                    stats = {"n": int(valores.size),
                             "media": float(np.mean(valores)),
                             "p90": float(np.percentile(valores, 90)),
                             "max": float(np.max(valores))}
                    stats["passou"] = stats["media"] <= limite
                else:
                    stats = {"n": int(valores.size), "media": None,
                             "p90": None, "max": None, "passou": False}
                dados["poses"][indice] = stats
                if not stats["passou"]:
                    ok_lado = False
                    resultado["poses_ruins"].append(
                        (lado, indice + 1, nome, stats.get("media")))
            dados["fechado"] = ok_lado
            resultado["lados"][lado] = dados
        resultado["fechado"] = all(
            dados.get("fechado") for dados in resultado["lados"].values()
        )
        return resultado


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
#: Quanto (m, camera space) uma junta tem de estar ABAIXO do fim do tronco
#: (SpineBase) para ser DESCARTADA. A MESA bloqueia a vista do Kinect abaixo do
#: tronco (bancada 09/10/2026, pedido do usuario): o SDK infere quadril/joelho/
#: pe ATRAVES da mesa (state Inferred, que o desenho ACEITAVA) e esses pontos sao
#: fantasma -- esqueleto desenhado atravessando a mesa e medida errada se
#: consumidos (ancora de profundidade do punho, modelo do braco). Margem de
#: 0,10 m: joelhos/pes (~0,25 m abaixo) somem; quadril (~0,05 m) e PUNHO NA MESA
#: (mesma altura do tronco sentado) continuam. 0 = mascara desligada.
MASCARA_ABAIXO_TRONCO_M = 0.10
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

@dataclass(frozen=True)
class AuxCameraView:
    """Calibracao de UMA webcam auxiliar, pronta para usar SEM trocar a ativa.

    Motivo (bancada de 23/09/2026): com DUAS auxiliares (uma de cada lado, ~45
    graus do Kinect) o overlay tem de projetar as DUAS no MESMO frame do Kinect.
    Os campos vivos do `CameraAlignment` (auxiliary_matrix, stereo_rotation, ...)
    descrevem UMA camera so': chamar `set_active_aux` dentro do desenho seria
    estado global mudando no meio do frame (e a camera ativa deixaria de ser a
    que o protocolo escolheu). A vista e' IMUTAVEL e autocontida: quem desenha
    passa a vista, nao o alignment.

    Convencao (a mesma da calibracao salva): X_aux = R X_kin + T, com X em metros
    no referencial de CAMERA do Kinect (Z para frente, X para a direita).
    """

    index: int
    kinect_matrix: Optional[np.ndarray] = None
    kinect_dist: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    rms: Optional[float] = None
    stereo_usable: bool = False
    runtime_matrix: Optional[np.ndarray] = None
    runtime_size: Optional[tuple] = None
    calibrated_size: Optional[tuple] = None

    @property
    def usable(self):
        """True quando da' para projetar/triangular com esta camera."""
        return bool(
            self.stereo_usable
            and self.matrix is not None and self.dist is not None
            and self.rotation is not None and self.translation is not None
            and self.kinect_matrix is not None and self.kinect_dist is not None
        )

    @property
    def size(self):
        """(w, h) do frame em uso agora (runtime) ou o da calibracao."""
        return self.runtime_size or self.calibrated_size

    @property
    def effective_matrix(self):
        """Matriz intrinseca na resolucao em uso (a calibrada ou a escalada)."""
        return (self.runtime_matrix if self.runtime_matrix is not None
                else self.matrix)

    def pixels(self, landmarks):
        """(N,2) px DESTA camera a partir de landmarks normalizados do MediaPipe."""
        width, height = self.size or (0, 0)
        return np.array([(lm.x * width, lm.y * height) for lm in landmarks],
                        np.float64)

    def to_kinect_stereo(self, points, kinect_depth_m):
        """Projeta pixels da auxiliar no RGB do Kinect com profundidade dada.

        Mesma matematica de `CameraAlignment.auxiliary_to_kinect_stereo` (que
        agora delega para aqui). Devolve (N,2) px do Kinect, ou None quando a
        camera nao tem stereo usavel (o chamador decide o fallback 2D).
        """
        if not self.usable:
            return None
        matrix = self.effective_matrix
        auxiliary_points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(
            auxiliary_points, matrix, self.dist
        ).reshape(-1, 2)
        depths = np.asarray(kinect_depth_m, dtype=np.float32).reshape(-1)
        if len(depths) == 1:
            depths = np.full(len(normalized), depths[0], dtype=np.float32)
        if len(depths) != len(normalized):
            return None
        valid_depths = depths[np.isfinite(depths) & (depths > 0)]
        if valid_depths.size == 0:
            return None
        depths = np.where(
            np.isfinite(depths) & (depths > 0),
            depths,
            np.median(valid_depths),
        )
        # Profundidades invalidas (NaN, 0, fora do sensor) viram a mediana das
        # validas; o clip fisico (0.4-3.0 m) evita valores absurdos. NAO ha'
        # suavizacao por indice de landmark: os indices 0..20 do MediaPipe nao
        # sao espacialmente contiguos e isso destruiria os offsets 3D aprendidos.
        depths = np.clip(np.asarray(depths, np.float64), 0.4, 4.0)
        original = np.asarray(kinect_depth_m, dtype=np.float32).reshape(-1)
        original_finite = np.isfinite(original) & (original > 0)
        if original_finite.sum() < 0.5 * original.size:
            depths = np.full(original.size, float(np.median(depths)),
                             dtype=np.float64)
        kinect_3d = []
        rotation_inverse = np.asarray(self.rotation, np.float64).T
        translation = np.asarray(self.translation, np.float64).reshape(3)
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
        kinect_3d = np.asarray(kinect_3d, dtype=np.float64)
        projected, _ = cv2.projectPoints(
            kinect_3d.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            np.asarray(self.kinect_matrix, np.float64),
            np.asarray(self.kinect_dist, np.float64),
        )
        return projected.reshape(-1, 2)


    def triangulate_with_kinect(self, aux_pixels, kinect_pixels):
        """Triangula nos 3D (m, frame do Kinect) a partir das DUAS vistas.

        Mesma matematica de `CameraAlignment.triangulate_hand_points` (que agora
        delega para aqui): camera 1 = Kinect na origem (P1 = [I|0]), camera 2 =
        esta auxiliar (P2 = [R|T], com X_aux = R X_kin + T). Correspondencia
        trivial pelo indice do no do MediaPipe.

        Devolve (points_3d (N,3), reproj_px (N,)) ou (None, None).
        """
        if not self.usable:
            return None, None
        aux_px = np.asarray(aux_pixels, np.float64).reshape(-1, 1, 2)
        kin_px = np.asarray(kinect_pixels, np.float64).reshape(-1, 1, 2)
        if aux_px.shape[0] == 0 or aux_px.shape[0] != kin_px.shape[0]:
            return None, None
        rays_kin = cv2.undistortPoints(
            kin_px, np.asarray(self.kinect_matrix, np.float64),
            np.asarray(self.kinect_dist, np.float64),
        ).reshape(-1, 2)
        rays_aux = cv2.undistortPoints(
            aux_px, np.asarray(self.effective_matrix, np.float64),
            np.asarray(self.dist, np.float64),
        ).reshape(-1, 2)
        if not (np.isfinite(rays_kin).all() and np.isfinite(rays_aux).all()):
            return None, None
        R = np.asarray(self.rotation, np.float64).reshape(3, 3)
        T = np.asarray(self.translation, np.float64).reshape(3, 1)
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, T])
        points3d, plausible = _triangulate_rays(P1, P2, rays_kin, rays_aux)
        if points3d is None:
            return None, None
        # Erro de reprojecao por no em px do KINECT (mesma metrica do codigo
        # original: a qualidade da triangulacao e' lida em pixels).
        projected, _ = cv2.projectPoints(
            np.nan_to_num(points3d).reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            np.asarray(self.kinect_matrix, np.float64),
            np.asarray(self.kinect_dist, np.float64),
        )
        reproj = np.linalg.norm(
            projected.reshape(-1, 2) - kin_px.reshape(-1, 2), axis=1
        )
        points3d[~plausible] = np.nan
        return points3d, reproj


def _triangulate_rays(P1, P2, rays_1, rays_2):
    """Triangula raios normalizados (DLT) e valida a plausibilidade fisica.

    Devolve (points_3d (N,3) no referencial da camera 1, mascara_de_plausiveis)
    ou (None, None). Nos improvaveis (oclusao, deteccao trocada entre as maos)
    viram NaN -- e a mao toda e' recusada quando menos da METADE dos nos e'
    plausivel (mesma regra do `triangulate_hand_points` original). O erro de
    reprojecao NAO e' calculado aqui: cada chamador o mede em px, com as
    intrinsecas da camera cujo pixel ele tem.
    """
    rays_1 = np.asarray(rays_1, np.float64).reshape(-1, 2)
    rays_2 = np.asarray(rays_2, np.float64).reshape(-1, 2)
    points4 = cv2.triangulatePoints(
        np.asarray(P1, np.float64), np.asarray(P2, np.float64),
        rays_1.reshape(-1, 1, 2).astype(np.float64),
        rays_2.reshape(-1, 1, 2).astype(np.float64),
    )
    w = points4[3]
    if not np.isfinite(w).all() or (np.abs(w) < 1e-9).any():
        return None, None
    points3d = (points4[:3] / w).T
    z = points3d[:, 2]
    plausible = np.isfinite(points3d).all(axis=1) & (z > 0.3) & (z < 4.0)
    if plausible.sum() < max(5, int(0.5 * len(points3d))):
        return None, None
    return points3d, plausible


def erro_reprojecao_px(points_3d, pixels, matrix, dist):
    """Reprojecao mediana (px) de pontos 3D numa camera (metric de qualidade).

    CONVENCAO: `points_3d` vem no referencial DA PROPRIA CAMERA (o mesmo que a
    triangulacao devolve para a camera 1 de um par). Para medir num referencial
    do Kinect, converta antes:  X_cam = R X_kin + T.
    """
    points_3d = np.asarray(points_3d, np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    validos = np.isfinite(points_3d).all(axis=1)
    if not validos.any():
        return float("nan")
    projected, _ = cv2.projectPoints(
        points_3d[validos].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        np.asarray(matrix, np.float64), np.asarray(dist, np.float64),
    )
    erros = np.linalg.norm(projected.reshape(-1, 2) - pixels[validos], axis=1)
    if erros.size == 0 or not np.isfinite(erros).any():
        return float("nan")
    return float(np.nanmedian(erros))

def relative_transform(view_a, view_b):
    """(R_ab, T_ab) entre duas auxiliares: X_b = R_ab X_a + T_ab.

    As duas vistas estao calibradas contra o MESMO Kinect (X_i = R_i X_kin +
    T_i), entao a relacao entre elas sai por composicao:

        X_kin = R_a^T (X_a - T_a)
        X_b   = R_b R_a^T X_a + (T_b - R_b R_a^T T_a)

    Isso dispensa calibrar o par aux-aux separadamente: as duas calibracoes com o
    Kinect ja' definem a geometria entre elas.
    """
    R_a = np.asarray(view_a.rotation, np.float64).reshape(3, 3)
    T_a = np.asarray(view_a.translation, np.float64).reshape(3)
    R_b = np.asarray(view_b.rotation, np.float64).reshape(3, 3)
    T_b = np.asarray(view_b.translation, np.float64).reshape(3)
    R_ab = R_b @ R_a.T
    T_ab = T_b - R_ab @ T_a
    return R_ab, T_ab


def triangulate_pair_points(view_a, view_b, pixels_a, pixels_b):
    """Triangula a mao usando SO' AS DUAS auxiliares (sem o Kinect).

    E' o caminho que responde ao objetivo do projeto: quando o Kinect perde o
    video e a projecao do esqueleto, as duas auxiliares (nos dois lados, ~45
    graus) ainda dizem, por elas mesmas, ONDE a mao esta nos 3 eixos do espaco.

    `pixels_a`/`pixels_b`: (N,2) em pixels de cada camera (`view.pixels`).
    Devolve (points_3d (N,3) NO REFERENCIAL DO KINECT, erro_mediano_px) ou
    (None, None). O erro e' a media das reprojecoes medianas das duas vistas.
    """
    if not (view_a.usable and view_b.usable):
        return None, None
    if int(view_a.index) == int(view_b.index):
        return None, None
    px_a = np.asarray(pixels_a, np.float64).reshape(-1, 1, 2)
    px_b = np.asarray(pixels_b, np.float64).reshape(-1, 1, 2)
    if px_a.shape[0] == 0 or px_a.shape[0] != px_b.shape[0]:
        return None, None
    rays_a = cv2.undistortPoints(
        px_a, np.asarray(view_a.effective_matrix, np.float64),
        np.asarray(view_a.dist, np.float64),
    ).reshape(-1, 2)
    rays_b = cv2.undistortPoints(
        px_b, np.asarray(view_b.effective_matrix, np.float64),
        np.asarray(view_b.dist, np.float64),
    ).reshape(-1, 2)
    if not (np.isfinite(rays_a).all() and np.isfinite(rays_b).all()):
        return None, None
    R_ab, T_ab = relative_transform(view_a, view_b)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R_ab, T_ab.reshape(3, 1)])
    points_a, plausible = _triangulate_rays(P1, P2, rays_a, rays_b)
    if points_a is None:
        return None, None
    points_a = points_a.copy()
    points_a[~plausible] = np.nan
    # Volta para o referencial do KINECT (o MESMO que a triangulacao com o
    # Kinect devolve): X_kin = R_a^T (X_a - T_a).
    R_a = np.asarray(view_a.rotation, np.float64).reshape(3, 3)
    T_a = np.asarray(view_a.translation, np.float64).reshape(3)
    points_kin = (R_a.T @ (points_a - T_a.reshape(1, 3)).T).T
    points_b = (R_ab @ points_a.T).T + T_ab.reshape(1, 3)
    erro_a = erro_reprojecao_px(points_a, px_a.reshape(-1, 2),
                                view_a.effective_matrix, view_a.dist)
    erro_b = erro_reprojecao_px(points_b, px_b.reshape(-1, 2),
                                view_b.effective_matrix, view_b.dist)
    erros = [e for e in (erro_a, erro_b) if np.isfinite(e)]
    if not erros or not np.isfinite(points_kin).any():
        return None, None
    return points_kin, float(np.mean(erros))


def triangulate_aux_pairs(alignment, deteccoes):
    """Melhor triangulacao usando APENAS pares de webcams auxiliares.

    `deteccoes` = {indice: landmarks do MediaPipe (21) ou None}; a mao JA' vem
    escolhida pelo lado da trial (o casamento por lado e' do chamador). Todas as
    combinacoes de pares com deteccao E calibracao utilizavel sao trianguladas;
    vence o MENOR erro mediano de reprojecao (px).

    Devolve (points_3d (N,3) no frame do Kinect, info) com
    info = {"indices": (a, b), "erro_mediano_px": float} ou (None, None).
    """
    indices = sorted(int(i) for i, maos in (deteccoes or {}).items() if maos)
    if len(indices) < 2:
        return None, None
    melhor = None
    melhor_erro = float("inf")
    melhor_indices = None
    for posicao, index_a in enumerate(indices):
        view_a = alignment.aux_view(index_a)
        if view_a is None or not view_a.usable:
            continue
        for index_b in indices[posicao + 1:]:
            view_b = alignment.aux_view(index_b)
            if view_b is None or not view_b.usable:
                continue
            points, erro = triangulate_pair_points(
                view_a, view_b,
                view_a.pixels(deteccoes[index_a]),
                view_b.pixels(deteccoes[index_b]),
            )
            if points is None or not np.isfinite(erro):
                continue
            if erro < melhor_erro:
                melhor, melhor_erro = points, erro
                melhor_indices = (index_a, index_b)
    if melhor is None:
        return None, None
    return melhor, {"indices": melhor_indices, "erro_mediano_px": melhor_erro}





def _cinza_para_detectar_tabuleiro(gray, largura_alvo):
    """(cinza reduzido com CLAHE, fator para voltar a resolucao cheia).

    Ver o comentario de BOARD_DETECTION_WIDTHS: a deteccao roda numa largura
    reduzida (mais robusta e ~8x mais barata) e os cantos achados sao
    multiplicados pelo fator devolvido. Largura 0 = sem reducao.
    """
    if largura_alvo and gray.shape[1] > int(largura_alvo):
        fator = gray.shape[1] / float(largura_alvo)
        reduzido = cv2.resize(
            gray,
            (int(largura_alvo), max(1, int(round(gray.shape[0] / fator)))),
            interpolation=cv2.INTER_AREA)
    else:
        reduzido = gray
    if BOARD_DETECTION_CLAHE_CLIP > 0:
        reduzido = cv2.createCLAHE(
            clipLimit=BOARD_DETECTION_CLAHE_CLIP,
            tileGridSize=(BOARD_DETECTION_CLAHE_GRID,
                          BOARD_DETECTION_CLAHE_GRID)).apply(reduzido)
    return reduzido, gray.shape[1] / float(reduzido.shape[1])


def grupos_centro_borda(cantos, fracao_borda=0.75):
    """Separa os cantos do tabuleiro em CENTRO e BORDAS pela GEOMETRIA.

    Devolve `(indice_do_canto_central, mascara_das_bordas, raio_de_cada_canto)`,
    onde `raio` e' a distancia do canto ao centroide do tabuleiro.

    Por que por geometria e nao por indice fixo: o resumo do diagnostico (tecla
    T) usava `errors[24]` para "centro" e `errors[:8]`/`errors[-8:]` para
    "bordas" -- indices da placa ANTIGA, de 11x7 = 77 cantos. Com a placa atual
    (5x3 = 15) a tecla T estourava `IndexError` DEPOIS de acertar o tabuleiro e
    medir a profundidade (medido 28/09/2026). Assim o resumo acompanha a placa
    sozinho, como `CHECKERBOARD_SIZE` manda.
    """
    pontos = np.asarray(cantos, np.float64).reshape(-1, 2)
    raios = np.linalg.norm(pontos - pontos.mean(axis=0), axis=1)
    indice_centro = int(np.argmin(raios))
    if float(raios.max()) > 0.0:
        bordas = raios >= float(fracao_borda) * float(raios.max())
    else:                                   # tabuleiro degenerado (1 canto)
        bordas = np.ones(raios.shape, bool)
    return indice_centro, bordas, raios


def simetrias_do_tabuleiro(board_size=None):
    """As 4 simetrias da grade -> permutacao de indices (com o nome de cada uma).

    Espelha `stereo_calibration.symmetry_indices` (mesmos nomes e mesma
    convencao row-major): 'id', 'rot180', 'hflip' e 'vflip'. O sensor de
    tabuleiro e' simetrico, entao a ORDEM que o detector devolve nao diz qual
    canto fisico e' o primeiro -- e cada camera escolhe o seu. Quem resolve isso
    e' `ordem_canonica_do_tabuleiro` (a regra da calibracao) ou, no diagnostico,
    testar as quatro e ficar com a de menor erro (como o solver faz).
    """
    cols, rows = CHECKERBOARD_SIZE if board_size is None else board_size
    cols, rows = int(cols), int(rows)
    total = cols * rows
    rr, cc = np.mgrid[0:rows, 0:cols]
    return {
        "id": np.arange(total),
        "rot180": (total - 1) - np.arange(total),
        "hflip": (rr * cols + (cols - 1 - cc)).reshape(-1),
        "vflip": ((rows - 1 - rr) * cols + cc).reshape(-1),
    }


def ordem_canonica_do_tabuleiro(cantos, board_size=None):
    """Cantos na ORDEM CANONICA da calibracao: `(pontos, nome_da_simetria)`.

    Espelha `stereo_calibration.canonical_order` (mesma regra, mesmo desempate):
    e' a convencao que o solver usa para casar os dois tabuleiros -- o primeiro
    canto e' o canto externo mais proximo do canto superior esquerdo da IMAGEM,
    a primeira linha segue o eixo mais horizontal e as linhas avancam para
    baixo.

    Por que nao comparar as listas CRUAS: cada camera escolhe a ORDEM de deteccao
    por conta propria (o cv2 nao fixa a origem e o padrao e' simetrico). Comparar
    ou projetar as duas listas cruas compara cantos que NAO se correspondem e o
    erro sai na casa das centenas de px mesmo com a calibracao perfeita (medido
    na bancada em 28/09/2026: 'id' 491 px, 'rot180' 494 px e a canonica 494 px,
    com uma homografia NOVA do mesmo par fechando em 0,3 px -- a calibracao
    gravada e' que era de outra montagem).

    Existe aqui, e nao importada de `stereo_calibration`, porque aquele modulo
    importa ESTE -- a duplicacao e' travada por `test_deteccao_cascata.py`, que
    compara as duas implementacoes em tabuleiros aleatorios.
    """
    cols, rows = CHECKERBOARD_SIZE if board_size is None else board_size
    cols, rows = int(cols), int(rows)
    pontos = np.asarray(cantos, np.float64).reshape(-1, 2)
    esperado = cols * rows
    if pontos.shape[0] != esperado:
        raise ValueError(
            "tabuleiro %dx%d cantos esperado (%d); recebidos %d -- calibracao "
            "de outra placa?" % (cols, rows, esperado, pontos.shape[0]))
    simetrias = simetrias_do_tabuleiro((cols, rows))
    melhor_nome, melhor, melhor_penalidade = "id", pontos, None
    for nome, indices in simetrias.items():
        candidato = pontos[indices]
        inicio = candidato[0]
        vetor_linha = candidato[cols - 1] - inicio
        vetor_coluna = candidato[cols] - inicio
        outros = np.array([candidato[cols - 1], candidato[cols], candidato[-1]])
        penalidade = (
            max(0.0, float(outros.sum(axis=1).min()) - float(inicio.sum()))
            + max(0.0, abs(vetor_linha[1]) - abs(vetor_linha[0]))
            + max(0.0, -vetor_coluna[1])
        )
        if melhor_penalidade is None or penalidade < melhor_penalidade - 1e-9:
            melhor_nome, melhor, melhor_penalidade = nome, candidato, penalidade
    return melhor, melhor_nome


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
        # Nome (Windows) da camera para a qual o npz carregado foi GRAVADO: e' o
        # unico jeito de pegar o npz carregado pela camera errada (ver
        # stereo_status).
        self.stereo_camera_name = None
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
            # O que o npz DIZ que foi calibrado (placa e quadrado): guardado
            # para `stereo_status` poder explicar a recusa por extenso.
            "board_size": None, "square_size_m": None, "camera_name": None,
            "stereo_ready": False, "stereo_usable": False,
            "auxiliary_size": None, "kinect_size": None,
            "hand_homography": None, "hand_residuals": None, "hand_rms": None,
            "scale_depth_params": None, "reference_scale": None,
            "node_z_offset": None, "hand_ready": False, "runtime_matrix": None,
            # Tamanho/aviso de escala ficam POR CAMERA (multicamera): o runtime
            # de uma auxiliar nunca pode contaminar a outra.
            "runtime_size": None, "scale_warned": False,
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
                model["board_size"] = saved_board
                model["square_size_m"] = saved_square
                # Nome da camera que GRAVOU o npz (vazio/ausente nos arquivos
                # anteriores a 29/09/2026: nesses nao ha' como conferir).
                gravado = (stereo_data["camera_name"]
                           if "camera_name" in stereo_data.files else None)
                model["camera_name"] = (str(gravado).strip() or None
                                        if gravado is not None else None)
                model["rms"] = (float(stereo_data["rms"])
                                if "rms" in stereo_data.files else None)
                model["stereo_ready"] = (
                    saved_board == CHECKERBOARD_SIZE
                    and saved_square is not None
                    and abs(saved_square - SQUARE_SIZE_M) < 1e-6
                    and model["rms"] is not None
                    and model["rms"] <= MAX_STEREO_RMS_PX
                    # e o arquivo tem de ser DESTA camera (se o npz diz o nome):
                    # geometria de outro par nao fica "pronta" nem com RMS bom.
                    and nome_de_camera_confere(model["camera_name"], aux_index))
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
        self.stereo_camera_name = model["camera_name"]
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

    def camera_name_conflict(self, aux_index=None):
        """Nome gravado no npz quando esse npz NAO e' da camera de agora.

        Devolve o nome da camera para a qual o arquivo foi gravado quando os dois
        nomes (gravado x o daquele indice hoje) sao conhecidos e DIFERENTES;
        None quando nao da' para saber (npz anterior a 29/09/2026, sem comtypes,
        indice fora da lista) ou quando conferem -- nunca inventa conflito.

        Motivo: o indice do OpenCV e' a POSICAO na enumeracao do DirectShow e
        muda quando uma webcam entra/sai do USB. Sem esta checagem, um npz pode
        ser carregado pela camera errada sem sintoma nenhum alem de um erro
        geometrico que o RMS do proprio arquivo nao ve'.
        """
        index = self.active_aux if aux_index is None else int(aux_index)
        model = None if index is None else self.aux_models.get(int(index))
        gravado = None if model is None else model.get("camera_name")
        if nome_de_camera_confere(gravado, index):
            return None
        return gravado

    def stereo_status(self, aux_index=None):
        """(estado, motivo) do stereo de uma auxiliar -- com o motivo POR EXTENSO.

        Motivo de existir: a mensagem antiga so' dizia "RMS alto", e desde a
        troca do tabuleiro (placa nova = 6x4 quadrados de 57 mm; os npz de
        aux1/aux2 foram gravados com a placa de 52 mm) a recusa NAO e' de
        qualidade -- e' de COMPATIBILIDADE com a placa. Sem separar as duas
        causas, o dia da recalibracao comeca com um diagnostico enganoso.
        `aux_index` None consulta a camera ATIVA.
        """
        index = self.active_aux if aux_index is None else int(aux_index)
        model = None if index is None else self.aux_models.get(int(index))
        if model is None or model.get("matrix") is None:
            return "sem calibracao", "arquivo stereo ausente ou sem intrinsecas"
        # DE QUAL CAMERA E' ESTE ARQUIVO? O npz novo grava o nome do Windows da
        # camera calibrada (ver camera_name_conflict). Nome desconhecido de um
        # dos lados (npz antigo, sem comtypes, indice fora da lista) = sem
        # checagem, como antes.
        conflito = self.camera_name_conflict(index)
        if conflito:
            return ("rejeitada",
                    "gravado para a camera '%s', mas o indice %d agora e' '%s' "
                    "-> confira o mapa (tools\\diagnostico_cameras.py) e "
                    "RECALIBRE se o par mudou"
                    % (conflito, index, nome_da_camera_do_indice(index)))
        board = model.get("board_size")
        square = model.get("square_size_m")
        rms = model.get("rms")
        if board is not None and tuple(board) != tuple(CHECKERBOARD_SIZE):
            return ("rejeitada",
                    "tabuleiro gravado %dx%d != %dx%d cantos da placa atual "
                    "-> RECALIBRE" % (board[0], board[1], CHECKERBOARD_SIZE[0],
                                      CHECKERBOARD_SIZE[1]))
        if square is not None and abs(float(square) - SQUARE_SIZE_M) > 1e-6:
            erro = abs(float(square) - SQUARE_SIZE_M) / SQUARE_SIZE_M * 100.0
            return ("rejeitada",
                    "quadrado gravado %g mm != %g mm (escala ~%.0f%% errada) "
                    "-> RECALIBRE" % (float(square) * 1000.0,
                                      SQUARE_SIZE_M * 1000.0, erro))
        if rms is None:
            return "rejeitada", "npz sem RMS -> RECALIBRE"
        if model.get("stereo_ready"):
            return "aceita", "RMS %.2f px <= %g px" % (rms, MAX_STEREO_RMS_PX)
        if model.get("stereo_usable"):
            return ("aproximada",
                    "RMS %.2f px entre %g e %g px (recalibre para < %g px)"
                    % (rms, MAX_STEREO_RMS_PX, STEREO_OVERLAY_MAX_RMS_PX,
                       MAX_STEREO_RMS_PX))
        return ("rejeitada",
                "RMS %.2f px > %g px -> RECALIBRE"
                % (rms, STEREO_OVERLAY_MAX_RMS_PX))

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
        # DESTINO por camera (ver aux_hand_output_file): gravar sempre no nome
        # legado fazia a calibracao de uma auxiliar != 0 nunca ser carregada.
        destino = aux_hand_output_file(
            self.active_aux if self.active_aux is not None else 0)
        np.savez(destino, **save_dict)
        if verbose:
            print(f"  calibracao da mao salva em {destino}")
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
        """Cantos internos do tabuleiro (N,2) NA ESCALA DO FRAME, ou None.

        Caminho ROBUSTO de proposito (ver BOARD_DETECTION_WIDTHS): CLAHE +
        cascata de larguras (SB EXHAUSTIVE, com o detector classico como
        fallback na mesma escala) + refinamento na resolucao cheia. Antes rodava
        so' na resolucao cheia, sem CLAHE e com CALIB_CB_ACCURACY, e respondia
        "Diagnostico: tabuleiro nao visto nas duas cameras" com a placa na mesa
        e as tres cameras abertas (medido 28/09/2026, tecla T).
        """
        if frame is None:
            return None
        gray_cheio = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                      if frame.ndim == 3 else frame)
        for largura in BOARD_DETECTION_WIDTHS:
            gray, fator = _cinza_para_detectar_tabuleiro(gray_cheio, largura)
            achou, cantos = False, None
            if hasattr(cv2, "findChessboardCornersSB"):
                achou, cantos = cv2.findChessboardCornersSB(
                    gray,
                    CHECKERBOARD_SIZE,
                    cv2.CALIB_CB_EXHAUSTIVE,
                )
            if not achou:
                achou, cantos = cv2.findChessboardCorners(
                    gray,
                    CHECKERBOARD_SIZE,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
            if not achou:
                continue
            pontos = cantos.reshape(-1, 2).astype(np.float64) * fator
            try:
                # Refina na resolucao CHEIA: a deteccao reduzida sozinha deixa
                # ~1-2 px de erro (era o que estourava o RMS do stereo).
                pontos = cv2.cornerSubPix(
                    gray_cheio,
                    pontos.astype(np.float32).reshape(-1, 1, 2),
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                     0.001),
                ).reshape(-1, 2).astype(np.float64)
            except cv2.error:               # mantem os cantos da deteccao
                pass
            return pontos
        return None

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
        # Guarda as duas listas na MESMA ordem canonica que o solver usa (o cv2
        # nao fixa a origem da ordem de deteccao e o padrao e' simetrico): um
        # `findHomography` com os pares trocados sai errado e iria direto para o
        # arquivo de calibracao.
        auxiliary_corners, _ = ordem_canonica_do_tabuleiro(auxiliary_corners)
        kinect_corners, _ = ordem_canonica_do_tabuleiro(kinect_corners)
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

    def aux_view(self, aux_index):
        """Vista IMUTAVEL de UMA camera auxiliar (usar VARIAS no mesmo frame).

        Ver `AuxCameraView`: com duas auxiliares (uma de cada lado) o overlay
        precisa das DUAS calibracoes ao mesmo tempo, sem mexer na camera ativa.
        Devolve None se o indice nao estiver registrado.
        """
        model = self.aux_models.get(int(aux_index))
        if model is None:
            return None
        return AuxCameraView(
            index=int(aux_index),
            kinect_matrix=model["kinect_matrix"],
            kinect_dist=model["kinect_dist"],
            matrix=model["matrix"],
            dist=model["dist"],
            rotation=model["rotation"],
            translation=model["translation"],
            rms=model["rms"],
            stereo_usable=bool(model["stereo_usable"]),
            runtime_matrix=model.get("runtime_matrix"),
            runtime_size=model.get("runtime_size"),
            calibrated_size=model["auxiliary_size"],
        )

    def active_view(self):
        """Vista da camera ATIVA, montada a partir dos CAMPOS VIVOS.

        Diferenca que importa: `aux_view(indice)` devolve a calibracao GRAVADA
        daquela camera (o que o multicamera precisa para as duas auxiliares ao
        mesmo tempo), enquanto esta vista usa os campos vivos do alignment
        (`auxiliary_matrix`, `stereo_rotation`, ...). Isso preserva o contrato
        historico de mano unica: quem sobrescreve os campos vivos (os testes
        sinteticos fazem isso) continua mandando no overlay da camera ativa.
        """
        if self.active_aux is None:
            return None
        model = self.aux_models.get(self.active_aux) or {}
        return AuxCameraView(
            index=int(self.active_aux),
            kinect_matrix=self.kinect_matrix,
            kinect_dist=self.kinect_dist,
            matrix=self.auxiliary_matrix,
            dist=self.auxiliary_dist,
            rotation=self.stereo_rotation,
            translation=self.stereo_translation,
            rms=self.stereo_rms,
            stereo_usable=bool(self.stereo_usable),
            runtime_matrix=self.auxiliary_matrix_runtime,
            runtime_size=model.get("runtime_size"),
            calibrated_size=(self.calibrated_auxiliary_size
                             or (self.auxiliary_width, self.auxiliary_height)),
        )

    def auxiliary_to_kinect_stereo(self, points, kinect_depth_m):
        """Projeta pixels da AUXILIAR ATIVA no RGB do Kinect (stereo 3D).

        Delega para `AuxCameraView.to_kinect_stereo` (a matematica vive la', para
        poder ser usada com qualquer camera, ativa ou nao). Sem stereo usavel cai
        na homografia 2D do tabuleiro -- comportamento historico preservado.
        """
        view = self.active_view()
        projected = None
        if view is not None and view.usable:
            projected = view.to_kinect_stereo(points, kinect_depth_m)
        if projected is None:
            return self.auxiliary_to_kinect(points)
        return projected

    def triangulate_hand_points(self, aux_pixels, kinect_pixels):
        """Triangula nos 3D (m, frame do Kinect) a partir das duas vistas.

        Delega para `AuxCameraView.triangulate_with_kinect` da camera ATIVA.
        Convencao: X_aux = R X_kin + T; camera 1 = Kinect (na origem,
        P1=[I|0]); camera 2 = auxiliar (P2=[R|T]). Correspondencia trivial pelo
        indice do no MediaPipe.

        Retorna (points_3d (N,3), reproj_px (N,)) ou (None, None) se a
        triangulacao nao for possivel/confiavel.
        """
        view = self.active_view()
        if view is None:
            return None, None
        return view.triangulate_with_kinect(aux_pixels, kinect_pixels)


    # (A triangulacao vive em AuxCameraView.triangulate_with_kinect; o metodo
    # equivalente de CameraAlignment esta' ACIMA e delega para la' -- este bloco
    # duplicado foi removido para nao voltar a divergir.)

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
        model = (self.aux_models.get(self.active_aux)
                 if self.active_aux is not None else None)
        # A checagem e' POR CAMERA (antes era pelo tamanho GLOBAL, "da ultima"):
        # com duas auxiliares do MESMO tamanho, a segunda nunca guardava o seu
        # `runtime_size` -- e a conversao normalizado -> pixel passava a usar o
        # tamanho/matriz da outra camera.
        if model is not None \
                and tuple(model.get("runtime_size") or ()) == (width, height):
            self.auxiliary_height, self.auxiliary_width = height, width
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
                # O aviso e' POR CAMERA (no multicamera cada uma tem a sua
                # resolucao) e sai uma vez so'.
                avisado = bool(model.get("scale_warned")) if model else False
                if not avisado:
                    if model is not None:
                        model["scale_warned"] = True
                    self.auxiliary_scale_warned = True
                    print(
                        f"AVISO: camera auxiliar {self.active_aux} em "
                        f"{width}x{height}, mas a calibracao foi feita em "
                        f"{cal_width}x{cal_height}; usando matriz intrinseca "
                        "escalada. Recalibre na mesma resolucao para melhor "
                        "precisao.", flush=True
                    )
        # persistencia da matriz em escala por camera (troca de aux mantem).
        # O TAMANHO tambem fica por camera: com duas auxiliares no mesmo frame e'
        # preciso converter normalizado -> pixel com o tamanho de CADA uma (ver
        # AuxCameraView.pixels/triangulate_aux_pairs).
        if self.active_aux in self.aux_models:
            self.aux_models[self.active_aux]["runtime_matrix"] = runtime
            self.aux_models[self.active_aux]["runtime_size"] = (width, height)


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


def mascara_dentro_do_frame(pixels, largura, altura,
                            margem=OVERLAY_FRAME_MARGIN_PX):
    """Mascara dos nos cujo pixel cai (com folga `margem`) dentro do frame.

    Porta de validez do overlay: projecao longe do frame = profundidade invalida
    ou mao fora do volume calibrado. Sem isso o esqueleto projetado vira "lixo"
    de milhares de px e contamina o log de desvio.
    """
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    return (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= -margem) & (pixels[:, 0] < largura + margem)
        & (pixels[:, 1] >= -margem) & (pixels[:, 1] < altura + margem)
    )


def desenhar_esqueleto_projetado(frame, pixels, cor_nos=(0, 0, 255),
                                 cor_linhas=(0, 0, 220), rotulo=None,
                                 raio=4, espessura=2):
    """Desenha nos + conexoes de UMA mao projetada no frame BGR do Kinect.

    FONTE UNICA do tracado: a mao do Kinect, cada webcam auxiliar e o par de
    auxiliares usam a MESMA tabela HAND_CONNECTIONS e o mesmo gating de borda
    (`mascara_dentro_do_frame`). Antes o laco de desenho existia dentro de um
    metodo unico e cada camera nova teria de copiar o bloco.

    Devolve a mascara (N,) dos nos elegiveis (ja' usada por quem mede desvio).
    """
    altura, largura = frame.shape[:2]
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    dentro = mascara_dentro_do_frame(pixels, largura, altura)
    for pixel, ok in zip(pixels, dentro):
        if not ok:
            continue
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        if 0 <= x < largura and 0 <= y < altura:
            cv2.circle(frame, (x, y), raio, cor_nos, -1)
    for start, end in HAND_CONNECTIONS:
        if not (dentro[start] and dentro[end]):
            continue
        start_x, start_y = map(int, np.round(pixels[start]))
        end_x, end_y = map(int, np.round(pixels[end]))
        if all((0 <= start_x < largura, 0 <= start_y < altura,
                0 <= end_x < largura, 0 <= end_y < altura)):
            cv2.line(frame, (start_x, start_y), (end_x, end_y), cor_linhas,
                     espessura)
    if rotulo is not None and dentro.any():
        # Rotulo junto da palma (no 9), para o experimentador saber de QUE
        # camera e' cada esqueleto sem consultar o terminal.
        ancora = pixels[9] if dentro.size > 9 else pixels[int(np.argmax(dentro))]
        x, y = int(round(ancora[0])), int(round(ancora[1]))
        if 0 <= x < largura and 0 <= y < altura:
            cv2.putText(frame, rotulo, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_linhas, 2)
    return dentro
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
        # Contadores de diagnostico do frame (log do vetor da palma, item 111).
        self.diag = {"mapeamento": 0, "depth": 0, "pontos": 0,
                     "fallbacks": 0, "resgatados": 0, "profundidades": []}
        if self.kinect.has_new_color_frame():
            data = self.kinect.get_last_color_frame()
            self.last_color = data.reshape((self.color_height, self.color_width, 4))
            self._last_color_capture_time = getattr(
                self.kinect, "_last_color_frame_time", 0)
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
        # SINCRONIA color x depth (item 105): os dois streams NAO sao
        # amarrados; quando o par vem de instantes diferentes o ponto 3D do
        # landmark cai no lugar errado e a normal cambaleia. Exponho o delta
        # em ms para o tool recusar o par (None = desconhecido/absurdo).
        tempo_color = float(getattr(self, "_last_color_capture_time", 0) or 0)
        tempo_depth = float(getattr(self, "_last_depth_capture_time", 0) or 0)
        if tempo_color and tempo_depth and abs(tempo_color - tempo_depth) < 5.0:
            self.sync_color_depth_ms = abs(tempo_color - tempo_depth) * 1000.0
        else:
            self.sync_color_depth_ms = None
        if self.kinect.has_new_body_frame():
            self.last_bodies = self.kinect.get_last_body_frame()
        # Maos do esqueleto 3D (lado geometrico + prior de profundidade, item 113)
        self._pontos_mao_corpo = self._body_hand_camera_points()
        self._z_por_lado = {
            lado: float(ponto[2])
            for lado, ponto in self._pontos_mao_corpo.items()}
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
        """Pixel de depth para um pixel de cor, com FALLBACK em vizinhos.

        Por que o fallback (item 111): nos testes de 22/10 apareciam
        "mapeo color->depth falhou para a palma (1118, 266)" e o vetor verde
        SUMIA. Um unico pixel mapeado para fora do FOV de depth (mao alta/perto
        da borda) derrubava a normal inteira. Se o pixel exato falha, procura em
        aneis crescentes (1..8 px) um vizinho com mapeamento valido -- o
        landmark do MediaPipe tem 1-2 px de incerteza, entao isso e' fisicamente
        razoavel e recupera a maioria dos frames.
        """
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
        except BaseException as exc:
            if not self._map_error_reported:
                self._map_error_reported = True
                print(
                    f"ERRO mapeo color->depth ({type(exc).__name__}): {exc}\n"
                    "  (relativo ao caminho MapColorFrameToDepthSpace; pode ser "
                    "rigido de FOV/posicion do depth ou del proprio mapper)"
                )
            return None

        def _ler(px, py):
            if not (0 <= px < self.color_width and 0 <= py < self.color_height):
                return None
            ponto = self._color_to_depth_output[py * self.color_width + px]
            try:
                if not math.isfinite(float(ponto.x)) or not math.isfinite(float(ponto.y)):
                    return None
            except (ValueError, TypeError):
                return None
            dx, dy = int(round(ponto.x)), int(round(ponto.y))
            if 0 <= dx < self.depth_width and 0 <= dy < self.depth_height:
                return dx, dy
            return None

        achado = _ler(color_x, color_y)
        if achado is not None:
            return achado
        # FALLBACK: aneis crescentes em volta do pixel original
        for raio in (1, 2, 3, 4, 6, 8):
            for dy in range(-raio, raio + 1):
                for dx in range(-raio, raio + 1):
                    if max(abs(dx), abs(dy)) != raio:
                        continue
                    achado = _ler(color_x + dx, color_y + dy)
                    if achado is not None:
                        try:
                            self.diag["fallbacks"] = self.diag.get("fallbacks", 0) + 1
                        except (AttributeError, TypeError):
                            pass
                        return achado
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
        mediana = float(np.median(valid))
        # CLUSTER em volta da mediana: na borda dos dedos o patch 11x11
        # mistura mao (0,5-1,5 m) com FUNDO (2-4 m) e a mediana pulava entre
        # os dois -> ponto 3D saltava -> normal cambaleava (item 105).
        # Mantem so' o aglomerado de profundidade em volta da mediana.
        cluster = valid[np.abs(valid - mediana) <= 30.0]
        if cluster.size >= 3:
            return float(np.median(cluster))
        return mediana if math.isfinite(mediana) else None

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
            #: Mascara ABAIXO_DO_TRONCO: a MESA bloqueia a vista do Kinect ali e
            #: o SDK infere perna/pe ATRAVES da tampo -- desenhar isso seria
            #: linha lixo atravessando a mesa (pedido da bancada 09/10/2026).
            y_base = self._tronco_base_y(body.joints)
            points = {}
            for joint_type in range(PyKinectV2.JointType_Count):
                joint = body.joints[joint_type]
                if joint.TrackingState == PyKinectV2.TrackingState_NotTracked:
                    continue
                if self._abaixo_do_tronco(joint.Position.y, y_base):
                    continue
                point = self.mapper.MapCameraPointToDepthSpace(joint.Position)
                if point is None:
                    continue
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

    def _landmark_camera_point(self, landmark, lado=None):
        """Ponto 3D do landmark no camera space, com RESGATE pelo esqueleto.

        `lado` escolhe o prior de profundidade (`self._z_por_lado`, item 113):
        se o depth do landmark cair no fundo/for invalido, usa o z da mao do
        ESQUELETO como profundidade plausivel, em vez de descartar o ponto (que
        derrubava a normal inteira e a seta sumia). Sem `lado`, usa a EMA da
        ultima profundidade da palma como referencia.
        """
        color_x = int(np.clip(landmark.x * self.color_width, 0, self.color_width - 1))
        color_y = int(np.clip(landmark.y * self.color_height, 0, self.color_height - 1))
        depth_pixel = self._color_to_depth_pixel(color_x, color_y)
        if depth_pixel is None:
            try:
                self.diag["mapeamento"] = self.diag.get("mapeamento", 0) + 1
            except (AttributeError, TypeError):
                pass
            return None
        # Parche mas grande que el default (11x11): los dedos son contornos
        # finos y el depth del Kinect es ruidoso ahi; un parche mayor estabiliza
        # la profundidad por landmark (menos huecos/fondo).
        depth_mm = self._median_depth(*depth_pixel, radius=5)
        z_priors = getattr(self, "_z_por_lado", None) or {}
        prior = None
        if lado in z_priors:
            prior = z_priors.get(lado)
        referencia = (prior if prior is not None
                      else getattr(self, "last_hand_depth_m", None))
        if depth_mm is None:
            try:
                self.diag["depth"] = self.diag.get("depth", 0) + 1
            except (AttributeError, TypeError):
                pass
        resgate = resgatar_profundidade(depth_mm, list(z_priors.values()),
                                        referencia_m=referencia)
        if resgate is not None:
            depth_mm = resgate
            try:
                self.diag["resgatados"] = (self.diag.get("resgatados", 0) + 1)
            except (AttributeError, TypeError):
                pass
        if depth_mm is None:
            return None
        try:
            self.diag["pontos"] = self.diag.get("pontos", 0) + 1
            self.diag["profundidades"].append(float(depth_mm))
        except (AttributeError, TypeError):
            pass
        depth_point = PyKinectV2._DepthSpacePoint()
        depth_point.x, depth_point.y = map(float, depth_pixel)
        camera_point = self.mapper.MapDepthPointToCameraSpace(
            depth_point, int(round(depth_mm))
        )
        if camera_point is None:
            return None
        try:
            xyz = np.array([camera_point.x, camera_point.y, camera_point.z],
                           dtype=float)
        except (AttributeError, TypeError, ValueError):
            return None
        #: O mapper devolve inf/NaN para profundidades/pixels invalidos (ex.:
        #: mao no limite superior do FOV, item 121). Ponto nao-finito aqui
        #: virava inf na projecao p/ cor e estourava int(round(inf)) no desenho.
        if not np.isfinite(xyz).all() or xyz[2] <= 0.05:
            return None
        return xyz

    @staticmethod
    def palm_normal_from_points(wrist, index_mcp, pinky_mcp, eps=1e-9,
                                lado=None):
        """Normal unitaria do plano da palma a partir de 3 pontos 3D (m).

        COM `lado` ('right'/'left'): orientacao ANATOMICA -- aponta para FORA da
        PALMA em QUALQUER pose (item 79 do historico). Medido na bancada
        (19/10/2026, sessao 16:07:43, com os pontos 3D crus): nesta camera o
        produto vetorial `(index_mcp-punho) x (pinky_mcp-punho)` sai do lado da
        PALMA na mao DIREITA e do lado do DORSO na ESQUERDA (as maos sao
        espelhadas), logo o sinal e' +1 para 'right' e -1 para 'left'.
        Validacao fisica INDEPENDENTE (nao depende de saber a montagem da
        camera): com esse sinal as normais das DUAS maos ficam a 11 graus na
        fase MESA (as duas palmas para baixo -> normais iguais), a 28-30 graus
        em PREPARO/KINECT (as duas palmas para o sensor) e a 153 graus em
        REZANDO (palmas uma contra a outra -> normais opostas) -- exatamente o
        que a pose exige. Com o sinal antigo (nz<0 forcado) davam 169/151/27
        graus, ou seja, INVERTIDAS.

        SEM `lado` (None): comportamento ANTIGO (sinal forcado para nz < 0), que
        so' e' a palma na pose "palma virada para o Kinect"; mantido para nao
        mudar os chamadores antigos (ex.: o log de angulos do paradigma).
        """
        normal = np.cross(
            np.asarray(index_mcp, np.float64) - np.asarray(wrist, np.float64),
            np.asarray(pinky_mcp, np.float64) - np.asarray(wrist, np.float64),
        )
        length = float(np.linalg.norm(normal))
        if length < eps:
            return None
        normal = normal / length
        if lado == "right":
            return normal
        if lado == "left":
            return -normal
        if normal[2] > 0:
            normal = -normal
        return normal

    def palm_normal_camera(self, landmarks, lado=None):
        """Normal da palma medida pelo Kinect (um lookup de profundidade por
        landmark; mais caro que a versao sobre a triangulacao).

        COM `lado` a normal e' ANATOMICA (palma para fora em qualquer pose --
        e' a verdade do diagnostico e da calibracao da luva); SEM `lado` mantem
        o comportamento antigo (sinal forcado para a camera).
        """
        wrist = self._landmark_camera_point(landmarks[0], lado)
        index_mcp = self._landmark_camera_point(landmarks[5], lado)
        pinky_mcp = self._landmark_camera_point(landmarks[17], lado)
        if wrist is None or index_mcp is None or pinky_mcp is None:
            return None
        return self.palm_normal_from_points(wrist, index_mcp, pinky_mcp,
                                            lado=lado)

    def palm_orientation_camera(self, landmarks, lado=None):
        """(normal, direcao_dedos) da palma pelo depth do Kinect.

        COM `lado` a normal e' ANATOMICA (ver `palm_normal_from_points`).

        Quatro lookups de profundidade (punho 0, MCP medio 9, MCP indicador 5,
        MCP minimo 17). Devolve (None, None) se os tres pontos do plano
        falharem; a direcao dos dedos pode vir None isoladamente.
        """
        wrist = self._landmark_camera_point(landmarks[0], lado)
        if wrist is None:
            return None, None
        index_mcp = self._landmark_camera_point(landmarks[5], lado)
        pinky_mcp = self._landmark_camera_point(landmarks[17], lado)
        middle_mcp = self._landmark_camera_point(landmarks[9], lado)
        return palm_frame(wrist, index_mcp, pinky_mcp, middle_mcp, lado=lado)

    def palm_plane_camera(self, landmarks, lado=None, normal_anterior=None,
                          rms_max_m=0.030):
        """(normal, rms_m) do PLANO MEDIO do dorso pelo depth do Kinect.

        Usa 5 lookups de profundidade (punho + MCPs indicador/medio/anelar/
        minimo) e `palm_normal_from_plane` -- a verdade mais ESTAVEL para a
        calibracao da palma (item 102: tecla J do tool). Devolve (None, None)
        se os pontos nao fecharem um plano confiavel (borda/oclusao/ruido) --
        nunca uma normal inventada.
        """
        pontos = {}
        for nome, indice in PONTOS_DORSO_PALMA.items():
            try:
                ponto = self._landmark_camera_point(landmarks[indice], lado)
            except (AttributeError, IndexError, TypeError):
                ponto = None
            if ponto is None:
                continue
            ponto = np.asarray(ponto, np.float64).reshape(3)
            if np.isfinite(ponto).all():
                pontos[nome] = ponto
        return palm_normal_from_plane(pontos, lado=lado,
                                       normal_anterior=normal_anterior,
                                       rms_max_m=rms_max_m)

    @staticmethod
    def _tronco_base_y(joints):
        """y (camera space, Y pra cima) do fim do tronco (SpineBase) ou None.

        Referencia da mascara ABAIXO_DO_TRONCO: sem SpineBase valido (corpo sem
        tronco tracked, ou `joints` sem aquela junta) devolve None e a mascara
        nao descarta nada (melhor nao mexer do que descartar certo).
        """
        try:
            joint = joints[PyKinectV2.JointType_SpineBase]
        except (IndexError, KeyError, TypeError):
            return None
        not_tracked = getattr(PyKinectV2, "TrackingState_NotTracked", 0)
        if getattr(joint, "TrackingState", not_tracked) == not_tracked:
            return None
        try:
            y = float(joint.Position.y)
        except (AttributeError, TypeError, ValueError):
            return None
        return y if np.isfinite(y) else None

    @staticmethod
    def _abaixo_do_tronco(y, y_base):
        """True quando a junta esta' claramente ABAIXO do fim do tronco.

        `y`/`y_base` em camera space (Y pra cima); ver MASCARA_ABAIXO_TRONCO_M.
        """
        if y_base is None or not MASCARA_ABAIXO_TRONCO_M:
            return False
        try:
            y = float(y)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(y) and y < y_base - MASCARA_ABAIXO_TRONCO_M)

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
        #: Mascara ABAIXO_DO_TRONCO: joelho/pé/etc inferidos ATRAVES da mesa
        #: nao servem nem pro modelo do braco (punho na mesa fica: a margem e'
        #: MASCARA_ABAIXO_TRONCO_M).
        if KinectHandTracker._abaixo_do_tronco(
                y, KinectHandTracker._tronco_base_y(joints)):
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

    def medir_plano_mesa(self, passo=6, limiar_m=0.02):
        """Plano da MESA no referencial da camera (item 115).

        Amostra o depth em grade, converte para camera space e ajusta o plano
        dominante (RANSAC). Devolve dict de `ajustar_plano_mesa` ou None.
        """
        if self.last_depth is None:
            return None
        passo = max(2, int(passo))
        pontos = []
        for py in range(0, self.depth_height, passo):
            for px in range(0, self.depth_width, passo):
                try:
                    mm = int(self.last_depth[py, px])
                except (IndexError, TypeError, ValueError):
                    continue
                if not 350 < mm < 2500:
                    continue
                ponto = PyKinectV2._DepthSpacePoint()
                ponto.x, ponto.y = float(px), float(py)
                try:
                    cam = self.mapper.MapDepthPointToCameraSpace(ponto, mm)
                except Exception:
                    continue
                xyz = (float(cam.x), float(cam.y), float(cam.z))
                if np.isfinite(xyz).all() and 0.35 < xyz[2] < 2.5:
                    pontos.append(xyz)
        return ajustar_plano_mesa(pontos, limiar_m=limiar_m)

    def _body_hand_camera_points(self):
        """{"left": (3,), "right": (3,)} das maos do esqueleto 3D (m).

        Usado para (a) decidir o LADO de cada mao do Kinect pela geometria
        (o rotulo do MediaPipe pisca e o sinal anatomico e espelhado por lado:
        lado errado = seta verde invertida 180 graus, item 113) e (b) prior de
        profundidade para resgatar landmarks cujo depth caiu no fundo.
        """
        pontos = {}
        if getattr(self, "last_bodies", None) is None:
            return pontos
        for body in self.last_bodies.bodies:
            if not body.is_tracked:
                continue
            joints = getattr(body, "joints", None)
            if joints is None:
                continue
            for lado, juntas in (
                    ("right", (PyKinectV2.JointType_WristRight,
                               PyKinectV2.JointType_HandRight)),
                    ("left", (PyKinectV2.JointType_WristLeft,
                              PyKinectV2.JointType_HandLeft))):
                for tipo in juntas:
                    try:
                        junta = joints[tipo]
                    except (IndexError, KeyError, TypeError):
                        continue
                    if (junta.TrackingState
                            == PyKinectV2.TrackingState_NotTracked):
                        continue
                    pos = junta.Position
                    xyz = np.array([float(pos.x), float(pos.y), float(pos.z)])
                    if not np.isfinite(xyz).all() or not 0.3 < xyz[2] < 4.0:
                        continue
                    pontos[lado] = xyz
                    break
            if pontos:
                break
        return pontos

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
            #: Linha do fim do tronco: punho/HandTip ABAIXO dela (mais de
            #: MASCARA_ABAIXO_TRONCO_M) esta' atras da MESA -- o SDK infere
            #: atraves da tampo e a "medida" e' fantasma; usa-la corromperia a
            #: profundidade (inclusive o EMA last_hand_depth_m).
            y_base = self._tronco_base_y(joints)
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
                if self._abaixo_do_tronco(y, y_base):
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
    def _palm_pixel(result, index, tamanho):
        """(x, y) em px da PALMA (no' 9) da deteccao `index`, ou None."""
        try:
            palma = result.hand_landmarks[index][9]
            return (float(palma.x) * float(tamanho[0]),
                    float(palma.y) * float(tamanho[1]))
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    @staticmethod
    def _nearest_hand(result, ancora_px, tamanho, raio_px, indices=None):
        """Indice da deteccao cuja PALMA esta' mais perto da ancora (px).

        Devolve None quando a mais perto esta' FORA de `raio_px` -- mao que saiu
        do quadro e voltou longe e' outra cena, e prender o rastreamento nela
        seria pior do que reescolher pelo rotulo.
        """
        melhor, melhor_dist = None, float(raio_px)
        for index in (range(len(result.hand_landmarks))
                      if indices is None else indices):
            pixel = KinectHandTracker._palm_pixel(result, index, tamanho)
            if pixel is None:
                continue
            dist = float(np.hypot(pixel[0] - ancora_px[0],
                                  pixel[1] - ancora_px[1]))
            if dist <= melhor_dist:
                melhor, melhor_dist = index, dist
        return melhor

    @staticmethod
    def _pick_hand_index(result, want_side=None, flip=False,
                         ancora_px=None, tamanho=None, raio_px=0.0):
        """Indice da mao a rastrear (None se nao houver candidata).

        Sem `want_side`, escolhe a mao mais confiante (o MediaPipe devolve a
        lista ordenada por escore), que era o comportamento anterior.

        `ancora_px` + `tamanho` (largura, altura do RGB) ativam a CONTINUIDADE:
        quando a deteccao da ancora ainda esta' no quadro (dentro de `raio_px`),
        e' ELA que vale, mesmo que o rotulo dela agora diga o lado oposto. O
        rotulo do MediaPipe e' uma classificacao de APARENCIA do quadro: com as
        duas maos visiveis ele oscila (palma virada, oclusao, maos que se
        cruzam) e, como a escolha era refeita a cada quadro, o rastreamento
        trocava de mao NO MEIO DA TRIAL (relato de 21/10/2026). A continuidade
        so' vale DENTRO do raio: mao perdida/reaparecida longe volta a ser
        escolhida pelo rotulo.
        """
        total = len(getattr(result, "hand_landmarks", None) or [])
        if total == 0:
            return None
        if want_side not in ("left", "right"):
            return 0
        if ancora_px is not None and tamanho is not None and raio_px > 0:
            index = KinectHandTracker._nearest_hand(result, ancora_px,
                                                    tamanho, raio_px)
            if index is not None:
                return index
        for index in range(total):
            if KinectHandTracker._handedness_of(result, index,
                                                flip) == want_side:
                return index
        return None

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
                               want_side=None, flip=None):
        """Detecta maos na webcam auxiliar.

        `aux_index` None usa a camera primaria (compatibilidade com chamadas
        legadas); com indice usa o detector/timestamp daquela camera.
        `want_side` ('left'/'right') coloca a mao daquele lado em [0] e
        devolve [] se o detector viu somente a outra mao.

        `flip` None usa a inversao APRENDIDA daquela camera
        (`aux_handedness_flip`): o pipeline espelha a auxiliar, mas com duas
        webcams de modelos diferentes o driver de uma delas pode ja' entregar o
        fluxo espelhado, e entao SO' aquela camera sai com o rotulo trocado.
        Antes daqui sair `False` fixo, o rotulo invertido de UMA camera trocava
        as maos nela e nada no programa percebia.
        """
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
            if flip is None:
                flip = aux_handedness_flip(index)
            return self._order_hands(result, want_side, flip)
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
            self.last_draw_recusa = "sem stereo/hand_ready"
            return None
        first_deviation = None
        first_red = None
        first_green = None
        #: Motivo da recusa do ultimo draw (diagnostico de oclusao 14/10: o
        #: tool imprimia "SEM vermelho" mas nao sabia SE o draw recusou ou nem
        #: foi chamado -- o usuario leu aux_hand=sim/3D=sim e o vermelho mesmo
        #: assim nao saiu). O tool le via getattr; None = desenhou.
        recusa = None
        finitos3 = -1
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
                finitos3 = int(finite3.sum())
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
                recusa = (f"sem pixels (3D "
                          + (f"finitos={finitos3}" if finitos3 >= 0
                             else "ausente")
                          + f"; depth_frame="
                          f"{'sim' if self.last_depth is not None else 'NAO'})")
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
            in_frame = (finite & mascara_dentro_do_frame(
                kinect_pixels, self.color_width, self.color_height))
            if in_frame.sum() < 5:
                recusa = (f"projecao fora do quadro "
                          f"({int(in_frame.sum())} de 21 dentro)")
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
            # (16/10: este registro estava DENTRO do `if ... last_kinect_hands`
            #  -- com o Kinect cego o bloco pulava, first_red nunca era setado,
            #  a funcao devolvia None e o overlay_deviation_log PERDIA as frames
            #  cegas; medido no overlay_diag.log: recusa_draw="nenhuma mao
            #  desenhavel" com 3D/aux_hand presentes. Fora do bloco, como o
            #  proprio comentario promete.)
            if first_red is None:
                red_copy = kinect_pixels.copy()
                red_copy[~in_frame] = np.nan
                first_red = red_copy
            # Tracado: fonte unica (nos + conexoes + gating de borda).
            desenhar_esqueleto_projetado(frame, kinect_pixels)
        if first_red is None:
            self.last_draw_recusa = recusa or (
                "nenhuma mao desenhavel" if auxiliary_hands
                else "lista de maos vazia")
            return None
        self.last_draw_recusa = None
        if first_deviation is None:
            # Sem mao verde (Kinect cego): desvio indefinido, mas a trajetoria
            # vermelha segue disponivel para o log de diagnostico.
            return (None, None, None, first_red, None)
        return (*first_deviation, first_red, first_green)
#: Cores (BGR) por POSICAO da webcam auxiliar na lista de cameras. O
    #: esqueleto do Kinect e' verde (desenhado em detect_hand/detect_hands_all);
    #: a primeira auxiliar fica vermelha -- a mesma cor do overlay de um par so'.
    CORES_AUXILIARES = ((0, 0, 255), (255, 0, 255), (0, 165, 255),
                        (255, 255, 0))

    def _projetar_mao_no_kinect(self, view, hand_landmarks, kinect_depth_m=None,
                                triangulado=None, trim=None):
        """(21,2) px do Kinect para a mao de UMA auxiliar (ou None).

        Prioridade:
        0) TRIANGULACAO 3D ja' calculada (Kinect+aux, ou o PAR de auxiliares
           quando o Kinect esta' cego) -- e' medida, nao estimativa;
        1) stereo 3D com UMA profundidade rigida para a mao (regra estavel: a
           profundidade por no espalha os dedos, porque o modelo de escala
           aparente tem ~12 cm de residuo).

        SEM nenhuma das duas (mao perdida: nem 3D nem profundidade daquele lado)
        devolve None -- nao desenha. Uma camera so' da' o RAIO onde a mao esta';
        a distancia ao longo do raio nao e' observavel, e projetar com a
        profundidade NOMINAL (1,0 m) jogava o esqueleto a centenas de px
        (bug de bancada de 19/10/2026: Kinect + camera ATIVA cegos num lado e a
        outra auxiliar, que ainda via a mao, desenhava em 1,0 m fixo -- medido
        744 px de erro a 0,65 m na Trust e 1836 px na SIGMA).

        NAO depende do depth FRAME do Kinect (so' da profundidade em metros que o
        chamador passar): e' isso que permite desenhar as auxiliares justamente
        quando o Kinect nao esta' entregando nada.
        """
        if view is None or not view.usable or hand_landmarks is None:
            return None
        deslocamento = (np.asarray(trim, np.float64)
                        if trim is not None else np.zeros(2))
        if triangulado is not None:
            tri = np.asarray(triangulado, np.float64).reshape(-1, 3)
            if np.isfinite(tri).all(axis=1).sum() >= 5:
                projected, _ = cv2.projectPoints(
                    np.nan_to_num(tri).reshape(-1, 1, 3),
                    np.zeros(3),
                    np.zeros(3),
                    np.asarray(view.kinect_matrix, np.float64),
                    np.asarray(view.kinect_dist, np.float64),
                )
                pixels = projected.reshape(-1, 2)
                if np.any(deslocamento):
                    pixels = pixels + deslocamento[None, :]
                return pixels
        pixels = view.pixels(hand_landmarks)
        init_depth = None
        if kinect_depth_m is not None:
            valores = np.asarray(kinect_depth_m, np.float64).reshape(-1)
            validos = valores[np.isfinite(valores) & (valores > 0.1)]
            if validos.size > 0:
                init_depth = float(np.median(validos))
        if init_depth is None:
            # Mao PERDIDA: sem triangulacao e sem profundidade deste lado nao ha'
            # medida nenhuma -- o lado fica SEM esqueleto (melhor que inventar a
            # posicao). Ver o docstring: e' o fix do "esqueleto desgovernado"
            # quando o Kinect e a auxiliar ativa perdem a mao e a outra auxiliar
            # ainda a ve.
            return None
        pixels = view.to_kinect_stereo(pixels, [init_depth])
        if pixels is None:
            return None
        pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
        if np.any(deslocamento):
            pixels = pixels + deslocamento[None, :]
        return pixels

    def draw_auxiliary_cameras_on_color(self, frame, hands_por_camera,
                                        alignment, depths=None,
                                        triangulados=None, rotulos=True,
                                        cores=None, ignorar=()):
        """Desenha as maos de TODAS as auxiliares no RGB do Kinect.

        Objetivo (bancada de 23/09/2026): na janela do Kinect tem de aparecer o
        esqueleto do KINECT (verde, ja' desenhado por detect_hand*) MAIS o de
        CADA webcam auxiliar, cada um na sua cor e com rotulo. E' a verificacao
        visual de que as duas auxiliares enxergam a mao e de onde cada uma a
        coloca no espaco (calibracao de cada camera).

        `hands_por_camera` = {indice: [landmarks, ...]};
        `depths`           = {indice: profundidades (m, frame do Kinect)};
        `triangulados`     = {indice: points_3d (21,3) no frame do Kinect} -- aqui
        entra TAMBEM a triangulacao do PAR DE AUXILIARES, que e' o caminho que
        substitui o Kinect quando ele perde a mao;
        `cores`            = {indice: (B,G,R)}, para fixar a cor de uma camera
        (ex.: pular a cor da camera cujo esqueleto outra rotina ja' desenhou);
        `ignorar`          = indices a NAO desenhar (quem ja' desenhou essa mao).

        Devolve {indice: (desvio_px|None, dx, dy, red_pixels, green_pixels)} no
        MESMO formato de `draw_auxiliary_hands_on_color` (por camera; a primeira
        mao de cada camera e' a que alimenta o desvio).
        """
        if frame is None or not hands_por_camera:
            return {}
        ignorar = {int(i) for i in ignorar}
        verde_todas = self.last_kinect_hands or []
        resultados = {}
        for posicao, index in enumerate(sorted(hands_por_camera)):
            if int(index) in ignorar:
                continue
            maos = hands_por_camera[index]
            if not maos:
                continue
            # A camera ATIVA usa os CAMPOS VIVOS (mesmo contrato da rotina de
            # camera unica: quem os sobrescreve -- ex. testes sinteticos ou o
            # trim em tempo real -- manda); as outras usam a calibracao GRAVADA
            # daquela camera (`aux_view`).
            ativo = getattr(alignment, "active_aux", None)
            view = None
            if (ativo is not None and int(index) == int(ativo)
                    and hasattr(alignment, "active_view")):
                view = alignment.active_view()
            if view is None:
                view = alignment.aux_view(int(index))
            if view is None or not view.usable:
                continue
            cor_nos = (cores or {}).get(int(index))
            if cor_nos is None:
                cor_nos = self.CORES_AUXILIARES[
                    posicao % len(self.CORES_AUXILIARES)]
            cor_nos = tuple(int(canal) for canal in cor_nos)
            cor_linhas = tuple(int(canal * 0.75) for canal in cor_nos)
            rotulo = None
            if rotulos:
                rotulo = "aux %s" % nome_auxiliar(index)
            for hand_index, landmarks in enumerate(maos):
                # REFERENCIA POR MAO quando o chamador passou uma lista do
                # tamanho da lista de maos (bimanual: cada lado leva o SEU
                # triangulated_3d/z). Formato ANTIGO continua igual: tri (ndarray)
                # so' na mao 0 e z ([z]) broadcast para todas. Antes TODAS as
                # maos recebiam o dado da PRIMARIA: com as duas maos na tela a
                # nao primaria era desenhada na profundidade/3D da outra (roxo
                # offset na mao esquerda, bancada 09/10/2026).
                tri_all = (triangulados or {}).get(int(index))
                z_all = (depths or {}).get(int(index))
                if isinstance(tri_all, (list, tuple)):
                    tri_mao = (tri_all[hand_index]
                               if hand_index < len(tri_all) else None)
                else:
                    tri_mao = tri_all if hand_index == 0 else None
                if isinstance(z_all, (list, tuple)):
                    z_mao = (z_all[hand_index] if hand_index < len(z_all)
                             else (z_all[0] if z_all else None))
                else:
                    z_mao = z_all
                pixels = self._projetar_mao_no_kinect(
                    view, landmarks, z_mao, tri_mao, alignment.overlay_trim,
                )
                if pixels is None or pixels.shape[0] < 21:
                    continue
                dentro = (np.isfinite(pixels).all(axis=1)
                          & mascara_dentro_do_frame(pixels, self.color_width,
                                                    self.color_height))
                if dentro.sum() < 5:
                    continue
                desvio = None
                verde_mao = None
                if hand_index < len(verde_todas):
                    verde_mao = np.array(
                        [(lm.x * self.color_width, lm.y * self.color_height)
                         for lm in verde_todas[hand_index]], np.float64)
                    if verde_mao.shape == pixels.shape:
                        validos = dentro & np.isfinite(verde_mao).all(axis=1)
                        if validos.any():
                            difs = pixels - verde_mao
                            desvio = (
                                float(np.mean(np.linalg.norm(difs[validos],
                                                             axis=1))),
                                float(np.mean(difs[validos, 0])),
                                float(np.mean(difs[validos, 1])),
                            )
                desenhar_esqueleto_projetado(
                    frame, pixels, cor_nos, cor_linhas,
                    rotulo if (hand_index == 0 and rotulo) else None)
                if int(index) not in resultados:
                    vermelho = pixels.copy()
                    vermelho[~dentro] = np.nan
                    if desvio is None:
                        resultados[int(index)] = (None, None, None, vermelho,
                                                  None)
                    else:
                        resultados[int(index)] = (*desvio, vermelho, verde_mao)
        return resultados

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
            #: Mascara ABAIXO_DO_TRONCO (mesma da depth): sem ela o esqueleto
            #: do SDK desenha quadril/joelho/pe inferidos ATRAVES da MESA --
            #: pontos fantasma abaixo do tronco (bancada 09/10/2026).
            y_base = self._tronco_base_y(body.joints)
            points = {}
            for joint_type in range(PyKinectV2.JointType_Count):
                joint = body.joints[joint_type]
                if joint.TrackingState == PyKinectV2.TrackingState_NotTracked:
                    continue
                if self._abaixo_do_tronco(joint.Position.y, y_base):
                    continue
                point = self.mapper.MapCameraPointToColorSpace(joint.Position)
                if point is None:
                    continue
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

    def _palm_camera_position(self, landmarks):
        """Palma (landmark 9) no espaco de camera do Kinect, ou None.

        Mesmo caminho de detect_hand: pixel de cor -> mapeamento color->depth
        -> mediana de depth -> MapDepthPointToCameraSpace (y invertido).
        """
        palm = landmarks[9]
        color_x = int(np.clip(palm.x * self.color_width, 0, self.color_width - 1))
        color_y = int(np.clip(palm.y * self.color_height, 0, self.color_height - 1))
        depth_pixel = self._color_to_depth_pixel(color_x, color_y)
        if depth_pixel is None:
            self._log_map_failure(color_x, color_y)
            return None
        depth_mm = self._median_depth(*depth_pixel)
        if depth_mm is None:
            return None
        depth_point = PyKinectV2._DepthSpacePoint()
        depth_point.x = float(depth_pixel[0])
        depth_point.y = float(depth_pixel[1])
        camera_point = self.mapper.MapDepthPointToCameraSpace(
            depth_point, int(round(depth_mm))
        )
        if camera_point is None:
            return None
        position = np.array(
            [camera_point.x, -camera_point.y, camera_point.z], dtype=float
        )
        if not np.isfinite(position).all():
            return None
        return position

    def detect_hands_all(self, color_frame, depth_frame, flip=None):
        """Detecta TODAS as maos no RGB do Kinect (groundtruth bimanual).

        Devolve (color_frame, maos); cada item e' um dict:
            {"side": "right"/"left"/"",
             "landmarks": [21 landmarks],
             "position": np.array(3) | None}  # palma no espaco de camera
        `side` vem do rotulo do MediaPipe corrigido pelo espelho (o RGB do
        Kinect nao e' espelhado). Todas as maos sao DESENHADAS com destaque:
        o groundtruth bimanual nao escolhe uma unica mao ativa.
        """
        if color_frame is None or depth_frame is None:
            self.last_hand_reason = "sem frame Kinect"
            return None, []
        if flip is None:
            flip = MEDIAPIPE_KINECT_HANDEDNESS_FLIP
        raw_color_frame = color_frame.copy()
        self._draw_skeleton(color_frame)
        rgb = cv2.cvtColor(raw_color_frame, cv2.COLOR_BGRA2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.kinect_frame_timestamp_ms = max(
            self.kinect_frame_timestamp_ms + 1, int(time.monotonic() * 1000)
        )
        result = self.hands.detect_for_video(image, self.kinect_frame_timestamp_ms)
        detected = list(getattr(result, "hand_landmarks", None) or [])
        self.last_kinect_hands = detected
        maos = []
        lados_usados = set()
        pontos_corpo = getattr(self, "_pontos_mao_corpo", None) or {}
        for index, hand_landmarks in enumerate(detected):
            side = self._handedness_of(result, index, flip)
            side_mediapipe = side
            position = self._palm_camera_position(hand_landmarks)
            if position is not None:
                if self.last_hand_depth_m is None:
                    self.last_hand_depth_m = float(position[2])
                else:
                    self.last_hand_depth_m = (
                        0.8 * self.last_hand_depth_m + 0.2 * float(position[2])
                    )
            if position is not None and pontos_corpo:
                lado_geometrico = lado_mais_proximo(position, pontos_corpo)
                if (lado_geometrico is not None
                        and lado_geometrico not in lados_usados):
                    if lado_geometrico != side_mediapipe:
                        if not getattr(self, "_aviso_lado_geo", False):
                            self._aviso_lado_geo = True
                            print("[lado] mao do Kinect: lado pelo "
                                  f"ESQUELETO 3D ({lado_geometrico}) em vez "
                                  "do rotulo do MediaPipe "
                                  f"({side_mediapipe}) -- evita seta "
                                  "invertida (item 113)", flush=True)
                    side = lado_geometrico
            if side:
                lados_usados.add(side)
            self._draw_hand(color_frame, hand_landmarks, destaque=True)
            maos.append(
                {
                    "side": side,
                    "landmarks": hand_landmarks,
                    "position": position,
                }
            )
        self.last_hand_reason = (
            f"{len(maos)} mao(s) no Kinect" if maos else "MediaPipe sem deteccao"
        )
        return color_frame, maos

    def detect_auxiliary_hands_sides(self, auxiliary_frame, aux_index=None,
                                     flip=None):
        """Detecta TODAS as maos na auxiliar com a lateralidade de cada uma.

        Devolve lista de {"side": ..., "side_bruto": ..., "landmarks": [...]},
        em ORDEM CANONICA (direita primeiro, depois esquerda, depois as sem
        rotulo): o pareamento do groundtruth bimanual e' PELO LADO, e a ordem
        estavel e' o que faz "mao 1" e "mao 2" significarem a MESMA coisa em
        todas as cameras (antes a lista vinha na ordem de ESCORE do MediaPipe,
        que muda de camera para camera e de frame para frame).

        A auxiliar e' espelhada no pipeline, entao o rotulo do MediaPipe ja'
        corresponde a mao fisica -- EXCETO na camera cujo driver ja' entrega o
        fluxo espelhado (o bug relatado na bancada em 05/10/2026). Por isso o
        `flip` e' POR CAMERA: None usa o valor aprendido por
        `ArbitroDeLateralidade` (`aux_handedness_flip.json`).

        `aux_index` None usa o detector PRIMARIO (compatibilidade); com indice usa
        o detector/timestamp DAQUELA camera -- necessario com duas auxiliares: o
        VideoMode do MediaPipe nao admite intercalar dois fluxos no mesmo
        landmarker.
        """
        if auxiliary_frame is None:
            return []
        rgb = cv2.cvtColor(auxiliary_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if aux_index is not None:
            index = int(aux_index)
            detector = self.auxiliary_detectors.get(index)
            if detector is None:
                raise KeyError(f"Sem detector para a camera auxiliar {index}.")
            stamp = max(self.auxiliary_timestamps_ms[index] + 1,
                        int(time.monotonic() * 1000))
            self.auxiliary_timestamps_ms[index] = stamp
            result = detector.detect_for_video(image, stamp)
        else:
            index = None
            self.auxiliary_frame_timestamp_ms = max(
                self.auxiliary_frame_timestamp_ms + 1,
                int(time.monotonic() * 1000)
            )
            result = self.auxiliary_hands.detect_for_video(
                image, self.auxiliary_frame_timestamp_ms
            )
        if flip is None:
            flip = aux_handedness_flip(index) if index is not None else False
        return ordena_maos_por_lado([
            {
                "side": self._handedness_of(result, posicao, flip),
                "side_bruto": self._handedness_of(result, posicao, False),
                "landmarks": hand_landmarks,
            }
            for posicao, hand_landmarks in enumerate(
                getattr(result, "hand_landmarks", None) or []
            )
        ])

    def detect_hand(self, color_frame, depth_frame, want_side=None,
                    ancora_px=None, raio_px=0.0):
        """Detecta a mao no RGB do Kinect.

        `want_side` ('left'/'right') seleciona a mao DAQUELE lado (necessario
        com as duas maos visiveis: a trial define a mao do movimento). Sem
        `want_side` escolhe a mais confiante. A lateralidade e' corrigida pela
        convencao de espelho do MediaPipe (o RGB do Kinect nao e' espelhado).

        `ancora_px`/`raio_px` mantem a MESMA mao do quadro anterior mesmo que o
        rotulo dela tenha oscilado (ver `_pick_hand_index`). Nesse caso
        `last_hand_side` devolve o rotulo ATUAL da mao escolhida -- que pode ser
        o lado oposto ao pedido: e' assim que o chamador sabe que o detector
        discordou (e pode marcar a trial).
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
        index = self._pick_hand_index(
            result, want_side, MEDIAPIPE_KINECT_HANDEDNESS_FLIP,
            ancora_px=ancora_px, raio_px=raio_px,
            tamanho=(self.color_width, self.color_height))
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
        if camera_point is None:
            self.last_hand_reason = "mapeamento depth->camera falhou"
            return color_frame, None, landmarks
        position = np.array(
            [camera_point.x, -camera_point.y, camera_point.z], dtype=float)
        if not np.isfinite(position).all() or position[2] <= 0.05:
            self.last_hand_reason = "posicao 3D invalida (inf/NaN)"
            return color_frame, None, landmarks
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
                f"use 'T' com o tabuleiro para diagnosticar (debe ver "
                f"{CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1]} cantos)"
            )

    def camera_para_pixel_color(self, posicao):
        """Ponto de camera -> pixel do RGB, ou None se a projecao for invalida.

        O mapper do Kinect devolve inf/NaN/valores absurdos para pontos que
        caem fora do FOV da cor (mao no limite do quadro, item 121). Toda a
        conversao que DEPOIS vira int(round(...)) ou cv2 passa por aqui: sem
        isso o programa morria com OverflowError: float infinity.
        """
        if posicao is None:
            return None
        try:
            vetor = np.asarray(posicao, np.float64).reshape(3)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(vetor).all():
            return None
        try:
            ponto_camera = PyKinectV2._CameraSpacePoint(
                float(vetor[0]), float(vetor[1]), float(vetor[2]))
            ponto = self.mapper.MapCameraPointToColorSpace(ponto_camera)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        try:
            x = float(ponto.x)
            y = float(ponto.y)
        except (AttributeError, TypeError, ValueError):
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        largura = float(getattr(self, "color_width", 0) or 0)
        altura = float(getattr(self, "color_height", 0) or 0)
        if largura <= 0 or altura <= 0:
            return None
        #: margem generosa: projecao coerente pode cair fora do quadro, mas um
        #: valor a dezenas de vezes o tamanho do quadro e' lixo numerico.
        if abs(x) > 10.0 * largura or abs(y) > 10.0 * altura:
            return None
        return int(round(x)), int(round(y))

    def draw_palm_vector_custom(self, frame, landmarks, direction_camera,
                                cor=(0, 0, 255), rotulo="direcao da palma",
                                espessura=5):
        """Seta de um vetor da palma (camera space) no video, com cor/rotulo.

        `direction_camera` e' QUALQUER versor (a previsao do IMU, a normal do
        plano do Kinect...): e' o desenho que permite comparar as duas lado a
        lado na calibracao por plano do dorso (tecla J, item 102).
        """
        if direction_camera is None or frame is None or not landmarks:
            return
        try:
            direcao = np.asarray(direction_camera, np.float64).reshape(3)
        except (TypeError, ValueError):
            return
        norma = float(np.linalg.norm(direcao))
        if not np.isfinite(direcao).all() or norma < 1e-9:
            return
        direcao = direcao / norma
        try:
            palm = self._landmark_camera_point(landmarks[9])
        except (AttributeError, IndexError, TypeError):
            palm = None
        if palm is None:
            return
        palm = np.asarray(palm, np.float64).reshape(3)
        if not np.isfinite(palm).all():
            return
        endpoint = palm + direcao * PALM_VECTOR_LENGTH_M
        start_pixel = self.camera_para_pixel_color(palm)
        end_pixel = self.camera_para_pixel_color(endpoint)
        if start_pixel is None or end_pixel is None:
            return
        try:
            altura, largura = frame.shape[:2]
        except (AttributeError, ValueError):
            return
        if largura <= 0 or altura <= 0:
            return
        #: recorta o destino para o quadro (seta continua visivel no limite em
        #: vez de sumir quando a ponta sai do video).
        end_pixel = (min(max(end_pixel[0], 0), largura - 1),
                     min(max(end_pixel[1], 0), altura - 1))
        if not (0 <= start_pixel[0] < largura
                and 0 <= start_pixel[1] < altura):
            return
        cv2.arrowedLine(frame, start_pixel, end_pixel, cor, espessura,
                        tipLength=0.2)
        cv2.putText(frame, rotulo, end_pixel, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, cor, 2)

    def draw_palm_vector(self, frame, landmarks, direction_camera):
        self.draw_palm_vector_custom(frame, landmarks, direction_camera)

    def close(self):
        self.hands.close()
        self.auxiliary_hands.close()
        self.kinect.close()


# --- LATERALIDADE DAS MAOS AUXILIARES: ORDEM CANONICA + ARBITRO GEOMETRICO ----
#: Ordem canonica dos lados -- a MESMA do laco por mao do groundtruth/tool
#: ("right" antes de "left", porque a direita e' a mao PRIMARIA dos recursos de
#: mao unica: teclas H/K/L).
ORDEM_DOS_LADOS = ("right", "left")
#: Indice do landmark da PALMA: o MESMO no' que o resto do modulo usa como "a mao
#: esta aqui" (profundidade da palma, vetor da palma, ancoras). O arbitro compara
#: esse ponto e nao os 21 nos: os dedos tem ruido proprio e sao justamente o que a
#: calibracao de esqueleto existe para corrigir.
PALM_LANDMARK_INDEX = 9


def ordena_maos_por_lado(maos):
    """Maos (dicts com "side") em ordem canonica e ESTAVEL (sorted e' estavel).

    A lista do MediaPipe vem ordenada por ESCORE, que muda de camera para camera
    e de frame para frame: "mao 1" podia ser a ESQUERDA numa auxiliar e a DIREITA
    na outra no MESMO instante, o que torna a conferencia visual (e qualquer log)
    impossivel de ler. Sem rotulo (`side` "") vai para o fim.
    """
    def chave(entrada):
        lado = (entrada or {}).get("side") or ""
        return (ORDEM_DOS_LADOS.index(lado) if lado in ORDEM_DOS_LADOS
                else len(ORDEM_DOS_LADOS))

    return sorted(maos or [], key=chave)


def pontos_normalizados(landmarks):
    """(N,2) normalizado (0..1) dos landmarks do MediaPipe, ou None se vazio."""
    pontos = [(float(ponto.x), float(ponto.y)) for ponto in (landmarks or [])]
    if not pontos:
        return None
    return np.asarray(pontos, np.float64)


def projeta_mao_aux_no_kinect(homography, residuals, landmarks, kinect_size):
    """Palma (px) no RGB do Kinect a partir dos landmarks da AUXILIAR, ou None.

    Mesma matematica de `CameraAlignment.auxiliary_to_kinect_landmark` (homografia
    dos 21 nos + residual por no), mas com a homografia PASSADA como argumento:
    o arbitro precisa comparar TODAS as auxiliares no MESMO frame, e os campos
    vivos do alignment (`hand_landmark_homography`) descrevem UMA camera so'.
    """
    if homography is None or kinect_size is None:
        return None
    pontos = pontos_normalizados(landmarks)
    if pontos is None:
        return None
    try:
        projetado = cv2.perspectiveTransform(
            pontos.reshape(-1, 1, 2), np.asarray(homography, np.float64)
        ).reshape(-1, 2)
    except cv2.error:                      # homografia degenerada no npz
        return None
    if residuals is not None:
        residual = np.asarray(residuals, np.float64).reshape(-1, 2)
        contagem = min(len(projetado), len(residual))
        projetado[:contagem] += residual[:contagem]
    if len(projetado) <= PALM_LANDMARK_INDEX or not np.isfinite(projetado).all():
        return None
    largura, altura = float(kinect_size[0]), float(kinect_size[1])
    palma = projetado[PALM_LANDMARK_INDEX]
    return np.array([palma[0] * largura, palma[1] * altura], np.float64)


#: Limites do arbitro, em px do RGB do Kinect. MAX_DIST: uma projecao a mais de
#: X px de QUALQUER mao do Kinect nao e' um casamento -- melhor nao decidir do que
#: decidir errado (o pior erro possivel e' associar/triangular a mao errada).
#: MIN_MARGEM: os punhos da bancada ficam a ~40 cm (~500 px com fx ~1050) um do
#: outro e a projecao 2D erra ~10-20 px (RMS da calibracao de esqueleto), entao
#: 40 px de folga entre a melhor e a segunda colocada e' conservador.
ARBITRO_MAX_DIST_PX = 120.0
ARBITRO_MIN_MARGEM_PX = 40.0
#: Erro maximo (px) para um frame ENSINAR (votar pela inversao de uma camera). A
#: margem de `ARBITRO_MAX_DIST_PX` serve para DECIDIR no frame que o usuario ve; o
#: que fica gravado em `aux_handedness_flip.json` vale para TODOS os programas, e a
#: projecao boa erra 1-2 px -- 60 px ja' e' folga larga. (Medido ao vivo em
#: 05/10/2026: sem esse portao, uma coleta com UMA mao visivel de cada vez encheu o
#: contador de votos de uma camera cujo rotulo era, sim, invertido -- mas por
#: evidencia que nao distinguia "rotulo trocado" de "as duas cameras viram maos
#: diferentes".)
ARBITRO_MAX_DIST_VOTO_PX = 60.0
#: Consenso para GRAVAR a inversao de uma camera: N vereditos e uma fracao minima
#: deles pedindo a MESMA correcao (uma leitura ruim nao grava arquivo).
ARBITRO_MIN_VOTOS = 20
ARBITRO_MIN_FRACAO = 0.9
#: POR ONDE o arbitro projeta a palma auxiliar no RGB do Kinect -- e o relatorio
#: diz QUAL via foi usada, porque o numero so' vale o que vale a via:
#:  * `stereo 3D` e' o caminho do overlay (extrinsecas + profundidade do punho do
#:    Kinect; RMS de 1-2 px quando a calibracao esta' boa);
#:  * `homografia 2D da mao` e' a projecao que o proprio modulo documenta como
#:    IMPRECISA (centenas de px: a mao e' um objeto 3D visto de angulos
#:    diferentes). Medido na bancada em 05/10/2026: com ela o arbitro ficava
#:    indeciso em TODOS os frames ("projecao a ~300 px") e a lateralidade voltava
#:    a ser o rotulo -- o usuario via o esqueleto projetado confuso e o terminal
#:    inundado pela mesma linha.
VIA_STEREO = "stereo 3D"
VIA_HOMOGRAFIA = "homografia 2D da mao"


class ArbitroDeLateralidade:
    """Diz o LADO de cada mao auxiliar pela GEOMETRIA e APRENDE a inversao.

    Por que nao confiar no rotulo do MediaPipe: a lateralidade dele e' uma
    classificacao da APARENCIA do frame e assume imagem espelhada (convencao
    "selfie"). O pipeline espelha todas as auxiliares com o mesmo
    `AUXILIARY_MIRROR_HORIZONTAL`, mas o DRIVER de uma camera pode ja' entregar o
    fluxo espelhado (nesta maquina ha' dois modelos: SIGMA-W780M e Trust USB
    Camera) -- e entao o nosso flip desespelha, a premissa cai e SO' aquela
    camera sai com as maos trocadas. Foi o relato da bancada em 05/10/2026: uma
    auxiliar concordava com o Kinect e a outra estava invertida, o que troca as
    maos justamente no casamento por lado (o que liga cada mao ao IMU certo).

    A geometria nao depende de aparencia: cada auxiliar tem o stereo calibrado com
    o Kinect (`stereo_calibration_aux{N}.npz`) e o Kinect da' o punho em 3D.
    Projetamos a PALMA de cada deteccao auxiliar no RGB do Kinect pelo MESMO
    caminho do overlay (`AuxCameraView.to_kinect_stereo` com a profundidade do
    punho do Kinect) e perguntamos qual mao do Kinect esta' ali: essa e' a verdade
    daquele frame. A homografia no-a-no da mao (`hand_landmark_calibration*.npz`)
    fica de RESERVA, e o relatorio diz qual via foi usada -- porque ela e' a
    projecao que o modulo documenta como imprecisa (centenas de px: a mao e' um
    objeto 3D visto de angulos diferentes, e a homografia mapeia um plano). O
    rotulo passa a ser palpite, e a discordancia alimenta o aprendizado -- que
    fica GRAVADO (`aux_handedness_flip.json`) e vale tambem para os programas que
    so' tem o rotulo (protocolo, sand, demo).

    Quando o Kinect nao ve' mao -- exatamente o caso em que o PAR de auxiliares
    substitui o Kinect -- nao ha' verdade para comparar: cai no rotulo ja'
    corrigido pelo que foi aprendido, e o relatorio diz que a fonte foi o rotulo
    (nunca finge que mediu).
    """

    def __init__(self, ativo=True, max_dist=ARBITRO_MAX_DIST_PX,
                 min_margem=ARBITRO_MIN_MARGEM_PX, min_votos=ARBITRO_MIN_VOTOS,
                 min_fracao=ARBITRO_MIN_FRACAO, verbose=True, persistir=True):
        self.ativo = bool(ativo)
        self.max_dist = float(max_dist)
        self.min_margem = float(min_margem)
        self.min_votos = int(min_votos)
        self.min_fracao = float(min_fracao)
        self.verbose = bool(verbose)
        #: False = aprende em memoria mas NAO grava o json (usado por probes que
        #: nao devem mexer no que a sessao real usa).
        self.persistir = bool(persistir)
        #: {indice: {"oposto": n, "igual": n}} -- votos ACUMULADOS por camera.
        #: "oposto" = o rotulo discorda da geometria (pede o flip OPOSTO ao
        #: vigente); "igual" = concorda (confirma o valor vigente).
        self.votos = {}
        #: {indice: {...}} do ULTIMO frame -- e' o que a interface mostra.
        self.ultimo_relatorio = {}
        #: Cameras que ja' tiveram a confirmacao anunciada (evita log repetido).
        self.confirmados = set()


    # ------------------------------------------------------------- API publica
    def arbitrar(self, alignment, kinect_hands, aux_por_camera, kinect_size):
        """Corrige o lado das deteccoes auxiliares (e aprende a inversao).

        Devolve `(aux_por_camera_corrigido, relatorio)`. As listas do chamador
        NAO sao alteradas (copia). `relatorio[indice]` diz, por camera, a FONTE
        do lado ("geometria", "rotulo (...)" ou "sem mao"), o erro em px, quantas
        maos foram decididas e se o rotulo do MediaPipe estava invertido -- e' o
        que o tool e o `tools\\probe_lateralidade.py` mostram para o usuario saber
        se o programa MEDIU ou apenas acreditou no rotulo.
        """
        corrigido = {int(indice): [dict(entrada) for entrada in (entradas or [])]
                     for indice, entradas in (aux_por_camera or {}).items()}
        relatorio = {}
        maos_kinect = [mao for mao in (kinect_hands or [])
                       if (mao or {}).get("side") in ORDEM_DOS_LADOS
                       and mao.get("landmarks")]
        for indice in sorted(corrigido):
            entradas = corrigido[indice]
            if not entradas:
                relatorio[indice] = {"fonte": "sem mao", "maos": 0}
                continue
            if not self.ativo or kinect_size is None or not maos_kinect:
                relatorio[indice] = {
                    "fonte": "rotulo (Kinect sem mao para comparar)",
                    "maos": len(entradas),
                    "invertido": aux_handedness_flip(indice),
                }
                for entrada in entradas:
                    entrada.setdefault("side_rotulo", entrada.get("side") or "")
                    entrada.setdefault("lado_fonte", "rotulo")
                corrigido[indice] = ordena_maos_por_lado(entradas)
                continue
            relatorio[indice] = self._arbitrar_uma_camera(
                alignment, indice, entradas, maos_kinect, kinect_size)
            # Auditoria uniforme em TODOS os caminhos: a mao sabe dizer qual era
            # o rotulo do MediaPipe e quem decidiu o lado (geometria ou rotulo).
            for entrada in entradas:
                entrada.setdefault("side_rotulo", entrada.get("side") or "")
                entrada.setdefault("lado_fonte", "rotulo")
            corrigido[indice] = ordena_maos_por_lado(entradas)
        self.ultimo_relatorio = relatorio
        return corrigido, relatorio

    def resumo(self, nomes=None):
        """(linhas) legiveis do veredito do ULTIMO frame, uma por camera."""
        nomes = nomes or {}
        linhas = []
        for indice in sorted(self.ultimo_relatorio):
            info = self.ultimo_relatorio[indice]
            nome = nomes.get(int(indice)) or nomes.get(str(indice)) or ""
            rotulo = f"aux {indice}" + (f" ({nome})" if nome else "")
            fonte = info.get("fonte", "")
            if fonte == "sem mao":
                linhas.append(f"Lateralidade {rotulo}: sem mao no quadro")
            elif fonte.startswith("rotulo"):
                detalhe = info.get("detalhe")
                linhas.append(
                    f"Lateralidade {rotulo}: {fonte}"
                    + (f" -- {detalhe}" if detalhe else "")
                    + (" | com inversao aprendida"
                       if info.get("invertido") else ""))
            else:
                via = info.get("via")
                linhas.append(
                    f"Lateralidade {rotulo}: lado por GEOMETRIA "
                    f"({info.get('maos', 0)} mao(s) decidida(s), erro "
                    f"{info.get('dist_px', float('nan')):.0f} px"
                    + (f", via {via}" if via else "") + ")"
                    + (" | ROTULO DO MEDIAPIPE INVERTIDO -- corrigido"
                       if info.get("inverteu") else " | rotulo confere"))
        return linhas

    # --------------------------------------------------------------- internos
    def _arbitrar_uma_camera(self, alignment, indice, entradas, maos_kinect,
                             kinect_size):
        """Corrige (no lugar) o lado das `entradas` de UMA camera auxiliar."""
        modelo = (getattr(alignment, "aux_models", None) or {}).get(int(indice))
        homografia = (modelo or {}).get("hand_homography")
        vista = self._vista_da_camera(alignment, indice)
        if vista is None and homografia is None:
            return {"fonte": "rotulo (sem calibracao da mao desta camera)",
                    "detalhe": ("calibre o STEREO desta camera "
                                "(stereo_calibration.py, com ela ativa) -- a "
                                "tecla H do tool (calibracao do esqueleto) e' "
                                "so' a reserva imprecisa"),
                    "maos": len(entradas),
                    "invertido": aux_handedness_flip(indice)}
        # A via PREFERIDA e' a mesma do overlay (stereo 3D com a profundidade do
        # punho que o Kinect mediu -- `AuxCameraView.to_kinect_stereo`); a
        # homografia 2D do esqueleto fica de reserva e o relatorio diz qual foi.
        via = VIA_STEREO if vista is not None else VIA_HOMOGRAFIA
        maos_por_lado = {mao["side"]: mao for mao in maos_kinect}
        palmas_kinect = {lado: self._palma_no_kinect(mao, kinect_size)
                         for lado, mao in maos_por_lado.items()}
        lados = [lado for lado in ORDEM_DOS_LADOS
                 if palmas_kinect.get(lado) is not None]
        if not lados:
            return {"fonte": "rotulo (palma do Kinect indisponivel)",
                    "maos": len(entradas), "via": via,
                    "invertido": aux_handedness_flip(indice)}
        trim = getattr(alignment, "overlay_trim", None)
        trim = (np.zeros(2, np.float64) if trim is None
                else np.asarray(trim, np.float64).reshape(-1)[:2])
        padrao = self._profundidade_padrao(maos_por_lado.values())
        # Custo POR PAR (deteccao auxiliar x mao do Kinect): com o stereo a
        # projecao depende da profundidade TESTADA, entao cada candidato tem a
        # sua propria palma projetada -- e um casamento errado acumula dois erros
        # (o pixel e a profundidade).
        custos = {
            posicao: {lado: self._custo_de_casamento(
                vista, homografia, (modelo or {}).get("hand_residuals"),
                entrada.get("landmarks"), maos_por_lado[lado], kinect_size,
                trim, padrao)
                for lado in lados}
            for posicao, entrada in enumerate(entradas)
        }
        decisoes, erro_px, detalhe = self._casa(custos, lados)
        # O rotulo (do jeito que o pipeline o produz HOJE, com o flip vigente)
        # vale como `side_rotulo`: e' contra ele que o aprendizado compara.
        for entrada in entradas:
            entrada["side_rotulo"] = entrada.get("side") or ""
            entrada["lado_fonte"] = "rotulo"
        for posicao, lado in decisoes:
            entradas[posicao]["side"] = lado
            entradas[posicao]["lado_fonte"] = "geometria"
        # Um lado NAO pode aparecer duas vezes NA MESMA camera. Acontece de fato
        # (medido em 05/10/2026): a auxiliar ve 2 maos e o Kinect ve 1 -- a
        # geometria corrige a deteccao mais proxima para o lado do Kinect e a
        # outra fica com o rotulo, e as duas acabam no MESMO lado. Deixar passar
        # seria pior que nao decidir: quem casa por lado (`aux_lado_por_camera`,
        # `aux_por_lado`) guarda UM landmarks por lado e descartaria uma das
        # deteccoes em silencio. Aqui a atribuicao inteira e' descartada (e o
        # frame tambem nao vota: sem decisao nao ha' evidencia de inversao).
        if decisoes:
            lados_finais = [entrada.get("side") for entrada in entradas
                            if entrada.get("side") in ORDEM_DOS_LADOS]
            if len(lados_finais) != len(set(lados_finais)):
                for entrada in entradas:
                    entrada["side"] = entrada.get("side_rotulo") or ""
                    entrada["lado_fonte"] = "rotulo"
                decisoes = []
                detalhe = ("duas deteccoes no mesmo lado nesta camera (o Kinect "
                           f"viu {len(lados)} mao(s)): atribuicao descartada")
        if not decisoes and via == VIA_HOMOGRAFIA:
            # Nao decidiu E a via do numero foi a imprecisa: centenas de px de
            # erro sao ESPERADOS dela (a homografia 2D mapeia um plano; a mao sai
            # dele). O que falta e' a calibracao que o overlay usa, o stereo 3D --
            # dizer isso evita o caminho errado de "recalibrar o esqueleto" quando
            # o esqueleto nem e' o que projeta o overlay.
            detalhe = (f"{detalhe} | via {VIA_HOMOGRAFIA}: calibre o STEREO "
                       "desta camera (stereo_calibration.py) para a geometria "
                       "medir")
        # Sem decisao NAO e' "geometria": e' a geometria DIZENDO que nao da' para
        # separar as maos (juntas no quadro, projecao longe ou lado repetido). O
        # relatorio tem de poder dizer isso -- o usuario precisa saber quando o
        # programa NAO mediu.
        fonte = "geometria" if decisoes else "rotulo (geometria indecisa)"
        oposto = igual = 0
        for entrada in entradas:
            if entrada.get("lado_fonte") != "geometria":
                continue
            rotulo = entrada.get("side_rotulo") or ""
            if rotulo not in ORDEM_DOS_LADOS:
                continue               # sem rotulo confiavel: nao vota
            if rotulo == entrada.get("side"):
                igual += 1
            else:
                oposto += 1
        # SO' VOTA o frame que pode ser conferido INTEIRO -- e inteiro, aqui, e' o
        # caso BIMANUAL: as DUAS maos vistas pelo Kinect E pela auxiliar, todas
        # atribuidas. As duas restricoes vem da bancada de 05/10/2026 (ao vivo):
        #   * com a auxiliar vendo DUAS maos e o Kinect UMA, a correcao do frame
        #     pega o par mais proximo e o que sobra no rotulo nao foi conferido por
        #     ninguem;
        #   * com UMA mao de cada lado o frame NAO distingue "o ROTULO esta' trocado"
        #     de "as duas cameras viram maos diferentes" -- foi assim que uma coleta
        #     com uma mao por vez fez o contador oscilar (trocado -> confere -> ...);
        #   * e o erro tem de ser APERTADO (`ARBITRO_MAX_DIST_VOTO_PX`): 60 px ainda
        #     DECIDE o frame (limite 120), mas nao serve de prova.
        # Fora disso o frame segue corrigido e mostrado -- so' nao alimenta o
        # aprendizado nem o arquivo.
        if (len(decisoes) == 2 and len(entradas) == 2 and len(lados) == 2
                and np.isfinite(erro_px)
                and erro_px <= ARBITRO_MAX_DIST_VOTO_PX):
            self._votar(indice, oposto, igual)
        inverteu = any(entrada.get("lado_fonte") == "geometria"
                       and entrada.get("side_rotulo") != entrada.get("side")
                       for entrada in entradas)
        return {"fonte": fonte, "maos": len(decisoes),
                "dist_px": erro_px, "detalhe": detalhe, "inverteu": inverteu,
                "via": via, "invertido": aux_handedness_flip(indice)}

    @staticmethod
    def _vista_da_camera(alignment, indice):
        """`AuxCameraView` da camera, SE ela puder projetar pelo stereo 3D.

        Devolve None quando o alignment nao expoe `aux_view` (implementacoes
        minimas, como a das suites de teste) ou quando o stereo daquela camera
        nao esta' utilizavel (`usable` = stereo aceito + extrinsecas): e' o que
        faz o arbitro cair na homografia 2D do esqueleto -- e o relatorio DIZER
        que caiu, em vez de apresentar o numero como medida fina.
        """
        construtor = getattr(alignment, "aux_view", None)
        if not callable(construtor):
            return None
        try:
            vista = construtor(int(indice))
        except Exception:                       # calibracao corrompida
            return None
        if vista is None or not getattr(vista, "usable", False):
            return None
        return vista

    @staticmethod
    def _profundidade_da_mao(mao):
        """Z (m, referencial de camera do Kinect) do punho de UMA mao, ou None.

        Vem do proprio detector (`mao["position"]`, a palma em 3D): e' a mesma
        profundidade que o overlay usa para desenhar o esqueleto auxiliar, e por
        isso o arbitro mede a lateralidade no espaco em que o vermelho aparece.
        """
        posicao = (mao or {}).get("position")
        if posicao is None:
            return None
        valores = np.asarray(posicao, np.float64).reshape(-1)
        if valores.size < 3 or not np.isfinite(valores[2]) or valores[2] <= 0:
            return None
        return float(valores[2])

    def _profundidade_padrao(self, maos):
        """Z (m) de reserva para quando a mao do Kinect nao tem posicao 3D."""
        valores = [self._profundidade_da_mao(mao) for mao in maos]
        valores = [valor for valor in valores if valor is not None]
        if valores:
            return float(np.median(valores))
        return None

    def _palma_por_stereo(self, vista, landmarks, mao_kinect, trim, padrao):
        """Palma (px) no RGB do Kinect projetando a AUXILIAR pelo stereo 3D.

        A profundidade que entra e' a da PROPRIA mao candidata do Kinect
        (`position[2]`): um casamento errado acumula entao DOIS erros (o pixel e
        a profundidade), que e' o que separa "esta' na mao" de "esta' perto".
        Aplica o MESMO `overlay_trim` do overlay -- assim a medida do arbitro e
        o esqueleto vermelho que o usuario ve falam do mesmo ponto.
        """
        if vista is None or not getattr(vista, "usable", False):
            return None
        profundidade = self._profundidade_da_mao(mao_kinect)
        if profundidade is None:
            profundidade = (padrao if padrao is not None
                            else NOMINAL_HAND_DEPTH_M)
        try:
            pixels = vista.pixels(landmarks)
            projetado = vista.to_kinect_stereo(pixels, [float(profundidade)])
        except Exception:                       # calibracao degenerada
            return None
        if projetado is None:
            return None
        projetado = np.asarray(projetado, np.float64).reshape(-1, 2)
        if len(projetado) <= PALM_LANDMARK_INDEX:
            return None
        if trim is not None and np.any(trim):
            projetado = projetado + np.asarray(trim, np.float64)[None, :]
        palma = projetado[PALM_LANDMARK_INDEX]
        if not np.isfinite(palma).all():
            return None
        return palma

    def _custo_de_casamento(self, vista, homografia, residuals, landmarks, mao,
                            kinect_size, trim, padrao):
        """Distancia (px) entre a palma projetada da deteccao e a mao candidata.

        Devolve None quando nao ha' como projetar (nem stereo, nem homografia).
        """
        palma_kinect = self._palma_no_kinect(mao, kinect_size)
        if palma_kinect is None or not landmarks:
            return None
        palma_aux = self._palma_por_stereo(vista, landmarks, mao, trim, padrao)
        if palma_aux is None:
            # Reserva: homografia 2D do esqueleto, a projecao que o modulo
            # documenta como imprecisa (centenas de px). So' decide com as maos
            # MUITO separadas -- e o relatorio marca a via.
            palma_aux = projeta_mao_aux_no_kinect(homografia, residuals,
                                                  landmarks, kinect_size)
        if palma_aux is None:
            return None
        return self._distancia(palma_aux, palma_kinect)

    def _casa(self, custos, lados):
        """(decisoes, erro_px, detalhe) escolhendo a atribuicao de menor custo.

        `custos[posicao][lado]` = distancia (px) entre a palma projetada da
        deteccao auxiliar `posicao` e a mao do Kinect `lado` (None = nao da' para
        projetar). O custo DEPENDE do lado quando a projecao vem do stereo (a
        profundidade testada e' a da propria mao candidata do Kinect).

        Com DUAS maos no Kinect e DUAS deteccoes na auxiliar testa as duas
        atribuicoes possiveis e exige MARGEM entre elas: e' o que separa "as maos
        sao inconfundiveis" de "as duas estao lado a lado no quadro e qualquer
        casamento serve". Com um lado so' no Kinect (ou uma deteccao so' na
        auxiliar) decide apenas o par mais proximo -- o resto fica no rotulo.

        `detalhe` explica POR QUE nao decidiu (margem pequena, projecao longe):
        sem esse numero, "INDECISO" nao diz ao experimentador se ele precisa
        separar mais as maos ou se a calibracao daquela camera esta' velha.
        """
        # Uma deteccao so' entra se projetar contra TODOS os lados candidatos:
        # meio casamento nao pode virar "mais barato" por falta de dado.
        validos = [posicao for posicao, linha in custos.items()
                   if linha and all(linha.get(lado) is not None
                                    for lado in lados)]
        if not validos:
            return [], float("nan"), "projecao da palma indisponivel"
        melhor = None
        melhor_custo, segundo_custo = float("inf"), float("inf")
        if len(lados) == 2 and len(validos) >= 2:
            for posicao_d in validos:
                for posicao_e in validos:
                    if posicao_d == posicao_e:
                        continue
                    total = (custos[posicao_d][lados[0]]
                             + custos[posicao_e][lados[1]])
                    if total < melhor_custo:
                        segundo_custo, melhor_custo = melhor_custo, total
                        melhor = ((posicao_d, lados[0]), (posicao_e, lados[1]))
                    elif total < segundo_custo:
                        segundo_custo = total
            if melhor is None:
                return [], float("nan"), "sem par valido para casar"
            custos_da_escolha = [custos[posicao][lado]
                                 for posicao, lado in melhor]
            erro = float(np.mean(custos_da_escolha))
            pior = float(max(custos_da_escolha))
            if pior > self.max_dist:
                return [], erro, (f"projecao a {pior:.0f} px "
                                  f"(limite {self.max_dist:.0f} px): conferir a "
                                  "calibracao da mao desta camera")
            margem = segundo_custo - melhor_custo
            if margem < self.min_margem:
                return [], erro, (f"margem {margem:.0f} px entre as duas "
                                  f"atribuicoes (< {self.min_margem:.0f} px): as "
                                  "maos estao juntas demais no quadro")
            return list(melhor), erro, "ok"
        # Um lado so' (ou uma deteccao so'): o par mais proximo, sem ambiguidade.
        for posicao in validos:
            for lado in lados:
                custo = custos[posicao][lado]
                if custo < melhor_custo:
                    melhor_custo, melhor = custo, (posicao, lado)
        if melhor is None:
            return [], float("nan"), "sem par valido para casar"
        if melhor_custo <= self.max_dist:
            return [melhor], melhor_custo, "ok"
        return [], melhor_custo, (f"projecao a {melhor_custo:.0f} px "
                                  f"(limite {self.max_dist:.0f} px)")

    @staticmethod
    def _distancia(palma_aux, palma_kinect):
        return float(np.linalg.norm(np.asarray(palma_aux, np.float64)
                                    - np.asarray(palma_kinect, np.float64)))

    @staticmethod
    def _palma_no_kinect(mao, kinect_size):
        """Palma (px) da mao do KINECT a partir dos landmarks normalizados."""
        pontos = pontos_normalizados(mao.get("landmarks"))
        if pontos is None or len(pontos) <= PALM_LANDMARK_INDEX:
            return None
        largura, altura = float(kinect_size[0]), float(kinect_size[1])
        palma = pontos[PALM_LANDMARK_INDEX]
        return np.array([palma[0] * largura, palma[1] * altura], np.float64)

    def _votar(self, indice, oposto, igual):
        """Acumula o voto da camera e grava a inversao quando ha' consenso.

        Voto por FRAME (e nao por mao): um frame com as duas maos concordando em
        que o rotulo esta' trocado vale UM voto, senao duas maos contariam dobrado
        e o consenso chegaria na metade do tempo com a mesma evidencia.
        """
        votos = self.votos.setdefault(int(indice), {"oposto": 0, "igual": 0})
        if oposto and not igual:
            votos["oposto"] += 1
        elif igual and not oposto:
            votos["igual"] += 1
        total = votos["oposto"] + votos["igual"]
        if total < self.min_votos:
            return None
        if votos["oposto"] / total >= self.min_fracao:
            # Pede o OPOSTO do que esta' vigente: e' o caso do driver que entrega
            # o fluxo espelhado -- e tambem o de uma inversao aprendida que ficou
            # velha (por exemplo depois de trocar a webcam de porta USB).
            self._grava(indice, not aux_handedness_flip(indice), votos["oposto"],
                        total)
        elif votos["igual"] / total >= self.min_fracao:
            if int(indice) not in self.confirmados:
                self.confirmados.add(int(indice))
                if self.verbose:
                    print(f"Lateralidade da camera auxiliar {indice}: rotulo do "
                          f"MediaPipe CONFERIDO com a geometria do Kinect em "
                          f"{total} frames (nada a corrigir)", flush=True)
            self.votos[int(indice)] = {"oposto": 0, "igual": 0}
        return None

    def _grava(self, indice, valor, votos, total):
        """Grava a inversao aprendida (com o motivo, para o arquivo explicar-se)."""
        if valor:
            motivo = (f"{votos}/{total} frames com o rotulo do MediaPipe trocado "
                      "em relacao ao Kinect, medido pela geometria (punho em 3D "
                      "do Kinect projetado pelo stereo daquela camera)")
        else:
            motivo = (f"{votos}/{total} frames concordando com a geometria do "
                      "Kinect -- a inversao gravada antes NAO se confirma e foi "
                      "desfeita")
        set_aux_handedness_flip(indice, valor, motivo=motivo,
                                silencioso=not self.verbose,
                                persistir=self.persistir)
        self.votos[int(indice)] = {"oposto": 0, "igual": 0}


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


if __name__ == "__main__":                        # pragma: no cover
    # Rodar a BIBLIOTECA direto nao faz nada: quem tenta (ja' aconteceu, e o
    # silencio nao ensina) tem de receber a instrucao e um codigo de saida
    # diferente de zero, para nao passar por "rodou e nao fez nada".
    raise SystemExit(
        "Este arquivo e' a BIBLIOTECA do projeto (nao tem entrada propria).\n"
        "  ferramenta de ground truth: .venv\\Scripts\\python.exe "
        "tools\\kinect_groundtruth_tool.py\n"
        "  probe de lateralidade:      .venv\\Scripts\\python.exe "
        "tools\\probe_lateralidade.py"
    )
