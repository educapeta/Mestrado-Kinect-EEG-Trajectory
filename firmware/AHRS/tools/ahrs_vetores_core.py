"""Nucleo matematico do visualizador de vetores AHRS (item 142).

O firmware publica quaternions corpo->mundo (w, x, y, z) dos dois MPU6050.
Aqui ficam o parser do protocolo serial e a matematica de rotacao -- sem
matplotlib/serial, para poder testar sem hardware.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

#: Ordem dos campos da linha `A,...` do firmware (item 142).
CAMPOS_LINHA_AHRS = (
    "t_us",
    "qw_left", "qx_left", "qy_left", "qz_left",
    "qw_right", "qx_right", "qy_right", "qz_right",
    "ax_left", "ay_left", "az_left",
    "ax_right", "ay_right", "az_right",
)


def quat_para_matriz(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion corpo->mundo para matriz de rotacao 3x3 (R @ v_corpo = v_mundo)."""
    valores = np.asarray([qw, qx, qy, qz], dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("quaternion com NaN/Infinity")
    norma = float(np.linalg.norm(valores))
    if norma < 1e-9:
        return np.eye(3, dtype=float)
    w, x, y, z = valores / norma
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
         2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def extrair_linhas(buffer: bytes):
    """Separa linhas completas (terminadas em \n) de um buffer cru.

    Usado no lugar de `readline()`: no Windows/CP210x o readline entrega uma
    linha por vez com latencia ~47 ms enquanto o ESP manda a cada 27 ms,
    criando uma fila de 10+ segundos. Lendo em bloco, todas as linhas
    disponiveis sao drenadas de uma vez.
    Retorna (lista_de_linhas_decodificadas, resto_do_buffer).
    """
    if not isinstance(buffer, (bytes, bytearray)):
        return [], b""
    resto = bytes(buffer)
    linhas = []
    while b"\n" in resto:
        bruta, resto = resto.split(b"\n", 1)
        linhas.append(bruta.decode("utf-8", "replace").strip())
    return linhas, resto


def parse_linha_ahrs(linha: str) -> Optional[dict]:
    """Interpreta uma linha `A,...` do firmware; devolve dict ou None.

    As linhas `#` (cabecalho), `E,` (erro) e texto de log sao ignoradas.
    """
    if not isinstance(linha, str):
        return None
    linha = linha.strip()
    if not linha.startswith("A,"):
        return None
    partes = [p.strip() for p in linha.split(",")]
    if len(partes) < 1 + len(CAMPOS_LINHA_AHRS):
        return None
    try:
        valores = [float(valor) for valor in partes[1:1 + len(CAMPOS_LINHA_AHRS)]]
    except ValueError:
        return None
    dados = dict(zip(CAMPOS_LINHA_AHRS, valores))
    if not np.isfinite(list(dados.values())).all():
        return None
    dados["left"] = np.array([dados["qw_left"], dados["qx_left"],
                              dados["qy_left"], dados["qz_left"]])
    dados["right"] = np.array([dados["qw_right"], dados["qx_right"],
                               dados["qy_right"], dados["qz_right"]])
    dados["accel_left"] = np.array([dados["ax_left"], dados["ay_left"],
                                    dados["az_left"]])
    dados["accel_right"] = np.array([dados["ax_right"], dados["ay_right"],
                                     dados["az_right"]])
    # A propria serial pode ter perdido o prefixo no inicio do stream.
    dados["t_s"] = dados["t_us"] * 1.0e-6
    return dados


def vetor_corpo_para_baixo(R0: np.ndarray,
                           baixo: np.ndarray = np.array([0.0, 0.0, -1.0])):
    """Vetor no CORPO que, na orientacao inicial R0, aponta para `baixo` no mundo."""
    R0 = np.asarray(R0, dtype=float).reshape(3, 3)
    baixo = np.asarray(baixo, dtype=float).reshape(3)
    return R0.T @ (baixo / np.linalg.norm(baixo))


def vetor_mundo(R: np.ndarray, v_corpo: np.ndarray) -> np.ndarray:
    """Rotaciona o vetor fixo do corpo para o mundo (unidade)."""
    vetor = np.asarray(R, dtype=float).reshape(3, 3) @ np.asarray(v_corpo, float).reshape(3)
    norma = float(np.linalg.norm(vetor))
    if norma < 1e-12:
        return np.zeros(3, dtype=float)
    return vetor / norma


def angulo_com_baixo(v_mundo: np.ndarray) -> float:
    """Angulo (graus) entre o vetor e o -Z do mundo (0 = apontando para o chao)."""
    vetor = np.asarray(v_mundo, dtype=float).reshape(3)
    norma = float(np.linalg.norm(vetor))
    if norma < 1e-12:
        return 0.0
    # dot com o "baixo" do mundo (0, 0, -1): 0 grau = apontando para o chao.
    cosseno = float(np.clip(np.dot(vetor / norma, [0.0, 0.0, -1.0]),
                            -1.0, 1.0))
    return math.degrees(math.acos(cosseno))


def quat_axis_angle(eixo: np.ndarray, angulo_rad: float) -> np.ndarray:
    """Quaternion (w,x,y,z) de uma rotacao de `angulo_rad` em torno de `eixo`."""
    eixo = np.asarray(eixo, dtype=float).reshape(3)
    norma = float(np.linalg.norm(eixo))
    if norma < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    eixo = eixo / norma
    metade = 0.5 * float(angulo_rad)
    return np.concatenate(([math.cos(metade)], math.sin(metade) * eixo))


def quat_multiplicar(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Produto de quaternions no formato (w,x,y,z)."""
    aw, ax, ay, az = np.asarray(a, dtype=float).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=float).reshape(4)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_simulado(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Quaternions de demonstracao para `--simular` (cada lado com fase propria)."""
    eixo_x = np.array([1.0, 0.0, 0.0])
    eixo_y = np.array([0.0, 1.0, 0.0])
    q_esq = quat_multiplicar(
        quat_axis_angle(eixo_x, 0.9 * math.sin(0.7 * t)),
        quat_axis_angle(eixo_y, 0.9 * math.sin(0.4 * t + 0.8)))
    q_dir = quat_multiplicar(
        quat_axis_angle(eixo_x, 0.9 * math.sin(0.7 * t + 0.5)),
        quat_axis_angle(eixo_y, 0.9 * math.sin(0.4 * t + 2.0)))
    return q_esq, q_dir
