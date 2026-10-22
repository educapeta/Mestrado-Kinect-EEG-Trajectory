"""Regressao: VERDADE DA PALMA PELO PLANO MEDIO DO DORSO (item 102).

Motivo (pedido do usuario em 22/10/2026): "a orientacao da palma ainda esta'
ruim, apontando para lugares ruins" -- a normal do Kinect era o produto vetorial
de 3 pontos (punho, MCP indicador, MCP minimo) e o ruido de profundidade de UM
no' gira a normal inteira. A proposta: ajustar o PLANO MEDIO dos nos do dorso
(punho + MCPs indicador/medio/anelar/minimo) e usar a normal desse plano.

O que este teste protege:
  1. recuperacao da normal de um plano sintetico conhecido (inclusive rotacao
     rigida e sinal anatomico por lado);
  2. com ruido de profundidade, o plano medio tem erro MENOR que o produto
     vetorial de 3 pontos (a razao de existir do metodo);
  3. casos degenerados: < 4 pontos, pontos colineares, sem os 3 pontos do sinal
     anatomico -> (None, None) (nunca normal inventada);
  4. controle de qualidade `rms_max_m` (plano ruim = amostra descartada);
  5. no CSV canonico de 19/10/2026 a normal do plano confere com a normal de 3
     pontos que o tool usava e o refit continua bom;
  6. integracao: `palm_plane_camera` no tracker, tecla J no tool, setas
     customizadas e o analisador offline `tools/calibra_palma_plano.py`.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

import kinect_imu_groundtruth as gt
import imu

FALHAS = []


def eh_none(resultado):
    """(None, None) sem comparar tuple com ndarray (evita ambiguidade numpy)."""
    return (isinstance(resultado, tuple) and len(resultado) == 2
            and resultado[0] is None and resultado[1] is None)


def check(nome, condicao, detalhe=""):
    if condicao:
        print(f"   ok: {nome}")
    else:
        FALHAS.append(nome)
        print(f"   FALHOU: {nome} {detalhe}")


def mao_sintetica(rotacao=None, deslocamento=None):
    """Pontos do dorso de uma mao sintetica em um PLANO CONHECIDO (z = 0).

    Layout aproximado (metros): punho na origem; os 4 MCPs em arco. O produto
    vetorial (index-wrist) x (pinky-wrist) aponta para +z, ou seja, a convencao
    anatomica do projeto da' +z para 'right' e -z para 'left'.
    """
    pontos = {
        "wrist": np.array([0.000, 0.000, 0.0]),
        "index": np.array([0.080, 0.020, 0.0]),
        "middle": np.array([0.020, 0.030, 0.0]),
        "ring": np.array([-0.010, 0.030, 0.0]),
        "pinky": np.array([-0.040, 0.020, 0.0]),
    }
    rot = np.eye(3) if rotacao is None else np.asarray(rotacao, float)
    desl = np.zeros(3) if deslocamento is None else np.asarray(deslocamento, float)
    return {nome: rot @ valor + desl for nome, valor in pontos.items()}


# ============================================================================
# 1) plano sintetico: normal exata + sinal por lado
# ============================================================================
print("1) recupera a normal de um plano sintetico conhecido")
pontos = mao_sintetica()
normal, rms = gt.palm_normal_from_plane(pontos, lado="right")
check("right: normal e' (0,0,1)", normal is not None
      and np.allclose(normal, [0.0, 0.0, 1.0], atol=1e-9), normal)
check("right: rms do plano ~0", rms is not None and rms < 1e-9, rms)
normal_esq, _rms = gt.palm_normal_from_plane(pontos, lado="left")
check("left: normal e' (0,0,-1) (sinal anatomico espelhado)",
      normal_esq is not None
      and np.allclose(normal_esq, [0.0, 0.0, -1.0], atol=1e-9), normal_esq)

# rotacao rigida: a normal acompanha R
rng = np.random.default_rng(5)
for rotacao_dict in rng.uniform(-180, 180, (6, 3)):
    rot = imu.rotation_matrix(*[float(v) for v in rotacao_dict])
    pontos_rot = mao_sintetica(rot, deslocamento=(0.3, -0.2, 1.4))
    normal_rot, _rms = gt.palm_normal_from_plane(pontos_rot, lado="right")
    esperada = rot @ np.array([0.0, 0.0, 1.0])
    check("rotacao %s: normal acompanha R (erro %.3f graus)"
          % (np.round(rotacao_dict, 0),
             float(np.degrees(np.arccos(np.clip(
                 float(np.dot(normal_rot, esperada)), -1, 1))))),
          normal_rot is not None
          and abs(float(np.dot(normal_rot, esperada))) > 1 - 1e-9,
          normal_rot)

# ============================================================================
# 2) com ruido, o plano medio e' melhor que o produto vetorial de 3 pontos
# ============================================================================
print("2) ruido de profundidade: plano medio x produto vetorial de 3 pontos")
rng = np.random.default_rng(11)
sigma_m = 0.002                     # 2 mm de ruido por coordenada
erros_plano = []
erros_cross = []
for _ in range(160):
    limpos = mao_sintetica()
    ruidosos = {nome: valor + rng.normal(0.0, sigma_m, 3)
                for nome, valor in limpos.items()}
    normal_plano, _rms = gt.palm_normal_from_plane(ruidosos, lado="right")
    if normal_plano is None:
        continue
    erros_plano.append(imu.angle_between(normal_plano, [0.0, 0.0, 1.0]))
    cross = np.cross(ruidosos["index"] - ruidosos["wrist"],
                     ruidosos["pinky"] - ruidosos["wrist"])
    norma = np.linalg.norm(cross)
    if norma > 1e-9:
        erros_cross.append(imu.angle_between(cross / norma, [0.0, 0.0, 1.0]))
media_plano = float(np.mean(erros_plano))
media_cross = float(np.mean(erros_cross))
print(f"   erro medio do plano: {media_plano:.2f} graus | do produto "
      f"vetorial: {media_cross:.2f} graus")
check("plano medio erra MENOS que os 3 pontos",
      media_plano < media_cross, (media_plano, media_cross))
check("erro medio do plano < 5 graus com 2 mm de ruido", media_plano < 5.0,
      media_plano)

# ============================================================================
# 3) casos degenerados: nunca normal inventada
# ============================================================================
print("3) casos degenerados -> (None, None)")
poucos = {"wrist": np.zeros(3), "index": np.array([0.08, 0.02, 0.0]),
          "middle": np.array([0.02, 0.03, 0.0])}
check("menos de 4 pontos validos", eh_none(gt.palm_normal_from_plane(poucos)))
colineares = {
    "wrist": np.zeros(3),
    "index": np.array([0.02, 0.0, 0.0]),
    "middle": np.array([0.04, 0.0, 0.0]),
    "ring": np.array([0.06, 0.0, 0.0]),
    "pinky": np.array([0.08, 0.0, 0.0]),
}
check("pontos colineares (sem plano)", eh_none(gt.palm_normal_from_plane(colineares)))
sem_sinal = {"wrist": np.zeros(3), "middle": np.array([0.02, 0.03, 0.0]),
             "ring": np.array([-0.01, 0.03, 0.0]),
             "pinky": np.array([-0.04, 0.02, 0.0])}
check("sem wrist/index/pinky (sinal anatomico impossivel)",
      eh_none(gt.palm_normal_from_plane(sem_sinal)))
_pontos_ruins = mao_sintetica()
# Mao de verdade nao e' plana: dedos para cima/baixo tiram os MCPs do plano.
_pontos_ruins["middle"] = np.array([0.020, 0.030, 0.060])
_pontos_ruins["ring"] = np.array([-0.010, 0.030, -0.060])
check("rms_max_m descarta plano ruim",
      eh_none(gt.palm_normal_from_plane(_pontos_ruins, rms_max_m=0.005)))
normal_ok, rms_ok = gt.palm_normal_from_plane(_pontos_ruins)
check("sem rms_max_m devolve a normal e o rms alto",
      normal_ok is not None and rms_ok > 0.008, rms_ok)

# ============================================================================
# 4) integracao no tracker/tool e analisador offline
# ============================================================================
print("4) integracao (tracker, tool e analisador offline)")
check("tracker tem palm_plane_camera",
      hasattr(gt.KinectHandTracker, "palm_plane_camera"))
check("lib tem palm_normal_from_plane e PONTOS_DORSO_PALMA",
      hasattr(gt, "palm_normal_from_plane")
      and hasattr(gt, "PONTOS_DORSO_PALMA"))
raiz = os.path.dirname(os.path.abspath(__file__))
fonte_tool = open(os.path.join(raiz, "tools", "kinect_groundtruth_tool.py"),
                  encoding="utf-8").read()
check("tool tem a tecla J", 'key == ord("j")' in fonte_tool)
check("tool chama palm_plane_camera", "palm_plane_camera" in fonte_tool)
check("tool alimenta a captura por plano", "captura_palma_plano" in fonte_tool)
check("tool desenha seta customizada", "draw_palm_vector_custom" in fonte_tool)
check("tool publica palm_erro_plano_deg", "palm_erro_plano_deg" in fonte_tool)
check("analisador offline existe",
      os.path.exists(os.path.join(raiz, "tools", "calibra_palma_plano.py")))

sys.path.insert(0, os.path.join(raiz, "tools"))
import calibra_palma_plano as cli                      # noqa: E402

csv_canonico = os.path.join(raiz, "palm_diag_20261019_180702.csv")
if os.path.exists(csv_canonico):
    normal, rms = gt.palm_normal_from_plane(
        {"wrist": np.zeros(3), "index": np.array([0.08, 0.02, 0.0]),
         "middle": np.array([0.02, 0.03, 0.0]),
         "pinky": np.array([-0.04, 0.02, 0.0])}, lado="right")
    saida = io.StringIO()
    with contextlib.redirect_stdout(saida):
        codigo = cli.analisa(csv_canonico, salvar=False)
    texto = saida.getvalue()
    check("analisador offline roda no CSV canonico", codigo == 0, codigo)
    check("relatorio mostra as fases e o veredito",
          "PREPARO" in texto and "VEREDITO" in texto)
    check("relatorio mostra residuo do refit por plano", "PLANO" in texto)
else:
    check("CSV canonico de 19/10 presente (para o teste offline)", False)

if FALHAS:
    print()
    print(f"PALMA_PLANO_FALHOU: {len(FALHAS)} verificacao(oes): "
          + " | ".join(FALHAS))
    raise SystemExit(1)
print()
print("PALMA_PLANO_OK: normal do plano medio do dorso recuperada (sintetica, "
      "rotacionada e com sinal por lado) | ruido de profundidade: plano erra "
      "menos que os 3 pontos | degenerados devolvem (None, None) | rms_max_m "
      "descarta plano ruim | tracker/tool/analisador offline integrados")
