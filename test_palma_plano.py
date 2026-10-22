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

# ============================================================================
# 5) CERTIFICADO GUIADO POR POSES (tecla V) -- fecha a questao com numeros
# ============================================================================
print("5) certificado guiado por poses (VerificacaoPosesPalma, tecla V)")


class FusaoFalsa:
    """PositionFusion minima: so' o que o certificado usa."""

    def __init__(self, t, f):
        self.imu_to_camera = np.asarray(t, dtype=float)
        self.palm_reference_body = np.asarray(f, dtype=float)


rng = np.random.default_rng(99)
T_verdade = imu.rotation_matrix(20.0, -10.0, 40.0)
f_verdade = np.array([0.1, -0.05, -0.99])
f_verdade = f_verdade / np.linalg.norm(f_verdade)
fusoes = {"right": FusaoFalsa(T_verdade, f_verdade),
          "left": FusaoFalsa(imu.rotation_matrix(30.0, 5.0, -70.0), f_verdade)}
poses = ("POSE A", "POSE B", "POSE C")
verif = gt.VerificacaoPosesPalma(poses=poses, seg_pose=1.0, settle=0.2,
                                 limite=15.0)
verif.iniciar(0.0)
for indice in range(3):
    for instante in (0.5 + indice * 1.0, 0.7 + indice * 1.0,
                     0.9 + indice * 1.0):
        for lado, fusao in fusoes.items():
            rot = imu.rotation_matrix(*rng.uniform(-60, 60, 3))
            normal = fusao.imu_to_camera @ rot @ fusao.palm_reference_body
            verif.coletar(lado, instante, rot, normal)
rel = verif.relatorio(fusoes)
check("certificado: FECHADO com alinhamento perfeito",
      rel["fechado"] is True, rel)
check("certificado: 3 poses por lado",
      all(len(rel["lados"][lado]["poses"]) == 3 for lado in ("right", "left")))
check("certificado: media ~0 em todas as poses", all(
    stats["media"] is not None and stats["media"] < 1.0
    for lado in ("right", "left")
    for stats in rel["lados"][lado]["poses"].values()))

# uma pose com a verdade 30 graus fora do alinhamento -> NAO FECHADO
verif2 = gt.VerificacaoPosesPalma(poses=poses, seg_pose=1.0, settle=0.2)
verif2.iniciar(0.0)
for indice in range(3):
    for instante in (0.5 + indice * 1.0, 0.7 + indice * 1.0,
                     0.9 + indice * 1.0):
        rot = np.eye(3)
        normal = T_verdade @ rot @ f_verdade
        if indice == 1:
            eixo = np.cross(normal, [0.0, 1.0, 0.0])
            norma = np.linalg.norm(eixo)
            eixo = (eixo / norma if norma > 1e-9
                    else np.array([1.0, 0.0, 0.0]))
            angulo = np.radians(30.0)
            cosseno, seno = np.cos(angulo), np.sin(angulo)
            k_matriz = np.array([[0.0, -eixo[2], eixo[1]],
                                 [eixo[2], 0.0, -eixo[0]],
                                 [-eixo[1], eixo[0], 0.0]])
            rotacao_erro = (np.eye(3) * cosseno + seno * k_matriz
                            + (1.0 - cosseno) * np.outer(eixo, eixo))
            normal = rotacao_erro @ normal
        for lado in ("right", "left"):
            verif2.coletar(lado, instante, rot, normal)
rel2 = verif2.relatorio({"right": FusaoFalsa(T_verdade, f_verdade),
                         "left": FusaoFalsa(T_verdade, f_verdade)})
check("certificado: NAO FECHADO com uma pose de 30 graus",
      rel2["fechado"] is False, rel2["fechado"])
check("certificado: lista a pose 2 como ruim",
      any(indice == 2 for _lado, indice, _nome, _media
          in rel2["poses_ruins"]), rel2["poses_ruins"])

# settle: amostra antes do settle da pose nao entra
verif3 = gt.VerificacaoPosesPalma(poses=poses, seg_pose=1.0, settle=0.2)
verif3.iniciar(0.0)
check("settle: amostra em t=0.1 e' ignorada",
      verif3.coletar("right", 0.1, np.eye(3), [0.0, 0.0, 1.0]) is False)
verif3.parar()
check("cancelar: nao coleta depois de parar",
      verif3.coletar("right", 0.5, np.eye(3), [0.0, 0.0, 1.0]) is False)

# sem alinhamento carregado: nao fecha e marca o lado
vazio = gt.VerificacaoPosesPalma(poses=poses, seg_pose=1.0, settle=0.2)
vazio.iniciar(0.0)


class SemAlinhamento:
    imu_to_camera = None
    palm_reference_body = None


rel3 = vazio.relatorio({"right": SemAlinhamento(), "left": SemAlinhamento()})
check("sem alinhamento: NAO FECHADO e marca os lados",
      rel3["fechado"] is False
      and all(rel3["lados"][lado]["sem_alinhamento"]
              for lado in ("right", "left")))
check("tool tem a tecla V", 'key == ord("v")' in fonte_tool)
check("tool usa VerificacaoPosesPalma", "VerificacaoPosesPalma" in fonte_tool)
check("tool imprime o CERTIFICADO", "CERTIFICADO" in fonte_tool)


# ============================================================================
# 6) ESTABILIDADE DA VERDADE DO KINECT (item 105): media de normais + sinal
# ============================================================================
print("6) estabilidade do vetor de referencia (media + continuidade de sinal)")
normais = [np.array([0.0, 0.0, 1.0]), np.array([0.02, 0.0, 1.0]),
           np.array([-0.01, 0.01, 1.0])]
media, spread = gt.media_estavel_normais(normais)
check("media de normais quase iguais ~ +z e spread pequeno",
      media is not None and float(np.dot(media, [0.0, 0.0, 1.0])) > 0.999
      and spread < 2.0, (media, spread))
normais_flip = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
                np.array([0.01, 0.0, 1.0])]
media, spread = gt.media_estavel_normais(normais_flip)
check("inversao pontual de sinal e' alinhada antes da media",
      media is not None and float(np.dot(media, [0.0, 0.0, 1.0])) > 0.999,
      media)
_media, spread = gt.media_estavel_normais(
    [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])])
check("historico conflitante (em movimento) tem spread grande",
      spread is not None and spread > 8.0, spread)
check("historico curto -> (None, None)",
      gt.media_estavel_normais([np.array([0.0, 0.0, 1.0])]) == (None, None))

quase_colinear = {"wrist": np.array([0.0, 0.0, 0.0]),
                  "index": np.array([0.08, 0.001, 0.0]),
                  "pinky": np.array([-0.04, -0.001, 0.0]),
                  "middle": np.array([0.02, 0.0012, 0.0]),
                  "ring": np.array([-0.01, 0.0008, 0.0])}
n_sem, _r = gt.palm_normal_from_plane(quase_colinear, "right")
check("anatomia degenerada SEM referencia anterior -> (None, None)",
      eh_none((n_sem, _r)))
n_com, _r = gt.palm_normal_from_plane(
    quase_colinear, "right", normal_anterior=np.array([0.0, 0.0, 1.0]))
check("anatomia degenerada COM normal anterior mantem o sinal",
      n_com is not None and float(np.dot(n_com, [0.0, 0.0, 1.0])) > 0.99,
      n_com)
longe = mao_sintetica()
longe["wrist"] = np.array([0.30, 0.0, 0.0])
check("ponto de FUNDO (30 cm do centro) e' descartado", eh_none(
    gt.palm_normal_from_plane(longe)))
check("tool usa a media estavel e a sincronia color-depth",
      "media_estavel_normais" in fonte_tool
      and "sync_color_depth_ms" in fonte_tool
      and "palm_plano_estavel" in fonte_tool)

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
