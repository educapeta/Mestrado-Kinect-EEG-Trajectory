"""Testes da CALIBRACAO GEOMETRICA da palma (item 127, tecla X).

Contexto: o J usa a verdade `direcao mao->lente` (posicao 3D), que e' precisa
mas so' cobre direcoes perto do eixo optico (as poses de mira ficam todas num
cone estreito em volta de -Z). A calibracao geometrica usa ALVOS EXATOS do
referencial da camera (normal da palma em +-X, +-Y, +-Z e diagonais), com o
usuario alinhando o plano da mao a um plano-guia no video; a verdade nao tem
erro de MEDIDA (so' de execucao, controlada por QA). No fim entra uma pose de
MIRA DESLOCADA (verdade por posicao) para ancorar a direcao absoluta, e o
ajuste T/f e' hibrido e PONDERADO.

Casos:
  1. solver com pesos recupera T/f de dados limpos e ignora outlier com peso
     baixo (a mesma conta usada pelo ajuste hibrido);
  2. `CalibracaoGeometrica` grava as 9 poses sinteticas (8 geometricas + 1
     mira), fica completa e recupera T/f;
  3. QA: normal fora do alvo nao grava take; movimento insuficiente desde o
     ultimo take nao arma o proximo; `pular_pose` avanca sem gravar;
  4. integracao no tool: teclas X/N, overlay do plano, finalizacao e desenho
     do plano-guia presentes (static check, sem hardware).

Harness: aritmetica pura (sem Qt, Kinect ou rede).
"""
from __future__ import annotations

import numpy as np

import imu
import kinect_imu_groundtruth as k

FALHAS = []


def check(nome, condicao, detalhe=None):
    if condicao:
        print(f"   ok: {nome}")
    else:
        FALHAS.append(nome)
        print(f"   FALHOU: {nome} | {detalhe!r}")


# ============================================================================
# 1) solver com pesos
# ============================================================================
print("1) solver ponderado: recupera T/f limpos e ignora outlier de peso baixo")
rng = np.random.default_rng(11)
T_verdade = imu.rotation_matrix(25.0, -12.0, 40.0)
f_verdade = np.array([0.15, -0.25, 0.95])
f_verdade /= np.linalg.norm(f_verdade)
rotacoes = [imu.rotation_matrix(*(rng.uniform(-180, 180, 3)))
            for _ in range(9)]
direcoes = [T_verdade @ R @ f_verdade for R in rotacoes]
pesos = [1.0] * 9
solucao = imu.solve_palm_alignment_pesos(rotacoes, direcoes, pesos=pesos)
check("solver ponderado devolve solucao", solucao is not None)
if solucao is not None:
    T_est, f_est, residuo = solucao
    check("residuo limpo ~0", residuo < 1e-5, residuo)
    check("T recuperado", np.allclose(T_est, T_verdade, atol=1e-6),
          np.abs(T_est - T_verdade).max())
    check("f recuperado", np.allclose(f_est, f_verdade, atol=1e-6),
          np.abs(f_est - f_verdade).max())
# outlier com peso baixo nao pode estragar o ajuste
direcoes_ruins = list(direcoes)
direcoes_ruins[0] = np.array([0.0, 0.0, 1.0])
sol_pond = imu.solve_palm_alignment_pesos(
    rotacoes, direcoes_ruins, pesos=[0.01] + [1.0] * 8)
check("outlier com peso baixo: solucao perto da verdade",
      sol_pond is not None and
      np.allclose(sol_pond[0], T_verdade, atol=2e-2),
      None if sol_pond is None else np.abs(sol_pond[0] - T_verdade).max())

# ============================================================================
# 2) state machine sintetica: 9 poses, completa, recupera T/f
# ============================================================================
print("2) CalibracaoGeometrica: 9 poses sinteticas -> soluçao")
calib = k.CalibracaoGeometrica("right", avisar=lambda *_a: None)
pos_geometrica = np.array([0.10, 0.05, 0.45])
pos_mira = np.array([0.20, 0.06, 0.45])
agora = 0.0
for pose in calib.poses:
    if pose["tipo"] == "posicao":
        posicao = pos_mira
        verdade = (-posicao / np.linalg.norm(posicao)
                   * float(pose.get("sinal", 1.0)))
    else:
        posicao = pos_geometrica
        verdade = np.asarray(pose["alvo"], np.float64)
        verdade = verdade / np.linalg.norm(verdade)
    R = k._rotacao_a_para_b(f_verdade, T_verdade.T @ verdade)
    for _ in range(25):
        agora += 0.1
        calib.amostra(agora, R, posicao, verdade)
check("gravou 9 takes", len(calib.capturas) == 9, len(calib.capturas))
check("completa", calib.completa is True)
resultado = calib.resultado()
check("resultado sem erro", "erro" not in resultado, resultado.get("erro"))
if "erro" not in resultado:
    check("residuo ponderado ~0", resultado["residual_deg"] < 1e-5,
          resultado["residual_deg"])
    check("T recuperado na calibracao",
          np.allclose(np.asarray(resultado["rotation"]), T_verdade,
                      atol=1e-6))
    check("f recuperado na calibracao",
          np.allclose(np.asarray(resultado["body"]), f_verdade, atol=1e-6))
    check("validacao da mira presente",
          resultado.get("validacao_mira") is not None)
    check("validacao cruzada presente",
          resultado.get("loo_mediana_deg") is not None)

# ============================================================================
# 3) QA de execucao, movimento minimo e pular pose
# ============================================================================
print("3) QA de execucao, gate de movimento e pular pose")
# QA: normal 30 graus fora do alvo nao grava (usa poses GEOMETRICAS; as
# posicoes por posicao nao tem alvo fixo)
poses_geo_teste = [pose for pose in k.POSES_CALIBRACAO_GEOMETRICA
                   if pose["tipo"] == "geometrica"][:2]
calib_qa = k.CalibracaoGeometrica("right", poses=poses_geo_teste,
                                 avisar=lambda *_a: None)
pose0 = calib_qa.pose_atual
alvo0 = np.asarray(pose0["alvo"], np.float64)
normal_ruim = alvo0 + np.array([0.0, 0.5, 0.5])
normal_ruim /= np.linalg.norm(normal_ruim)
agora = 0.0
R0 = k._rotacao_a_para_b(f_verdade, T_verdade.T @ alvo0)
for _ in range(40):
    agora += 0.1
    calib_qa.amostra(agora, R0, pos_geometrica, normal_ruim)
check("QA alto NAO grava", len(calib_qa.capturas) == 0,
      len(calib_qa.capturas))
check("QA alto mantem status de alinhamento",
      "ALINHE" in calib_qa.status.upper(), calib_qa.status)

# movimento minimo: depois do 1o take, repetir a MESMA rotacao nao arma o 2o
calib_mov = k.CalibracaoGeometrica("right", poses=poses_geo_teste,
                                 avisar=lambda *_a: None)
agora = 0.0
for _ in range(25):
    agora += 0.1
    calib_mov.amostra(agora, R0, pos_geometrica, alvo0)
check("movimento: 1o take gravado", len(calib_mov.capturas) == 1)
pose1 = calib_mov.pose_atual
alvo1 = np.asarray(pose1["alvo"], np.float64)
for _ in range(40):
    agora += 0.1
    calib_mov.amostra(agora, R0, pos_geometrica, alvo1)
check("movimento: sem mover a mao NAO grava o 2o take",
      len(calib_mov.capturas) == 1, len(calib_mov.capturas))
R1 = k._rotacao_a_para_b(f_verdade, T_verdade.T @ alvo1)
for _ in range(30):
    agora += 0.1
    calib_mov.amostra(agora, R1, pos_geometrica, alvo1)
check("movimento: apos mover, grava o 2o take",
      len(calib_mov.capturas) == 2, len(calib_mov.capturas))

# pular pose avanca sem gravar
calib_skip = k.CalibracaoGeometrica("right", avisar=lambda *_a: None)
evento = calib_skip.pular_pose()
check("pular pose devolve evento", evento is not None and
      evento["tipo"] == "pulada")
check("pular pose avanca sem gravar",
      calib_skip.indice == 1 and len(calib_skip.capturas) == 0,
      (calib_skip.indice, len(calib_skip.capturas)))

# ============================================================================
# 3b) amostra com rotacao NaN/inf nao pode gravar take (item 127, achado M2)
# ============================================================================
print("3b) rotacao NaN/inf nao entra na captura")
calib_nan = k.CalibracaoGeometrica("right", poses=poses_geo_teste[:1],
                                 avisar=lambda *_a: None)
alvo_nan = np.asarray(calib_nan.pose_atual["alvo"], np.float64)
R_valida = k._rotacao_a_para_b(f_verdade, T_verdade.T @ alvo_nan)
agora = 0.0
for indice in range(30):
    agora += 0.1
    if indice % 2 == 0:
        R_usar = np.full((3, 3), np.nan)
    else:
        R_usar = R_valida
    calib_nan.amostra(agora, R_usar, pos_geometrica, alvo_nan)
check("rotacao NaN nao grava take", len(calib_nan.capturas) == 0,
      len(calib_nan.capturas))
check("sequencia com NaN continua sem crash", calib_nan.indice == 0)

# ============================================================================
# 3b2) normal_robusta: virada REAL de 180 graus entre poses (item 131)
# ============================================================================
print("3b2) normal_robusta permite virada real de 180 graus (modo X)")
anterior = np.array([0.0, 0.0, 1.0])
rng_sinal = np.random.default_rng(31)
hist_flip = [np.array([0.0, 0.0, -1.0])
             + rng_sinal.normal(0.0, 0.005, 3) for _ in range(8)]
med_antigo, _frac = k.normal_robusta(hist_flip, normal_anterior=anterior)
check("continuidade antiga trava o sinal (documenta o bug)",
      med_antigo is not None and float(np.dot(med_antigo, anterior)) > 0.9,
      med_antigo)
med_novo, _frac = k.normal_robusta(
    hist_flip, normal_anterior=anterior,
    referencia_sinal=np.array([0.0, 0.0, -1.0]))
check("referencia anatomica permite a virada de 180 graus",
      med_novo is not None and float(np.dot(med_novo, anterior)) < -0.9,
      med_novo)
check("tool usa referencia_sinal no modo X",
      "referencia_sinal=(normal_plano_lado" in open(
          "tools/kinect_groundtruth_tool.py", encoding="utf-8").read())

# ============================================================================
# 3d) poses de POSICAO ignoram o QA do Kinect (item 135)
# ============================================================================
print("3d) pose de POSICAO grava mesmo com Kinect invertendo palma/dorso")
pose_posicao = [pose for pose in k.POSES_CALIBRACAO_GEOMETRICA
                if pose["tipo"] == "posicao"][0]
verdade_pos = (-pos_mira / np.linalg.norm(pos_mira)
               * float(pose_posicao.get("sinal", 1.0)))
R_pos = k._rotacao_a_para_b(f_verdade, T_verdade.T @ verdade_pos)
calib_pos = k.CalibracaoGeometrica("right", poses=[pose_posicao],
                                  avisar=lambda *_a: None)
agora = 0.0
for _ in range(25):
    agora += 0.1
    # normal INVERTIDA (como o Kinect "achando" que a mao e' o dorso)
    calib_pos.amostra(agora, R_pos, pos_mira, -verdade_pos)
check("posicao: normal invertida NAO bloqueia o take",
      len(calib_pos.capturas) == 1, len(calib_pos.capturas))
if calib_pos.capturas:
    cap = calib_pos.capturas[0]
    check("posicao: QA registrado ~180 (informativo)",
          cap["qa_deg"] is not None and cap["qa_deg"] > 150.0,
          cap["qa_deg"])
    check("posicao: take nao depende da normal",
          cap["normal_medida"] is not None)
# geometrica continua exigindo o plano certo
calib_geo_ruim = k.CalibracaoGeometrica("right", poses=poses_geo_teste[:1],
                                        avisar=lambda *_a: None)
alvo_geo = np.asarray(calib_geo_ruim.pose_atual["alvo"], np.float64)
R_geo = k._rotacao_a_para_b(f_verdade, T_verdade.T @ alvo_geo)
agora = 0.0
for _ in range(30):
    agora += 0.1
    calib_geo_ruim.amostra(agora, R_geo, pos_geometrica, -alvo_geo)
# posicao com plano PERPENDICULAR a linha de mira (QA dobrado ~90) bloqueia
calib_pos_ruim = k.CalibracaoGeometrica("right", poses=[pose_posicao],
                                        avisar=lambda *_a: None)
normal_perp = np.cross(verdade_pos, np.array([0.0, 0.0, 1.0]))
if np.linalg.norm(normal_perp) < 1e-6:
    normal_perp = np.cross(verdade_pos, np.array([1.0, 0.0, 0.0]))
normal_perp /= np.linalg.norm(normal_perp)
agora = 0.0
for _ in range(30):
    agora += 0.1
    calib_pos_ruim.amostra(agora, R_pos, pos_mira, normal_perp)
check("posicao: plano perpendicular a linha de mira bloqueia",
      len(calib_pos_ruim.capturas) == 0, len(calib_pos_ruim.capturas))
check("geometrica: normal invertida continua bloqueando",
      len(calib_geo_ruim.capturas) == 0, len(calib_geo_ruim.capturas))

# ============================================================================
# 3c) finalizar_calibracao_geometrica: gate, JSON piloto e oficial + .bak
# ============================================================================
print("3c) finalizar: gate grava piloto + oficial e preserva .bak")
import importlib.util
import json
import os
import tempfile
from pathlib import Path

raiz = Path(__file__).resolve().parent
spec_tool = importlib.util.spec_from_file_location(
    "kgt_tool_para_teste", raiz / "tools" / "kinect_groundtruth_tool.py")
tool = importlib.util.module_from_spec(spec_tool)
spec_tool.loader.exec_module(tool)

calib_final = k.CalibracaoGeometrica("right", avisar=lambda *_a: None)
agora = 0.0
for pose in calib_final.poses:
    if pose["tipo"] == "posicao":
        posicao = pos_mira
        verdade = (-posicao / np.linalg.norm(posicao)
                   * float(pose.get("sinal", 1.0)))
    else:
        posicao = pos_geometrica
        verdade = np.asarray(pose["alvo"], np.float64)
        verdade = verdade / np.linalg.norm(verdade)
    R = k._rotacao_a_para_b(f_verdade, T_verdade.T @ verdade)
    for _ in range(25):
        agora += 0.1
        calib_final.amostra(agora, R, posicao, verdade)
check("finalizar: calibracao sintetica completa",
      calib_final.completa and len(calib_final.capturas) == 9)

cwd_anterior = os.getcwd()
avisos_final = []
with tempfile.TemporaryDirectory(prefix="calib_geo_") as tmp:
    tmp_path = Path(tmp)
    os.chdir(tmp_path)
    tool.PALM_ALIGNMENT_FILES = {
        "right": str(tmp_path / "imu_palm_alignment_right.json"),
        "left": str(tmp_path / "imu_palm_alignment_left.json"),
    }
    fusao = imu.PositionFusion()
    try:
        res1 = tool.finalizar_calibracao_geometrica(
            calib_final, fusao, participante="teste", avisar=avisos_final.append)
        check("finalizar: ok", res1.get("ok") is True, res1)
        check("finalizar: salvou oficial (gate passou)",
              res1.get("salvou_oficial") is True, res1)
        pilotos = list(tmp_path.glob("calibracao_geometrica_right_*.json"))
        check("finalizar: JSON piloto gravado", len(pilotos) == 1, pilotos)
        oficial = tmp_path / "imu_palm_alignment_right.json"
        check("finalizar: oficial gravado", oficial.exists())
        if pilotos:
            piloto = json.loads(pilotos[0].read_text(encoding="utf-8"))
            check("finalizar: piloto com 9 posicoes",
                  len(piloto.get("posicoes", ())) == 9)
            check("finalizar: piloto sem NaN/Infinity",
                  "NaN" not in pilotos[0].read_text(encoding="utf-8")
                  and "Infinity" not in pilotos[0].read_text(encoding="utf-8"))
        if oficial.exists():
            dados_oficial = json.loads(oficial.read_text(encoding="utf-8"))
            check("finalizar: oficial tem T/f/residuo",
                  set(dados_oficial) >= {"rotation", "body", "residual_deg"})
            check("finalizar: T oficial perto da verdade",
                  np.allclose(np.asarray(dados_oficial["rotation"]), T_verdade,
                              atol=1e-6))
        # roda de novo: deve criar .bak do oficial anterior
        res2 = tool.finalizar_calibracao_geometrica(
            calib_final, fusao, participante="teste", avisar=avisos_final.append)
        check("finalizar: segunda vez ok", res2.get("ok") is True)
        check("finalizar: .bak do oficial criado",
              (tmp_path / "imu_palm_alignment_right.json.bak").exists())
    finally:
        os.chdir(cwd_anterior)

# ============================================================================
# 4) integracao no tool (estatico)
# ============================================================================
print("4) integracao no tool (teclas X/N, overlay, finalizacao, desenho)")
fonte_tool = open("tools/kinect_groundtruth_tool.py",
                  encoding="utf-8").read()
check("tool tem handler da tecla X", 'ord("x")' in fonte_tool)
check("tool tem handler da tecla N", 'ord("n")' in fonte_tool)
check("tool instancia CalibracaoGeometrica",
      "CalibracaoGeometrica(" in fonte_tool)
check("tool finaliza gravando JSON/solucao",
      "finalizar_calibracao_geometrica(" in fonte_tool)
check("tool desenha o plano-guia",
      "desenhar_guia_plano_palma(" in fonte_tool)
check("tool sabe pular pose", ".pular_pose()" in fonte_tool)
check("tool tem --log-limpo (tela limpa no X)",
      "--log-limpo" in fonte_tool and "LOG_LIMPO" in fonte_tool)
check("tool avisa quando mao/depth nao detectados (plano ancorado na mao)",
      "MAO NAO DETECTADA" in fonte_tool and
      "mao/plano NAO detectado" in fonte_tool)
fonte_lib = open("kinect_imu_groundtruth.py", encoding="utf-8").read()
check("biblioteca define a classe/pesos/poses",
      "class CalibracaoGeometrica" in fonte_lib and
      "POSES_CALIBRACAO_GEOMETRICA" in fonte_lib and
      "CALIB_GEO_QA_MAX_DEG" in fonte_lib)
check("solver ponderado importado na biblioteca",
      "solve_palm_alignment_pesos" in fonte_lib)

if FALHAS:
    print()
    print(f"CALIBRACAO_GEOMETRICA_FALHOU: {len(FALHAS)} verificacao(oes): "
          + " | ".join(FALHAS))
    raise SystemExit(1)
print()
print("CALIBRACAO_GEOMETRICA_OK: solver ponderado (outlier com peso baixo), "
      "9 poses sinteticas recuperam T/f, QA alto nao grava, gate de "
      "movimento, pular pose, rotacao NaN rejeitada, finalizar grava "
      "piloto/oficial/.bak, integracao no tool (X/N/overlay/finalizacao)")
