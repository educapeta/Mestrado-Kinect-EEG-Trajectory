"""Testes do nucleo do visualizador de vetores AHRS (item 142)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import ahrs_vetores_core as core  # noqa: E402

FALHAS = []


def check(nome, condicao, detalhe=None):
    if condicao:
        print(f"   ok: {nome}")
    else:
        FALHAS.append(nome)
        print(f"   FALHOU: {nome} | {detalhe!r}")


print("1) quaternion -> matriz")
R_ident = core.quat_para_matriz(1.0, 0.0, 0.0, 0.0)
check("identidade", np.allclose(R_ident, np.eye(3), atol=1e-12))
q_z90 = core.quat_axis_angle(np.array([0.0, 0.0, 1.0]), math.pi / 2.0)
R_z90 = core.quat_para_matriz(*q_z90)
check("+90 em Z leva X para Y",
      np.allclose(R_z90 @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0],
                  atol=1e-9), R_z90 @ np.array([1.0, 0.0, 0.0]))
check("quaternion normalizado",
      np.allclose(core.quat_para_matriz(2.0, 0.0, 0.0, 0.0), np.eye(3)))
try:
    core.quat_para_matriz(float("nan"), 0.0, 0.0, 0.0)
    check("NaN rejeitado", False)
except ValueError:
    check("NaN rejeitado", True)

print("2) parser do protocolo serial")
linha = ("A,1234567,1,0,0,0,1,0,0,0,"
         "0.01,-0.02,1.0,0.0,0.01,0.99")
dados = core.parse_linha_ahrs(linha)
check("linha valida parseada", dados is not None)
if dados:
    check("t_us/t_s", dados["t_us"] == 1234567.0 and
          abs(dados["t_s"] - 1.234567) < 1e-9)
    check("quaternions L/R", dados["left"].shape == (4,) and
          dados["right"].shape == (4,))
    check("accel L/R", dados["accel_left"].shape == (3,) and
          dados["accel_right"].shape == (3,))
check("cabecalho ignorado", core.parse_linha_ahrs("# AHRS_VETOR v1") is None)
check("linha E ignorada", core.parse_linha_ahrs("E,0,D") is None)
check("linha curta ignorada", core.parse_linha_ahrs("A,1,0,0") is None)
check("NaN na linha ignorada",
      core.parse_linha_ahrs("A,1,1,0,0,0,1,0,0,0,1,0,0,1,0,nan") is None)

print("3) vetor comeca em -Z e gira com o sensor")
v_corpo = core.vetor_corpo_para_baixo(R_ident)
check("tara identidade -> vetor corpo -Z", np.allclose(v_corpo, [0, 0, -1]))
v_inicial = core.vetor_mundo(R_ident, v_corpo)
check("vetor inicial no mundo -Z", np.allclose(v_inicial, [0, 0, -1]))
check("angulo inicial com chao = 0",
      abs(core.angulo_com_baixo(v_inicial)) < 1e-9)
# sensor inclinado 90 graus em X (face vira para +Y): vetor deve ir para +Y
R_x90 = core.quat_para_matriz(*core.quat_axis_angle(
    np.array([1.0, 0.0, 0.0]), math.pi / 2.0))
v_girado = core.vetor_mundo(R_x90, v_corpo)
check("rotacao 90 em X leva o vetor para +Y",
      np.allclose(v_girado, [0.0, 1.0, 0.0], atol=1e-9), v_girado)
check("angulo com chao ~90",
      abs(core.angulo_com_baixo(v_girado) - 90.0) < 1e-6)
# tara com orientacao inicial arbitraria continua -Z
v_corpo2 = core.vetor_corpo_para_baixo(R_x90)
check("tara em orientacao arbitraria -> mundo -Z",
      np.allclose(core.vetor_mundo(R_x90, v_corpo2), [0, 0, -1], atol=1e-9))

print("4) simulacao")
q_esq, q_dir = core.quat_simulado(1.0)
check("quaternions simulados unitarios",
      abs(np.linalg.norm(q_esq) - 1.0) < 1e-12 and
      abs(np.linalg.norm(q_dir) - 1.0) < 1e-12)
check("matrizes ortogonais",
      np.allclose(core.quat_para_matriz(*q_esq).T @
                  core.quat_para_matriz(*q_esq), np.eye(3), atol=1e-9))

print("4b) extrair linhas de buffer fragmentado")
linhas, resto = core.extrair_linhas(b"A,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0\nA,2,1")
check("uma linha completa + resto",
      linhas == ["A,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0"] and resto == b"A,2,1")
linhas2, resto2 = core.extrair_linhas(resto + b"0,0,0,0,0,0,0,0,0,1,0,0\n")
check("completa a segunda linha",
      len(linhas2) == 1 and linhas2[0].startswith("A,2") and resto2 == b"")
linhas3, resto3 = core.extrair_linhas(b"")
check("buffer vazio", linhas3 == [] and resto3 == b"")

print("5) arquivos/integracao")
raiz = Path(__file__).resolve().parent
fw_main = raiz / "firmware" / "esp32_teste_ahrs" / "src" / "main.cpp"
fw_ini = raiz / "firmware" / "esp32_teste_ahrs" / "platformio.ini"
visual = raiz / "tools" / "visualiza_vetores_ahrs.py"
check("firmware de teste existe", fw_main.exists())
if fw_main.exists():
    fonte_fw = fw_main.read_text(encoding="utf-8")
    check("firmware publica AHRS_VETOR", "AHRS_VETOR" in fonte_fw)
    check("firmware usa os dois enderecos",
          "ADDR_DIREITO" in fonte_fw and "ADDR_ESQUERDO" in fonte_fw)
    check("firmware tem getter de quaternion",
          "obter_quaternion" in fonte_fw)
    check("firmware tem comando de recalibracao",
          "calibrarTudo()" in fonte_fw and "c == 'c'" in fonte_fw)
check("platformio do firmware existe",
      fw_ini.exists() and "[env:esp32_teste_ahrs]" in
      fw_ini.read_text(encoding="utf-8"))
check("atalho de log existe",
      (raiz / "AHRS_grafico_log.bat").exists())
check("analisador de log existe",
      (raiz / "tools" / "analisa_log_ahrs.py").exists())
check("visualizador existe", visual.exists())
if visual.exists():
    fonte_vis = visual.read_text(encoding="utf-8")
    check("visualizador tem parser/rotacao",
          "parse_linha_ahrs" in fonte_vis and "quat_para_matriz" in fonte_vis)
    check("visualizador tem modo simulado", "--simular" in fonte_vis)
    check("visualizador tem tara Z e comandos",
          '("z", "Z")' in fonte_vis and '("c", "C")' in fonte_vis)

if FALHAS:
    print()
    print("AHRS_VETORES_FALHOU: " + " | ".join(FALHAS))
    raise SystemExit(1)
print()
print("AHRS_VETORES_OK: parser, quaternion->matriz, tara -Z, rotacao e "
      "simulacao")
