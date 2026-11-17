"""Testes do analisador OFFLINE da calibracao geometrica (item 127).

Contexto: `tools/analisa_calibracao_geometrica.py` reanalisa o JSON piloto
`calibracao_geometrica_<lado>_<data>.json` (gravado pela tecla X) sem
hardware: erro por pose com a solucao do piloto, LOO ponderado e comparacao
com o alinhamento oficial. Este teste gera um piloto SINTETICO com T/f
VERDADEIROS conhecidos (mesma matematica de test_calibracao_geometrica.py:
R = rotacao minima de f_true para T_true^T @ verdade; verdade geometrica =
alvo do referencial da camera; pose de mira = -normalize(posicao da mao)) e
verifica a ferramenta de ponta a ponta.

Casos:
  1. gerador sintetico grava JSON no schema (9 poses, solucao, constantes) e
     o solver ponderado recupera T/f;
  2. funcoes da ferramenta (import por importlib, pois esta em tools/): erro
     por pose ~0, resumo por tipo/global, LOO ~0 em dados limpos, oficial
     ausente nao quebra, relatorio legivel;
  3. comparacao com o oficial: oficial = solucao do piloto -> "equivalente";
     oficial girado 30 graus -> "oficial pior" com diferenca de mediana
     negativa; arquivo ausente -> aviso e segue;
  4. CLI via subprocess (.venv\\Scripts\\python.exe): exit 0 com --json-out
     (JSON sem NaN), default pega o mais recente do cwd e avisa oficial
     ausente, arquivo malformado / sem posicoes / inexistente -> exit 1.

Harness: aritmetica pura + subprocess (sem Qt, Kinect ou rede). Nada e'
gravado fora do diretorio temporario (os imu_palm_alignment_*.json reais nao
sao tocados).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import imu  # noqa: E402

CAMINHO_TOOL = RAIZ / "tools" / "analisa_calibracao_geometrica.py"
PY_VENV = RAIZ / ".venv" / "Scripts" / "python.exe"
PYTHON = PY_VENV if PY_VENV.is_file() else Path(sys.executable)

FALHAS = []


def check(nome, condicao, detalhe=None):
    if condicao:
        print(f"   ok: {nome}")
    else:
        FALHAS.append(nome)
        print(f"   FALHOU: {nome} | {detalhe!r}")


def roda_cli(argumentos, cwd):
    """Roda o CLI da ferramenta com o Python do projeto (subprocess)."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [str(PYTHON), str(CAMINHO_TOOL)] + [str(a) for a in argumentos],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(cwd), timeout=600, env=env)


# ============================================================================
# gerador sintetico (mesma matematica de kinect_imu_groundtruth /
# test_calibracao_geometrica.py, sem importar o modulo pesado)
# ============================================================================
def _matriz_rotacao_eixo(eixo, angulo):
    """Igual a kinect_imu_groundtruth._matriz_rotacao_eixo (Rodrigues)."""
    x, y, z = eixo
    c, s = np.cos(angulo), np.sin(angulo)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], np.float64)


def _rotacao_a_para_b(a, b):
    """Rotacao minima de a em b (igual a k._rotacao_a_para_b)."""
    a = np.asarray(a, np.float64) / np.linalg.norm(a)
    b = np.asarray(b, np.float64) / np.linalg.norm(b)
    eixo = np.cross(a, b)
    cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(eixo) < 1e-12:
        if cos > 0.0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-9:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return _matriz_rotacao_eixo(perp, np.pi)
    eixo = eixo / np.linalg.norm(eixo)
    return _matriz_rotacao_eixo(eixo, float(np.arccos(cos)))


def _desvia(vetor, angulo_deg, rng):
    """Gira `vetor` angulo_deg em torno de um eixo aleatorio perpendicular."""
    eixo = np.cross(vetor, rng.normal(size=3))
    eixo = eixo / np.linalg.norm(eixo)
    return _matriz_rotacao_eixo(eixo, np.radians(angulo_deg)) @ vetor


_R2 = 0.7071067811865476
# Nomes/tipos/alvos copiados de POSES_CALIBRACAO_GEOMETRICA (item 127):
# 8 poses geometricas (alvo EXATO no referencial da camera) + 1 mira.
POSES_SINTETICAS = (
    ("PALMA PARA A CAMERA", "geometrica", (0.0, 0.0, -1.0)),
    ("DORSO PARA A CAMERA", "geometrica", (0.0, 0.0, 1.0)),
    ("PALMA PARA A DIREITA", "geometrica", (1.0, 0.0, 0.0)),
    ("PALMA PARA A ESQUERDA", "geometrica", (-1.0, 0.0, 0.0)),
    ("PALMA PARA CIMA (cam)", "geometrica", (0.0, 1.0, 0.0)),
    ("PALMA PARA BAIXO (cam)", "geometrica", (0.0, -1.0, 0.0)),
    ("DIAGONAL FRENTE-DIREITA", "geometrica", (_R2, 0.0, -_R2)),
    ("DIAGONAL FRENTE-CIMA", "geometrica", (0.0, _R2, -_R2)),
    ("MIRA DESLOCADA (palma na lente)", "posicao", None),
)


def gerar_piloto_sintetico(diretorio, lado="right", seed=11):
    """Grava o JSON piloto sintetico e devolve (caminho, T_verdade, f_verdade).

    Mesmo gerador de test_calibracao_geometrica.py: para cada pose, a rotacao
    do IMU e' a rotacao minima que leva f_verdade em T_verdade^T @ verdade
    (entao `T_verdade @ R @ f_verdade = verdade` EXATAMENTE); na mira, a
    verdade e' -normalize(posicao da mao). QA sintetico = desvio pequeno da
    normal medida.
    """
    rng = np.random.default_rng(seed)
    T_verdade = imu.rotation_matrix(25.0, -12.0, 40.0)
    f_verdade = np.array([0.15, -0.25, 0.95])
    f_verdade = f_verdade / np.linalg.norm(f_verdade)
    pos_geo = np.array([0.10, 0.05, 0.45])
    pos_mira = np.array([0.20, 0.06, 0.45])
    entradas, rotacoes, verdades, pesos = [], [], [], []
    for nome, tipo, alvo in POSES_SINTETICAS:
        if tipo == "posicao":
            posicao = pos_mira
            verdade = -posicao / np.linalg.norm(posicao)
        else:
            posicao = pos_geo
            verdade = np.asarray(alvo, np.float64)
            verdade = verdade / np.linalg.norm(verdade)
        rotacao = _rotacao_a_para_b(f_verdade, T_verdade.T @ verdade)
        qa = float(rng.uniform(0.5, 3.0))
        normal = _desvia(verdade, qa, rng)
        entradas.append({
            "posicao": nome, "tipo": tipo,
            "alvo_camera": None if alvo is None else list(alvo),
            "verdade_camera": verdade.tolist(),
            "qa_deg": qa,
            "rotacao": rotacao.tolist(),
            "posicao_mao_camera": posicao.tolist(),
            "normal_medida_camera": normal.tolist(),
        })
        rotacoes.append(rotacao)
        verdades.append(verdade)
        pesos.append(1.5 if tipo == "posicao" else 1.0)
    T_est, f_est, residuo = imu.solve_palm_alignment_pesos(
        rotacoes, verdades, pesos=pesos)
    payload = {
        "modo": "calibracao_geometrica", "lado": lado,
        "participante": "sintetico", "quando": "2099-12-31 12:00:00",
        "posicoes": entradas,
        "solucao": {"rotation": np.asarray(T_est).tolist(),
                    "body": np.asarray(f_est).tolist(),
                    "residual_deg": float(residuo)},
        "diagnostico": {"n_takes": len(entradas),
                        "gerador": "test_analisa_calibracao_geometrica.py"},
        "constantes": {"TAKE_SEC": 2.0, "QA_MAX_DEG": 8.0,
                       "PESO_GEOMETRICA": 1.0, "PESO_POSICAO": 1.5},
    }
    caminho = (Path(diretorio) /
               f"calibracao_geometrica_{lado}_20991231_120000.json")
    caminho.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    return caminho, T_verdade, f_verdade


# ============================================================================
# import da ferramenta por caminho (esta' em tools/)
# ============================================================================
_spec = importlib.util.spec_from_file_location(
    "analisa_calibracao_geometrica", str(CAMINHO_TOOL))
ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ana)

# ============================================================================
# 1) gerador sintetico: JSON piloto no schema
# ============================================================================
print("1) gerador sintetico: JSON piloto no schema (T/f verdadeiros)")
tmp = Path(tempfile.mkdtemp(prefix="test_analisa_geo_"))
try:
    caminho, T_verdade, f_verdade = gerar_piloto_sintetico(tmp)
    check("json piloto criado no tmp", caminho.is_file())
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    check("9 posicoes no schema",
          isinstance(dados.get("posicoes"), list)
          and len(dados["posicoes"]) == 9, len(dados.get("posicoes", [])))
    chaves = {"posicao", "tipo", "alvo_camera", "verdade_camera", "qa_deg",
              "rotacao", "posicao_mao_camera", "normal_medida_camera"}
    check("campos por posicao completos",
          all(chaves <= set(p) for p in dados["posicoes"]))
    solucao = dados.get("solucao") or {}
    check("solucao com rotation/body/residual",
          all(k in solucao for k in ("rotation", "body", "residual_deg")))
    check("solucao sintetica recupera T/f verdadeiros",
          np.allclose(np.asarray(solucao["rotation"]), T_verdade, atol=1e-6)
          and np.allclose(np.asarray(solucao["body"]), f_verdade, atol=1e-6))
    check("modo/lado corretos",
          dados.get("modo") == "calibracao_geometrica"
          and dados.get("lado") == "right")

    # ========================================================================
    # 2) funcoes da ferramenta (import por caminho)
    # ========================================================================
    print("2) funcoes da ferramenta (importlib por caminho)")
    check("modulo carregado com analisar/carregar_piloto",
          hasattr(ana, "analisar") and hasattr(ana, "carregar_piloto"))
    check("pesos do item 127 (1.0 / 1.5)",
          ana.PESO_POR_TIPO == {"geometrica": 1.0, "posicao": 1.5},
          ana.PESO_POR_TIPO)
    dados_carregados, avisos = ana.carregar_piloto(caminho)
    check("carregar_piloto sem avisos", avisos == [], avisos)
    resultado = ana.analisar(dados_carregados, avisos=avisos,
                             caminho_arquivo=str(caminho),
                             caminho_oficial=None)
    erros = [linha["erro_deg"] for linha in resultado["por_pose"]]
    check("tabela com 9 poses", len(erros) == 9, len(erros))
    check("erro por pose ~0 (solucao do piloto)", max(erros) < 1e-4,
          max(erros))
    resumo = resultado["resumo"]
    check("resumo geometrica n=8", resumo["geometrica"]["n"] == 8,
          resumo["geometrica"])
    check("resumo posicao n=1", resumo["posicao"]["n"] == 1,
          resumo["posicao"])
    check("resumo global mediana ~0",
          resumo["global"]["n"] == 9
          and resumo["global"]["mediana_deg"] < 1e-4, resumo["global"])
    loo = resultado["loo"]
    check("LOO presente (9/9 re-solvidas)", loo["n_resolvidos"] == 9, loo)
    check("LOO ~0 em dados limpos",
          loo["mediana_deg"] is not None and loo["mediana_deg"] < 1e-3
          and loo["max_deg"] < 1e-3, loo)
    check("sem --oficial: comparacao pulada com motivo",
          resultado["oficial"]["presente"] is False
          and bool(resultado["oficial"].get("motivo")))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ana.imprimir_relatorio(resultado, verbose=True)
    relatorio = buffer.getvalue()
    check("relatorio legivel (tabela/resumo/LOO/veredito ausente)",
          "erro por pose" in relatorio and "resumo" in relatorio
          and "deixa-um-de-fora" in relatorio
          and "sem comparacao" in relatorio)

    # ========================================================================
    # 3) comparacao com o alinhamento oficial
    # ========================================================================
    print("3) comparacao com o alinhamento oficial")
    oficial_bom = tmp / "oficial_igual.json"
    oficial_bom.write_text(json.dumps(
        {"rotation": solucao["rotation"], "body": solucao["body"],
         "residual_deg": solucao["residual_deg"]}), encoding="utf-8")
    res_igual = ana.analisar(dados_carregados, caminho_oficial=oficial_bom)
    of = res_igual["oficial"]
    check("oficial presente", of["presente"] is True)
    check("erro por pose com oficial identico ~0",
          max(linha["erro_deg"] for linha in of["por_pose"]) < 1e-4)
    check("veredito 'equivalente' com oficial identico",
          of["veredito"] == "equivalente", of["veredito"])

    # oficial "errado": T girado 30 graus num eixo nao alinhado a nenhuma
    # verdade -> todo erro por pose fica ~20-25 graus
    eixo = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    T_ruim = (_matriz_rotacao_eixo(eixo, np.radians(30.0))
              @ np.asarray(solucao["rotation"], np.float64))
    oficial_ruim = tmp / "oficial_girado.json"
    oficial_ruim.write_text(json.dumps(
        {"rotation": T_ruim.tolist(), "body": solucao["body"],
         "residual_deg": 30.0}), encoding="utf-8")
    res_ruim = ana.analisar(dados_carregados, caminho_oficial=oficial_ruim)
    of2 = res_ruim["oficial"]
    mediana_of2 = of2["resumo"]["global"]["mediana_deg"]
    check("oficial girado: mediana > 10 graus",
          mediana_of2 is not None and mediana_of2 > 10.0, mediana_of2)
    check("veredito 'oficial pior'", of2["veredito"] == "oficial pior",
          of2["veredito"])
    check("diferenca de mediana negativa (piloto - oficial)",
          of2["diferenca_mediana_deg"] is not None
          and of2["diferenca_mediana_deg"] < -10.0,
          of2["diferenca_mediana_deg"])

    res_ausente = ana.analisar(dados_carregados,
                               caminho_oficial=tmp / "nao_existe.json")
    check("oficial ausente: aviso e segue",
          res_ausente["oficial"]["presente"] is False
          and any("oficial" in aviso for aviso in res_ausente["aviso"]))

    # ========================================================================
    # 4) CLI via subprocess
    # ========================================================================
    print("4) CLI via subprocess (python do projeto)")
    saida_json = tmp / "resultado_analise.json"
    proc = roda_cli(["--arquivo", caminho, "--oficial", oficial_bom,
                     "--json-out", saida_json], cwd=RAIZ)
    check("CLI exit 0", proc.returncode == 0,
          (proc.stderr or "")[-300:])
    check("CLI imprime o relatorio",
          "ANALISE OFFLINE DA CALIBRACAO GEOMETRICA" in proc.stdout)
    check("--json-out criado", saida_json.is_file())
    if saida_json.is_file():
        texto_saida = saida_json.read_text(encoding="utf-8")
        saida = json.loads(texto_saida)
        check("json-out: 9 por_pose", len(saida.get("por_pose", [])) == 9,
              len(saida.get("por_pose", [])))
        check("json-out: LOO com mediana",
              saida.get("loo", {}).get("mediana_deg") is not None)
        check("json-out: oficial com veredito",
              saida.get("oficial", {}).get("veredito") == "equivalente")
        check("json-out: solucao_usada origem 'piloto'",
              saida.get("solucao_usada", {}).get("origem")
              == "piloto (do arquivo)")
        check("json-out: sem NaN/Infinity",
              "NaN" not in texto_saida and "Infinity" not in texto_saida)

    # default --arquivo (mais recente no cwd) + oficial padrao ausente
    proc2 = roda_cli(["--verbose"], cwd=tmp)
    check("CLI default (mais recente no cwd) exit 0", proc2.returncode == 0,
          (proc2.stderr or "")[-300:])
    check("CLI default usou o piloto do tmp",
          "calibracao_geometrica_right_20991231_120000.json" in proc2.stdout)
    check("CLI default: lado veio do arquivo", "lado: right" in proc2.stdout)
    check("CLI default: oficial padrao ausente avisada",
          "AVISO" in proc2.stdout and "oficial" in proc2.stdout.lower())

    quebrado = tmp / "quebrado.json"
    quebrado.write_text("{ isto nao e' json", encoding="utf-8")
    proc3 = roda_cli(["--arquivo", quebrado], cwd=RAIZ)
    check("JSON malformado -> exit 1", proc3.returncode == 1,
          proc3.returncode)
    check("mensagem clara de malformado",
          "ERRO" in proc3.stderr and "malformado" in proc3.stderr.lower(),
          (proc3.stderr or "")[:200])

    sem_pos = tmp / "sem_posicoes.json"
    sem_pos.write_text(json.dumps({"modo": "calibracao_geometrica",
                                   "lado": "right"}), encoding="utf-8")
    proc4 = roda_cli(["--arquivo", sem_pos], cwd=RAIZ)
    check("JSON valido sem posicoes -> exit 1", proc4.returncode == 1,
          proc4.returncode)
    check("mensagem aponta 'posicoes'",
          "posicoes" in proc4.stderr.lower(), (proc4.stderr or "")[:200])

    proc5 = roda_cli(["--arquivo", tmp / "fantasma.json"], cwd=RAIZ)
    check("arquivo inexistente -> exit 1", proc5.returncode == 1,
          proc5.returncode)
    check("mensagem de arquivo nao encontrado",
          "nao encontrado" in proc5.stderr.lower(), (proc5.stderr or "")[:200])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if FALHAS:
    print()
    print(f"ANALISA_CALIBRACAO_GEOMETRICA_FALHOU: {len(FALHAS)} "
          "verificacao(oes): " + " | ".join(FALHAS))
    raise SystemExit(1)
print()
print("ANALISA_CALIBRACAO_GEOMETRICA_OK: piloto sintetico no schema "
      "(9 poses, T/f recuperados), erro por pose ~0 com a solucao do piloto, "
      "resumo por tipo/global, LOO ~0 em dados limpos, oficial identico -> "
      "'equivalente', oficial girado 30 graus -> 'oficial pior' com "
      "diferenca negativa, oficial ausente avisa e segue, CLI exit 0 com "
      "--json-out sem NaN, default pega o mais recente do cwd, malformado/"
      "sem posicoes/inexistente -> exit 1")
