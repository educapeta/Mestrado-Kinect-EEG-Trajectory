"""Analisador OFFLINE da calibracao geometrica da palma (item 127, tecla X).

Por que existe: `finalizar_calibracao_geometrica`
(tools/kinect_groundtruth_tool.py) grava o JSON piloto
`calibracao_geometrica_<lado>_<data>.json` com os takes brutos, a solucao T/f
do momento e as validacoes. Este CLI REANALISA esse piloto sem hardware, sem
rede e sem Kinect:

  1. recalcula o erro por pose com a solucao do PROPRIO piloto (modelo
     `v = T R f`, verdade = alvo do referencial da camera);
  2. refaz a validacao cruzada DEIXA-UM-DE-FORA (LOO) com o mesmo solver
     ponderado (`imu.solve_palm_alignment_pesos`, pesos do item 127:
     pose geometrica = CALIB_GEO_PESO_GEOMETRICA, mira = CALIB_GEO_PESO_POSICAO);
  3. compara com o alinhamento OFICIAL atual (`imu_palm_alignment_<lado>.json`):
     erro por pose com o T/f oficial, diferenca de mediana e veredito simples
     ("oficial melhor/pior/equivalente", criterio em TOLERANCIA_EQUIVALENCIA_DEG).

Schema do JSON piloto (gerado por `finalizar_calibracao_geometrica`):

    {"modo": "calibracao_geometrica", "lado": "right|left",
     "participante": "...", "quando": "YYYY-MM-DD HH:MM:SS",
     "posicoes": [{"posicao": "NOME", "tipo": "geometrica|posicao",
                   "alvo_camera": [...]|null, "verdade_camera": [...],
                   "qa_deg": ..., "rotacao": [[...],[...],[...]],
                   "posicao_mao_camera": [...],
                   "normal_medida_camera": [...]}],
     "solucao": {"rotation": [[...]], "body": [...], "residual_deg": ...},
     "diagnostico": {...}, "constantes": {...}}

Referencial da camera (Kinect v2, camera space): X = direita na imagem,
Y = CIMA na imagem, Z = saindo da lente -- os alvos seguem essa convencao.

Este modulo foi feito para rodar OFFLINE (inclusive em maquina sem o Kinect):
`imu.py` e' puro (numpy/math/socket). As constantes de peso tentam vir de
`kinect_imu_groundtruth` (FONTE UNICA do item 127); esse import carrega
OpenCV/mediapipe/pykinect e pode faltar numa maquina so' de analise -- se
falhar, o fallback usa os valores DOCUMENTADOS (1.0 / 1.5) e segue.

Uso (da raiz do projeto):

    .venv\\Scripts\\python.exe tools\\analisa_calibracao_geometrica.py
    .venv\\Scripts\\python.exe tools\\analisa_calibracao_geometrica.py --verbose
    .venv\\Scripts\\python.exe tools\\analisa_calibracao_geometrica.py --arquivo calibracao_geometrica_right_20260519_120000.json
    .venv\\Scripts\\python.exe tools\\analisa_calibracao_geometrica.py --oficial imu_palm_alignment_right.json --json-out analise.json

Sem `--arquivo` ele pega o `calibracao_geometrica_*.json` MAIS RECENTE da
pasta atual. Codigos de saida: 0 = analise concluida; 1 = arquivo ausente,
malformado ou sem posicoes validas.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# A biblioteca leve vive na raiz do projeto (um nivel acima de tools/).
RAIZ = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from imu import (PALM_ALIGNMENT_MIN_SAMPLES,  # noqa: E402
                 angle_between, solve_palm_alignment_pesos)

# Pesos do ajuste hibrido: FONTE UNICA em kinect_imu_groundtruth (item 127).
# O import e' pesado (OpenCV/mediapipe/pykinect) e pode faltar numa maquina
# so' de analise; se falhar, usa os valores documentados nas constantes do
# item 127 (1.0 / 1.5) e segue offline.
try:
    from kinect_imu_groundtruth import (  # noqa: E402
        CALIB_GEO_PESO_GEOMETRICA, CALIB_GEO_PESO_POSICAO)
    FONTE_PESOS = "kinect_imu_groundtruth"
except (Exception, SystemExit) as _exc:  # SystemExit = dependencias ausentes
    CALIB_GEO_PESO_GEOMETRICA = 1.0
    CALIB_GEO_PESO_POSICAO = 1.5
    FONTE_PESOS = ("fallback documentado (import de kinect_imu_groundtruth "
                   f"falhou: {type(_exc).__name__})")

#: Peso por tipo de pose no ajuste/LOO (mesma conta do item 127).
PESO_POR_TIPO = {"geometrica": float(CALIB_GEO_PESO_GEOMETRICA),
                 "posicao": float(CALIB_GEO_PESO_POSICAO)}

#: Tolerancia (graus) na mediana GLOBAL para declarar o oficial "equivalente"
#: ao piloto: o ruido normal da normal do Kinect e' ~1-2 graus (ver
#: PALM_ALIGNMENT_MAX_RESIDUAL_DEG em imu.py), entao abaixo disso nao ha'
#: vencedor confiavel. Acima: quem tem a mediana MENOR ganha.
TOLERANCIA_EQUIVALENCIA_DEG = 1.0

_PADRAO_PILOTO = "calibracao_geometrica_*.json"
_PADRAO_OFICIAL = "imu_palm_alignment_{lado}.json"


# ============================================================================
# carga/validacao
# ============================================================================
def mais_recente_padrao(diretorio=None):
    """Caminho do `calibracao_geometrica_*.json` mais recente (ou None)."""
    pasta = Path(diretorio) if diretorio is not None else Path.cwd()
    candidatos = [p for p in pasta.glob(_PADRAO_PILOTO) if p.is_file()]
    if not candidatos:
        return None
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def carregar_piloto(caminho):
    """Le e valida o JSON piloto; devolve (dados, avisos).

    Levanta ValueError com mensagem clara se o arquivo nao existir, nao for
    JSON valido, nao for um objeto ou nao tiver `posicoes` (lista nao vazia).
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ValueError(f"arquivo nao encontrado: {caminho}")
    try:
        texto = caminho.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"nao consegui ler ({exc})") from None
    try:
        dados = json.loads(texto)
    except ValueError as exc:
        raise ValueError(f"JSON malformado ({exc})") from None
    if not isinstance(dados, dict):
        raise ValueError("o JSON nao e' um objeto (esperado dict com 'posicoes')")
    posicoes = dados.get("posicoes")
    if not isinstance(posicoes, list) or not posicoes:
        raise ValueError("sem 'posicoes' (lista nao vazia) no JSON piloto")
    avisos = []
    modo = dados.get("modo")
    if modo is not None and modo != "calibracao_geometrica":
        avisos.append(f"modo inesperado {modo!r} (esperado "
                      "'calibracao_geometrica')")
    return dados, avisos


def _vetor(valor, tamanho):
    """ndarray (tamanho,) float finito ou None se invalido."""
    if valor is None:
        return None
    try:
        vetor = np.asarray(valor, np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vetor.shape != (tamanho,) or not np.isfinite(vetor).all():
        return None
    return vetor


def _matriz(valor):
    """ndarray (3, 3) float finito ou None se invalido."""
    if valor is None:
        return None
    try:
        matriz = np.asarray(valor, np.float64)
    except (TypeError, ValueError):
        return None
    if matriz.shape != (3, 3) or not np.isfinite(matriz).all():
        return None
    return matriz


def _numero(valor):
    """float finito ou None (aceita None/NaN/string nao numerica)."""
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if math.isfinite(numero) else None


def _normaliza_lado(valor):
    """'Right ' -> 'right'; None se vazio."""
    if valor is None:
        return None
    texto = str(valor).strip().lower()
    return texto or None


def validar_posicoes(posicoes_brutas, avisos):
    """Normaliza as entradas do piloto; descarta invalidas com aviso.

    Cada entrada valida ganha `rotacao` (3x3), `verdade` (unitario) e `peso`
    (por tipo, valores do item 127). A verdade e' obrigatoria (e' o alvo da
    conta `v = T R f`); `alvo_camera`/`normal_medida_camera`/`qa_deg` sao
    informativos (QA de execucao) e podem faltar.
    """
    validas = []
    for indice, bruta in enumerate(posicoes_brutas):
        if not isinstance(bruta, dict):
            avisos.append(f"posicoes[{indice}] nao e' objeto; ignorada")
            continue
        rotacao = _matriz(bruta.get("rotacao"))
        verdade = _vetor(bruta.get("verdade_camera"), 3)
        problema = None
        if rotacao is None:
            problema = "rotacao nao e' 3x3 finita"
        elif verdade is None or np.linalg.norm(verdade) < 1e-9:
            problema = "verdade_camera ausente/zero"
        if problema:
            avisos.append(f"posicoes[{indice}] "
                          f"({bruta.get('posicao')!r}): {problema}; ignorada")
            continue
        tipo = str(bruta.get("tipo") or "geometrica")
        if tipo not in ("geometrica", "posicao"):
            avisos.append(f"posicoes[{indice}]: tipo {tipo!r} desconhecido "
                          "(peso de 'geometrica' aplicado)")
        validas.append({
            "indice": indice,
            "posicao": str(bruta.get("posicao") or f"pose_{indice}"),
            "tipo": tipo,
            "qa_deg": _numero(bruta.get("qa_deg")),
            "rotacao": rotacao,
            "verdade": verdade / np.linalg.norm(verdade),
            "peso": PESO_POR_TIPO.get(tipo, CALIB_GEO_PESO_GEOMETRICA),
        })
    return validas


# ============================================================================
# contas (mesmo modelo do CalibracaoGeometrica.resultado)
# ============================================================================
def erro_por_pose(posicoes, T, f):
    """Tabela [{posicao, tipo, qa_deg, erro_deg}] no modelo `v = T R f`."""
    tabela = []
    for entrada in posicoes:
        previsto = T @ entrada["rotacao"] @ f
        tabela.append({
            "posicao": entrada["posicao"],
            "tipo": entrada["tipo"],
            "qa_deg": entrada["qa_deg"],
            "erro_deg": float(angle_between(previsto, entrada["verdade"])),
        })
    return tabela


def resumo_erro(tabela):
    """Mediana/max por tipo ('geometrica', 'posicao') e global, em graus."""
    def _fechar(valores):
        finitos = [v for v in valores
                   if v is not None and math.isfinite(v)]
        return {"n": len(finitos),
                "mediana_deg": (float(np.median(finitos))
                                if finitos else None),
                "max_deg": (float(max(finitos)) if finitos else None)}
    resumo = {}
    for tipo in ("geometrica", "posicao"):
        resumo[tipo] = _fechar([linha["erro_deg"] for linha in tabela
                                if linha["tipo"] == tipo])
    resumo["global"] = _fechar([linha["erro_deg"] for linha in tabela])
    return resumo


def resolver_de_novo(posicoes):
    """Re-resolve T/f com o solver ponderado do item 127 (ou None)."""
    return solve_palm_alignment_pesos(
        [p["rotacao"] for p in posicoes],
        [p["verdade"] for p in posicoes],
        pesos=[p["peso"] for p in posicoes])


def loo_deixa_um_de_fora(posicoes):
    """Validacao cruzada LOO: resolve sem a pose i e mede NELA.

    Mesmo calculo de `CalibracaoGeometrica.resultado()` (item 127): a pose i
    sai do ajuste e volta como unica prova. Devolve (erros_deg, n_resolvidos).
    """
    erros = []
    for i in range(len(posicoes)):
        resto = [p for j, p in enumerate(posicoes) if j != i]
        solucao = solve_palm_alignment_pesos(
            [p["rotacao"] for p in resto],
            [p["verdade"] for p in resto],
            pesos=[p["peso"] for p in resto])
        if solucao is None:
            continue
        T, f, _residuo = solucao
        erros.append(float(angle_between(
            T @ posicoes[i]["rotacao"] @ f, posicoes[i]["verdade"])))
    return erros, len(erros)


def carregar_oficial(caminho):
    """Le o alinhamento oficial; devolve (T, f, residual_deg, motivo).

    `motivo` None = ok; string = razao pela qual o arquivo nao serve (o
    schema e' {"rotation": 3x3, "body": 3, "residual_deg": ...}).
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        return None, None, None, f"arquivo ausente: {caminho}"
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, None, None, f"arquivo invalido ({exc})"
    if not isinstance(dados, dict):
        return None, None, None, "arquivo invalido (JSON nao e' objeto)"
    T = _matriz(dados.get("rotation"))
    f = _vetor(dados.get("body"), 3)
    if T is None or f is None or np.linalg.norm(f) < 1e-9:
        return None, None, None, ("sem 'rotation' 3x3 / 'body' 3 validos")
    return T, f / np.linalg.norm(f), _numero(dados.get("residual_deg")), None


def _veredito(mediana_piloto, mediana_oficial):
    """'oficial melhor/pior/equivalente' pela mediana GLOBAL (mesmas poses).

    Criterio: |mediana_piloto - mediana_oficial| <= TOLERANCIA_EQUIVALENCIA_DEG
    -> "equivalente"; senao vence quem tem a mediana menor.
    """
    if mediana_piloto is None or mediana_oficial is None:
        return None
    diferenca = mediana_piloto - mediana_oficial
    if abs(diferenca) <= TOLERANCIA_EQUIVALENCIA_DEG:
        return "equivalente"
    return "oficial melhor" if diferenca > 0.0 else "oficial pior"


# ============================================================================
# analise completa
# ============================================================================
def analisar(dados, avisos=None, caminho_arquivo=None, caminho_oficial=None,
             lado=None):
    """Reanalise completa do piloto; devolve dict JSON-serializavel.

    `caminho_oficial=None` pula a comparacao (registrada com motivo). Levanta
    ValueError se nenhuma posicao for valida ou se o piloto nao tem solucao
    utilizavel e o refit tambem falha.
    """
    avisos = list(avisos or [])
    resultado = {"aviso": avisos}
    resultado["arquivo"] = (str(caminho_arquivo)
                            if caminho_arquivo is not None else None)
    resultado["modo"] = dados.get("modo")
    resultado["participante"] = dados.get("participante")
    resultado["quando"] = dados.get("quando")

    lado_resolvido = _normaliza_lado(lado) or _normaliza_lado(dados.get("lado"))
    if not lado_resolvido:
        lado_resolvido = "right"
        avisos.append("lado ausente no arquivo e na linha de comando; "
                      "assumido 'right'")
    resultado["lado"] = lado_resolvido

    posicoes = validar_posicoes(dados.get("posicoes", []), avisos)
    if not posicoes:
        raise ValueError("nenhuma 'posicoes' valida no JSON piloto "
                         "(rotacao 3x3 finita e verdade_camera 3D sao "
                         "obrigatorias)")
    resultado["n_poses"] = len(posicoes)
    resultado["n_por_tipo"] = {
        tipo: sum(1 for p in posicoes if p["tipo"] == tipo)
        for tipo in ("geometrica", "posicao")}
    resultado["fonte_pesos"] = FONTE_PESOS
    resultado["pesos"] = dict(PESO_POR_TIPO)

    # -- solucao do piloto (ou refit a partir dos takes) -------------------
    bruta = dados.get("solucao")
    solucao = bruta if isinstance(bruta, dict) else {}
    T_piloto = _matriz(solucao.get("rotation"))
    f_piloto = _vetor(solucao.get("body"), 3)
    if f_piloto is not None and np.linalg.norm(f_piloto) < 1e-9:
        f_piloto = None
    origem = "piloto (do arquivo)"
    if T_piloto is None or f_piloto is None:
        avisos.append("solucao do piloto ausente/invalida; re-resolvendo "
                      "T/f a partir dos takes")
        refit = resolver_de_novo(posicoes)
        if refit is None:
            raise ValueError(
                "sem solucao utilizavel: a 'solucao' do arquivo e' invalida "
                f"e o refit falhou (o solver exige >= "
                f"{PALM_ALIGNMENT_MIN_SAMPLES} takes validos)")
        T_piloto, f_piloto, _residuo = refit
        origem = "refit (re-resolvido a partir dos takes)"
    f_piloto = f_piloto / np.linalg.norm(f_piloto)
    resultado["solucao_usada"] = {
        "origem": origem,
        "residual_deg_arquivo": _numero(solucao.get("residual_deg")),
        "rotation": T_piloto.tolist(),
        "body": f_piloto.tolist(),
    }

    # -- 1) erro por pose com a solucao do piloto ---------------------------
    por_pose = erro_por_pose(posicoes, T_piloto, f_piloto)
    resultado["por_pose"] = por_pose
    resultado["resumo"] = resumo_erro(por_pose)

    # -- 2) validacao cruzada LOO -------------------------------------------
    loo = {"n_poses": len(posicoes), "n_resolvidos": 0, "mediana_deg": None,
           "max_deg": None, "erros_deg": []}
    if len(posicoes) >= PALM_ALIGNMENT_MIN_SAMPLES + 1:
        erros_loo, n_loo = loo_deixa_um_de_fora(posicoes)
        finitos = [e for e in erros_loo if math.isfinite(e)]
        loo.update({"n_resolvidos": n_loo, "erros_deg": finitos,
                    "mediana_deg": (float(np.median(finitos))
                                    if finitos else None),
                    "max_deg": (float(max(finitos)) if finitos else None)})
        if n_loo < len(posicoes):
            avisos.append(f"LOO: {len(posicoes) - n_loo} pose(s) ficaram "
                          "sem subconjunto solucionavel")
    else:
        avisos.append(f"LOO pulado: {len(posicoes)} poses < "
                      f"{PALM_ALIGNMENT_MIN_SAMPLES + 1}")
    resultado["loo"] = loo

    # -- 3) comparacao com o alinhamento oficial ----------------------------
    oficial = {"presente": False,
               "arquivo": (str(caminho_oficial)
                           if caminho_oficial is not None else None)}
    if caminho_oficial is None:
        oficial["motivo"] = "comparacao nao solicitada (sem --oficial)"
    else:
        T_of, f_of, residual_of, motivo = carregar_oficial(caminho_oficial)
        if motivo is not None:
            oficial["motivo"] = motivo
            avisos.append(f"alinhamento oficial nao comparado ({motivo})")
        else:
            por_pose_of = erro_por_pose(posicoes, T_of, f_of)
            resumo_of = resumo_erro(por_pose_of)
            mediana_piloto = resultado["resumo"]["global"]["mediana_deg"]
            mediana_oficial = resumo_of["global"]["mediana_deg"]
            diferenca = (None if mediana_piloto is None
                         or mediana_oficial is None
                         else float(mediana_piloto - mediana_oficial))
            oficial.update({
                "presente": True,
                "residual_deg": residual_of,
                "por_pose": por_pose_of,
                "resumo": resumo_of,
                "diferenca_mediana_deg": diferenca,
                "veredito": _veredito(mediana_piloto, mediana_oficial),
                "criterio": (f"|mediana piloto - mediana oficial| <= "
                             f"{TOLERANCIA_EQUIVALENCIA_DEG:g} graus = "
                             "'equivalente'; senao vence a mediana menor"),
            })
    resultado["oficial"] = oficial

    constantes = dados.get("constantes")
    resultado["constantes_arquivo"] = (constantes
                                       if isinstance(constantes, dict)
                                       else None)
    return resultado


# ============================================================================
# saida
# ============================================================================
def _limpa(objeto):
    """Converte numpy/NaN/Inf para tipos JSON puros (recursivo)."""
    if isinstance(objeto, np.ndarray):
        return _limpa(objeto.tolist())
    if isinstance(objeto, (np.floating, np.integer)):
        return _limpa(objeto.item())
    if isinstance(objeto, float):
        return objeto if math.isfinite(objeto) else None
    if isinstance(objeto, dict):
        return {str(chave): _limpa(valor)
                for chave, valor in objeto.items()}
    if isinstance(objeto, (list, tuple)):
        return [_limpa(valor) for valor in objeto]
    return objeto


def _fmt(valor, formato="{:6.2f}", vazio="   -- "):
    """Valor formatado ou marcador de vazio (colunas alinhadas)."""
    if valor is None or not math.isfinite(valor):
        return vazio
    return formato.format(valor)


def imprimir_relatorio(resultado, verbose=False):
    """Imprime o relatorio legivel da reanalise no terminal."""
    por_tipo = resultado.get("n_por_tipo") or {}
    print("")
    print("===== ANALISE OFFLINE DA CALIBRACAO GEOMETRICA (item 127) =====")
    print(f"arquivo: {resultado.get('arquivo') or '?'}")
    print(f"lado: {resultado.get('lado', '?')} | "
          f"poses: {resultado.get('n_poses', 0)} "
          f"(geometrica {por_tipo.get('geometrica', 0)}, "
          f"posicao {por_tipo.get('posicao', 0)}) | "
          f"quando: {resultado.get('quando') or '?'}")
    if verbose:
        print(f"pesos LOO: geometrica {PESO_POR_TIPO['geometrica']:g} / "
              f"posicao {PESO_POR_TIPO['posicao']:g} "
              f"(fonte: {resultado.get('fonte_pesos', FONTE_PESOS)})")
        residual = (resultado.get("solucao_usada") or {}).get(
            "residual_deg_arquivo")
        if residual is not None:
            print(f"residuo ponderado gravado no piloto: {residual:.2f} graus")
    for aviso in resultado.get("aviso") or ():
        print(f"AVISO: {aviso}")

    solucao_usada = resultado.get("solucao_usada") or {}
    print("--- erro por pose (v = T R f | solucao: "
          f"{solucao_usada.get('origem', '?')}) ---")
    print(f"  {'pose':<33} {'tipo':<11} {'QA':>6} {'erro':>8}")
    for linha in resultado.get("por_pose") or ():
        print(f"  {str(linha.get('posicao', ''))[:33]:<33} "
              f"{str(linha.get('tipo', '')):<11} "
              f"{_fmt(linha.get('qa_deg'), '{:6.1f}'):>6} "
              f"{_fmt(linha.get('erro_deg')):>8}")
    print("--- resumo (mediana / max, graus) ---")
    resumo = resultado.get("resumo") or {}
    for chave in ("geometrica", "posicao", "global"):
        bloco = resumo.get(chave) or {"n": 0, "mediana_deg": None,
                                      "max_deg": None}
        if not bloco.get("n"):
            print(f"  {chave:<11}: (sem poses)")
            continue
        print(f"  {chave:<11}: mediana {_fmt(bloco.get('mediana_deg'))} | "
              f"max {_fmt(bloco.get('max_deg'))} | n={bloco['n']}")

    loo = resultado.get("loo") or {}
    print("--- validacao cruzada (deixa-um-de-fora, pesos do item 127) ---")
    if loo.get("n_resolvidos"):
        print(f"  LOO: mediana {_fmt(loo.get('mediana_deg'))} | max "
              f"{_fmt(loo.get('max_deg'))} | n={loo['n_resolvidos']}/"
              f"{loo.get('n_poses', '?')} poses")
        if verbose:
            print("  erros LOO (ordem das poses): "
                  + ", ".join(f"{erro:.2f}"
                              for erro in loo.get("erros_deg") or ()))
    else:
        print("  LOO indisponivel (poses insuficientes p/ re-solver)")

    oficial = resultado.get("oficial") or {}
    print("--- comparacao com o alinhamento oficial ---")
    if not oficial.get("presente"):
        print(f"  sem comparacao: {oficial.get('motivo') or 'oficial ausente'}")
    else:
        bloco_of = (oficial.get("resumo") or {}).get("global") or {}
        linha_arquivo = f"  arquivo: {oficial.get('arquivo')}"
        if oficial.get("residual_deg") is not None:
            linha_arquivo += (f" (residual gravado: "
                              f"{_fmt(oficial.get('residual_deg'))} graus)")
        print(linha_arquivo)
        print(f"  oficial: mediana {_fmt(bloco_of.get('mediana_deg'))} | max "
              f"{_fmt(bloco_of.get('max_deg'))} | n={bloco_of.get('n', 0)}")
        print(f"  diferenca de mediana (piloto - oficial): "
              f"{_fmt(oficial.get('diferenca_mediana_deg'))} graus")
        veredito = oficial.get("veredito")
        if veredito:
            print(f"  veredito: {veredito.upper()} "
                  f"[criterio: {oficial.get('criterio')}]")
            leituras = {
                "oficial melhor": "o oficial atual e' melhor que o piloto",
                "oficial pior": ("o PILOTO e' melhor: o gate ao vivo deveria "
                                 "ter promovido (ou o oficial e' antigo)"),
                "equivalente": "sem vencedor dentro da tolerancia",
            }
            if veredito in leituras:
                print(f"  leitura: {leituras[veredito]}")
        if verbose:
            print("  erro por pose com o oficial:")
            for linha in oficial.get("por_pose") or ():
                print(f"    {str(linha.get('posicao', ''))[:33]:<33} "
                      f"{str(linha.get('tipo', '')):<11} "
                      f"{_fmt(linha.get('erro_deg')):>8}")
    if verbose and resultado.get("constantes_arquivo"):
        print(f"constantes gravadas no piloto: "
              f"{resultado['constantes_arquivo']}")
    print("")


# ============================================================================
# CLI
# ============================================================================
def montar_parser():
    """Argumentos: --arquivo, --lado, --oficial, --json-out, --verbose."""
    parser = argparse.ArgumentParser(
        description=("Reanalisa offline o JSON piloto da calibracao "
                     "geometrica da palma (item 127) e compara com o "
                     "alinhamento oficial (sem hardware/rede)."))
    parser.add_argument("--arquivo", default=None, metavar="JSON",
                        help=f"JSON piloto (default: {_PADRAO_PILOTO} mais "
                             "recente na pasta atual)")
    parser.add_argument("--lado", choices=("right", "left"), default=None,
                        help="lado da mao (default: o lado do arquivo)")
    parser.add_argument("--oficial", default=None, metavar="JSON",
                        help="alinhamento oficial para comparar (default: "
                             + _PADRAO_OFICIAL.format(lado="<lado>") + ")")
    parser.add_argument("--json-out", default=None, metavar="JSON",
                        help="grava o resultado completo em JSON "
                             "(sem numpy cru)")
    parser.add_argument("--verbose", action="store_true",
                        help="mostra pesos, LOO por pose e erros do oficial")
    return parser


def main(argv=None):
    """CLI; devolve 0 (analise concluida) ou 1 (arquivo invalido)."""
    args = montar_parser().parse_args(argv)

    if args.arquivo:
        arquivo = Path(args.arquivo)
    else:
        arquivo = mais_recente_padrao()
        if arquivo is None:
            print(f"ERRO: nenhum {_PADRAO_PILOTO} na pasta atual "
                  f"({Path.cwd()}); use --arquivo", file=sys.stderr)
            return 1
    if not arquivo.is_file():
        print(f"ERRO: arquivo nao encontrado: {arquivo}", file=sys.stderr)
        return 1
    try:
        dados, avisos = carregar_piloto(arquivo)
    except ValueError as exc:
        print(f"ERRO: {arquivo}: {exc}", file=sys.stderr)
        return 1

    lado = _normaliza_lado(args.lado) or _normaliza_lado(dados.get("lado"))
    if lado is None:
        lado = "right"
        avisos.append("lado ausente no arquivo; assumido 'right'")
    lado_arquivo = _normaliza_lado(dados.get("lado"))
    if args.lado and lado_arquivo and lado_arquivo != args.lado:
        avisos.append(f"--lado {args.lado} difere do lado do arquivo "
                      f"({lado_arquivo}); usando {args.lado}")
    oficial = (Path(args.oficial) if args.oficial
               else Path(_PADRAO_OFICIAL.format(lado=lado)))

    try:
        resultado = analisar(dados, avisos=avisos,
                             caminho_arquivo=str(arquivo),
                             caminho_oficial=oficial, lado=lado)
    except ValueError as exc:
        print(f"ERRO: {arquivo}: {exc}", file=sys.stderr)
        return 1

    imprimir_relatorio(resultado, verbose=args.verbose)

    if args.json_out:
        try:
            Path(args.json_out).write_text(
                json.dumps(_limpa(resultado), indent=2, ensure_ascii=False),
                encoding="utf-8")
            print(f"resultado completo gravado: {args.json_out}")
        except OSError as exc:
            print(f"AVISO: nao gravei --json-out ({exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
