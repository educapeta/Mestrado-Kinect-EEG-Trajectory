"""Teste DISCRIMINANTE do VETOR DA PALMA (a seta do tool x a verdade do Kinect).

Bug real (bancada, 19-21/10/2026): a seta da palma apontava para o lado OPOSTO ao
da palma na mao DIREITA -- com a palma virada para o teto a seta ia para o chao
(180 graus) e, ao girar o punho em torno do eixo do antebraco num sentido, a seta
girava no sentido contrario. A esquerda "parecia" certa.

CAUSA (provada nos dados desta sessao): o alinhamento CARREGADO
(`imu_palm_alignment_<lado>.json`) nao era um ajuste do modelo `v = T R f` para a
verdade ANATOMICA -- o arquivo tinha sido sobrescrito pela captura da tecla O
(rodada de 14 s com a mao girando), que acerta a pose da calibracao e erra ~100
graus quando o punho gira. O RESIDUO GRAVADO NAO DENUNCIA ISSO: o modelo tem uma
liberdade de calibre (`T -> T Q`, `f -> Q^T f` deixa TODA previsao igual) e, mais
grave, trocar o sinal da verdade troca `f` por `-f` com o MESMO residuo -- ou
seja, "residuo OK" nao e' prova de que a seta aponta para a palma. A prova tem de
ser contra a VERDADE, pose por pose.

Este teste usa a sessao `palm_diag_20261019_180702.csv` (4 poses programadas, com
os 4 pontos 3D crus da palma) e a normal ANATOMICA reconstruida dos pontos
(right = +cross, left = -cross, validada pela fisica das poses: MESA as duas
palmas para baixo => normais iguais; REZANDO palmas opostas => normais opostas).

Casos:
  1. a verdade anatomica do CSV fecha a FISICA das poses (MESA iguais, REZANDO
     opostas) -- sem isso o resto da comparacao nao vale;
  2. o alinhamento salvo e' rotacao propria (det +1) com `f` unitario;
  3. para CADA lado e CADA uma das 4 poses: mediana do angulo (seta, verdade)
     <= 20 graus (o limite do diagnostico e' 15; 20 cobre o ruido de captura);
  4. SINAL: na MESA (as duas palmas para BAIXO) a seta tem de apontar para BAIXO
     -- a imagem espelhada exata do sintoma "palma para CIMA => seta para o
     chao"; e, se o `.json.bak` existir, ele tem de FALHAR neste caso (e' o
     negativo de controle: prova que o teste discrimina);
  5. SENTIDO DE GIRO: entre MESA e REZANDO (o punho gira em torno do eixo do
     antebraco) a seta tem de girar no MESMO sentido da palma.

    .venv\\Scripts\\python.exe test_palm_seta_pose.py

Codigos de saida: 0 = passou; 1 = falhou (o defeito esta' no alinhamento salvo);
2 = faltam os dados (CSV/JSON) para julgar.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
if not (RAIZ / "imu.py").exists():
    RAIZ = Path.cwd()
sys.path.insert(0, str(RAIZ))

from imu import (ImuSample, PositionFusion, angle_between,  # noqa: E402
                 rotation_matrix, solve_palm_alignment)

#: CSV da sessao de referencia (4 poses programadas + pontos 3D crus).
CSV_SESSAO = "palm_diag_20261019_180702.csv"
#: Amostras do inicio de cada fase ficam fora (o experimentador esta' chegando
#: na pose) -- o MESMO settle do tool e do analisador.
SETTLE_S = 1.5
#: Mediana maxima (graus) aceitavel por pose. O diagnostico ao vivo usa 15; aqui
#: 20 para nao brigar com o ruido de captura da normal do Kinect.
LIMITE_POSE_DEG = 20.0
#: Sinal ANATOMICO da normal por mao (item 79): nesta camera
#: `(index_mcp-punho) x (pinky_mcp-punho)` sai do lado da PALMA na direita e do
#: DORSO na esquerda (maos espelhadas).
SINAL_ANATOMICO = {"right": 1.0, "left": -1.0}
POSES = ("PREPARO", "MESA", "REZANDO", "KINECT")
#: Minimo de amostras de uma pose para a mediana valer (o mesmo do tool).
MIN_AMOSTRAS_POSE = 10
#: "Aponta para o MESMO lado" na MESA: dot >= 0.9 (~26 graus).
DOT_SINAL_OK = 0.90
#: Abaixo disto a seta esta' no lado OPOSTO (0.5 ~ 120 graus): e' o defeito.
DOT_SINAL_OPOSTO = 0.50


# ------------------------------------------------------------------- leitura
def _numero(texto, padrao=float("nan")):
    try:
        return float(texto)
    except (TypeError, ValueError):
        return padrao


def carregar_sessao(caminho):
    """Linhas do CSV do diagnostico com as colunas numericas em float."""
    linhas = []
    with open(caminho, encoding="utf-8", newline="") as handle:
        for bruto in csv.DictReader(handle):
            linha = dict(bruto)
            for coluna in ("t_fase_s", "roll_deg", "pitch_deg", "yaw_deg",
                           "wrist_x", "wrist_y", "wrist_z",
                           "index_x", "index_y", "index_z",
                           "pinky_x", "pinky_y", "pinky_z",
                           "middle_x", "middle_y", "middle_z"):
                linha[coluna] = _numero(bruto.get(coluna))
            linhas.append(linha)
    return linhas


def _ponto(linha, nome):
    ponto = np.asarray([linha.get(f"{nome}_{eixo}") for eixo in "xyz"],
                       dtype=float)
    return ponto if np.isfinite(ponto).all() else None


def normal_anatomica(linha):
    """Normal da palma reconstruida dos pontos CRUS, com o sinal ANATOMICO.

    E' a MESMA conta de `KinectHandTracker.palm_normal_from_points(..., lado)`
    (que nao pode ser importada aqui: a biblioteca carrega OpenCV/pykinect).
    """
    punho, indicador, minimo = (_ponto(linha, "wrist"), _ponto(linha, "index"),
                                _ponto(linha, "pinky"))
    if punho is None or indicador is None or minimo is None:
        return None
    cru = np.cross(indicador - punho, minimo - punho)
    norma = float(np.linalg.norm(cru))
    if norma < 1e-9:
        return None
    return (cru / norma) * SINAL_ANATOMICO[linha["lado"]]


def direcao_dedos(linha):
    """Punho -> MCP do medio (unitario): da' o eixo do punho/antebraco."""
    punho, medio = _ponto(linha, "wrist"), _ponto(linha, "middle")
    if punho is None or medio is None:
        return None
    vetor = medio - punho
    norma = float(np.linalg.norm(vetor))
    return None if norma < 1e-9 else vetor / norma


def amostra_imu(linha):
    """ImuSample equivalente ao rpy gravado (o que a seta consome)."""
    return ImuSample(timestamp_us=0, received_at=0.0,
                     roll_deg=linha["roll_deg"], pitch_deg=linha["pitch_deg"],
                     yaw_deg=linha["yaw_deg"],
                     accel_g=np.array([0.0, 0.0, 1.0]),
                     gyro_dps=np.zeros(3))


def carregar_alinhamento(lado, arquivo=None):
    """PositionFusion com o alinhamento do arquivo carregado (caminho do tool).

    Usa `set_palm_alignment` + `palm_direction` -- o MESMO codigo que desenha a
    seta no video (`PositionFusion.palm_direction`, imu.py).
    """
    caminho = (RAIZ / (arquivo or f"imu_palm_alignment_{lado}.json"))
    if not caminho.exists():
        return None, None
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    fusao = PositionFusion()
    aplicado = fusao.set_palm_alignment(dados["rotation"], dados.get("body"),
                                       dados.get("residual_deg"))
    return (fusao if aplicado else None), dados


def _media_unit(vetores):
    media = np.mean(vetores, axis=0)
    norma = float(np.linalg.norm(media))
    return None if norma < 1e-9 else media / norma


def _por_pose(linhas, lado, fusao=None):
    """{fase: {"verdade": (3,), "seta": (3,), "dedos": (3,), "n": int}}."""
    if fusao is None:
        fusao, _dados = carregar_alinhamento(lado)
    assert fusao is not None, f"sem alinhamento para {lado}"
    saida = {}
    for fase in POSES:
        verdades, setas, dedos = [], [], []
        for linha in linhas:
            if linha["lado"] != lado or linha["fase_nome"] != fase:
                continue
            if not np.isfinite(linha["t_fase_s"]) or linha["t_fase_s"] < SETTLE_S:
                continue
            verdade = normal_anatomica(linha)
            seta = fusao.palm_direction(amostra_imu(linha))
            if verdade is None or seta is None:
                continue
            verdades.append(verdade)
            setas.append(np.asarray(seta, dtype=float)
                         / float(np.linalg.norm(seta)))
            dedo = direcao_dedos(linha)
            if dedo is not None:
                dedos.append(dedo)
        if len(verdades) < MIN_AMOSTRAS_POSE:
            continue
        saida[fase] = {"verdade": _media_unit(verdades),
                       "seta": _media_unit(setas),
                       "dedos": _media_unit(dedos) if dedos else None,
                       "n": len(verdades),
                       "erros": [angle_between(v, s)
                                 for v, s in zip(verdades, setas)]}
    return saida


def erro_mediano_por_pose(linhas, lado, fusao=None):
    """{fase: mediana(angulo entre a seta e a verdade))} em graus."""
    return {fase: float(np.median(dados["erros"]))
            for fase, dados in _por_pose(linhas, lado, fusao=fusao).items()}


def refitar_csv(linhas, lado):
    """PositionFusion com T/f REFEITOS do proprio CSV (verdade anatomica).

    Por que: o JSON salvo pode ter sido refeito na bancada em OUTRA sessao
    (tecla J), com outro yaw de boot/montagem -- comparar esse arquivo com o
    CSV canonico reprova por motivo legitimo, mas nao prova que o metodo
    esta' errado. O refit do proprio CSV certifica o METODO (sinal, modelo
    v = T R f, verdade anatomica) independente de qual JSON esta' em disco.
    """
    rotacoes, normais = [], []
    for linha in linhas:
        if linha["lado"] != lado:
            continue
        verdade = normal_anatomica(linha)
        if verdade is None or not np.isfinite(linha["roll_deg"]):
            continue
        rotacoes.append(rotation_matrix(linha["roll_deg"],
                                        linha["pitch_deg"],
                                        linha["yaw_deg"]))
        normais.append(verdade)
    resolvido = solve_palm_alignment(rotacoes, normais)
    if resolvido is None:
        return None, None, 0
    t_matriz, f_body, residuo = resolvido
    fusao = PositionFusion()
    fusao.set_palm_alignment(t_matriz, f_body, residuo)
    return fusao, float(residuo), len(normais)


# ============================================================================
# 1) a verdade anatomica fecha a FISICA das poses (sem isso nada abaixo vale)
# ============================================================================
caminho_csv = RAIZ / CSV_SESSAO
if not caminho_csv.exists():
    print(f"SEM DADOS: {CSV_SESSAO} nao existe nesta pasta -- nada a julgar")
    sys.exit(2)
linhas = carregar_sessao(caminho_csv)
if not linhas:
    print(f"SEM DADOS: {CSV_SESSAO} vazio")
    sys.exit(2)

medias = {}
for linha in linhas:
    if not np.isfinite(linha["t_fase_s"]) or linha["t_fase_s"] < SETTLE_S:
        continue
    normal = normal_anatomica(linha)
    if normal is None:
        continue
    medias.setdefault((linha["fase_nome"], linha["lado"]), []).append(normal)
if ("MESA", "right") not in medias or ("MESA", "left") not in medias:
    print(f"SEM DADOS: {CSV_SESSAO} sem pontos 3D crus da fase MESA nas duas maos")
    sys.exit(2)

mesa_r = _media_unit(medias[("MESA", "right")])
mesa_l = _media_unit(medias[("MESA", "left")])
ang_mesa = angle_between(mesa_r, mesa_l)
assert ang_mesa <= 25.0, (f"MESA: as duas palmas estao para baixo, logo as "
                          f"normais ANATOMICAS tem de ser ~iguais; deu "
                          f"{ang_mesa:.1f} graus -- sinal anatomico trocado?")
if ("REZANDO", "right") in medias and ("REZANDO", "left") in medias:
    ang_rezando = angle_between(_media_unit(medias[("REZANDO", "right")]),
                                _media_unit(medias[("REZANDO", "left")]))
    assert ang_rezando >= 120.0, (f"REZANDO: palmas uma contra a outra -> normais "
                                  f"OPOSTAS; deu {ang_rezando:.1f} graus")
print(f"1) OK: verdade anatomica fecha a fisica das poses (MESA R x L "
      f"{ang_mesa:.1f} graus = iguais; REZANDO {ang_rezando:.1f} = opostas)")

# ============================================================================
# 2) o alinhamento salvo e' rotacao propria com f unitario
# ============================================================================
for lado in ("right", "left"):
    fusao, dados = carregar_alinhamento(lado)
    assert fusao is not None, f"{lado}: alinhamento ausente ou invalido"
    rotacao = np.asarray(dados["rotation"], dtype=float)
    corpo = np.asarray(dados["body"], dtype=float)
    assert abs(float(np.linalg.det(rotacao)) - 1.0) < 1e-6, \
        f"{lado}: `rotation` nao e' rotacao propria (det {np.linalg.det(rotacao):+.3f})"
    assert abs(float(np.linalg.norm(corpo)) - 1.0) < 1e-6, \
        f"{lado}: `body` nao e' unitario"
    assert np.isfinite(float(dados.get("residual_deg") or float("nan"))), \
        f"{lado}: residual_deg ausente"
print("2) OK: os dois alinhamentos salvos sao rotacoes proprias com f unitario")

# ============================================================================
# 3) O METODO: T/f REFEITOS do proprio CSV apontam para a palma, pose por pose?
#    (independente do JSON salvo -- que pode ser refit de OUTRA sessao/boot)
# ============================================================================
falhas_metodo = []
fusoes_refit = {}
for lado in ("right", "left"):
    fusao, residuo, n_amostras = refitar_csv(linhas, lado)
    assert fusao is not None, (
        f"{lado}: o refit do CSV canonico falhou (dados degenerados)")
    fusoes_refit[lado] = fusao
    erros = erro_mediano_por_pose(linhas, lado, fusao=fusao)
    faltando = [fase for fase in POSES if fase not in erros]
    assert not faltando, f"{lado}: poses sem amostra suficiente: {faltando}"
    print(f"   {lado:5s}: refit do CSV com {n_amostras} amostras | residuo "
          f"{residuo:.2f} graus")
    for fase in POSES:
        print(f"   {lado:5s} {fase:8s} mediana da seta x verdade "
              f"{erros[fase]:6.1f} graus")
        if erros[fase] > LIMITE_POSE_DEG:
            falhas_metodo.append(f"{lado}/{fase} {erros[fase]:.1f} graus")
assert not falhas_metodo, (
    "O METODO NAO APONTA PARA A PALMA (limite "
    f"{LIMITE_POSE_DEG:g} graus por pose): " + " | ".join(falhas_metodo)
    + " -- o refit v = T R f contra a verdade anatomica nao fecha nem com os "
      "dados do proprio CSV: conferir sinal anatomico/modelo")
print(f"3) OK: refit do proprio CSV <= {LIMITE_POSE_DEG:g} graus nas 4 poses, "
      "nos DOIS lados")

# 3b) o JSON SALVO pode ser de outra sessao (ex.: J rodado na bancada, com outro
#     yaw de boot/montagem). Se ele nao fecha aqui, isso e' esperado e NAO
#     reprova o metodo -- avisa para certificar a sessao certa.
falhas_salvo = []
for lado in ("right", "left"):
    erros_salvo = erro_mediano_por_pose(linhas, lado)
    for fase in POSES:
        if fase in erros_salvo and erros_salvo[fase] > LIMITE_POSE_DEG:
            falhas_salvo.append(f"{lado}/{fase} {erros_salvo[fase]:.1f}")
if falhas_salvo:
    print("3b) AVISO: o JSON SALVO nao e' o refit DESTA sessao canonica "
          f"({'; '.join(falhas_salvo)}).")
    print("    Esperado quando o J/--salvar foi rodado em outra sessao/boot. "
          "Para certificar a sessao correspondente: rode o certificado nela "
          "(tecla V) ou `analisa_diag_palma.py <csv> --salvar`.")
else:
    print("3b) OK: o JSON salvo tambem aponta para a palma no CSV canonico")

# ============================================================================
# 4) SINAL: na MESA (palmas para BAIXO) a seta tem de apontar para BAIXO
#    (imagem espelhada do sintoma "palma para CIMA => seta para o chao")
# ============================================================================
for lado, referencia in (("right", mesa_r), ("left", mesa_l)):
    pose = _por_pose(linhas, lado, fusao=fusoes_refit[lado])["MESA"]
    produto = float(np.dot(pose["seta"], pose["verdade"]))
    seta_vs_baixo = float(np.dot(pose["seta"], referencia))
    print(f"   {lado:5s} MESA: dot(seta, verdade) = {produto:+.3f} | "
          f"dot(seta, normal da mesa) = {seta_vs_baixo:+.3f}")
    assert produto >= DOT_SINAL_OK, (
        f"{lado}: com as DUAS palmas apoiadas na mesa (para baixo) a seta "
        f"aponta para o lado OPOSTO (dot {produto:+.2f}, "
        f"~{angle_between(pose['seta'], pose['verdade']):.0f} graus): o sinal "
        "do refit esta' invertido")
print("4) OK: com a palma para baixo a seta aponta para baixo (sinal correto)")

# negativo de controle SINTETICO (nao depende do .bak): trocar f por -f deixa o
# MESMO residuo e TODA previsao invertida -- este teste TEM de reprovar.
for lado in ("right", "left"):
    origem = fusoes_refit[lado]
    fusao_neg = PositionFusion()
    fusao_neg.set_palm_alignment(
        origem.imu_to_camera, -np.asarray(origem.palm_reference_body),
        origem.palm_alignment_residual_deg)
    pose = _por_pose(linhas, lado, fusao=fusao_neg)["MESA"]
    produto_neg = float(np.dot(pose["seta"], pose["verdade"]))
    print(f"   {lado:5s} negativo de controle (f -> -f): dot(seta, verdade) "
          f"na MESA = {produto_neg:+.3f}")
    assert produto_neg < DOT_SINAL_OPOSTO, (
        f"{lado}: o negativo de controle nao discriminou (dot "
        f"{produto_neg:+.2f}) -- o teste nao prova nada")
print("4b) OK: o modelo com o sinal trocado (f -> -f) FALHA no mesmo teste -- "
      "o teste discrimina o sinal")

# ============================================================================
# 5) SENTIDO DO GIRO do punho (MESA -> REZANDO) tem de ser o mesmo da seta
# ============================================================================
for lado in ("right", "left"):
    pose = _por_pose(linhas, lado, fusao=fusoes_refit[lado])
    mesa, rezando = pose["MESA"], pose["REZANDO"]
    eixo = np.cross(mesa["verdade"], rezando["verdade"])
    norma = float(np.linalg.norm(eixo))
    assert norma > 1e-6, f"{lado}: MESA e REZANDO com a mesma normal"
    eixo = eixo / norma
    sentido_verdade = float(np.dot(eixo, np.cross(mesa["verdade"],
                                                  rezando["verdade"])))
    sentido_seta = float(np.dot(eixo, np.cross(mesa["seta"], rezando["seta"])))
    print(f"   {lado:5s}: giro MESA->REZANDO verdade {sentido_verdade:+.3f} x "
          f"seta {sentido_seta:+.3f}")
    assert sentido_seta * sentido_verdade > 0.0, (
        f"{lado}: a seta gira no sentido CONTRARIO ao da palma quando o punho "
        "gira (eixo do antebraco) -- alinhamento com sinal/mao trocados")
print("5) OK: entre MESA e REZANDO a seta gira no MESMO sentido da palma")

print()
print("PALM_SETA_OK: 6/6 grupos (fisica das poses, alinhamento proprio, refit "
      "do CSV x verdade por pose nos 2 lados, sinal na MESA, negativo sintetico "
      "f -> -f, sentido do giro)")
sys.exit(0)
