"""Analisador OFFLINE do diagnostico da palma (`palm_diag_*.csv`, tecla D).

Por que existe (item 74 do historico -- "a calibracao da palma esta' dando
ruim"): a ferramenta ao vivo grava, uma linha por (frame, lado), a normal da
palma medida pelo Kinect (a VERDADE), a direcao prevista pelo IMU COM
alinhamento (`T R f`, a seta do video) e a prevista SEM alinhamento (`R f`),
com os dois erros em graus. O numero ao vivo (`palm_err_deg`) diz QUE esta'
errado, nunca POR QUE. Com as poses PROGRAMADAS (mesa, rezando, kinect) este
CLI responde a pergunta:

  1. e' SISTEMATICO? se a mediana do erro e' parecida nas tres poses, o
     problema esta' em `T`/`f` (arquivo de outra luva, captura O ruim, normal
     do Kinect com plano/sinal trocados);
  2. depende da ROTACAO? se mesa/rezando estouram e kinect fecha, o alinhamento
     DECORA a pose da tecla O (que e' justamente "palma para o Kinect") ->
     refaca O girando a mao devagar;
  3. o MODELO fecha? `refitar` resolve `T`/`f` DE NOVO so' com o CSV
     (`solve_palm_alignment`, o mesmo modelo `v_i = T R_i f`) e compara o
     residuo do refit com o erro ao vivo. Residuo do refit ALTO quer dizer que
     NENHUMA rotacao explica estes dados: o problema e' o dado capturado
     (normal do Kinect ruidosa/fora do quadro, luva trocada), nao o arquivo de
     alinhamento que esta' carregado.

IMPORTANTE: este modulo NAO importa `kinect_imu_groundtruth` nem `cv2` (aquele
import carrega OpenCV, pykinect e as webcams). Ele roda offline -- inclusive em
outra maquina, so' com o CSV -- e por isso a estatistica e os criterios de
veredito sao DUPLICADOS do `DiagnosticoPalma` (tools/kinect_groundtruth_tool.py)
em vez de importados: e' o preco de manter este arquivo sem OpenCV.

Uso (da raiz do projeto):

    .venv\\Scripts\\python.exe tools\\analisa_diag_palma.py
    .venv\\Scripts\\python.exe tools\\analisa_diag_palma.py palm_diag_20261019_153000.csv
    .venv\\Scripts\\python.exe tools\\analisa_diag_palma.py CAMINHO --settle 1.5 --limite 15

Sem argumento ele pega o `palm_diag_*.csv` MAIS RECENTE da pasta atual (e o
`..._meta.json` de mesmo nome, quando existir). Codigos de saida: 0 = analisou;
2 = nenhum CSV / CSV sem amostras (nao ha' o que concluir).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# A biblioteca vive na raiz do projeto (um nivel acima de tools/). `imu.py` e'
# puro (numpy/math/socket): nada de OpenCV aqui.
RAIZ = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(RAIZ))

from imu import (PALM_ALIGNMENT_MAX_RESIDUAL_DEG,  # noqa: E402
                 PALM_ALIGNMENT_MIN_SAMPLES, angle_between, rotation_matrix,
                 solve_palm_alignment)

#: Prefixo do arquivo gravado pela tecla D (`DiagnosticoPalma`).
PREFIXO_CSV = "palm_diag_"
#: Abaixo disto (linhas de DADOS) o CSV e' uma sessao ABORTADA -- D apertado
#: duas vezes no mesmo segundo ou ESC logo no inicio. Escolher o "mais recente"
#: cru fazia o analisador abrir justamente o arquivo vazio: medido em
#: 19/10/2026, a segunda sessao de 2 s (3,6 KB) virou a "mais recente" e o
#: relatorio saiu todo sem amostras, escondendo a sessao COMPLETA de 21 s (57 KB)
#: que estava do lado. O operador concluia "o diagnostico nao gravou nada".
MIN_LINHAS_CSV_VALIDO = 30
#: Espelho de `PALM_DIAG_SETTLE_S` do tool: segundos iniciais de cada fase que
#: ficam fora do resumo (o experimentador leva ~1 s para chegar na pose). O
#: valor REALMENTE usado vem do meta.json, que e' o que estava valendo na
#: sessao; este e' so' o padrao quando nao ha' meta.
SETTLE_PADRAO_S = 1.5
#: Espelho de `PALM_DIAG_MIN_SAMPLES_FASE`: abaixo disto o veredito avisa que a
#: mao saiu do quadro em vez de fingir que a estatistica vale.
MIN_AMOSTRAS_FASE = 10
#: As TRES poses programadas que o veredito cobra (PREPARO e' acomodacao: a mao
#: ainda esta' chegando na pose e o ruido ali nao informa).
POSES_PROGRAMADAS = ("MESA", "REZANDO", "KINECT")
#: Fases na ordem em que o tool as executa (so' para imprimir a tabela na mesma
#: ordem, mesmo que alguma nao tenha amostra).
ORDEM_FASES = ("PREPARO", "MESA", "REZANDO", "KINECT")
#: Colunas que viram float na leitura. `fase_nome`/`lado` ficam texto.
COLUNAS_NUMERICAS = ("t_s", "fase_idx", "t_fase_s", "roll_deg", "pitch_deg",
                     "yaw_deg", "nx", "ny", "nz", "prev_x", "prev_y", "prev_z",
                     "err_deg", "raw_x", "raw_y", "raw_z", "err_raw_deg",
                     "alinhado",
                     # evidencia crua (item 78): giroscopio/acelerometro do ESP32
                     # e os 4 pontos 3D crus da palma. CSV antigo nao tem estas
                     # colunas -> viram NaN e as secoes novas so' sao omitidas.
                     "imu_timestamp_us",
                     "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps",
                     "wrist_x", "wrist_y", "wrist_z", "index_x", "index_y",
                     "index_z", "pinky_x", "pinky_y", "pinky_z", "middle_x",
                     "middle_y", "middle_z")
#: Direcao ESPERADA da normal da palma em cada fase, no referencial da camera
#: do Kinect (Y para CIMA, Z para dentro da cena; a normal da palma e' forcada
#: para z < 0, ou seja, aponta para a CAMERA quando a palma esta' virada para
#: ela). `None` = a fase nao tem uma direcao unica (rezando: as duas palmas
#: apontam para lados opostos, o que se cobra e' ficar na HORIZONTAL).
DIRECOES_ESPERADAS = {
    "PREPARO": ("palma para o Kinect", (0.0, 0.0, -1.0)),
    "MESA": ("palma para BAIXO", (0.0, -1.0, 0.0)),
    "REZANDO": ("palma na HORIZONTAL (|ny| pequeno)", None),
    "KINECT": ("palma para o Kinect", (0.0, 0.0, -1.0)),
}
#: Limite de |ny| para considerar a fase REZANDO "na horizontal". Heuristica de
#: EXECUCAO (ver `angulo_medio_normal`), nao uma medida da normal.
LIMITE_NY_HORIZONTAL = 0.4


# --------------------------------------------------------------- utilidades
def _numero(texto, padrao=float("nan")):
    """float() tolerante para CSV: vazio/ilegivel vira NaN (nunca 0)."""
    try:
        return float(texto)
    except (TypeError, ValueError):
        return padrao


def caminho_meta(caminho):
    """`palm_diag_X.csv` -> `palm_diag_X_meta.json` (o nome que o tool grava)."""
    texto = str(caminho)
    if texto.lower().endswith(".csv"):
        return Path(texto[:-4] + "_meta.json")
    return Path(texto + "_meta.json")


def _conta_linhas_dados(caminho):
    """Linhas de DADOS do CSV (sem cabecalho); 0 se o arquivo nao abre."""
    try:
        with open(caminho, encoding="utf-8", errors="replace") as arquivo:
            return max(0, sum(1 for _linha in arquivo) - 1)
    except OSError:
        return 0


def mais_recente(pasta=None):
    """`palm_diag_*.csv` mais recente COM AMOSTRAS da pasta, ou None.

    Preferencia: o mais recente cujo CSV tem >= `MIN_LINHAS_CSV_VALIDO` linhas
    de dados; so' se NENHUM tiver (todos abortados) cai para o mais recente cru,
    para o operador ainda ver o erro "sem amostras" em vez de "nao achei arquivo".
    """
    pasta = Path.cwd() if pasta is None else Path(pasta)
    candidatos = list(pasta.glob(PREFIXO_CSV + "*.csv"))
    if not candidatos:
        return None
    candidatos.sort(key=lambda caminho: caminho.stat().st_mtime, reverse=True)
    for caminho in candidatos:
        if _conta_linhas_dados(caminho) >= MIN_LINHAS_CSV_VALIDO:
            return caminho
    return candidatos[0]


def estatisticas(valores):
    """n/media/mediana/desvio-padrao/p90/min/max (graus) de uma lista."""
    dados = np.asarray([valor for valor in valores
                        if valor is not None and np.isfinite(valor)],
                       dtype=float)
    if dados.size == 0:
        return {"n": 0}
    return {
        "n": int(dados.size),
        "media": float(np.mean(dados)),
        "mediana": float(np.median(dados)),
        "desvio": float(np.std(dados)),
        "p90": float(np.percentile(dados, 90)),
        "minimo": float(np.min(dados)),
        "maximo": float(np.max(dados)),
    }


def carregar(caminho):
    """(linhas, meta) do diagnostico.

    `linhas` e' a lista de dicts do `csv.DictReader` com as colunas numericas ja'
    convertidas em float (NaN quando vazias: "nao deu para medir" nunca vira 0).
    `meta` e' o JSON ao lado do CSV ({} se nao existir/ilegivel -- o CSV sozinho
    continua analisavel).
    """
    caminho = Path(caminho)
    linhas = []
    with caminho.open("r", newline="", encoding="utf-8") as arquivo:
        for bruta in csv.DictReader(arquivo):
            linha = dict(bruta)
            for coluna in COLUNAS_NUMERICAS:
                linha[coluna] = _numero(bruta.get(coluna))
            linha["fase_nome"] = (bruta.get("fase_nome") or "").strip()
            linha["lado"] = (bruta.get("lado") or "").strip()
            linhas.append(linha)
    meta = {}
    meta_caminho = caminho_meta(caminho)
    if meta_caminho.exists():
        try:
            meta = json.loads(meta_caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"AVISO: meta {meta_caminho.name} ilegivel ({exc}); seguindo so' "
                  "com o CSV")
            meta = {}
    return linhas, meta


# ---------------------------------------------------------------- estatistica
def resumo_por_fase(linhas, settle=SETTLE_PADRAO_S):
    """{(fase, lado): estatisticas de `err_deg`} -- so' amostras apos o settle.

    O filtro de settle e' o MESMO do resumo impresso pelo tool: as primeiras
    amostras de cada fase sao a mao CHEGANDO na pose (entram no CSV para
    auditoria, mas envenenariam a mediana).
    """
    grupos = {}
    for linha in linhas:
        fase, lado = linha.get("fase_nome"), linha.get("lado")
        if not fase or not lado:
            continue
        t_fase = _numero(linha.get("t_fase_s"))
        if not np.isfinite(t_fase) or t_fase < float(settle):
            continue
        valor = _numero(linha.get("err_deg"))
        if not np.isfinite(valor):
            continue
        grupos.setdefault((fase, lado), []).append(valor)
    return {chave: estatisticas(valores) for chave, valores in grupos.items()}


def resumo_por_lado(linhas, coluna="err_raw_deg"):
    """{lado: estatisticas da coluna} -- sem filtro de settle (o veredito
    "o alinhamento nao esta' sendo aplicado" compara o erro BRUTO com o
    alinhado na sessao inteira)."""
    grupos = {"right": [], "left": []}
    for linha in linhas:
        lado = linha.get("lado")
        if lado not in grupos:
            continue
        valor = _numero(linha.get(coluna))
        if np.isfinite(valor):
            grupos[lado].append(valor)
    return {lado: estatisticas(valores) for lado, valores in grupos.items()}


def erro_refit_por_fase(linhas, lado, alinhamento, corpo,
                        settle=SETTLE_PADRAO_S):
    """{(fase, lado): estatisticas} do erro com o alinhamento do REFIT.

    E' o que mostra se o modelo explica CADA pose: se `T_refit`/`f_refit` fecham
    em 2-3 graus na mesa e no rezando mas o erro AO VIVO la' e' 30, o arquivo de
    alinhamento carregado e' que esta' errado (nao o modelo).
    """
    alinhamento = np.asarray(alinhamento, np.float64).reshape(3, 3)
    corpo = np.asarray(corpo, np.float64).reshape(3)
    grupos = {}
    for linha in linhas:
        if linha.get("lado") != lado:
            continue
        t_fase = _numero(linha.get("t_fase_s"))
        if not np.isfinite(t_fase) or t_fase < float(settle):
            continue
        fase = linha.get("fase_nome")
        rpy = [_numero(linha.get(coluna))
               for coluna in ("roll_deg", "pitch_deg", "yaw_deg")]
        normal = np.array([_numero(linha.get(coluna))
                           for coluna in ("nx", "ny", "nz")], dtype=float)
        if not fase or not np.isfinite(rpy).all() or not np.isfinite(normal).all():
            continue
        previsto = alinhamento @ rotation_matrix(*rpy) @ corpo
        norma = np.linalg.norm(previsto)
        if norma < 1e-9:
            continue
        erro = angle_between(normal, previsto / norma)
        if not np.isfinite(erro):
            continue
        grupos.setdefault((fase, lado), []).append(erro)
    return {chave: estatisticas(valores) for chave, valores in grupos.items()}


def angulo_medio_normal(linhas, settle=SETTLE_PADRAO_S):
    """Cluster da normal do KINECT por (fase, lado) + conferencia de EXECUCAO.

    Devolve {(fase, lado): {"media": (3,), "dispersao_deg": float,
    "esperado_deg": float|None, "esperado": texto, "nota": texto}}.

    A normal media e' a direcao verdadeira media da palma naquela pose; a
    `dispersao_deg` (angulo medio das amostras ate' a media) diz se o
    experimentador ficou PARADO (poucos graus) ou se a fase mediu movimento.

    ATENCAO -- HEURISTICA DE EXECUCAO, NAO VERDADE ABSOLUTA: a comparacao com as
    direcoes esperadas supoe (a) a convencao da camera do Kinect (Y para CIMA,
    Z para dentro da cena, normal da palma forcada para z < 0) e (b) que a
    instrucao da fase foi seguida. Ela serve para descobrir que a mao NAO estava
    na pose pedida (ex.: mesa com o dorso virado para o Kinect, que inverte a
    normal), nunca para "corrigir" a normal -- a normal do Kinect e' a
    referencia do sistema.
    """
    grupos = {}
    for linha in linhas:
        fase, lado = linha.get("fase_nome"), linha.get("lado")
        if not fase or not lado:
            continue
        t_fase = _numero(linha.get("t_fase_s"))
        if not np.isfinite(t_fase) or t_fase < float(settle):
            continue
        normal = np.array([_numero(linha.get(coluna))
                           for coluna in ("nx", "ny", "nz")], dtype=float)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
            continue
        grupos.setdefault((fase, lado), []).append(normal / np.linalg.norm(normal))
    saida = {}
    for chave, normais in grupos.items():
        media = np.mean(normais, axis=0)
        norma = np.linalg.norm(media)
        if norma < 1e-9:
            continue
        media = media / norma
        dispersao = float(np.mean([angle_between(normal, media)
                                   for normal in normais]))
        nome, esperado = DIRECOES_ESPERADAS.get(chave[0], ("?", None))
        esperado_deg = None
        nota = ""
        if esperado is not None:
            esperado_deg = angle_between(media, esperado)
            if esperado_deg > 45.0:
                nota = (f"FUGIU da pose pedida em {esperado_deg:.0f} graus "
                        f"(esperado {nome}) -- confira a execucao")
        else:
            horizontal = abs(float(media[1])) <= LIMITE_NY_HORIZONTAL
            nota = ("na horizontal" if horizontal else
                    f"|ny| = {abs(float(media[1])):.2f} (esperado <= "
                    f"{LIMITE_NY_HORIZONTAL:g}) -- a mao nao estava rezando")
        saida[chave] = {"media": media, "dispersao_deg": dispersao,
                        "esperado_deg": esperado_deg, "esperado": nome,
                        "nota": nota}
    return saida


def refitar(linhas, lado):
    """(T, f, residuo) resolvido SO' com o CSV, ou None.

    `solve_palm_alignment` com R = `rotation_matrix(roll, pitch, yaw)` e
    v = normal do Kinect -- o MESMO modelo `v_i = T R_i f` da tecla O, refeito
    offline. Se o residuo sai alto, NENHUMA rotacao `T` explica estes dados: o
    problema esta' na CAPTURA (normal ruidosa, mao fora do quadro, luva
    trocada), nao no arquivo de alinhamento que esta' carregado.

    Cuidado ao interpretar (e' o bug que este diagnostico investiga): uma
    rotacao CONSTANTE errada em `T` e' absorvida pelo refit -- ele vai achar
    outro `T` e residuo baixo. O que o refit denuncia e' erro que DEPENDE da
    pose (o modelo nao fecha em todas as poses ao mesmo tempo).
    """
    rotacoes, direcoes = [], []
    for linha in linhas:
        if linha.get("lado") != lado:
            continue
        rpy = [_numero(linha.get(coluna))
               for coluna in ("roll_deg", "pitch_deg", "yaw_deg")]
        normal = np.array([_numero(linha.get(coluna))
                           for coluna in ("nx", "ny", "nz")], dtype=float)
        if not np.isfinite(rpy).all() or not np.isfinite(normal).all():
            continue
        if np.linalg.norm(normal) < 1e-9:
            continue
        rotacoes.append(rotation_matrix(*rpy))
        direcoes.append(normal)
    return solve_palm_alignment(rotacoes, direcoes)


# ------------------------------------------------- consistencia da ATITUDE (IMU)
def rotacao_media(linhas):
    """Rotacao media (3,3) de uma lista de amostras, ou None.

    A media componente-a-componente de matrizes de rotacao pode nao ser
    rotacao; projeta-se de volta (SVD) para a rotacao propria mais proxima.
    """
    matrizes = []
    for linha in linhas:
        rpy = [linha.get(coluna) for coluna in ("roll_deg", "pitch_deg", "yaw_deg")]
        if not all(valor is not None and np.isfinite(valor) for valor in rpy):
            continue
        matrizes.append(rotation_matrix(*rpy))
    if not matrizes:
        return None
    media = np.mean(np.asarray(matrizes, np.float64), axis=0)
    u, _s, vt = np.linalg.svd(media)
    return u @ vt


def _angulo_rotacao_graus(rotacao):
    """Angulo (graus) de uma rotacao pela formula do traco (robusto a ruido)."""
    traco = float(np.clip((np.trace(rotacao) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(traco)))


def consistencia_pose_repetida(linhas, fases=("PREPARO", "KINECT"),
                               settle=SETTLE_PADRAO_S):
    """A MESMA pose fisica medida duas vezes: quanto o IMU e o Kinect divergem.

    PREPARO e KINECT sao a MESMA pose pedida ("palmas para o Kinect"). As duas
    medidas sao do mesmo corpo, entao o angulo entre as duas atitudes do IMU
    teria de ser ~o angulo entre as duas normais do Kinect. Medido na bancada em
    19/10/2026 (sessao 15:55:51): Kinect 11-20 graus, IMU 74-171 graus --
    assinatura de yaw INCONSISTENTE, nao de ruido de captura. O firmware da luva
    integra `yaw += gz*dt` (gyro no eixo Z do CORPO), o que so' vale com o sensor
    nivelado; com a mao inclinada o yaw verdadeiro mistura gx/gy.

    Devolve {lado: {"imu_deg": ..., "kinect_deg": ..., "n": ...}}. Lados sem as
    duas fases (ou poucas amostras) ficam de fora.
    """
    resultado = {}
    for lado in ("right", "left"):
        amostras = {fase: [] for fase in fases}
        for linha in linhas:
            if linha.get("lado") != lado:
                continue
            fase = linha.get("fase_nome")
            if fase not in amostras:
                continue
            t_fase = linha.get("t_fase_s")
            if t_fase is None or not np.isfinite(t_fase) or t_fase < settle:
                continue
            amostras[fase].append(linha)
        primeira, segunda = fases
        if min(len(amostras[primeira]), len(amostras[segunda])) < 3:
            continue
        rotacao_primeira = rotacao_media(amostras[primeira])
        rotacao_segunda = rotacao_media(amostras[segunda])
        normais = []
        for fase in fases:
            vetores = [np.asarray([linha["nx"], linha["ny"], linha["nz"]],
                                  np.float64) for linha in amostras[fase]]
            vetores = [vetor for vetor in vetores if np.isfinite(vetor).all()]
            if not vetores:
                normais = []
                break
            normal = np.mean(vetores, axis=0)
            norma = float(np.linalg.norm(normal))
            if norma < 1e-9:
                normais = []
                break
            normais.append(normal / norma)
        if (rotacao_primeira is None or rotacao_segunda is None
                or len(normais) != 2):
            continue
        resultado[lado] = {
            "imu_deg": _angulo_rotacao_graus(rotacao_primeira.T @ rotacao_segunda),
            "kinect_deg": angle_between(normais[0], normais[1]),
            "n": len(amostras[primeira]) + len(amostras[segunda]),
        }
    return resultado


def veredito_palm_diag(estat_por_fase, bruto, limite=PALM_ALIGNMENT_MAX_RESIDUAL_DEG,
                       minimo=MIN_AMOSTRAS_FASE):
    """Linhas do veredito de UM lado -- os MESMOS criterios do tool.

    `estat_por_fase` = {fase: estatisticas de `err_deg`} (ja' sem o settle);
    `bruto` = estatisticas de `err_raw_deg` do lado (o erro SEM alinhamento).

    A ordem das causas e' a ordem em que elas se escondem uma atras da outra:
    poucas amostras invalidam qualquer conclusao; "o alinhamento nao esta'
    aplicado" explica erro grande em TODAS as poses; so' depois faz sentido
    separar sistematico (T/f) de dependente da rotacao (modelo/captura).
    """
    notas = []
    poses = [fase for fase in POSES_PROGRAMADAS
             if estat_por_fase.get(fase, {}).get("n")]
    if not poses:
        return ["sem amostras validas: nenhuma pose programada foi medida"]
    for fase, dados in estat_por_fase.items():
        if 0 < dados.get("n", 0) < minimo:
            notas.append(f"sem amostras suficientes na fase {fase} "
                         f"({dados['n']} < {minimo}): a mao saiu do quadro OU "
                         "a outra mao cobriu esta (nao encoste as maos)")
    mediana = {fase: estat_por_fase[fase]["mediana"] for fase in poses}
    medio = float(np.mean([estat_por_fase[fase]["media"] for fase in poses]))
    bruto_medio = (bruto or {}).get("media")
    if bruto_medio is not None and np.isfinite(bruto_medio) \
            and bruto_medio <= medio + 2.0:
        notas.append("o alinhamento NAO esta' sendo aplicado (sem "
                     "imu_palm_alignment_<lado>.json / bootstrap de 1 amostra): "
                     f"erro SEM alinhamento {bruto_medio:.1f} graus x "
                     f"{medio:.1f} com ele")
    ok = {fase: mediana[fase] <= limite for fase in poses}
    if len(poses) == 3 and all(ok.values()):
        notas.append("ALINHAMENTO OK nas 3 poses programadas")
    elif ok.get("KINECT") and (not ok.get("MESA", True)
                               or not ok.get("REZANDO", True)):
        notas.append("o alinhamento DECORA a pose da tecla O: T/f estao errados "
                     "(captura O quase parada ou sempre na mesma orientacao) -> "
                     "refaca O girando a mao devagar")
    elif not ok.get("KINECT", True):
        notas.append("nem a pose da calibracao fecha: conferir "
                     "enquadramento/profundidade da mao, luva trocada "
                     "(--imu-portas) ou normal do Kinect ruidosa")
    # Numeros para quem for conferir a mao: a mediana de cada pose medida.
    notas.append("medianas: " + ", ".join(
        f"{fase} {mediana[fase]:.1f}" for fase in poses) + " graus")
    return notas


# --------------------------------------------------------------------- saida
def _linha_tabela(lado, fase, dados):
    """Uma linha da tabela ASCII (traco curto quando a fase nao tem amostra)."""
    if not dados.get("n"):
        return f"{lado:5s} | {fase:8s} | {0:4d} | {'-':>7s} | {'-':>7s} | " \
               f"{'-':>6s} | {'-':>6s} | {'-':>6s} | {'-':>6s}"
    return (f"{lado:5s} | {fase:8s} | {dados['n']:4d} | {dados['media']:7.2f} | "
            f"{dados['mediana']:7.2f} | {dados['desvio']:6.2f} | "
            f"{dados['p90']:6.2f} | {dados['minimo']:6.2f} | "
            f"{dados['maximo']:6.2f}")


def _cabecalho_tabela():
    return (f"{'lado':5s} | {'fase':8s} | {'n':>4s} | {'media':>7s} | "
            f"{'mediana':>7s} | {'std':>6s} | {'p90':>6s} | {'min':>6s} | "
            f"{'max':>6s}")


def _formata_vetor(vetor):
    return "(" + ", ".join(f"{valor:+.2f}" for valor in np.asarray(vetor)) + ")"


def analisar(caminho, settle=None, limite=None):
    """Roda a analise completa e imprime; devolve o codigo de saida."""
    caminho = Path(caminho)
    linhas, meta = carregar(caminho)
    limite = (PALM_ALIGNMENT_MAX_RESIDUAL_DEG if limite is None
              else float(limite))
    if settle is None:
        settle = float(meta.get("settle_s", SETTLE_PADRAO_S))
    else:
        settle = float(settle)
    print("=" * 78)
    print(f"DIAGNOSTICO DA PALMA -- {caminho.name} ({len(linhas)} linhas)")
    print("=" * 78)
    if meta:
        print(f"meta: {caminho_meta(caminho).name} | sessao de "
              f"{meta.get('pc_time', '?')} | limite "
              f"{meta.get('limite_erro_graus', limite)} graus")
        for lado, dados in (meta.get("lados") or {}).items():
            estado = dados.get("alinhamento")
            situacao = ("sem alinhamento (bootstrap de 1 amostra)"
                        if estado is None else
                        f"residuo {estado.get('residual_deg')}")
            arquivo = (f"de {dados.get('arquivo_mtime')}"
                       if dados.get("arquivo_existe") else "AUSENTE")
            print(f"  {lado}: {situacao} | arquivo {dados.get('arquivo')} "
                  f"{arquivo}")
    print(f"settle: {settle:g} s (amostras do inicio de cada fase fora do resumo)"
          f" | limite por fase: {limite:g} graus")
    if not linhas:
        print("CSV SEM AMOSTRAS: a sessao nao gravou nenhuma linha -- nada a "
              "concluir. Rode a tecla D com a mao DENTRO do quadro do Kinect "
              "(a linha so' existe quando a mao E a normal do Kinect existem).")
        return 2

    resumo = resumo_por_fase(linhas, settle)
    resumo_bruto = resumo_por_lado(linhas, "err_raw_deg")
    cluster = angulo_medio_normal(linhas, settle)
    if not resumo:
        print(f"NENHUMA amostra passou o settle de {settle:g} s: a mao saiu do "
              "quadro logo no inicio de cada fase. Nada a concluir.")
        return 2

    # ---------------------------------------------------------- tabela
    print()
    print("ERRO AO VIVO (Kinect x IMU) POR FASE E LADO")
    print(_cabecalho_tabela())
    print("-" * len(_cabecalho_tabela()))
    for lado in ("right", "left"):
        for fase in ORDEM_FASES:
            if (fase, lado) not in resumo:
                continue
            print(_linha_tabela(lado, fase, resumo[(fase, lado)]))
        print("-" * len(_cabecalho_tabela()))
    if resumo_bruto:
        print("erro SEM alinhamento (R f, T = identidade), sessao inteira: "
              + " | ".join(f"{lado} {dados['media']:.1f} graus"
                           for lado, dados in resumo_bruto.items()
                           if dados.get("n")))

    # -------------------------------------------------- execucao das poses
    print()
    print("EXECUCAO DAS POSES (HEURISTICA, nao verdade absoluta: a normal do "
          "Kinect E' a referencia)")
    for lado in ("right", "left"):
        for fase in ORDEM_FASES:
            dados = cluster.get((fase, lado))
            if dados is None:
                continue
            angulo_txt = ("" if dados["esperado_deg"] is None else
                          f" (angulo {dados['esperado_deg']:.0f} graus)")
            print(f"  {lado:5s} {fase:8s} normal media "
                  f"{_formata_vetor(dados['media'])} | dispersao "
                  f"{dados['dispersao_deg']:.1f} graus | esperado: "
                  f"{dados['esperado']}{angulo_txt}"
                  + (f" | {dados['nota']}" if dados["nota"] else ""))

    # -------------------------------------------------------------- modelo
    print()
    print("MODELO: residuo do refit (v_i = T R_i f, resolvido SO' com este CSV)")
    refits = {}
    for lado in ("right", "left"):
        resolvido = refitar(linhas, lado)
        if resolvido is None:
            print(f"  {lado}: amostras insuficientes para refitar "
                  f"(minimo {PALM_ALIGNMENT_MIN_SAMPLES})")
            continue
        alinhamento, corpo, residuo = resolvido
        refits[lado] = resolvido
        veredito = ("OK" if residuo <= limite else "RESIDUO ALTO")
        print(f"  {lado}: residuo {residuo:.2f} graus ({veredito}) | "
              f"f_body {_formata_vetor(corpo)}")
        refit_fase = erro_refit_por_fase(linhas, lado, alinhamento, corpo, settle)
        comparacoes = []
        for fase in ORDEM_FASES:
            vivo = resumo.get((fase, lado))
            novo = refit_fase.get((fase, lado))
            if not vivo or not vivo.get("n") or not novo or not novo.get("n"):
                continue
            comparacoes.append(f"{fase} vivo {vivo['mediana']:.1f} x refit "
                               f"{novo['mediana']:.1f}")
        if comparacoes:
            print(f"    {lado}: " + " | ".join(comparacoes) + " (mediana, graus)")

    # ------------------------------------ consistencia da ATITUDE do IMU
    repetida = consistencia_pose_repetida(linhas, settle=settle)
    if repetida:
        print()
        print("CONSISTENCIA DA ATITUDE DO IMU: a MESMA pose medida duas vezes "
              "(PREPARO x KINECT)")
        for lado, dados in repetida.items():
            print(f"  {lado:5s}: o IMU diz que o sensor girou "
                  f"{dados['imu_deg']:.0f} graus | o Kinect diz que a palma "
                  f"girou {dados['kinect_deg']:.0f} graus")
            if dados["imu_deg"] > dados["kinect_deg"] + 40.0:
                print(f"    {lado}: IMPOSSIVEL para um corpo rigido -> o yaw do "
                      "IMU nao representa a atitude (o firmware integra o "
                      "giroscopio so' no eixo Z do corpo). Refazer a tecla O "
                      "NAO resolve; a correcao e' de firmware/filtro (item 78).")

    # ------------------------------------------------------------ veredito
    print()
    print("VEREDITO")
    problemas = []
    for lado in ("right", "left"):
        estat_lado = {fase: resumo[(fase, lado)]
                      for fase in ORDEM_FASES if (fase, lado) in resumo}
        if not estat_lado:
            print(f"  {lado}: sem amostras (a mao nao apareceu no Kinect "
                  "durante o diagnostico)")
            problemas.append("enquadramento")
            continue
        for nota in veredito_palm_diag(estat_lado, resumo_bruto.get(lado),
                                       limite):
            print(f"  {lado}: {nota}")
            if "DECORA" in nota:
                problemas.append("refazer O")
            elif "nao esta' sendo aplicado" in nota:
                problemas.append("sem alinhamento")
            elif "nem a pose" in nota:
                problemas.append("captura")
        resolvido = refits.get(lado)
        if resolvido is not None and resolvido[2] > limite:
            print(f"  {lado}: o MODELO nao fecha com os dados desta sessao "
                  f"(residuo do refit {resolvido[2]:.1f} > {limite:g}): a "
                  "captura e' que esta' ruim (normal do Kinect ruidosa, mao "
                  "fora do quadro/na borda, luva trocada), nao o arquivo de "
                  "alinhamento")
            problemas.append("captura")
        dados_repetida = repetida.get(lado)
        if (dados_repetida is not None
                and dados_repetida["imu_deg"] > dados_repetida["kinect_deg"] + 40.0):
            print(f"  {lado}: o YAW do IMU e' INCONSISTENTE para um corpo "
                  f"rigido (mesma pose: IMU {dados_repetida['imu_deg']:.0f} x "
                  f"Kinect {dados_repetida['kinect_deg']:.0f} graus)")
            problemas.append("yaw do IMU")

    # ------------------------------------------------------- proximos passos
    print()
    print("PROXIMOS PASSOS")
    if "yaw do IMU" in problemas:
        print("  0) PRIORIDADE: o yaw do IMU (firmware) NAO e' consistente -- "
              "nenhuma captura da tecla O vai fechar enquanto isso. Corrigir o "
              "firmware (integrar o giroscopio TAMBEM nos eixos gx/gy, pela "
              "atitude atual) ou aplicar o filtro no PC; detalhes no item 78 do "
              "historico.")
    print("  1) refaca a tecla O GIRANDO a mao devagar e em varias direcoes (com "
          "a palma virada para o Kinect, sem virar o dorso) -- captura quase "
          "parada ou sempre na mesma orientacao 'decora' uma pose so'.")
    print("  2) confira a luva de cada lado: --imu-portas \"4210,4211\" (ordem "
          "DIREITA,ESQUERDA). Luva trocada da' residuo alto em um lado so'.")
    print("  3) confira o enquadramento/profundidade: mao INTEIRA dentro do "
          "quadro do Kinect, longe das bordas, sem oclusao da outra mao -- a "
          "normal do Kinect sai do depth e ruido de profundidade vira erro de "
          "dezenas de graus.")
    if not problemas:
        print("  nada urgente: as tres poses fecharam dentro do limite.")
    print()
    print(f"arquivo analisado: {caminho}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default=None,
                        help="palm_diag_*.csv (padrao: o MAIS RECENTE da pasta atual)")
    parser.add_argument("--settle", type=float, default=None,
                        help="segundos iniciais de cada fase fora do resumo "
                             "(padrao: o do meta.json, senao 1,5)")
    parser.add_argument("--limite", type=float, default=None,
                        help=f"limite (graus) da mediana por fase "
                             f"(padrao {PALM_ALIGNMENT_MAX_RESIDUAL_DEG:g})")
    args = parser.parse_args(argv)
    caminho = None
    if args.csv:
        caminho = Path(args.csv)
        if not caminho.exists():
            print(f"arquivo nao encontrado: {caminho}")
            return 2
    else:
        caminho = mais_recente()
        if caminho is None:
            print(f"nenhum {PREFIXO_CSV}*.csv na pasta atual ({Path.cwd()}).\n"
                  "Rode a ferramenta com --log-overlay, aperte D no video e "
                  "volte com o CSV -- ou passe o caminho na linha de comando.")
            return 2
        print(f"(sem argumento: usando o mais recente da pasta -- {caminho.name})")
    return analisar(caminho, settle=args.settle, limite=args.limite)


if __name__ == "__main__":
    sys.exit(main())
