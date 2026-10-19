"""Teste do DIAGNOSTICO DA PALMA (tecla D do tool) e do analisador offline.

Por que existe (item 74 do historico -- "a calibracao da palma esta' dando
ruim"): o diagnostico nasceu como um LOG contra um MOVIMENTO PROGRAMADO --
quatro fases (preparo, palma na mesa, maos rezando, palma para o Kinect) para
descobrir se o erro do vetor da palma e' SISTEMATICO (T/f/arquivo errados) ou
DEPENDENTE DA ROTACAO (o alinhamento so' vale na pose da tecla O). Um
diagnostico que mede errado (fase trocada, settle nao aplicado, erro calculado
com a formula errada, CSV com colunas fora de ordem) levaria a conclusao errada
sobre a CALIBRACAO -- por isso ele e' testado ate' o fim, com CSV sintetico,
sem Kinect, sem IMU e sem cv2 no analisador.

Casos:
 1. `DiagnosticoPalma` com RELOGIO INJETADO: as 4 fases caem nos tempos certos
    (fase_atual/restante/restante_total), o filtro de settle tira do resumo as
    amostras do inicio de cada fase (mas elas ficam no CSV), `finalizar` grava o
    resumo e `cancelar` fecha sem quebrar; o objeto pode ser reiniciado.
 2. `registrar`: a linha do CSV tem as 20 colunas do cabecalho e os erros saem
    certos -- 0 grau quando a previsao E' a normal, 90 graus quando e'
    perpendicular, vazio quando nao da' para medir -- e `alinhado` diz se havia
    `palm_reference_body` (1) ou nao (0). Tambem respeita os 20 Hz por lado.
 3. ponta a ponta SINTETICO (sem hardware): um `palm_diag_*.csv` + meta gerado
    com T/f VERDADEIROS conhecidos (2 lados, 4 fases com normais plausiveis,
    40 amostras/fase, ruido deterministico) -> o resumo fecha dentro de 5 graus
    e o `refitar` do analisador acha o modelo com residuo <= 5 graus. Depois o
    MESMO CSV e' corrompido com um erro que DEPENDE DA POSE (a Kabsch NAO
    consegue absorver: um T errado constante seria invisivel, porque o refit
    acharia outro T) e o teste exige que o refit DENUNCIE (residuo > 10 graus) e
    que o resumo acuse mediana acima do limite -- e' o caso "a captura esta'
    ruim", que o veredito tem de separar de "o arquivo esta' errado".
 4. `tools/analisa_diag_palma.py` NAO pode importar cv2/pykinect nem o modulo
    do ground truth (ele roda offline, so' com o CSV): checado pelo AST.
 5. `mais_recente` (o CLI sem argumento) IGNORA a sessao abortada: medido em
    19/10/2026 que apertar D duas vezes deixa um CSV de 2 s como o "mais
    recente" e o relatorio saia sem amostras, escondendo a sessao completa.

Roda com o Python do projeto (sem pytest, sem hardware):
    .venv\\Scripts\\python.exe test_diagnostico_palma.py
"""
import ast
import contextlib
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(".", "tools"))

import analisa_diag_palma as analisador             # noqa: E402
import kinect_groundtruth_tool as tool              # noqa: E402
from imu import angle_between, rotation_matrix      # noqa: E402

#: Fases que o tool tem de executar, NESTA ordem (o teste trava o protocolo:
#: mudar a ordem muda o significado do diagnostico, nao e' detalhe interno).
FASES_ESPERADAS = ["PREPARO", "MESA", "REZANDO", "KINECT"]


# ------------------------------------------------------------------- apoio
class RelogioFalso:
    """Relogio injetavel: avanca 21 s de diagnostico sem dormir 21 s."""

    def __init__(self, agora=500.0):
        self.agora = float(agora)

    def __call__(self):
        return self.agora

    def avancar(self, segundos):
        self.agora += float(segundos)
        return self.agora


class AmostraFalsa:
    """Amostra minima de IMU (o que `rotation_matrix`/`palm_direction` leem)."""

    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll_deg = float(roll)
        self.pitch_deg = float(pitch)
        self.yaw_deg = float(yaw)
        self.timestamp_us = 1_000_000


class FusaoFalsa:
    """Espelha o que o `DiagnosticoPalma` usa de `PositionFusion`.

    `PALM_REFERENCIA` = None faz o papel da luva SEM alinhamento (bootstrap de 1
    amostra, `palm_reference_body is None`): e' o caso que o veredito tem de
    detectar ("o alinhamento nao esta' sendo aplicado").
    """

    def __init__(self, referencia=None, alinhamento=None, residuo=2.0):
        self.imu_to_camera = (np.eye(3) if alinhamento is None
                              else np.asarray(alinhamento, np.float64))
        self.palm_reference_body = (None if referencia is None
                                    else np.asarray(referencia, np.float64))
        self.palm_alignment_residual_deg = float(residuo)

    def palm_alignment_state(self):
        if self.palm_reference_body is None:
            return None
        return {"rotation": self.imu_to_camera.tolist(),
                "body": self.palm_reference_body.tolist(),
                "residual_deg": self.palm_alignment_residual_deg}

    def palm_direction(self, sample):
        if sample is None or self.palm_reference_body is None:
            return None
        direcao = (self.imu_to_camera
                   @ rotation_matrix(sample.roll_deg, sample.pitch_deg,
                                     sample.yaw_deg)
                   @ self.palm_reference_body)
        norma = np.linalg.norm(direcao)
        return None if norma < 1e-9 else direcao / norma


@contextlib.contextmanager
def pasta_temporaria():
    """CWD numa pasta TEMPORARIA: os `palm_diag_*` nao podem sujar a raiz."""
    anterior = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="diag_palma_") as pasta:
        os.chdir(pasta)
        try:
            yield Path(pasta)
        finally:
            os.chdir(anterior)


def rotacao_eixo_angulo(eixo, graus):
    """Rotacao propria por eixo + angulo (Rodrigues) -- T/f 'verdadeiros'."""
    eixo = np.asarray(eixo, np.float64)
    eixo = eixo / np.linalg.norm(eixo)
    angulo = math.radians(float(graus))
    k = np.array([[0.0, -eixo[2], eixo[1]],
                  [eixo[2], 0.0, -eixo[0]],
                  [-eixo[1], eixo[0], 0.0]])
    return np.eye(3) + math.sin(angulo) * k + (1.0 - math.cos(angulo)) * (k @ k)


def rpy_de_rotacao(rotacao):
    """(roll, pitch, yaw) que `rotation_matrix` converte de volta na matriz.

    `rotation_matrix` e' Rz(yaw) @ Ry(pitch) @ Rx(roll) (convencao do firmware)
    -- a MESMA inversao usada em test_palm_alignment.py.
    """
    pitch = -math.asin(float(np.clip(rotacao[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotacao[2, 1]), float(rotacao[2, 2]))
    yaw = math.atan2(float(rotacao[1, 0]), float(rotacao[0, 0]))
    return tuple(math.degrees(valor) for valor in (roll, pitch, yaw))


def rotacao_de_normal(normal_corpo, giro_graus=0.0):
    """Rotacao R (corpo -> mundo do IMU) com `f = (0,0,-1)` em `normal_corpo`.

    `giro_graus` e' um TWIST em torno da propria normal: NAO muda `R f` (o que o
    modelo usa) mas varia a matriz R -- e' o que da' variacao de atitude para o
    refit identificar `(T, f)` em vez de decorar uma pose (o mesmo papel que
    "girar a mao" tem na tecla O).
    """
    f_body = np.array([0.0, 0.0, -1.0])
    alvo = np.asarray(normal_corpo, np.float64)
    alvo = alvo / np.linalg.norm(alvo)
    eixo = np.cross(f_body, alvo)
    seno = float(np.linalg.norm(eixo))
    cosseno = float(np.clip(np.dot(f_body, alvo), -1.0, 1.0))
    if seno < 1e-9:
        alinhamento = np.eye(3)
    else:
        alinhamento = rotacao_eixo_angulo(eixo / seno,
                                          math.degrees(math.atan2(seno, cosseno)))
    return rotacao_eixo_angulo(alvo, giro_graus) @ alinhamento


def linhas_do_csv(caminho):
    """(cabecalho, linhas cruas) do CSV -- para conferir a ORDEM das colunas."""
    with Path(caminho).open("r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.reader(arquivo)
        cabecalho = next(leitor)
        return cabecalho, [linha for linha in leitor if linha]


# ------------------------------------------------------------------- teste 1
def teste_fases_settle_e_cancelar():
    """As 4 fases nos tempos certos, settle aplicado e cancelar sem estrago."""
    with pasta_temporaria():
        relogio = RelogioFalso(500.0)
        avisos = []
        diag = tool.DiagnosticoPalma(relogio=relogio, avisar=avisos.append)
        fusions = {"right": FusaoFalsa([0.0, 0.0, -1.0]),
                   "left": FusaoFalsa([0.0, 0.0, -1.0])}
        assert diag.iniciar(500.0, fusions) is True
        assert diag.ativa
        assert diag.iniciar(500.0, fusions) is False, \
            "iniciar duas vezes nao pode criar um segundo par de arquivos"
        assert Path(diag.caminho_csv).exists()
        assert Path(diag.caminho_meta).exists()
        meta = json.loads(Path(diag.caminho_meta).read_text(encoding="utf-8"))
        assert [fase["nome"] for fase in meta["fases"]] == FASES_ESPERADAS
        assert meta["settle_s"] == tool.PALM_DIAG_SETTLE_S
        assert meta["limite_erro_graus"] == tool.PALM_DIAG_MAX_ERR_DEG
        assert meta["lados"]["right"]["alinhamento"] is not None
        assert "imu_palm_alignment_right.json" in meta["lados"]["right"]["arquivo"]

        # --- tempos das fases: PREPARO 3 s, MESA 6 s, REZANDO 6 s, KINECT 6 s
        assert abs(diag.duracao_total_s - 21.0) < 1e-9
        assert [nome for nome, _s, _i in diag.fases] == FASES_ESPERADAS
        indice, nome, t_fase, restante, instrucao = diag.fase_atual(500.0)
        assert (indice, nome) == (0, "PREPARO")
        assert abs(t_fase) < 1e-9 and abs(restante - 3.0) < 1e-9
        assert "palmas viradas para o Kinect" in instrucao
        indice, nome, _t, restante, _i = diag.fase_atual(502.9)
        assert indice == 0 and abs(restante - 0.1) < 1e-6
        indice, nome, t_fase, _r, _i = diag.fase_atual(503.0)
        assert (indice, nome) == (1, "MESA") and abs(t_fase) < 1e-9
        indice, nome, _t, _r, _i = diag.fase_atual(508.99)
        assert (indice, nome) == (1, "MESA")
        indice, nome, _t, _r, _i = diag.fase_atual(509.0)
        assert (indice, nome) == (2, "REZANDO")
        indice, nome, _t, _r, _i = diag.fase_atual(515.0)
        assert (indice, nome) == (3, "KINECT")
        indice, nome, _t, restante, _i = diag.fase_atual(520.9)
        assert nome == "KINECT" and abs(restante - 0.1) < 1e-6
        assert diag.fase_atual(521.0) is None, "depois de 21 s nao ha' fase"
        # restante_total_s usa o relogio INJETADO (nao um time.monotonic novo)
        relogio.agora = 510.0
        assert abs(diag.restante_total_s - 11.0) < 1e-9

        # --- settle: em MESA, amostras com t_fase < 1,5 s NAO entram no resumo
        normal = np.array([0.0, -1.0, 0.0])
        amostra = AmostraFalsa()
        assert diag.registrar("right", 503.5, amostra, normal, fusions["right"])
        assert diag.registrar("right", 505.0, amostra, normal, fusions["right"])
        assert diag.registrar("right", 506.0, amostra, normal, fusions["right"])
        assert diag.registrar("right", 507.0, amostra, normal, fusions["right"])
        # 20 Hz POR LADO: 0,02 s depois da ultima a linha e' recusada
        assert diag.registrar("right", 507.02, amostra, normal,
                              fusions["right"]) is False
        assert diag.registrar("left", 507.02, amostra, normal,
                              fusions["left"]) is True
        caminho_resumo = diag.finalizar()
        assert not diag.ativa
        assert caminho_resumo is not None and Path(caminho_resumo).exists()
        assert Path(caminho_resumo).name.endswith(".csv_resumo.txt")
        texto = Path(caminho_resumo).read_text(encoding="utf-8")
        assert "VEREDITO" in texto and "MESA" in texto
        # As 5 linhas ficam no CSV (inclusive a de 0,5 s)...
        cabecalho, linhas = linhas_do_csv(diag.caminho_csv)
        assert len(linhas) == 5, len(linhas)
        # ...mas o resumo (o MESMO criterio do analisador) so' conta 3 em MESA.
        carregadas, meta_carregada = analisador.carregar(diag.caminho_csv)
        assert meta_carregada.get("settle_s") == tool.PALM_DIAG_SETTLE_S
        resumo = analisador.resumo_por_fase(carregadas, tool.PALM_DIAG_SETTLE_S)
        assert resumo[("MESA", "right")]["n"] == 3, resumo.get(("MESA", "right"))
        assert abs(resumo[("MESA", "right")]["mediana"] - 90.0) < 1e-6
        # normal (0,-1,0) x previsao (0,0,-1) = 90 graus (o caso perpendicular
        # do teste 2 aparece aqui de novo, agora pelo resumo).
        assert resumo[("MESA", "left")]["n"] == 1

        # --- reinicio depois de finalizar + cancelar (fecha sem resumo)
        assert diag.iniciar(600.0, fusions) is True
        csv_antigo = diag.caminho_csv
        diag.cancelar()
        assert not diag.ativa
        cabecalho, linhas = linhas_do_csv(csv_antigo)
        assert linhas == [], "cancelar nao pode deixar linha pela metade"
        assert not Path(str(csv_antigo) + "_resumo.txt").exists(), \
            "cancelar nao gera resumo"
        assert diag.registrar("right", 601.0, amostra, normal,
                              fusions["right"]) is False, \
            "registrar depois de cancelar tem de ser recusado (arquivo fechado)"

        # --- finalizar SEM nenhuma amostra: avisa e nao cria resumo vazio
        assert diag.iniciar(700.0, fusions) is True
        avisos.clear()
        assert diag.finalizar() is None
        assert any("nenhuma amostra" in aviso for aviso in avisos), avisos
        assert Path(diag.caminho_csv).exists()
    print("[ok] 1) fases/relogio, settle no resumo (CSV completo), cancelar e "
          "reinicio; finalizar sem amostras avisa")


# ------------------------------------------------------------------- teste 2
def teste_registrar_erros_e_colunas():
    """20 colunas na ordem do cabecalho e err_deg = 0/90 graus nos casos exatos."""
    with pasta_temporaria():
        relogio = RelogioFalso(100.0)
        avisos = []
        diag = tool.DiagnosticoPalma(relogio=relogio, avisar=avisos.append)
        normal = np.array([0.0, -1.0, 0.0])
        # previsao == normal (R = I, f = normal) -> erro 0 grau.
        fusao_igual = FusaoFalsa([0.0, -1.0, 0.0])
        # previsao perpendicular a normal -> erro 90 graus.
        fusao_perpendicular = FusaoFalsa([0.0, 0.0, -1.0])
        # luva sem alinhamento: nada a prever (err_deg vazio, alinhado = 0).
        fusao_sem = FusaoFalsa(None)
        assert diag.iniciar(100.0, {"right": fusao_igual, "left": fusao_sem})
        amostra = AmostraFalsa()
        assert diag.registrar("right", 100.5, amostra, normal, fusao_igual)
        assert diag.registrar("right", 101.0, amostra, normal,
                              fusao_perpendicular)
        assert diag.registrar("left", 101.5, amostra, normal, fusao_sem)
        # recusas: sem amostra de IMU / sem normal do Kinect
        assert diag.registrar("right", 102.0, None, normal, fusao_igual) is False
        assert diag.registrar("right", 102.1, amostra, None, fusao_igual) is False
        diag.finalizar()

        cabecalho, linhas = linhas_do_csv(diag.caminho_csv)
        assert cabecalho == tool.cabecalho_palm_diag(), cabecalho
        # 20 colunas de medida + as 19 de EVIDENCIA CRUA do item 78
        # (giroscopio/acelerometro/timestamp + 4 pontos 3D da palma).
        assert len(cabecalho) == len(tool.cabecalho_palm_diag()) == 39, \
            len(cabecalho)
        assert all(len(linha) == len(cabecalho) for linha in linhas), \
            [len(linha) for linha in linhas]
        linhas_lidas, _meta = analisador.carregar(diag.caminho_csv)
        direita = [linha for linha in linhas_lidas if linha["lado"] == "right"]
        esquerda = [linha for linha in linhas_lidas if linha["lado"] == "left"]
        assert len(direita) == 2 and len(esquerda) == 1
        # 0 grau: previsao E' a normal do Kinect
        assert abs(direita[0]["err_deg"]) < 1e-6, direita[0]["err_deg"]
        assert (abs(direita[0]["prev_x"]) < 1e-6
                and abs(direita[0]["prev_y"] + 1.0) < 1e-6)
        assert direita[0]["alinhado"] == 1
        # 90 graus: previsao perpendicular a normal
        assert abs(direita[1]["err_deg"] - 90.0) < 1e-6, direita[1]["err_deg"]
        # sem alinhamento: campos vazios (NaN depois da leitura) e alinhado = 0
        assert esquerda[0]["alinhado"] == 0
        assert not np.isfinite(esquerda[0]["err_deg"])
        assert not np.isfinite(esquerda[0]["prev_x"])
        assert not np.isfinite(esquerda[0]["err_raw_deg"])
        assert esquerda[0]["fase_nome"] == "PREPARO"
        assert esquerda[0]["fase_idx"] == 0
        # a normal do Kinect e' a verdade gravada: (0,-1,0)
        assert abs(esquerda[0]["ny"] + 1.0) < 1e-6
    print("[ok] 2) registrar: 20 colunas na ordem do cabecalho, err_deg 0/90 "
          "graus exatos, alinhado 1/0 e recusas (sem IMU/sem normal)")


# ------------------------------------------------------------------- teste 3
def _gerar_diagnostico_sintetico(caminho_csv, erro_por_fase=None,
                                 ruido_deg=1.5, semente=7, por_fase=40):
    """CSV `palm_diag_*` + meta com T/f VERDADEIROS conhecidos (sem hardware).

    O gerador segue o MODELO da tecla O: `v_i = T R_i f` com `T` = rotacao de
    20 graus em Z (o referencial do IMU nao e' o da camera -- e' justamente o
    que a tecla O resolve) e `f` = (0,0,-1) no corpo do sensor. Cada fase tem
    uma rotacao base que, SEM o T, daria a normal pedida:

        PREPARO/KINECT rpy (0, 0, 0)         -> normal (0, 0, -1)  (para o Kinect)
        MESA           rpy (90, 0, 0)        -> normal (0, -1, 0)  (palma p/ baixo)
        REZANDO        rpy (90, 0, -90)      -> normal (-1, 0, 0)  (horizontal)

    `erro_por_fase` injeta uma rotacao EXTRA na normal do Kinect daquela fase
    (a "captura ruim"): e' o erro que o refit tem de denunciar. Um T errado
    CONSTANTE nao serviria para o teste: a Kabsch absorveria (acharia outro T).
    """
    caminho_csv = Path(caminho_csv)
    rng = np.random.default_rng(semente)
    f_verdadeiro = np.array([0.0, 0.0, -1.0])
    t_verdadeiro = rotacao_eixo_angulo([0.0, 0.0, 1.0], 20.0)
    normal_por_fase = {"PREPARO": (0.0, 0.0, -1.0), "MESA": (0.0, -1.0, 0.0),
                       "REZANDO": (-1.0, 0.0, 0.0), "KINECT": (0.0, 0.0, -1.0)}
    linhas = []
    for lado in ("right", "left"):
        for indice_fase, (nome, segundos, _instrucao) in enumerate(
                tool.PALM_DIAG_PHASES):
            # Normal DESEJADA no referencial da camera -> `R f` tem de ser
            # `T^-1 v`, e assim `T R f` devolve exatamente `v`.
            alvo_corpo = t_verdadeiro.T @ np.asarray(normal_por_fase[nome],
                                                     np.float64)
            for amostra_idx in range(por_fase):
                # twist aleatorio (varia a ATITUDE sem mudar a normal) + jitter
                # de 1 grau (a mao "parada" nunca esta' perfeita)
                rotacao = rotacao_de_normal(alvo_corpo,
                                            rng.uniform(-180.0, 180.0))
                rotacao = (rotacao_eixo_angulo(rng.normal(size=3), 1.0)
                           @ rotacao)
                roll, pitch, yaw = rpy_de_rotacao(rotacao)
                # R lida de volta do CSV (rpy): e' SO' isso que o analisador
                # tem, entao o CSV tem de ser coerente com a ida e volta.
                rotacao = rotation_matrix(roll, pitch, yaw)
                normal_verdadeira = t_verdadeiro @ rotacao @ f_verdadeiro
                normal_verdadeira /= np.linalg.norm(normal_verdadeira)
                normal = (normal_verdadeira
                          + rng.normal(0.0, ruido_deg / 57.29578, 3))
                normal /= np.linalg.norm(normal)
                if erro_por_fase and nome in erro_por_fase:
                    eixo, graus = erro_por_fase[nome]
                    normal = rotacao_eixo_angulo(eixo, graus) @ normal
                    normal /= np.linalg.norm(normal)
                previsto = t_verdadeiro @ rotacao @ f_verdadeiro
                previsto /= np.linalg.norm(previsto)
                bruto = rotacao @ f_verdadeiro
                bruto /= np.linalg.norm(bruto)
                t_fase = (amostra_idx + 0.5) * float(segundos) / por_fase
                linhas.append({
                    "t_s": f"{100.0 + indice_fase * 10.0 + t_fase:.3f}",
                    "fase_idx": indice_fase, "fase_nome": nome,
                    "t_fase_s": f"{t_fase:.3f}", "lado": lado,
                    "roll_deg": f"{roll:.3f}", "pitch_deg": f"{pitch:.3f}",
                    "yaw_deg": f"{yaw:.3f}",
                    "nx": f"{normal[0]:.6f}", "ny": f"{normal[1]:.6f}",
                    "nz": f"{normal[2]:.6f}",
                    "prev_x": f"{previsto[0]:.6f}",
                    "prev_y": f"{previsto[1]:.6f}",
                    "prev_z": f"{previsto[2]:.6f}",
                    "err_deg": f"{angle_between(normal, previsto):.3f}",
                    "raw_x": f"{bruto[0]:.6f}", "raw_y": f"{bruto[1]:.6f}",
                    "raw_z": f"{bruto[2]:.6f}",
                    "err_raw_deg": f"{angle_between(normal, bruto):.3f}",
                    "alinhado": 1,
                })
    colunas = tool.cabecalho_palm_diag()
    with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas)
    meta = {
        "pc_time": "2026-10-19 15:00:00",
        "csv": caminho_csv.name,
        "fases": [{"nome": nome, "segundos": float(segundos),
                   "instrucao": instrucao}
                  for nome, segundos, instrucao in tool.PALM_DIAG_PHASES],
        "settle_s": tool.PALM_DIAG_SETTLE_S,
        "limite_erro_graus": tool.PALM_DIAG_MAX_ERR_DEG,
        "lados": {"right": {"alinhamento": {"rotation": t_verdadeiro.tolist(),
                                           "body": f_verdadeiro.tolist(),
                                           "residual_deg": 1.0},
                            "arquivo": "imu_palm_alignment_right.json",
                            "arquivo_existe": True,
                            "arquivo_mtime": "2026-10-19 14:00:00"},
                  "left": {"alinhamento": None,
                           "arquivo": "imu_palm_alignment_left.json",
                           "arquivo_existe": False,
                           "arquivo_mtime": None}},
    }
    Path(analisador.caminho_meta(caminho_csv)).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return caminho_csv


def teste_ponta_a_ponta_sintetico():
    """CSV sintetico: resumo <= 5 graus e refit <= 5; com captura ruim, ambos
    denunciam (mediana > limite e residuo do refit > 10 graus)."""
    with pasta_temporaria():
        caminho = _gerar_diagnostico_sintetico("palm_diag_20261019_150000.csv")
        linhas, meta = analisador.carregar(caminho)
        assert len(linhas) == 2 * 4 * 40, len(linhas)
        assert meta["settle_s"] == tool.PALM_DIAG_SETTLE_S
        resumo = analisador.resumo_por_fase(
            linhas, settle=tool.PALM_DIAG_SETTLE_S)
        for lado in ("right", "left"):
            for fase in FASES_ESPERADAS:
                dados = resumo.get((fase, lado))
                assert dados is not None and dados["n"] > 10, (fase, lado)
                assert dados["mediana"] <= 5.0, (fase, lado, dados["mediana"])
        # As 3 poses programadas fecham: veredito de sucesso (as DUAS copias do
        # criterio -- tool e analisador -- tem de dizer a MESMA coisa).
        brutos = analisador.resumo_por_lado(linhas, "err_raw_deg")
        for lado in ("right", "left"):
            estat_lado = {fase: resumo[(fase, lado)] for fase in FASES_ESPERADAS}
            notas = tool.veredito_palm_diag(estat_lado, brutos[lado])
            assert any("ALINHAMENTO OK" in nota for nota in notas), notas
            assert notas == analisador.veredito_palm_diag(estat_lado,
                                                          brutos[lado]), notas
        # O refit acha o modelo com residuo pequeno (o CSV e' coerente).
        for lado in ("right", "left"):
            resolvido = analisador.refitar(linhas, lado)
            assert resolvido is not None, f"{lado}: refit devolveu None"
            _alinhamento, corpo, residuo = resolvido
            assert residuo <= 5.0, (lado, residuo)
            assert angle_between(corpo, [0.0, 0.0, -1.0]) <= 5.0, corpo
        # Cluster de execucao: mesa ~ para baixo, kinect ~ para o Kinect,
        # rezando ~ horizontal (heuristica de EXECUCAO).
        cluster = analisador.angulo_medio_normal(
            linhas, settle=tool.PALM_DIAG_SETTLE_S)
        for lado in ("right", "left"):
            assert cluster[("MESA", lado)]["media"][1] < -0.8
            assert cluster[("KINECT", lado)]["media"][2] < -0.8
            assert abs(cluster[("REZANDO", lado)]["media"][1]) < 0.5
            assert cluster[("MESA", lado)]["dispersao_deg"] <= 5.0

        # ---------------- agora a CAPTURA ruim (erro que depende da pose) ----
        # MESA girada +30 graus em X e REZANDO -30 graus em Z: as duas normais
        # SAEM do lugar (girar em torno do proprio eixo nao mudaria nada -- a
        # normal da REZANDO e' (-1,0,0) e girar em X a deixaria intacta) e
        # nenhuma rotacao T explica as duas ao mesmo tempo. Um T errado
        # CONSTANTE seria absorvido pelo refit -- e' o que torna o teste
        # significativo.
        caminho_ruim = _gerar_diagnostico_sintetico(
            "palm_diag_20261019_160000.csv",
            erro_por_fase={"MESA": ([1.0, 0.0, 0.0], 30.0),
                           "REZANDO": ([0.0, 0.0, 1.0], -30.0)})
        linhas_ruins, _meta = analisador.carregar(caminho_ruim)
        resumo_ruim = analisador.resumo_por_fase(
            linhas_ruins, settle=tool.PALM_DIAG_SETTLE_S)
        limite = tool.PALM_DIAG_MAX_ERR_DEG
        for lado in ("right", "left"):
            assert resumo_ruim[("MESA", lado)]["mediana"] > limite, \
                (lado, resumo_ruim[("MESA", lado)])
            assert resumo_ruim[("REZANDO", lado)]["mediana"] > limite
            assert resumo_ruim[("KINECT", lado)]["mediana"] <= limite
            resolvido = analisador.refitar(linhas_ruins, lado)
            assert resolvido is not None
            _alinhamento, _corpo, residuo = resolvido
            assert residuo > 10.0, (lado, residuo)
            notas = tool.veredito_palm_diag(
                {fase: resumo_ruim[(fase, lado)] for fase in FASES_ESPERADAS},
                analisador.resumo_por_lado(linhas_ruins, "err_raw_deg")[lado])
            assert any("DECORA" in nota for nota in notas), notas
            # ...e o veredito NAO pode dizer que o alinhamento esta' OK.
            assert not any("ALINHAMENTO OK" in nota for nota in notas), notas
    print("[ok] 3) ponta a ponta sintetico: resumo/refit <= 5 graus com dado bom; "
          "captura ruim -> mediana > limite, residuo do refit > 10 e veredito "
          "'DECORA a pose da tecla O'")


# ------------------------------------------------------------------- teste 4
def teste_analisador_sem_cv2():
    """O analisador offline NAO pode importar cv2/pykinect/ground truth (AST)."""
    caminho = Path("tools") / "analisa_diag_palma.py"
    assert caminho.exists(), caminho
    arvore = ast.parse(caminho.read_text(encoding="utf-8"))
    importados = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            importados.update(alias.name.split(".")[0] for alias in no.names)
        elif isinstance(no, ast.ImportFrom) and no.module:
            importados.add(no.module.split(".")[0])
    proibidos = {"cv2", "pykinect2", "kinect_imu_groundtruth",
                 "kinect_groundtruth_tool", "mediapipe", "camera_uvc"}
    achados = proibidos & importados
    assert not achados, f"o analisador offline importa {sorted(achados)}"
    assert "imu" in importados, importados
    assert "numpy" in importados
    # As funcoes que o teste 3 usa existem e sao publicas (contrato do CLI).
    for nome in ("carregar", "resumo_por_fase", "angulo_medio_normal",
                 "refitar", "veredito_palm_diag", "mais_recente", "main"):
        assert callable(getattr(analisador, nome, None)), nome
    print("[ok] 4) tools/analisa_diag_palma.py sem cv2/pykinect/ground truth "
          "no AST (so' stdlib + numpy + imu)")


# ------------------------------------------------------------------- teste 5
def teste_mais_recente_ignora_abortada():
    """Sem argumento, o CLI escolhe o CSV COM amostras, nao a sessao abortada.

    Bug real medido em 19/10/2026: apertar D duas vezes deixa uma sessao de 2 s
    como a mais recente; o analisador sem argumento abria JUSTAMENTE ela e o
    relatorio saia sem amostras, parecendo que o diagnostico nao gravou nada.
    """
    def escreve(caminho, linhas):
        with open(caminho, "w", encoding="utf-8", newline="") as arquivo:
            arquivo.write("t_s,fase_nome,lado\n")
            for indice in range(linhas):
                arquivo.write(f"{indice},MESA,right\n")

    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        completa = pasta / "palm_diag_20261019_153836.csv"
        abortada = pasta / "palm_diag_20261019_153900.csv"
        escreve(completa, analisador.MIN_LINHAS_CSV_VALIDO + 10)
        escreve(abortada, 3)
        os.utime(completa, (1000.0, 1000.0))
        os.utime(abortada, (2000.0, 2000.0))       # a abortada e' a mais NOVA
        assert analisador.mais_recente(pasta) == completa, \
            "escolheu a sessao abortada so' por ser a mais recente"
        # Nenhuma com amostra: cai para a mais recente (para o operador ver o
        # aviso "sem amostras", e nao "nao achei arquivo").
        escreve(completa, 2)
        # Reescrever ATUALIZA o mtime: sem reajustar, a "completa" voltaria a
        # ser a mais nova e o assert abaixo mediria a ordem errada.
        os.utime(completa, (3000.0, 3000.0))
        os.utime(abortada, (4000.0, 4000.0))
        assert analisador.mais_recente(pasta) == abortada
        assert analisador.mais_recente(pasta / "vazia") is None
    print("[ok] 5) analisador sem argumento ignora sessao abortada de 2 s "
          "(escolhe o CSV com amostras)")


# --------------------------------------------------------------------- main
def main():
    grupos = (teste_fases_settle_e_cancelar,
              teste_registrar_erros_e_colunas,
              teste_ponta_a_ponta_sintetico,
              teste_analisador_sem_cv2,
              teste_mais_recente_ignora_abortada)
    falhas = []
    for funcao in grupos:
        try:
            funcao()
        except AssertionError as exc:
            falhas.append(funcao.__name__)
            print(f"[FALHA] {funcao.__name__}: {exc}")
        except Exception as exc:                      # noqa: BLE001
            falhas.append(funcao.__name__)
            print(f"[ERRO] {funcao.__name__}: {type(exc).__name__}: {exc}")
    print(f"\ndiagnostico da palma: {len(grupos) - len(falhas)}/{len(grupos)} "
          "grupos OK"
          + (f" -> FALHAS: {', '.join(falhas)}" if falhas else ""))
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
