"""Teste da LATERALIDADE das maos nas webcams auxiliares (sem hardware).

Bug relatado na bancada (05/10/2026): com as duas maos no quadro, o rotulo do
MediaPipe concordava com o Kinect numa auxiliar e vinha INVERTIDO na outra. O
groundtruth casa as maos PELO LADO (right -> IMU 4210, left -> IMU 4211, cada uma
com o seu overlay e a sua triangulacao), entao a camera invertida associava a mao
errada: overlay vermelho na mao errada, triangulacao com os pixels da outra mao e
o IMU da luva errada alimentando o lado errado.

Causa: o rotulo do MediaPipe e' uma classificacao da APARENCIA e assume imagem
espelhada (convencao "selfie"). O pipeline espelha TODAS as auxiliares com UM
unico `AUXILIARY_MIRROR_HORIZONTAL = True`, mas com duas webcams de MODELOS
DIFERENTES (nesta maquina: 0 = SIGMA-W780M e 1 = Trust USB Camera) o driver de
uma delas pode ja' entregar o fluxo espelhado -- e entao o nosso flip desespelha,
a premissa do MediaPipe cai e SO' aquela camera sai trocada. Nada no programa
percebia, porque o casamento por lado confiava no rotulo.

Conserto (o que este teste tranca): `ArbitroDeLateralidade` decide o lado pela
GEOMETRIA -- projeta a PALMA de cada deteccao auxiliar no RGB do Kinect com a
homografia no-a-no da mao daquela camera e pergunta qual mao do Kinect esta' ali
(o Kinect tem o punho em 3D) -- e APRENDE a inversao por camera, gravando
`aux_handedness_flip.json` para os programas que so' tem o rotulo (protocolo,
sand, demo).

Casos:
 1. ordem canonica da lista (direita, esquerda, sem rotulo) -- faz "mao 1" e
    "mao 2" significarem a MESMA coisa em todas as cameras;
 2. `projeta_mao_aux_no_kinect`: normalizado da auxiliar -> px do Kinect (no' 9);
 3. O BUG demonstrado: sem arbitro, o casamento por ROTULO pega a mao errada na
    camera invertida (medido em px);
 4. O CONSERTO: o arbitro corrige a camera invertida pela geometria e deixa a
    outra intacta;
 5. aprendizado: com consenso a inversao e' GRAVADA no json (e o proximo frame
    ja' sai certo); confirmacao nao regrava; inversao velha e' DESFEITA
    (auto-cura);
 6. Kinect cego (o caso em que o par de auxiliares substitui o Kinect): cai no
    rotulo, ja' corrigido pelo que foi aprendido, e o relatorio diz que a fonte
    foi o rotulo -- nunca finge que mediu;
 7. maos ambiguas (duas atribuicoes quase equivalentes): NAO decide, mantem o
    rotulo -- melhor nao decidir do que trocar as maos por chute;
 8. degradacoes: sem homografia da mao, sem mao no Kinect, entrada vazia e
    deteccao sem rotulo -> nunca levanta e nunca troca lado por conta propria;
 9. o `flip` por camera na deteccao (o rotulo do MediaPipe reescrito) e a ORDEM
    CANONICA da lista que ele entrega;
10. auxiliar vendo DUAS maos e o Kinect UMA: a atribuicao inteira e' descartada
    (nunca dois lados iguais na mesma camera) e o frame NAO vota;
11. a VIA da projecao (item 39): a homografia 2D do esqueleto -- a que o modulo
    documenta como imprecisa -- deixa a geometria INDECISA a ~300 px (o log real
    da bancada) e o motivo passa a NOMEAR a via; com o stereo 3D daquela camera a
    decisao sai com erro < 1 px, o relatorio diz `via stereo 3D` e a profundidade
    pedida e' a do punho CANDIDATO do Kinect (nao um chute);
12. camera SEM homografia da mao (Trust USB nesta bancada) medida pelo stereo;
13. o log nao inunda: o estado do veredito ignora os NUMEROS do motivo;
14. o que ENSINA tem de ser conferivel INTEIRO: o caso bimanual (as DUAS maos vistas
    pelas duas cameras, todas atribuidas) com erro <= `ARBITRO_MAX_DIST_VOTO_PX`.
    Contagens diferentes, uma mao de cada lado ou erro grande corrigem o frame mas
    NAO alimentam o aprendizado (nem o arquivo).

    .venv\\Scripts\\python.exe test_lateralidade_maos.py
"""
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))        # o tool e o probe vivem em tools/

import kinect_imu_groundtruth as gt                              # noqa: E402
import kinect_groundtruth_tool as tool                           # noqa: E402

LARGURA, ALTURA = 1920, 1080
TAMANHO_KINECT = (LARGURA, ALTURA)
#: Arquivo de trabalho: o teste NUNCA mexe no aux_handedness_flip.json real.
ARQUIVO_TESTE = Path(__file__).with_name("tmp_aux_handedness_test.json")
gt.AUX_HANDEDNESS_FLIP_FILE = ARQUIVO_TESTE
ARQUIVO_TESTE.unlink(missing_ok=True)
gt._AUX_HANDEDNESS_FLIP.clear()


class LM:
    """Landmark minimo do MediaPipe (x e y normalizados)."""

    def __init__(self, x, y):
        self.x, self.y = x, y


def mao(x, y):
    """21 landmarks com a PALMA (no' 9, o mesmo do resto do modulo) em (x, y)."""
    return [LM(x, y) for _ in range(21)]


def palma_px(landmarks):
    """Palma em px do Kinect a partir de landmarks normalizados."""
    return np.array([landmarks[9].x * LARGURA, landmarks[9].y * ALTURA])


class Alinhamento:
    """`CameraAlignment` minimo: o arbitro so' usa `aux_models[i]`, e so' a
    homografia da mao (mais o residual por no')."""

    def __init__(self, indice_para_homografia):
        self.aux_models = {
            int(indice): {"hand_homography": np.asarray(homografia, np.float64),
                          "hand_residuals": np.zeros((21, 2), np.float64)}
            for indice, homografia in indice_para_homografia.items()
        }


class VistaFalsa:
    """`AuxCameraView` minimo: stereo PERFEITO, sem passar pela homografia 2D.

    A auxiliar enxerga a MESMA cena normalizada do Kinect (a regra do teste: as
    duplas de landmarks sao as mesmas nas duas cameras), com intrinsicas que so'
    reescalam -- entao a projecao stereo devolve a cena exata. `profundidades`
    guarda as profundidades PEDIDAS a cada chamada: e' o que prova de onde veio o
    Z (da mao candidata do Kinect, nao de um chute).

    `usable=False` modela a camera SEM stereo aceito: ai o arbitro so' tem a
    homografia 2D do esqueleto (a projecao imprecisa) e deve DIZER isso.
    """

    def __init__(self, tamanho, usable=True):
        self.tamanho = tamanho
        self.usable = bool(usable)
        self.profundidades = []

    def pixels(self, landmarks):
        return np.array([(lm.x * self.tamanho[0], lm.y * self.tamanho[1])
                         for lm in landmarks], np.float64)

    def to_kinect_stereo(self, points, kinect_depth_m):
        self.profundidades.append(
            [float(valor)
             for valor in np.asarray(kinect_depth_m, np.float64).reshape(-1)])
        pontos = np.asarray(points, np.float64).reshape(-1, 2).copy()
        pontos[:, 0] = pontos[:, 0] / self.tamanho[0] * LARGURA
        pontos[:, 1] = pontos[:, 1] / self.tamanho[1] * ALTURA
        return pontos


class AlinhamentoComVista:
    """`CameraAlignment` minimo com as DUAS vias: homografia 2D e `aux_view`."""

    def __init__(self, homografias, vistas, trim=None):
        self.aux_models = {
            int(indice): {
                "hand_homography": (None if homografia is None
                                    else np.asarray(homografia, np.float64)),
                "hand_residuals": np.zeros((21, 2), np.float64)}
            for indice, homografia in homografias.items()}
        self.vistas = {int(indice): vista for indice, vista in vistas.items()}
        self.overlay_trim = (np.zeros(2, np.float64) if trim is None
                             else np.asarray(trim, np.float64))

    def aux_view(self, indice):
        return self.vistas.get(int(indice))


class Resultado:
    """Resultado minimo do MediaPipe (so' o que `_handedness_of` le')."""

    def __init__(self, rotulos):
        self.handedness = [
            [type("Categoria", (), {"category_name": rotulo})()]
            for rotulo in rotulos
        ]
        self.hand_landmarks = [mao(0.5, 0.5) for _ in rotulos]


# =============================================================================
# 1) Ordem canonica
# =============================================================================
entradas = [{"side": "left"}, {"side": "right"}, {"side": ""},
            {"side": "left"}, {"side": "right"}]
ordenadas = gt.ordena_maos_por_lado(entradas)
assert [e["side"] for e in ordenadas] == ["right", "right", "left", "left", ""], \
    [e["side"] for e in ordenadas]
# Estavel: quem tem o mesmo lado mantem a ordem relativa (o mais confiante do
# MediaPipe continua primeiro DENTRO do lado).
assert ordenadas[0] is entradas[1] and ordenadas[2] is entradas[0]
assert gt.ordena_maos_por_lado([]) == [] and gt.ordena_maos_por_lado(None) == []
assert gt.ORDEM_DOS_LADOS == ("right", "left")
print("1) OK: ordem canonica (direita, esquerda, sem rotulo) estavel")

# =============================================================================
# 2) Projecao da palma auxiliar -> Kinect
# =============================================================================
palma = gt.projeta_mao_aux_no_kinect(np.eye(3), None, mao(0.25, 0.50),
                                     TAMANHO_KINECT)
assert palma is not None and np.allclose(palma, [480.0, 540.0]), palma
# Homografia com deslocamento + residual aplicados NA ORDEM do no'.
H = np.array([[2.0, 0.0, 0.1], [0.0, 2.0, 0.05], [0.0, 0.0, 1.0]])
residual = np.zeros((21, 2))
residual[9] = [0.01, -0.02]
palma = gt.projeta_mao_aux_no_kinect(H, residual, mao(0.20, 0.25), TAMANHO_KINECT)
assert np.allclose(palma, [(0.20 * 2 + 0.1 + 0.01) * LARGURA,
                           (0.25 * 2 + 0.05 - 0.02) * ALTURA]), palma
# Sem homografia / sem landmarks / landmarks de menos -> None (nunca excecao).
assert gt.projeta_mao_aux_no_kinect(None, None, mao(0.2, 0.5),
                                    TAMANHO_KINECT) is None
assert gt.projeta_mao_aux_no_kinect(np.eye(3), None, [], TAMANHO_KINECT) is None
assert gt.projeta_mao_aux_no_kinect(np.eye(3), None, mao(0.2, 0.5)[:5],
                                    TAMANHO_KINECT) is None
assert gt.PALM_LANDMARK_INDEX == 9
print("2) OK: projeta a palma (no' 9) da auxiliar no RGB do Kinect, com residual")

# =============================================================================
# 3) O BUG: casar por ROTULO pega a mao errada na camera invertida
# =============================================================================
# Cena: a mao DIREITA do usuario aparece a' esquerda da imagem do Kinect (x
# menor) e a ESQUERDA a' direita -- o normal de quem fica de frente. A auxiliar
# ve a MESMA cena (homografia identidade, para o teste medir em px).
X_DIREITA, X_ESQUERDA = 0.40, 0.75
maos_kinect = [
    {"side": "right", "landmarks": mao(X_DIREITA, 0.5), "position": None},
    {"side": "left", "landmarks": mao(X_ESQUERDA, 0.5), "position": None},
]
alinhamento = Alinhamento({0: np.eye(3), 2: np.eye(3)})
separacao_px = (X_ESQUERDA - X_DIREITA) * LARGURA       # 672 px entre as maos

# Camera 0: rotulo CERTO. Camera 2: rotulo TROCADO (o relato da bancada).
aux_por_camera = {
    0: [{"side": "right", "landmarks": mao(X_DIREITA, 0.5)},
        {"side": "left", "landmarks": mao(X_ESQUERDA, 0.5)}],
    2: [{"side": "left", "landmarks": mao(X_DIREITA, 0.5)},
        {"side": "right", "landmarks": mao(X_ESQUERDA, 0.5)}],
}


def pareamento_por_rotulo(entradas_camera):
    """O casamento que o tool/protocolo faziam: por ROTULO do MediaPipe."""
    return {lado: next((e for e in entradas_camera if e["side"] == lado), None)
            for lado in gt.ORDEM_DOS_LADOS}


erros_px = {}
for indice, entradas_camera in aux_por_camera.items():
    pares = pareamento_por_rotulo(entradas_camera)
    erro = max(
        np.linalg.norm(palma_px(pares[lado]["landmarks"])
                       - palma_px(maos_kinect[posicao]["landmarks"]))
        for posicao, lado in enumerate(gt.ORDEM_DOS_LADOS)
        if pares[lado] is not None
    )
    erros_px[indice] = float(erro)
print(f"3) sem arbitro, erro do casamento por rotulo: camera 0 = "
      f"{erros_px[0]:.0f} px | camera 2 = {erros_px[2]:.0f} px "
      f"(separacao entre as maos = {separacao_px:.0f} px)")
assert erros_px[0] < 1.0, erros_px          # rotulo certo: casou certo
assert erros_px[2] > 0.9 * separacao_px, erros_px   # invertida: casou a OUTRA mao
print("   OK: a camera 2 (rotulo trocado) associava a mao errada -- o bug real")

# =============================================================================
# 4) O CONSERTO: o arbitro decide pela GEOMETRIA
# =============================================================================
arbitro = gt.ArbitroDeLateralidade(verbose=False)
corrigido, relatorio = arbitro.arbitrar(alinhamento, maos_kinect,
                                        aux_por_camera, TAMANHO_KINECT)
for indice in (0, 2):
    assert relatorio[indice]["fonte"] == "geometria", relatorio[indice]
    assert relatorio[indice]["maos"] == 2, relatorio[indice]
    assert relatorio[indice]["dist_px"] < 1.0, relatorio[indice]
    assert [e["side"] for e in corrigido[indice]] == ["right", "left"], \
        (indice, [e["side"] for e in corrigido[indice]])
assert relatorio[2]["inverteu"] is True
assert relatorio[0]["inverteu"] is False
# O rotulo original fica registrado: da' para auditar o que o MediaPipe disse.
assert [e["side_rotulo"] for e in corrigido[2]] == ["left", "right"]
assert [e["lado_fonte"] for e in corrigido[2]] == ["geometria", "geometria"]
# As listas do chamador NAO sao alteradas (o arbitro trabalha numa copia).
assert [e["side"] for e in aux_por_camera[2]] == ["left", "right"]
# E o casamento por lado passa a pegar a mao CERTA nas DUAS cameras.
for indice in (0, 2):
    pares = pareamento_por_rotulo(corrigido[indice])
    for posicao, lado in enumerate(gt.ORDEM_DOS_LADOS):
        erro = np.linalg.norm(palma_px(pares[lado]["landmarks"])
                              - palma_px(maos_kinect[posicao]["landmarks"]))
        assert erro < 1.0, (indice, lado, erro)
print("4) OK: o arbitro corrigiu a camera 2 pela geometria e deixou a 0 intacta")

# =============================================================================
# 5) Aprendizado: grava no json, confirma sem regravar, e se auto-cura
# =============================================================================
import json                                                     # noqa: E402

assert gt.aux_handedness_flip(2) is False        # nada aprendido ainda
aprende = gt.ArbitroDeLateralidade(min_votos=3, verbose=True)
for _ in range(3):
    aprende.arbitrar(alinhamento, maos_kinect, aux_por_camera, TAMANHO_KINECT)
assert gt.aux_handedness_flip(2) is True, "a inversao tinha de ser gravada"
assert ARQUIVO_TESTE.exists(), "o json da inversao tinha de existir"
assert gt.carrega_aux_handedness_flip() == {2: True}, \
    gt.carrega_aux_handedness_flip()          # le de volta do DISCO
conteudo = json.loads(ARQUIVO_TESTE.read_text(encoding="utf-8"))
assert conteudo["flip"] == {"2": True}, conteudo
assert "trocado" in conteudo["motivo"], conteudo    # o arquivo explica-se
assert gt.aux_handedness_flip(0) is False           # a camera 0 nao foi tocada

# Com a inversao aprendida, o pipeline passa a entregar o rotulo JA' corrigido:
# a geometria CONCORDA e o programa so' confirma (sem reescrever o arquivo).
aux_ja_corrigido = {
    2: [{"side": "right", "landmarks": mao(X_DIREITA, 0.5)},
        {"side": "left", "landmarks": mao(X_ESQUERDA, 0.5)}]}
antes = ARQUIVO_TESTE.read_text(encoding="utf-8")
confirma = gt.ArbitroDeLateralidade(min_votos=3, verbose=False)
for _ in range(3):
    _, relatorio_confirma = confirma.arbitrar(
        alinhamento, maos_kinect, aux_ja_corrigido, TAMANHO_KINECT)
assert relatorio_confirma[2]["inverteu"] is False
assert relatorio_confirma[2]["fonte"] == "geometria"
assert gt.aux_handedness_flip(2) is True
assert ARQUIVO_TESTE.read_text(encoding="utf-8") == antes, \
    "uma confirmacao nao pode reescrever o arquivo"
assert 2 in confirma.confirmados

# Auto-cura: a webcam mudou de porta USB (ou o driver voltou ao normal) e a
# inversao aprendida ficou VELHA -- a geometria percebe e desfaz.
aux_velho = {
    2: [{"side": "left", "landmarks": mao(X_DIREITA, 0.5)},
        {"side": "right", "landmarks": mao(X_ESQUERDA, 0.5)}]}
cura = gt.ArbitroDeLateralidade(min_votos=3, verbose=True)
for _ in range(3):
    cura.arbitrar(alinhamento, maos_kinect, aux_velho, TAMANHO_KINECT)
assert gt.aux_handedness_flip(2) is False, "a inversao velha tinha de ser desfeita"
assert gt.carrega_aux_handedness_flip() == {2: False}
print("5) OK: inversao gravada no json (com motivo), confirmacao sem regravar e "
      "auto-cura de inversao velha")

# =============================================================================
# 6) Kinect cego: cai no rotulo (ja' corrigido pelo que foi aprendido)
# =============================================================================
# Este e' o caso que importa para o objetivo do projeto: o PAR de auxiliares
# substituindo o Kinect. Sem mao no Kinect nao ha' verdade geometrica -- o
# programa tem de dizer que esta' usando o rotulo, nunca fingir que mediu.
gt.set_aux_handedness_flip(2, True, motivo="teste: aprendido na camera 2",
                           silencioso=True)
cego = gt.ArbitroDeLateralidade(verbose=False)
corrigido_cego, relatorio_cego = cego.arbitrar(
    alinhamento, [], aux_ja_corrigido, TAMANHO_KINECT)
assert relatorio_cego[2]["fonte"].startswith("rotulo"), relatorio_cego[2]
assert relatorio_cego[2]["invertido"] is True       # a inversao aprendida valendo
assert [e["side"] for e in corrigido_cego[2]] == ["right", "left"]
assert all(e["lado_fonte"] == "rotulo" for e in corrigido_cego[2])
assert "rotulo" in cego.resumo()[0]
print("6) OK: Kinect cego -> rotulo (com a inversao aprendida) e a fonte declarada")

# =============================================================================
# 7) Maos ambiguas: NAO decide (melhor nao decidir do que chutar a troca)
# =============================================================================
# Punhos a 40 px (maos quase sobrepostas no quadro) e deteccoes que projetam
# QUASE no mesmo ponto: as duas atribuicoes ficam equivalentes -> sem margem.
maos_juntas = [
    {"side": "right", "landmarks": mao(0.6000, 0.5), "position": None},
    {"side": "left", "landmarks": mao(0.6208, 0.5), "position": None},
]
aux_juntas = {2: [{"side": "right", "landmarks": mao(0.6104, 0.5)},
                  {"side": "left", "landmarks": mao(0.6104, 0.5)}]}
ambiguo = gt.ArbitroDeLateralidade(verbose=False)
corrigido_juntas, relatorio_juntas = ambiguo.arbitrar(
    alinhamento, maos_juntas, aux_juntas, TAMANHO_KINECT)
assert relatorio_juntas[2]["fonte"] == "rotulo (geometria indecisa)", \
    relatorio_juntas[2]
assert relatorio_juntas[2]["maos"] == 0
assert [e["side"] for e in corrigido_juntas[2]] == ["right", "left"]   # chute: nao
assert all(e["lado_fonte"] == "rotulo" for e in corrigido_juntas[2])
# Sem decisao nao pode haver voto NENHUM: nem a entrada do contador e' criada
# (a partir do item 39 o `_votar` so' e' chamado para frames conferiveis
# inteiros e com erro apertado -- ver o caso 14).
assert ambiguo.votos.get(2, {"oposto": 0, "igual": 0}) == {"oposto": 0,
                                                           "igual": 0}
assert "indecisa" in ambiguo.resumo()[0]
print("7) OK: maos ambiguas -> indeciso (mantem o rotulo, sem votar)")

# =============================================================================
# 8) Degradacoes: nunca levanta excecao, nunca troca lado sozinho
# =============================================================================
falho = gt.ArbitroDeLateralidade(verbose=False)
# (a) camera sem calibracao de mao registrada
corrigido_a, rel_a = falho.arbitrar(Alinhamento({}), maos_kinect,
                                    aux_por_camera, TAMANHO_KINECT)
assert rel_a[2]["fonte"].startswith("rotulo"), rel_a[2]
# O LADO nao e' tocado (continua o rotulo) -- o que muda e' apenas a ORDEM da
# lista, que sai canonica em todas as cameras (right antes de left).
assert [e["side"] for e in corrigido_a[2]] == ["right", "left"]
assert [e["side_rotulo"] for e in corrigido_a[2]] == ["right", "left"]
assert all(e["lado_fonte"] == "rotulo" for e in corrigido_a[2])
# (b) Kinect sem mao
corrigido_b, rel_b = falho.arbitrar(alinhamento, [], aux_por_camera,
                                    TAMANHO_KINECT)
assert all(info["fonte"].startswith("rotulo") for info in rel_b.values()), rel_b
assert [e["side"] for e in corrigido_b[2]] == ["right", "left"]
assert all(e["lado_fonte"] == "rotulo" for e in corrigido_b[2])
# (c) sem deteccao / dicionario vazio
assert falho.arbitrar(alinhamento, maos_kinect, {}, TAMANHO_KINECT) == ({}, {})
corrigido_c, rel_c = falho.arbitrar(alinhamento, maos_kinect, {2: []},
                                    TAMANHO_KINECT)
assert rel_c[2]["fonte"] == "sem mao" and corrigido_c[2] == []
# (d) tamanho do frame desconhecido
corrigido_d, rel_d = falho.arbitrar(alinhamento, maos_kinect, aux_por_camera,
                                    None)
assert all(info["fonte"].startswith("rotulo") for info in rel_d.values())
assert [e["side"] for e in corrigido_d[2]] == ["right", "left"]
# (e) deteccao SEM rotulo: a geometria ainda da' o lado, mas NAO vota (sem rotulo
#     nao existe evidencia de inversao para aprender)
sem_rotulo = {2: [{"side": "", "landmarks": mao(X_DIREITA, 0.5)},
                  {"side": "", "landmarks": mao(X_ESQUERDA, 0.5)}]}
corrigido_e, rel_e = falho.arbitrar(alinhamento, maos_kinect, sem_rotulo,
                                    TAMANHO_KINECT)
assert rel_e[2]["fonte"] == "geometria" and rel_e[2]["maos"] == 2, rel_e[2]
assert [e["side"] for e in corrigido_e[2]] == ["right", "left"]
assert all(e["side_rotulo"] == "" for e in corrigido_e[2])
assert falho.votos[2] == {"oposto": 0, "igual": 0}
# (f) so' uma das cameras tem homografia da mao
corrigido_f, rel_f = falho.arbitrar(Alinhamento({0: np.eye(3)}), maos_kinect,
                                    aux_por_camera, TAMANHO_KINECT)
assert rel_f[2]["fonte"].startswith("rotulo"), rel_f[2]
assert rel_f[0]["fonte"] == "geometria", rel_f[0]
assert [e["side"] for e in corrigido_f[0]] == ["right", "left"]
print("8) OK: degradacoes (sem homografia, sem Kinect, vazio, sem rotulo, "
      "metade calibrada) degradam para o rotulo sem excecao")

# =============================================================================
# 9) A matematica do rotulo que o `flip` por camera aplica na deteccao
# =============================================================================
resultado = Resultado(["Left", "Right"])        # como o MediaPipe devolve
assert gt.KinectHandTracker._handedness_of(resultado, 0, False) == "left"
assert gt.KinectHandTracker._handedness_of(resultado, 0, True) == "right"
assert gt.KinectHandTracker._handedness_of(resultado, 1, False) == "right"
assert gt.KinectHandTracker._handedness_of(resultado, 1, True) == "left"
# Sem rotulo confiavel / indice inexistente -> "" (nunca inventa mao).
assert gt.KinectHandTracker._handedness_of(Resultado([]), 0, True) == ""
assert gt.KinectHandTracker._handedness_of(Resultado(["Unknown"]), 0,
                                          True) == ""
# E a deteccao devolve na ORDEM CANONICA (right, left) em qualquer camera.
assert [e["side"] for e in gt.ordena_maos_por_lado(
    [{"side": gt.KinectHandTracker._handedness_of(resultado, i, False)}
     for i in range(2)])] == ["right", "left"]
print("9) OK: flip por camera na deteccao + ordem canonica da lista")

# =============================================================================
# 10) Auxiliar ve DUAS maos e o Kinect ve UMA: nada de lado repetido
# =============================================================================
# Medido ao vivo em 05/10/2026 (`tools\probe_lateralidade.py`): a geometria
# corrigia a deteccao mais proxima para o lado do Kinect e a OUTRA deteccao ficava
# com o rotulo -- as duas terminavam no MESMO lado. Quem casa por lado guarda UM
# landmarks por lado, entao uma deteccao seria descartada em silencio: melhor
# descartar a ATRIBUICAO inteira (e nao votar nada).
maos_uma = [{"side": "left", "landmarks": mao(X_ESQUERDA, 0.5), "position": None}]
aux_duas = {2: [{"side": "right", "landmarks": mao(X_ESQUERDA, 0.5)},
                {"side": "left", "landmarks": mao(X_DIREITA, 0.5)}]}
colisao = gt.ArbitroDeLateralidade(verbose=False)
corrigido_colisao, rel_colisao = colisao.arbitrar(
    alinhamento, maos_uma, aux_duas, TAMANHO_KINECT)
assert rel_colisao[2]["fonte"] == "rotulo (geometria indecisa)", rel_colisao[2]
assert rel_colisao[2]["maos"] == 0
assert "duas deteccoes no mesmo lado" in rel_colisao[2]["detalhe"], \
    rel_colisao[2]
# Voltou para os rotulos (ordem canonica) e NAO votou (nao ha' evidencia: a
# atribuicao foi descartada, entao nem a entrada do contador e' criada).
assert [e["side"] for e in corrigido_colisao[2]] == ["right", "left"]
assert all(e["lado_fonte"] == "rotulo" for e in corrigido_colisao[2])
assert colisao.votos.get(2, {"oposto": 0, "igual": 0}) == {"oposto": 0,
                                                          "igual": 0}
print("10) OK: lado repetido na mesma camera -> atribuicao descartada (sem votar)")

# =============================================================================
# 11) A VIA da projecao: o radio do arbitro tem de ser o do overlay
# =============================================================================
# Medido ao vivo em 05/10/2026 (bancada, SIGMA-W780M como auxiliar 0): com a
# homografia 2D do esqueleto a projecao da palma caia a ~300 px da mao do Kinect
# em TODOS os frames -> "rotulo (geometria indecisa) -- projecao a 300 px" e a
# lateralidade voltava a ser o rotulo, com o terminal inundado. Mas a projecao
# que o OVERLAY usa (a que desenha o esqueleto vermelho) e' o STEREO 3D daquele
# par de cameras: e' essa que o arbitro tem de medir. A homografia 2D mapeia um
# PLANO e a mao sai dele (o proprio modulo a documenta como imprecisa).
DESLOCAMENTO_2D_PX = 300.0
HOMOGRAFIA_2D_RUIM = np.array([
    [1.0, 0.0, DESLOCAMENTO_2D_PX / LARGURA], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
maos_kinect_3d = [
    {"side": "right", "landmarks": mao(X_DIREITA, 0.5),
     "position": np.array([0.10, 0.20, 0.90])},
    {"side": "left", "landmarks": mao(X_ESQUERDA, 0.5),
     "position": np.array([0.30, 0.25, 0.75])},
]
aux_com_rotulo_certo = {0: [{"side": "right", "landmarks": mao(X_DIREITA, 0.5)},
                            {"side": "left", "landmarks": mao(X_ESQUERDA, 0.5)}]}

# (a) SO' a homografia 2D (sem stereo utilizavel): indeciso a ~300 px -- o log.
vista_sem_stereo = VistaFalsa((1280, 720), usable=False)
so_homografia = AlinhamentoComVista({0: HOMOGRAFIA_2D_RUIM},
                                    {0: vista_sem_stereo})
corrigido_2d, rel_2d = gt.ArbitroDeLateralidade(verbose=False).arbitrar(
    so_homografia, maos_kinect_3d, aux_com_rotulo_certo, TAMANHO_KINECT)
assert rel_2d[0]["fonte"] == "rotulo (geometria indecisa)", rel_2d[0]
assert rel_2d[0]["via"] == gt.VIA_HOMOGRAFIA, rel_2d[0]
assert abs(rel_2d[0]["dist_px"] - DESLOCAMENTO_2D_PX) < 2.0, rel_2d[0]
# O motivo diz QUAL via produziu o numero e o que fazer: sem isso o usuario vai
# recalibrar o esqueleto quando o esqueleto nem e' o que projeta o overlay.
assert "projecao a" in rel_2d[0]["detalhe"], rel_2d[0]
assert gt.VIA_HOMOGRAFIA in rel_2d[0]["detalhe"], rel_2d[0]
assert "STEREO" in rel_2d[0]["detalhe"], rel_2d[0]
# E a lista sai intacta (rotulo preservado, ordem canonica, nada trocado).
assert [e["side"] for e in corrigido_2d[0]] == ["right", "left"]
assert [e["lado_fonte"] for e in corrigido_2d[0]] == ["rotulo", "rotulo"]
print(f"11a) OK: so' homografia 2D -> indeciso a "
      f"{rel_2d[0]['dist_px']:.0f} px, motivo aponta a via e o stereo")

# (b) Com o STEREO 3D daquela camera (a mesma homografia ruim no lugar): decide,
#     com erro de ~0 px, e o relatorio diz a via.
vista_stereo = VistaFalsa((1280, 720))
com_stereo = AlinhamentoComVista({0: HOMOGRAFIA_2D_RUIM}, {0: vista_stereo})
corrigido_st, rel_st = gt.ArbitroDeLateralidade(verbose=False).arbitrar(
    com_stereo, maos_kinect_3d, aux_com_rotulo_certo, TAMANHO_KINECT)
assert rel_st[0]["fonte"] == "geometria", rel_st[0]
assert rel_st[0]["via"] == gt.VIA_STEREO, rel_st[0]
assert rel_st[0]["maos"] == 2, rel_st[0]
assert rel_st[0]["dist_px"] < 1.0, rel_st[0]
assert [e["side"] for e in corrigido_st[0]] == ["right", "left"]
assert all(e["lado_fonte"] == "geometria" for e in corrigido_st[0])
# A profundidade PEDIDA ao stereo e' a da mao CANDIDATA do Kinect (0,90 m e
# 0,75 m vieram de `position`): e' o que faz a projecao ser do mesmo ponto que o
# overlay desenha -- e o que faz um casamento errado acumular dois erros.
profundidades = [valor for chamada in vista_stereo.profundidades
                 for valor in chamada]
assert profundidades, vista_stereo.profundidades
assert all(any(abs(valor - z) < 1e-9 for z in (0.90, 0.75))
           for valor in profundidades), profundidades
assert not any(abs(valor - 1.0) < 1e-9 for valor in profundidades), profundidades
# O rotulo TROCADO tambem e' corrigido por essa via (e o arbitro registra).
aux_com_rotulo_trocado = {0: [
    {"side": "left", "landmarks": mao(X_DIREITA, 0.5)},
    {"side": "right", "landmarks": mao(X_ESQUERDA, 0.5)}]}
_, rel_trocado = gt.ArbitroDeLateralidade(verbose=False).arbitrar(
    com_stereo, maos_kinect_3d, aux_com_rotulo_trocado, TAMANHO_KINECT)
assert rel_trocado[0]["fonte"] == "geometria", rel_trocado[0]
assert rel_trocado[0]["inverteu"] is True, rel_trocado[0]
print("11b) OK: stereo 3D decide (erro ~0 px) mesmo com a homografia 2D ruim, "
      "e a profundidade vem da mao candidata do Kinect")


# =============================================================================
# 12) Camera SEM a homografia da mao (Trust USB, nesta bancada) e' MEDIDA pelo
#     stereo -- antes ela ficava condenada ao rotulo
# =============================================================================
# `hand_landmark_calibration_aux1.npz` NAO existe nesta maquina: era exatamente o
# caso da camera 1, que o arbitro despachava para o rotulo ("sem homografia da
# mao desta camera") mesmo tendo `stereo_calibration_aux1.npz` aceito. Com a via
# do stereo ela passa a ser medida como qualquer outra -- e o `--aux-flip` da
# linha de comando deixa de ser a unica saida para ela.
vista_aux1 = VistaFalsa((640, 480))
so_stereo_aux1 = AlinhamentoComVista({1: None}, {1: vista_aux1})
aux_aux1 = {1: [{"side": "right", "landmarks": mao(X_DIREITA, 0.5)},
                {"side": "left", "landmarks": mao(X_ESQUERDA, 0.5)}]}
corrigido_aux1, rel_aux1 = gt.ArbitroDeLateralidade(verbose=False).arbitrar(
    so_stereo_aux1, maos_kinect_3d, aux_aux1, TAMANHO_KINECT)
assert rel_aux1[1]["fonte"] == "geometria", rel_aux1[1]
assert rel_aux1[1]["via"] == gt.VIA_STEREO, rel_aux1[1]
assert rel_aux1[1]["maos"] == 2, rel_aux1[1]
assert rel_aux1[1]["dist_px"] < 1.0, rel_aux1[1]
assert [e["side"] for e in corrigido_aux1[1]] == ["right", "left"]
# Sem homografia E sem stereo aceito: cai no rotulo, e o motivo diz o CAMINHO
# (antes so' dizia "sem homografia da mao", que manda calibrar o esqueleto quando
# quem projeta o esqueleto no overlay e' o stereo).
sem_nada = AlinhamentoComVista({1: None},
                               {1: VistaFalsa((640, 480), usable=False)})
_, rel_nada = gt.ArbitroDeLateralidade(verbose=False).arbitrar(
    sem_nada, maos_kinect_3d,
    {1: [{"side": "left", "landmarks": mao(X_DIREITA, 0.5)}]}, TAMANHO_KINECT)
assert rel_nada[1]["fonte"] == "rotulo (sem calibracao da mao desta camera)", \
    rel_nada[1]
assert "H" in rel_nada[1]["detalhe"], rel_nada[1]
assert rel_nada[1]["detalhe"].strip(), rel_nada[1]
print("12) OK: camera sem homografia da mao e' medida pelo stereo (e sem stereo "
      "o motivo diz o que calibrar)")

# =============================================================================
# 13) O log NAO inunda: o estado do veredito ignora os numeros do motivo
# =============================================================================
# O log do usuario (05/10/2026) tinha centenas de linhas iguais: a projecao ia de
# 297 a 302 px a cada frame e o numero entrava na CHAVE de estado, entao o tool
# achava que tudo tinha mudado e reimprimia as linhas das DUAS cameras. Agora a
# chave guarda a NATUREZA do motivo, e ha' um intervalo minimo entre impressoes.
assert tool._sem_numeros("projecao a 302 px (limite 120 px)") == \
    "projecao a ### px (limite ### px)"
assert tool._sem_numeros(None) == ""
assert tool._sem_numeros("") == ""
assert tool._sem_numeros("margem 0 px") == "margem # px"


def estado(detalhe, fonte="rotulo (geometria indecisa)", inverteu=False):
    """Chave de estado do relatorio de UMA camera."""
    return tool.estado_lateralidade(
        {0: {"fonte": fonte, "detalhe": detalhe, "inverteu": inverteu}})


assert estado("projecao a 302 px (limite 120 px)") == \
    estado("projecao a 297 px (limite 120 px)")        # mesmo motivo: silencio
assert estado("projecao a 302 px (limite 120 px)") != \
    estado("margem 12 px entre as duas atribuicoes")   # outro motivo: reimprime
assert estado("projecao a 302 px", fonte="geometria", inverteu=True) != \
    estado("projecao a 302 px", fonte="geometria", inverteu=False)
assert estado("x", fonte="sem mao") != estado("", fonte="sem mao")
assert tool.INTERVALO_LOG_LATERALIDADE_S > 0.0
print("13) OK: estado do log ignora os numeros (o MESMO motivo nao reimprime)")

# =============================================================================
# 14) Quem ENSINA tem de poder ser conferido INTEIRO (o aprendizado nao pega chute)
# =============================================================================
# Medido ao vivo em 05/10/2026 com o probe: a coleta tinha frames com UMA mao
# visivel por vez e votos de sobra. A correcao do frame pode usar esses frames
# (limite de 120 px + margem), mas GRAVAR `aux_handedness_flip.json` a partir
# deles e' outra coisa: com a auxiliar vendo DUAS maos e o Kinect UMA, o que sobra
# no rotulo nao foi conferido por ninguem; e um erro de 100 px ainda "decide" o
# frame, mas nao prova que o ROTULO esta' trocado.
# (a) contagens diferentes (aux 2 x Kinect 1): corrige o frame, NAO vota.
uma_mao_kinect = [{"side": "left", "landmarks": mao(X_ESQUERDA, 0.5),
                   "position": np.array([0.0, 0.0, 0.80])}]
arbitro_contagem = gt.ArbitroDeLateralidade(verbose=False)
contagem_corrigida, rel_contagem = arbitro_contagem.arbitrar(
    alinhamento, uma_mao_kinect, aux_com_rotulo_certo, TAMANHO_KINECT)
assert rel_contagem[0]["fonte"] == "geometria", rel_contagem[0]
assert rel_contagem[0]["maos"] == 1, rel_contagem[0]
assert [e["side"] for e in contagem_corrigida[0]] == ["right", "left"]
# Nenhum voto: nem a entrada do contador e' criada (o frame corrige e mostra, mas
# nao ensina).
assert arbitro_contagem.votos.get(0, {"oposto": 0, "igual": 0}) == \
    {"oposto": 0, "igual": 0}, arbitro_contagem.votos


# (b) UMA mao de cada lado (1 x 1): o frame NAO distingue "o rotulo esta' trocado"
#     de "as duas cameras viram maos diferentes" -- corrige e mostra, mas NAO vota.
#     (Foi este o caso que, ao vivo, fez o contador de votos oscilar.)
arbitro_uma = gt.ArbitroDeLateralidade(verbose=False)
corrigido_uma, rel_uma = arbitro_uma.arbitrar(
    alinhamento, uma_mao_kinect,
    {0: [{"side": "right", "landmarks": mao(X_ESQUERDA, 0.5)}]}, TAMANHO_KINECT)
assert rel_uma[0]["fonte"] == "geometria", rel_uma[0]
assert rel_uma[0]["inverteu"] is True, rel_uma[0]     # o frame mostrou a troca
assert [e["side"] for e in corrigido_uma[0]] == ["left"]
assert arbitro_uma.votos.get(0, {"oposto": 0, "igual": 0}) == {"oposto": 0,
                                                              "igual": 0}


# (c) erro grande (100 px: decide, mas nao ensina) x apertado (30 px: ensina).
def homografia_deslocada(px):
    """Homografia 2D com deslocamento horizontal de `px` no RGB do Kinect."""
    return np.array([[1.0, 0.0, px / LARGURA], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


arbitro_largo = gt.ArbitroDeLateralidade(verbose=False)
_, rel_largo = arbitro_largo.arbitrar(
    Alinhamento({0: homografia_deslocada(100.0)}), maos_kinect_3d, aux_com_rotulo_certo,
    TAMANHO_KINECT)
assert rel_largo[0]["fonte"] == "geometria", rel_largo[0]
assert abs(rel_largo[0]["dist_px"] - 100.0) < 1.0, rel_largo[0]
assert arbitro_largo.votos.get(0, {"oposto": 0, "igual": 0}) == \
    {"oposto": 0, "igual": 0}, arbitro_largo.votos
arbitro_apertado = gt.ArbitroDeLateralidade(verbose=False)
_, rel_apertado = arbitro_apertado.arbitrar(
    Alinhamento({0: homografia_deslocada(30.0)}), maos_kinect_3d, aux_com_rotulo_certo,
    TAMANHO_KINECT)
assert rel_apertado[0]["fonte"] == "geometria", rel_apertado[0]
# UM voto por FRAME (e nao por mao: duas maos concordando valem uma evidencia),
# entao o frame apertado conta 1 -- e o de 100 px, nenhum.
assert arbitro_apertado.votos[0]["igual"] == 1, arbitro_apertado.votos[0]
assert arbitro_apertado.votos[0]["oposto"] == 0, arbitro_apertado.votos[0]
assert gt.ARBITRO_MAX_DIST_VOTO_PX < gt.ARBITRO_MAX_DIST_PX
print("14) OK: o aprendizado so' conta frame conferivel INTEIRO e com erro "
      "apertado (a correcao do frame continua valendo)")

# =============================================================================
# Limpeza: o arquivo de trabalho nao pode sobrar no projeto.
# =============================================================================
ARQUIVO_TESTE.unlink(missing_ok=True)
gt.AUX_HANDEDNESS_FLIP_FILE = Path(__file__).with_name("aux_handedness_flip.json")
gt._AUX_HANDEDNESS_FLIP.clear()
assert not ARQUIVO_TESTE.exists()
print()
print("Teste lateralidade das maos: OK")




