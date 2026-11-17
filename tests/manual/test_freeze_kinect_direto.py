"""Prova de que o arranjo KINECT-ONLY destrava o congelamento (sem hardware).

O QUE ESTE TESTE PROVA (e por que ele existe)
---------------------------------------------
MEDIDO na bancada (30/09 e 01/10/2026): o caminho antigo de congelar
(`stereo_calibration.py --salvar-intrinsecas-kinect`) coleta as poses de uma
sessao de PAR. Nessa sessao quem manda na distancia do tabuleiro e' a webcam
auxiliar, e o resultado medido foi 11-13% da diagonal do quadro do Kinect --
abaixo do portao de 35%. Portao recusado nas duas sessoes de 01/10, e o paliativo
de 30/09 (medoide de 4 modelos de sessao) depois recusado por placa: sem modelo
congelado, cada sessao voltou a RE-ESTIMAR o Kinect e dois npz da MESMA bancada
saíram 68,8 px um do outro (RMS de 1,0-1,4 px em ambos -- o RMS so' mede a
coerencia interna do arquivo).

Este teste reproduz as duas situacoes com a MESMA maquinaria de
`freeze_kinect_direto.py` (nada de camera):

  1. `tenta_gravar` com poses na distancia do arranjo KINECT-ONLY (0,30-0,44 m,
     span 35-55%) -- tem de GRAVAR, e o modelo gravado tem de ser lido de volta
     por `carrega_intrinsecas_kinect` e estar perto do verdadeiro;
  2. as MESMAS poses na distancia que o PAR impoe (~1,3 m, span ~12%) -- tem de
     RECUSAR dizendo o numero medido. E' a prova de que o problema nao era o
     operador nem a placa: era o arranjo;
   3. o LOOP INTEIRO da captura (`coleta_poses`) com uma FakeKinect -- fita de
      cantos roteirizada, sem camera e sem GUI: com as poses boas tem de GRAVAR
      (e o arquivo tem de ser lido de volta), com as poses na distancia do PAR
      tem de RECUSAR (`gravou is False` e o arquivo NUNCA criado). E' a
      regressao exata do bug da linha 319, em que `if gravou_agora is not None`
      tratava a recusa como sucesso e o programa imprimia PRONTO sem gravar;
   4. a matematica do overlay: distancia pelo pinhole, centros das poses, raio
      do anel de separacao e os pixels efetivamente desenhados na janela.

Afirma tambem o portao de CAPTURA (`aceita_pose_kinect`: span e separacao entre
poses), a medida de movimento (`deslocamento_entre`), o roteamento dos overrides
de tabuleiro (`stereo_calibration.aplica_tabuleiro`, fonte unica) e que este
modulo NAO tem copia das constantes do tabuleiro -- a duplicata que, em
01/10/2026, deixou os dois modulos declarando placas diferentes.

    .venv\\Scripts\\python.exe test_freeze_kinect_direto.py
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np

import freeze_kinect_direto as fk
import stereo_calibration as sc

KINECT_SIZE = (1920, 1080)
#: Intrinsecas VERDADEIRAS do Kinect RGB 1080p (as mesmas de
#: test_intrinsecas_kinect.py): e' contra elas que se mede o desvio do modelo.
K1_GT = np.array([[1081.37, 0.0, 959.5], [0.0, 1081.37, 539.7], [0.0, 0.0, 1.0]])
D1_GT = np.array([-0.104, 0.062, 0.0005, -0.0005, 0.0])
POSES = 16
#: Ruido de deteccao: com o tabuleiro GRANDE no quadro o erro de canto MEDIDO na
#: bancada e' bem menor (a 1080p o SB+cornerSubPix fica em ~0,7 px). 0,4 px aqui
#: porque o portao do modelo pede RMS <= 0,6 px -- e o teste tem de mostrar que,
#: com o arranjo certo, esse portao e' alcancavel.
SIGMA_PX = 0.4
#: Distancia do arranjo KINECT-ONLY (m): a placa de 324,5 mm de diagonal ocupa
#: 35-55% do quadro aqui.
Z_SO_KINECT = (0.30, 0.44)
#: Distancia que o PAR impoe (m): o tabuleiro tem de caber NAS DUAS cameras ao
#: mesmo tempo, e a auxiliar fecha o quadro primeiro. MEDIDO na bancada: 11-13%
#: da diagonal do Kinect.
Z_DO_PAR = (1.25, 1.45)
TESTE = Path(__file__).with_name("_freeze_kinect_teste.npz")
#: Arquivo da RECUSA (coleta na distancia que o PAR impoe): tem de ficar
#: intocado -- se ele nascer, a recusa virou gravacao (bug da linha 319).
TESTE_MAU = Path(__file__).with_name("_freeze_kinect_mau.npz")


def rotacao(eixo, graus):
    """Matriz de rotacao (3,3) em torno de um eixo unitario."""
    eixo = np.asarray(eixo, np.float64)
    eixo = eixo / np.linalg.norm(eixo)
    angulo = np.radians(graus)
    k = np.array([[0.0, -eixo[2], eixo[1]],
                  [eixo[2], 0.0, -eixo[0]],
                  [-eixo[1], eixo[0], 0.0]])
    return np.eye(3) + np.sin(angulo) * k + (1.0 - np.cos(angulo)) * (k @ k)


def _visivel(pontos, tamanho, margem=12.0):
    """Todos os cantos projetados dentro do quadro, com margem nas bordas."""
    px = np.asarray(pontos, np.float64).reshape(-1, 2)
    return bool(np.all(px[:, 0] > margem) and np.all(px[:, 0] < tamanho[0] - margem)
                and np.all(px[:, 1] > margem)
                and np.all(px[:, 1] < tamanho[1] - margem))


def gera_poses(quantidade, rng, z_intervalo):
    """Poses do tabuleiro REAL (7x5 @ 45 mm) vistas pelo Kinect RGB.

    A placa da cena aqui e' a que `stereo_calibration` declara HOJE (ao contrario
    de test_intrinsecas_kinect.py, cuja cena e' um fixture da medicao de 30/09):
    este teste e' sobre a bancada de agora.
    """
    tabuleiro = sc.board_object_points()
    poses = []
    tentativas = 0
    while len(poses) < quantidade and tentativas < 6000:
        tentativas += 1
        z = rng.uniform(*z_intervalo)
        #: O afastamento lateral permitido cai com a distancia: e' o que mantem o
        #: tabuleiro inteiro dentro do quadro (e nas BORDAS, que ensinam a
        #: distorcao) sem sair dele.
        limite_x = min(0.22, 0.55 * z)
        limite_y = min(0.16, 0.40 * z)
        centro = np.array([rng.uniform(-limite_x, limite_x),
                           rng.uniform(-limite_y, limite_y), z])
        giro = (rotacao((1, 0, 0), rng.uniform(-30, 30))
                @ rotacao((0, 1, 0), rng.uniform(-30, 30))
                @ rotacao((0, 0, 1), rng.uniform(0, 360)))
        rvec, _ = cv2.Rodrigues(giro)
        tvec = centro.reshape(3, 1)
        px = cv2.projectPoints(tabuleiro, rvec, tvec, K1_GT, D1_GT)[0]
        if not _visivel(px, KINECT_SIZE):
            continue
        poses.append((rvec, tvec, px))
    if len(poses) < quantidade:
        raise RuntimeError("gerou apenas %d poses validas" % len(poses))
    return poses


def observacoes(poses, rng, sigma=SIGMA_PX):
    """(obj, kin) como o detector os entrega: ordem canonica + ruido de pixel.

    `sigma` e' o desvio do ruido de deteccao (padrao o MEDIDO na bancada, 0,4 px).
    O loop da coleta usa 0,2 px: com 12 poses o RMS fica folgado sob o portao de
    0,6 px e o teste exercita o CAMINHO DE GRAVACAO, nao o sorteio do ruido.
    """
    tabuleiro = sc.board_object_points()
    obj, kin, spans = [], [], []
    for rvec, tvec, px in poses:
        px = cv2.projectPoints(tabuleiro, rvec, tvec, K1_GT, D1_GT)[0]
        px = sc.canonical_order(px.reshape(-1, 2))[1]
        px = px + rng.normal(0.0, sigma, px.shape)
        obj.append(tabuleiro.copy())
        kin.append(px.reshape(-1, 1, 2))
        spans.append(sc.board_span(px) / float(np.hypot(*KINECT_SIZE)))
    return obj, kin, spans


def desvio_px(matriz, distorcao):
    """(mediana, max) px do modelo contra o VERDADEIRO, na regiao de trabalho."""
    return sc.compara_modelos_kinect(matriz, distorcao, K1_GT, D1_GT)


def main():
    rng = np.random.default_rng(11)
    #: O arquivo do teste NUNCA e' o da bancada: o modelo bom tem de sobreviver a
    #: qualquer execucao da suite.
    assert TESTE != sc.KINECT_INTRINSICS_FILE
    assert sc.KINECT_INTRINSICS_MIN_SPAN > sc.MIN_BOARD_SPAN_FRACTION
    TESTE.unlink(missing_ok=True)
    try:
        mediana = _so_kinect(rng)
        _distancia_do_par(rng, mediana)
        _portao_de_captura()
        _movimento()
        _fonte_unica()
        _overlay_e_distancia()
        _overlay_pixels()
        _resumo_do_span()
        _erros_do_rms()
        _coleta_com_fakekinect()
    finally:
        TESTE.unlink(missing_ok=True)
        TESTE_MAU.unlink(missing_ok=True)
    print("FREEZE_KINECT_DIRETO_OK: o arranjo Kinect-only grava (e o modelo sai "
          "perto do verdadeiro); na distancia que o PAR impoe o portao recusa "
          "citando o span medido; portao de captura, medida de movimento, fonte "
          "unica das constantes, distancia/anel do overlay, o SPAN MEDIANO "
          "(com aviso da pose fraca e motivo final do portao), o erro de CADA "
          "pose (pose isolada x captura inteira, com o descarte das piores "
          "gravando o conjunto que sobra) e o LOOP da coleta (FakeKinect: grava "
          "de verdade, a recusa nunca vira 'gravou' e as poses longe derrubam a "
          "mediana sem gravar -- regressao da linha 319) conferidos.")
    return 0


def _so_kinect(rng):
    """1) Com o tabuleiro perto (so' o Kinect), os portoes passam e GRAVA."""
    poses = gera_poses(POSES, rng, Z_SO_KINECT)
    obj, kin, spans = observacoes(poses, rng)
    mediana_span = float(np.median(spans))
    print("KINECT-ONLY: %d poses a %.2f-%.2f m | span mediano %.0f%% (portao "
          "%.0f%%) | menor span %.0f%%"
          % (len(poses), Z_SO_KINECT[0], Z_SO_KINECT[1], mediana_span * 100,
             sc.KINECT_INTRINSICS_MIN_SPAN * 100, min(spans) * 100))
    assert mediana_span >= sc.KINECT_INTRINSICS_MIN_SPAN, mediana_span
    gravou, rms, motivo = fk.tenta_gravar(obj, kin, KINECT_SIZE, str(TESTE))
    assert gravou, "o arranjo Kinect-only tinha de gravar: %s" % motivo
    assert rms <= sc.KINECT_INTRINSICS_MAX_RMS, rms
    # round-trip pelo caminho que a CALIBRACAO DE PAR usa
    matriz, distorcao, info = sc.carrega_intrinsecas_kinect(TESTE)
    assert matriz is not None, info
    assert info["poses"] == POSES, info
    assert abs(info["span"] - mediana_span) < 1e-9, (info, mediana_span)
    assert abs(info["rms"] - rms) < 1e-9, info
    assert info["aviso_placa"] == "", info["aviso_placa"]
    mediana, pior = desvio_px(matriz, distorcao)
    print("  -> gravado e lido de volta: RMS %.3f px | span %.0f%% | modelo a "
          "mediana %.2f px (pior %.2f) do VERDADEIRO"
          % (rms, info["span"] * 100, mediana, pior))
    #: O arquivo em disco guarda o MELHOR: com um `melhor_rms` ja' menor o
    #: programa NAO reescreve, porque mais poses quase iguais podem subir o RMS
    #: (a mesma regra do caminho do par, que salva se `rms < best_rms`).
    gravou_pior, rms_pior, motivo_pior = fk.tenta_gravar(
        obj, kin, KINECT_SIZE, str(TESTE), melhor_rms=0.0)
    assert gravou_pior is None and "NAO melhorou" in motivo_pior, motivo_pior
    assert rms_pior is not None, rms_pior
    assert np.allclose(sc.carrega_intrinsecas_kinect(TESTE)[0], matriz), \
        "o modelo em disco foi sobrescrito por um pior"
    assert mediana < 5.0, ("o modelo do arranjo Kinect-only ficou longe do "
                           "verdadeiro: %.2f px" % mediana)
    return mediana


def _distancia_do_par(rng, mediana_do_arranjo):
    """2) Na distancia que o PAR impoe, o MESMO programa recusa -- com o numero.

    E prova o que o arranjo vale: o modelo que sairia dali (estimativa da SESSAO,
    como o caminho antigo fazia) fica varias vezes mais longe do verdadeiro do que
    o modelo do arranjo Kinect-only.
    """
    #: A parte 1 gravou: apago agora para provar que RECUSAR nao escreve nada (o
    #: modelo bom da bancada nao pode ser sobrescrito por uma tentativa que falha).
    TESTE.unlink(missing_ok=True)
    poses = gera_poses(POSES, rng, Z_DO_PAR)
    obj, kin, spans = observacoes(poses, rng)
    mediana_span = float(np.median(spans))
    print("DISTANCIA DO PAR: %d poses a %.2f-%.2f m | span mediano %.0f%%"
          % (len(poses), Z_DO_PAR[0], Z_DO_PAR[1], mediana_span * 100))
    assert mediana_span < sc.KINECT_INTRINSICS_MIN_SPAN, mediana_span
    gravou, rms, motivo = fk.tenta_gravar(obj, kin, KINECT_SIZE, str(TESTE))
    assert not gravou, "o arranjo do par NAO podia passar no portao do span"
    assert "diagonal do quadro" in motivo, motivo
    assert ("%.0f%%" % (mediana_span * 100)) in motivo, motivo
    assert rms is None, ("sem span nao ha' motivo para pagar o calibrateCamera: %s"
                         % rms)
    assert not TESTE.exists(), "recusar NAO pode gravar arquivo nenhum"
    # o que o caminho antigo congelaria dali: a estimativa da SESSAO a 12% de span
    _, matriz_sessao, dist_sessao, _, _ = sc.resolve_modelo_kinect(
        obj, kin, KINECT_SIZE)
    mediana_sessao, _ = desvio_px(matriz_sessao, dist_sessao)
    print("  -> recusado (%s); o modelo estimado NAQUELA distancia fica a mediana "
          "%.2f px do verdadeiro -- %.1fx o desvio do arranjo Kinect-only "
          "(%.2f px)" % (motivo, mediana_sessao,
                         mediana_sessao / mediana_do_arranjo, mediana_do_arranjo))
    assert mediana_sessao > 2.0 * mediana_do_arranjo, (
        "o arranjo tinha de valer mais que isto: %.2f px x %.2f px"
        % (mediana_sessao, mediana_do_arranjo))
    return mediana_sessao



def _portao_de_captura():
    """3) `aceita_pose_kinect`: span, separacao entre poses e recusa explicita."""
    rng = np.random.default_rng(23)
    obj, kin, _ = observacoes(gera_poses(POSES, rng, Z_SO_KINECT), rng)
    perto = np.asarray(kin[0], np.float64).reshape(-1, 2)
    ok, motivo, fracao = sc.aceita_pose_kinect(perto, [], KINECT_SIZE)
    assert ok, motivo
    assert fracao >= sc.MIN_BOARD_SPAN_FRACTION, fracao
    #: O chao da CAPTURA (8%) e' mais frouxo que o portao do MODELO (35%) e isso e'
    #: de proposito: uma pose distante ainda ensina DISTORCAO, e recusar a captura
    #: seria perder trabalho. Quem reprova o conjunto e' o portao do modelo -- por
    #: isso a pose na distancia do PAR passa AQUI e nao passa la'.
    rng_par = np.random.default_rng(29)
    _, kin_par, _ = observacoes(gera_poses(3, rng_par, Z_DO_PAR), rng_par)
    ok_par, motivo_par, fracao_par = sc.aceita_pose_kinect(
        np.asarray(kin_par[0], np.float64).reshape(-1, 2), [], KINECT_SIZE)
    assert ok_par, motivo_par
    assert sc.MIN_BOARD_SPAN_FRACTION <= fracao_par < sc.KINECT_INTRINSICS_MIN_SPAN
    # tabuleiro longe demais ate' para a captura (abaixo do chao de 8%)
    longe = perto * 0.2 + 300.0
    ok_far, motivo_far, fracao_far = sc.aceita_pose_kinect(longe, [], KINECT_SIZE)
    assert not ok_far and "diagonal" in motivo_far, motivo_far
    assert fracao_far < sc.MIN_BOARD_SPAN_FRACTION, fracao_far
    # e COLADO na camera (o tabuleiro a ~0,15-0,20 m): sai de foco e domina fx
    colado = perto * 2.0
    ok_colado, motivo_colado, fracao_colado = sc.aceita_pose_kinect(
        colado, [], KINECT_SIZE)
    assert not ok_colado and "maximo" in motivo_colado, motivo_colado
    assert fracao_colado > sc.KINECT_INTRINSICS_MAX_SPAN > \
        sc.KINECT_INTRINSICS_MIN_SPAN, (fracao_colado, sc.KINECT_INTRINSICS_MAX_SPAN)
    # pose igual a uma ja' coletada (12 amostras de UMA pose = extrinseca falsa)
    ok_dup, motivo_dup, _ = sc.aceita_pose_kinect(perto, [kin[0]], KINECT_SIZE)
    assert not ok_dup and "quase igual" in motivo_dup, motivo_dup
    # e a pose mais AFASTADA da primeira tem de passar
    centros = [np.asarray(k, np.float64).reshape(-1, 2).mean(axis=0) for k in kin]
    centro0 = centros[0]
    longe_idx = int(np.argmax([np.linalg.norm(c - centro0) for c in centros]))
    span0 = sc.board_span(perto)
    assert (np.linalg.norm(centros[longe_idx] - centro0)
            > sc.MIN_POSE_SEPARATION * span0), "cenas de teste proximas demais"
    ok_nova, motivo_nova, _ = sc.aceita_pose_kinect(
        np.asarray(kin[longe_idx], np.float64).reshape(-1, 2), [kin[0]],
        KINECT_SIZE)
    assert ok_nova, motivo_nova
    print("Portao de captura: recusa com motivo explicito (span e separacao) e "
          "aceita pose nova a %.0f px da primeira"
          % np.linalg.norm(centros[longe_idx] - centro0))


def _movimento():
    """4) `deslocamento_entre`: e' o que impede entrar amostra borrada."""
    rng = np.random.default_rng(31)
    _, kin, _ = observacoes(gera_poses(2, rng, Z_SO_KINECT), rng)
    cantos = np.asarray(kin[0], np.float64).reshape(-1, 2)
    #: Deslocamento puro em X: a norma media e' exatamente o valor deslocado (mover
    #: em X e Y ao mesmo tempo daria sqrt(2) vezes o valor).
    andando = cantos + np.array([9.0, 0.0])
    assert fk.deslocamento_entre(cantos, None) == float("inf")
    assert fk.deslocamento_entre(None, cantos) == float("inf")
    assert fk.deslocamento_entre(cantos, cantos) == 0.0
    assert abs(fk.deslocamento_entre(cantos, andando) - 9.0) < 1e-9
    assert fk.deslocamento_entre(cantos, cantos[:5]) == float("inf")
    #: O portao tem de ser maior que o ruido de deteccao MEDIDO (~0,7 px na
    #: resolucao cheia), senao nenhuma pose seria aceita com o tabuleiro parado --
    #: e menor que o movimento que se quer barrar (aqui, 9 px).
    assert 2.0 * SIGMA_PX < fk.MAX_DESLOCAMENTO_PX < 9.0
    print("Movimento: 0 px parado | %.1f px com o tabuleiro andando 9 px | 'inf' "
          "sem os dois lados | portao %.1f px (ruido de deteccao %.1f px)"
          % (fk.deslocamento_entre(cantos, andando), fk.MAX_DESLOCAMENTO_PX,
             SIGMA_PX))


def _fonte_unica():
    """5) Nada de constante duplicada -- a duplicata que custou 01/10/2026."""
    assert fk.sc is sc, "o arranjo Kinect-only tem de usar o stereo_calibration"
    for nome in ("CHECKERBOARD_SIZE", "CHECKERBOARD_COLS", "CHECKERBOARD_ROWS",
                 "SQUARE_SIZE_M", "KINECT_INTRINSICS_MIN_SPAN",
                 "KINECT_INTRINSICS_MIN_POSES", "KINECT_INTRINSICS_MAX_RMS",
                 "MIN_BOARD_SPAN_FRACTION", "MIN_POSE_SEPARATION",
                 "STABLE_CAPTURE_SECONDS", "DETECTION_WIDTHS"):
        assert not hasattr(fk, nome), (
            "freeze_kinect_direto.py tem copia da constante %s: uma das duas vai "
            "divergir, como em 01/10/2026" % nome)
    placa = (tuple(sc.CHECKERBOARD_SIZE), float(sc.SQUARE_SIZE_M))
    try:
        ok, mensagem = sc.aplica_tabuleiro("5x3", 57.0)
        assert ok, mensagem
        assert tuple(sc.CHECKERBOARD_SIZE) == (5, 3), sc.CHECKERBOARD_SIZE
        assert abs(sc.SQUARE_SIZE_M - 0.057) < 1e-12, sc.SQUARE_SIZE_M
        assert not sc.aplica_tabuleiro("bogus", None)[0]
        assert "COLSxROWS" in sc.aplica_tabuleiro("bogus", None)[1]
        assert not sc.aplica_tabuleiro("2x2", None)[0]
        assert "invalido" in sc.aplica_tabuleiro("2x2", None)[1]
        assert not sc.aplica_tabuleiro(None, -1.0)[0]
        assert "> 0" in sc.aplica_tabuleiro(None, -1.0)[1]
    finally:
        sc.CHECKERBOARD_COLS, sc.CHECKERBOARD_ROWS = placa[0]
        sc.CHECKERBOARD_SIZE = placa[0]
        sc.SQUARE_SIZE_M = placa[1]
    assert tuple(sc.CHECKERBOARD_SIZE) == placa[0], "placa nao foi restaurada"
    assert abs(sc.SQUARE_SIZE_M - placa[1]) < 1e-12, "quadrado nao foi restaurado"
    print("Fonte unica: constantes vem de stereo_calibration (nenhuma copia) e os "
          "overrides passam por aplica_tabuleiro, com a placa %dx%d @ %g mm "
          "restaurada" % (placa[0][0], placa[0][1], placa[1] * 1000))


def _overlay_e_distancia():
    """6) Tudo que a janela desenha/exibe, como funcoes puras de array."""
    diagonal_px = float(np.hypot(*KINECT_SIZE))
    # --- distancia: Z = fx * placa / span_px = 0,159 / fracao ----------------
    d35 = fk.distancia_aproximada_m(0.35, KINECT_SIZE)
    esperado35 = (fk.FX_NOMINAL_PX * fk.placa_diagonal_m()
                  / (0.35 * diagonal_px))
    assert abs(d35 - esperado35) < 1e-12, (d35, esperado35)
    #: O portao do modelo (35%) tem de cair DENTRO da faixa de captura (m) e o
    #: portao maximo (60%) dentro da parte "nao cole" da faixa.
    assert 0.44 < d35 < 0.47, d35
    d60 = fk.distancia_aproximada_m(0.60, KINECT_SIZE)
    assert 0.25 < d60 < 0.29, d60
    assert d60 < d35, (d60, d35)
    #: span inexistente NAO vira distancia mentirosa na linha de estado
    assert np.isnan(fk.distancia_aproximada_m(0.0, KINECT_SIZE))
    assert np.isnan(fk.distancia_aproximada_m(None, KINECT_SIZE))
    #: o fx do modelo congelado entra no lugar do nominal (proporcional)
    z_fx = fk.distancia_aproximada_m(0.35, KINECT_SIZE, fx=1040.0)
    assert 0.0 < z_fx < d35, (z_fx, d35)
    #: fonte unica da placa: a mesma formula, derivada DAS constantes de sc
    placa = float(np.hypot((sc.CHECKERBOARD_COLS - 1) * sc.SQUARE_SIZE_M,
                           (sc.CHECKERBOARD_ROWS - 1) * sc.SQUARE_SIZE_M))
    assert abs(fk.placa_diagonal_m() - placa) < 1e-12, (fk.placa_diagonal_m(),
                                                        placa)
    #: modelo_em_disco: ausente => (None, None); o resto e' coberto na coleta
    assert fk.modelo_em_disco(str(TESTE_MAU)) == (None, None)
    # --- centros das poses salvas e raio do anel ----------------------------
    assert fk.centros_das_poses([]).shape == (0, 2)
    rng = np.random.default_rng(53)
    _, kin, _ = observacoes(gera_poses(2, rng, Z_SO_KINECT), rng)
    centros = fk.centros_das_poses(kin)
    assert centros.shape == (2, 2)
    for i, item in enumerate(kin):
        manual = np.asarray(item, np.float64).reshape(-1, 2).mean(axis=0)
        assert np.allclose(centros[i], manual), (centros[i], manual)
    #: O anel e' LITERALMENTE o portao da captura: raio = fracao * span
    assert abs(fk.raio_anel_separacao(800.0, 0.30) - 240.0) < 1e-12
    print("Overlay: 35%% -> %.2f m | 60%% -> %.2f m | anel = 0,30 x span "
          "(%.0f px @ 800 px de span) | centros/NaN/fx conferidos"
          % (d35, d60, fk.raio_anel_separacao(800.0, 0.30)))


def _overlay_pixels():
    """6b) Os pixels: X da pose salva, anel e centro ATUAL na janela reduzida."""
    rng = np.random.default_rng(59)
    _, kin, _ = observacoes(gera_poses(2, rng, Z_SO_KINECT), rng)
    centros = fk.centros_das_poses(kin)
    exibicao = np.zeros((int(1080 * sc.DISPLAY_SCALE),
                         int(1920 * sc.DISPLAY_SCALE), 3), np.uint8)
    saida = fk.desenha_marcadores(exibicao, kin[:1], kin[1], 0.30)
    assert saida is exibicao, "tem de devolver a propria imagem"
    escala = sc.DISPLAY_SCALE
    cx, cy = (np.asarray(centros[0]) * escala).astype(int)
    assert exibicao[int(cy), int(cx)].any(), "X da pose salva nao desenhado"
    raio_px = int(round(fk.raio_anel_separacao(sc.board_span(kin[1]), 0.30)
                        * escala))
    assert any(exibicao[int(cy), int(cx) + d].any()
               for d in range(raio_px - 2, raio_px + 3)), \
        "anel de separacao fora do raio esperado"
    atual = (np.asarray(kin[1], np.float64).reshape(-1, 2).mean(axis=0)
             * escala).astype(int)
    assert any(exibicao[int(atual[1]), int(atual[0]) + d].any()
               for d in range(5, 10)), "centro do tabuleiro ATUAL nao desenhado"
    #: sem pose atual nao ha' span: sobra o X; nao ha' anel nem ponto
    limpa = np.zeros_like(exibicao)
    fk.desenha_marcadores(limpa, kin[:1], None, 0.30)
    assert limpa[int(cy), int(cx)].any(), "o X tem de sobreviver sem candidato"
    assert not any(limpa[int(cy), int(cx) + d].any()
                   for d in range(raio_px - 2, raio_px + 3)), \
        "sem pose atual nao ha' span nem anel a desenhar"
    print("Overlay pixels: X/anel/atual conferidos na janela %dx%d (anel "
          "%.0f px)" % (limpa.shape[1], limpa.shape[0], raio_px))


def _resumo_do_span():
    """6c) O numero que DECIDE (o mediano) e o que o operador ve no console."""
    #: 3 poses dentro da faixa + 3 longe: o mediano NAO e' o da melhor pose, e
    #: e' ele que o portao do modelo usa. Sem o par (mediano, ok/total) a
    #: bancada de 05/10/2026 nao tinha como saber que 6 das 18 poses estavam
    #: derrubando o conjunto.
    rng = np.random.default_rng(67)
    _, perto, _ = observacoes(gera_poses(3, rng, Z_SO_KINECT), rng)
    _, longe, _ = observacoes(gera_poses(3, rng, (0.55, 0.62)), rng)
    pontos = perto + longe
    por_pose = fk.spans_das_poses(pontos, KINECT_SIZE)
    manual = np.array([sc.board_span(p) for p in pontos]) \
        / float(np.hypot(*KINECT_SIZE))
    assert np.allclose(por_pose, manual), (por_pose, manual)
    mediana, ok, total = fk.resumo_do_span(pontos, KINECT_SIZE)
    assert total == 6, total
    #: a mediana e' LITERALMENTE a do portao (`sc.span_do_tabuleiro`)
    assert mediana == sc.span_do_tabuleiro(pontos, KINECT_SIZE), mediana
    assert ok == int((manual >= sc.KINECT_INTRINSICS_MIN_SPAN).sum()), (ok, manual)
    assert 0 < ok < total, "a cena precisa de poses boas E frageis: %d/%d" \
        % (ok, total)
    #: o texto do console: 1 casa decimal (o portao reprova em 34,9% e o motivo
    #: do sc arredonda para "35%") e a contagem de quem cumpre a meta
    texto = fk.texto_do_span(pontos, KINECT_SIZE)
    assert "%.1f%%" % (mediana * 100.0) in texto, texto
    assert "meta %.1f%%" % (sc.KINECT_INTRINSICS_MIN_SPAN * 100.0) in texto, texto
    assert "%d/%d" % (ok, total) in texto, texto
    #: sem poses: nada de mediana mentirosa nem colchete vazio no motivo
    assert fk.spans_das_poses([], KINECT_SIZE).shape == (0,)
    assert fk.resumo_do_span([], KINECT_SIZE) == (0.0, 0, 0)
    assert fk.texto_do_span([], KINECT_SIZE) == ""
    assert fk.motivo_com_span("motivo", [], KINECT_SIZE) == "motivo"
    #: o motivo do portao sai com o mediano no colchete (era o que faltava
    #: para entender o "ocupou 35% (minimo 35%)")
    motivo = fk.motivo_com_span("o tabuleiro ocupou 35% (minimo 35%)", pontos,
                               KINECT_SIZE)
    assert motivo.startswith("o tabuleiro ocupou 35%") and "span mediano" in motivo
    #: ATENCAO na hora para a pose abaixo da meta, silencio para a boa
    fraco = float(fk.spans_das_poses(longe, KINECT_SIZE)[0])
    aviso = fk.aviso_span_fraco(fraco, pontos, KINECT_SIZE)
    assert "ATENCAO" in aviso and "MEDIANO" in aviso, aviso
    assert "%.1f%%" % (fraco * 100.0) in aviso, aviso
    assert fk.aviso_span_fraco(float(fk.spans_das_poses(perto, KINECT_SIZE)[0]),
                               pontos, KINECT_SIZE) == ""
    assert fk.aviso_span_fraco(0.50, pontos, KINECT_SIZE) == ""
    assert fk.aviso_span_fraco(None, pontos, KINECT_SIZE) == ""
    assert fk.aviso_span_fraco(float("nan"), pontos, KINECT_SIZE) == ""
    #: a linha de estado da JANELA mostra o mesmo par de numeros
    linha = fk.linha_de_status(6, 0.30, mediana, None, "motivo", 0.5, ok, total)
    assert "mediano %.1f%%" % (mediana * 100.0) in linha, linha
    assert "%d/%d ok" % (ok, total) in linha, linha
    assert "~0.50 m" in linha, linha
    vazia = fk.linha_de_status(0, 0.0, 0.0, None, "motivo")
    assert "--/-- ok" in vazia and "-- m" in vazia, vazia
    print("Span MEDIANO: %.1f%% com %d/%d poses cumprindo a meta | aviso da pose "
          "fraca, 1 casa decimal, contagem ok/total e linha de estado conferidos"
          % (mediana * 100.0, ok, total))


def _erros_do_rms():
    """6d) O erro de CADA pose: separa pose ruim de captura ruim (o RMS nao)."""
    #: MEDIDO na bancada (05/10/2026, 2a tentativa): 16 poses, RMS 1,77 px e nada
    #: gravado -- e o `calibrateCamera` nao diz QUAL pose estragou o conjunto.
    rng = np.random.default_rng(83)
    obj, kin, _ = observacoes(_cena_grade(), rng, sigma=0.2)
    cena = _cena_aceita(kin, 13)
    assert len(cena) >= 12, len(cena)
    objetos = [sc.board_object_points().copy() for _ in cena]
    ok, rms, matriz, _, _, _, erros, _ = fk.avalia_conjunto(objetos, cena,
                                                            KINECT_SIZE)
    assert ok and erros is not None, (ok, rms)
    assert erros.shape == (len(cena),)
    assert fk.piores_poses(erros, 1) == [int(np.argmax(erros))]
    assert fk.piores_poses(np.zeros(0), 3) == []
    assert fk.descartes_sugeridos(np.zeros(0), 0.6, 4) == []
    #: conjunto HOMOGENEO (todas parecidas, todas ruins): nao se descarta ninguem
    assert fk.descartes_sugeridos(np.full(15, 1.5), 0.6, 4) == []
    #: uma pose CLARAMENTE pior sai; as "quase" ficam (o corte e' 3x a mediana)
    misto = np.array([0.25] * 14 + [4.0])
    assert fk.descartes_sugeridos(misto, 0.6, 4) == [14]
    assert fk.descartes_sugeridos(np.array([0.25, 0.8, 0.9, 1.0]), 0.6, 4) == []
    #: o veredito separa os dois casos -- a acao do operador e' outra
    assert "CAPTURA INTEIRA" in fk.veredito_do_rms(np.full(15, 1.5), 0.6)
    veredito = fk.veredito_do_rms(misto, 0.6)
    assert "refaca SO'" in veredito and "#15" in veredito, veredito
    assert fk.veredito_do_rms(np.zeros(0), 0.6) == ""
    #: o traceback do pykinect2 ao fechar vira UMA linha (e nao engole outro erro)
    visto = {"anterior": []}
    gancho = fk.gancho_do_pykinect2(anterior=visto["anterior"].append,
                                    aviso=visto.setdefault("aviso", []).append)
    gancho(argparse.Namespace(exc_value=AttributeError(
        "'NoneType' object has no attribute 'GetFrameArrivedEventData'")))
    assert not visto["anterior"] and visto["aviso"], visto
    gancho(argparse.Namespace(exc_value=ValueError("outro erro de thread")))
    assert len(visto["anterior"]) == 1 and isinstance(
        visto["anterior"][0].exc_value, ValueError), visto
    print("Erro por pose: %d poses, mediana %.3f px, pior %.3f px | descarte "
          "(3x a mediana), veredito pose x captura e o wart do pykinect2 "
          "conferidos"
          % (erros.size, float(np.median(erros)), float(np.max(erros))))


# ---------------------------------------------------------------------------
# 7) O LOOP da captura com FakeKinect -- a REGRESSAO da linha 319
# ---------------------------------------------------------------------------
class _KinectFalso:
    """`kinect` de mentira: a "fita" de cantos roteirizada e' o detector.

    `has_new_color_frame`/`get_last_color_frame` entregam um buffer BGRA preto
    enquanto a fita nao acabar; quando acaba, nao ha' frame e `coleta_poses`
    dorme ate' o `--limite-s` (e' assim que o cenario da RECUSA termina sem
    gravar). O `find_corners` roteirizado e' trocado por `_roda_coleta`.
    """

    def __init__(self, fita):
        self.fita = fita
        self._quadro = np.zeros(KINECT_SIZE[1] * KINECT_SIZE[0] * 4, np.uint8)

    def has_new_color_frame(self):
        return bool(self.fita)

    def get_last_color_frame(self):
        return self._quadro

    def close(self):
        pass


def _cena_aceita(candidatas, limite):
    """Da fita de candidatas, as que o PROPIO `aceita_pose_kinect` aceita.

    Sem isso a cena mentiria para o portao ("quase igual" nunca chegaria em
    `tenta_gravar`), e o teste teria de reimplementar as regras -- aqui ele usa
    a MESMA funcao que a coleta usa.
    """
    cena = []
    for k in candidatas:
        if len(cena) >= limite:
            break
        ok, _, _ = sc.aceita_pose_kinect(np.asarray(k).reshape(-1, 2),
                                         cena, KINECT_SIZE)
        if ok:
            cena.append(k)
    return cena


def _roda_coleta(cena, saida, limite_s, descartes=0):
    """`coleta_poses` AO VIVO com a fita roteirizada: sem camera, sem GUI.

    Dois remendos pontuais no stereo_calibration (sempre restaurados):
    DETECTION_INTERVAL_S=0 faz a fita avancar um item por frame e
    STABLE_CAPTURE_SECONDS=0 tira os 2 s de espera de RELOGIO (o que se TESTA
    aqui e' o portao e a gravacao, nao o cronometro). Cada pose aparece DUAS
    vezes na fita: a 1a deteccao zera o cronometro (movimento > 3 px) e a 2a e'
    aceita -- o mesmo ciclo real de quem segura o tabuleiro parado.

    `descartes` vai para `--max-descartes` (0 = nenhuma pose descartada).
    """
    fita = []
    for k in cena:
        fita.extend([k, k])
    kinect = _KinectFalso(fita)
    original_find = sc.find_corners
    original_intervalo = sc.DETECTION_INTERVAL_S
    original_estavel = sc.STABLE_CAPTURE_SECONDS
    sc.find_corners = lambda quadro: (fita.pop(0) if fita else None)
    sc.DETECTION_INTERVAL_S = 0.0
    sc.STABLE_CAPTURE_SECONDS = 0.0
    args = argparse.Namespace(
        saida=str(saida), deslocamento_minimo=sc.MIN_POSE_SEPARATION,
        max_poses=sc.MAX_TOTAL_SAMPLES, sair_ao_gravar=False,
        sem_janela=True, limite_s=limite_s, max_descartes=descartes)
    try:
        return fk.coleta_poses(kinect, KINECT_SIZE, args)
    finally:
        sc.find_corners = original_find
        sc.DETECTION_INTERVAL_S = original_intervalo
        sc.STABLE_CAPTURE_SECONDS = original_estavel


def _cena_grade(z=0.43, colunas=(400, 670, 940, 1210, 1480),
                linhas=(270, 540, 810)):
    """Grade (px) de poses que o portao aceita SEM depender de sorte de semente.

    Aleatorio nao serve: MEDIDO, fechava no maximo 6-13 de 30 candidatas --
    o anel do portao e' 0,30 x span e o quadro e' finito. A grade poe os
    centros a 270 px (anel max 262 px a z = 0,43 m), tira o que nao couber
    (`_visivel`) e devolve o que sobra: 14 poses com span mediano ~36,5%.

    `z` (m) e' parametro porque o MESMO gerador monta a cena do fracasso de
    05/10/2026: a 0,47 m o span mediano cai para ~33% (abaixo do portao) e a
    0,80 m para ~20% (uma pose que DERRUBA o mediano) -- com o mesmo portao de
    separacao, que e' o que o teste precisa.

    O detalhe que quebra a ingenuidade: a origem de `board_object_points` e'
    um CANTO da placa (fixture do OpenCV), entao um tvec ingenuo joga o
    tabuleiro para o quadrante sudeste do alvo, a silhueta sai do quadro e o
    portao reprova metade da grade. E' preciso corrigir tvec pela projecao
    CENTRADA (iteracao ate' <0,5 px) -- parametrizacao medida na bancada.
    """
    cx0, cy0 = K1_GT[0, 2], K1_GT[1, 2]
    fx, fy = K1_GT[0, 0], K1_GT[1, 1]
    z = float(z)
    tabuleiro = sc.board_object_points()
    centro_placa = tabuleiro.reshape(-1, 3).mean(axis=0)
    giros = ((12, 12), (12, -12), (-12, 12), (-12, -12))
    cenas, k = [], 0
    for cy in linhas:
        for cx in colunas:
            gx, gy = giros[k % 4]
            k += 1
            rot = rotacao((1, 0, 0), gx) @ rotacao((0, 1, 0), gy)
            rvec, _ = cv2.Rodrigues(rot)
            tvec = np.array([[(cx - cx0) * z / fx],
                             [(cy - cy0) * z / fy],
                             [z]]) - (rot @ centro_placa).reshape(3, 1)
            for _ in range(4):
                px = cv2.projectPoints(tabuleiro, rvec, tvec, K1_GT, D1_GT)[0]
                centro = px.reshape(-1, 2).mean(axis=0)
                if abs(centro[0] - cx) < 0.5 and abs(centro[1] - cy) < 0.5:
                    break
                tvec[0, 0] += (cx - centro[0]) * z / fx
                tvec[1, 0] += (cy - centro[1]) * z / fy
            if _visivel(px.reshape(-1, 2), KINECT_SIZE):
                cenas.append((rvec, tvec, px))
    return cenas


def _coleta_grava_de_verdade():
    """7) O loop INTEIRO com poses boas: tem de GRAVAR -- e ser lido de volta."""
    TESTE.unlink(missing_ok=True)
    assert fk.modelo_em_disco(str(TESTE)) == (None, None)
    cena = _cena_aceita(
        observacoes(_cena_grade(), np.random.default_rng(41), sigma=0.2)[1],
        16)
    assert len(cena) >= 12, "a grade parou de caber no portao: %d" % len(cena)
    span12 = sc.span_do_tabuleiro(cena[:12], KINECT_SIZE)
    assert span12 >= sc.KINECT_INTRINSICS_MIN_SPAN, span12
    gravou, rms, motivo = _roda_coleta(cena, TESTE, limite_s=5.0)
    #: O bug (linha 319) fazia o ramo RECUSAR devolver `gravou=True` e o teste
    #: passaria SEM ARQUIVO -- por isso a leitura de volta e' obrigatoria.
    assert gravou, "coleta com poses boas nao gravou: %r" % (motivo,)
    assert rms is not None and motivo == "gravado", (rms, motivo)
    assert TESTE.exists(), "diz 'gravou' e o arquivo NAO existe (linha 319)"
    matriz, dist, info = sc.carrega_intrinsecas_kinect(TESTE)
    assert matriz is not None, info
    assert (info["poses"] >= 12
            and info["span"] >= sc.KINECT_INTRINSICS_MIN_SPAN), info
    assert rms <= sc.KINECT_INTRINSICS_MAX_RMS, rms
    melhor, fx_disco = fk.modelo_em_disco(str(TESTE))
    assert melhor is not None and abs(melhor - rms) < 1e-6, (melhor, rms)
    assert fx_disco is not None and fx_disco > 0.0, fx_disco
    mediana, _ = desvio_px(matriz, dist)
    assert mediana < 5.0, "modelo gravado longe do verdadeiro: %.2f px" % mediana
    print("FakeKinect GRAVOU: %d poses, span %.0f%%, RMS %.3f px -> %s "
          "(lido de volta: mediana %.2f px do verdadeiro)"
          % (info["poses"], info["span"] * 100, rms,
             Path(info["arquivo"]).name, mediana))


def _coleta_recusa_honestamente():
    """8) REGRESSAO da linha 319: poses longe => NUNCA 'gravou'/'PRONTO'."""
    TESTE_MAU.unlink(missing_ok=True)
    #: Aleatorio com pool de 30 fechava menos de 12 poses as vezes (mesma
    #: geometria do anel): pool maior e semente em retry ate' a cena fechar.
    cena = None
    for semente in range(4):
        rng = np.random.default_rng(47 + semente)
        pool = observacoes(gera_poses(60, rng, Z_DO_PAR), rng, sigma=0.2)[1]
        candidata = _cena_aceita(pool, 24)
        if len(candidata) >= 12:
            cena = candidata
            break
    assert cena is not None, "nao fechei 12 poses na distancia do PAR"
    span = sc.span_do_tabuleiro(cena[:12], KINECT_SIZE)
    assert span < sc.KINECT_INTRINSICS_MIN_SPAN, span
    gravou, rms, motivo = _roda_coleta(cena, TESTE_MAU, limite_s=1.0)
    #: O bug: `if gravou_agora is not None` aceitava o False da 1a pose (1/12),
    #: marcava gravou=True e o loop SAIA (`--sem-janela`): o main imprimia
    #: PRONTO/`MODELO GRAVADO` sem arquivo nenhum -- MEDIDO na bancada em
    #: 05/10/2026 com o span em 13%.
    assert not gravou, ("a recusa do portao virou sucesso: %r" % (motivo,))
    assert rms is None, ("sem span nao ha' motivo para resolver: %r" % (rms,))
    assert not TESTE_MAU.exists(), "RECUSAR criou arquivo"
    print("FakeKinect RECUSOU honestamente: %d poses a %.0f%% da diagonal | "
          "gravou=%s, rms=%s, arquivo nao criado | fim: %s"
          % (len(cena), span * 100, gravou, rms, motivo))


def _coleta_derruba_a_mediana():
    """9) O fracasso da bancada de 05/10/2026: poses BOAS + poses LONGE.

    18 poses aceitas, mediano de span 34,9% e NADA gravado -- e o console so'
    dizia "junte mais poses", com a recusa formatada em "ocupou 35% (minimo
    35%)". Aqui a mesma cena: a grade a 0,45 m (span mediano logo ABAIXO do
    portao) mais uma grade ENTRELACADA a 0,80 m (span ~20%, a pose que derruba
    o conjunto e que o portao de captura ACEITA de proposito -- ela ainda ajuda
    na distorcao, o que reprova o conjunto e' o portao do modelo).
    """
    TESTE_MAU.unlink(missing_ok=True)
    rng = np.random.default_rng(73)
    entrelacada = _cena_grade(z=0.80, colunas=(535, 805, 1075, 1345),
                              linhas=(405, 675, 945))
    _, cantos, _ = observacoes(_cena_grade(z=0.45) + entrelacada, rng, sigma=0.2)
    cena = _cena_aceita(cantos, 40)
    mediana, ok, total = fk.resumo_do_span(cena, KINECT_SIZE)
    assert total >= 20, "a cena nao fechou poses suficientes: %d" % total
    assert mediana < sc.KINECT_INTRINSICS_MIN_SPAN, (mediana, ok, total)
    assert 0 < ok < total, "faltam poses boas OU frageis na cena: %d/%d" % (ok, total)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        gravou, rms, motivo = _roda_coleta(cena, TESTE_MAU, limite_s=1.0)
    console = buffer.getvalue()
    assert not gravou and rms is None, (gravou, rms, motivo)
    assert not TESTE_MAU.exists(), "a recusa criou arquivo"
    #: O AVISO da pose fraca tem de sair NO CONSOLE, na hora: a bancada so'
    #: descobriu o problema no fim, e sem o numero.
    assert "ATENCAO" in console and "MEDIANO" in console, console[-1500:]
    assert "%.1f%%" % (mediana * 100.0) in console, console[-1500:]
    #: O motivo final e' o do PORTAO DO MODELO com o mediano -- nao o ultimo
    #: estado do frame ("tabuleiro nao visto no Kinect"), que foi o que a
    #: bancada leu como causa de nao ter gravado.
    assert "span mediano" in motivo, motivo
    assert "%.1f%%" % (mediana * 100.0) in motivo, (mediana, motivo)
    assert "tabuleiro nao visto" not in motivo, motivo
    print("FakeKinect com a MEDIANA derrubada: %d poses (mediano %.1f%%, %d/%d "
          "cumprem a meta) -> gravou=%s, arquivo nao criado | console avisou a "
          "pose fraca e o motivo final veio do PORTAO DO MODELO: %s"
          % (total, mediana * 100.0, ok, total, gravou, motivo[:60]))


def _descarte_recupera():
    """10) UMA pose borrada nao pode condenar as outras: o RMS x o conjunto.

    MEDIDO na bancada (05/10/2026, 2a tentativa): 16 poses, RMS 1,77 px, nada
    gravado -- e o programa nao dizia QUAL pose estava estragando o conjunto. A
    grade aqui recebe 2 poses com 8 px de ruido por canto (o que o detector
    devolve com a pose fora de foco/tremida) contra 0,2 px do resto: sem
    descartes a captura reprova citando as poses; com os descartes ligados o
    programa GRAVA sem elas, e o npz sai com o numero REAL de poses.
    """
    TESTE_MAU.unlink(missing_ok=True)
    rng = np.random.default_rng(79)
    _, kin, _ = observacoes(_cena_grade(), rng, sigma=0.2)
    cena = _cena_aceita(kin, 14)
    assert len(cena) >= 13, len(cena)
    ruido = np.random.default_rng(97)
    for k in (2, 9):
        cena[k] = cena[k] + ruido.normal(0.0, 8.0, np.asarray(cena[k]).shape)
    objetos = [sc.board_object_points().copy() for _ in cena]
    # --- o que o conjunto diz ANTES de qualquer descarte ---------------------
    ok, rms, _, _, _, _, erros, _ = fk.avalia_conjunto(objetos, cena,
                                                       KINECT_SIZE)
    assert not ok and erros is not None, (ok, rms)
    assert rms > sc.KINECT_INTRINSICS_MAX_RMS, rms
    assert set(fk.piores_poses(erros, 2)) == {2, 9}, (erros,)
    # --- descartes desligados: reprova, mas DIZ quais poses refazer ----------
    gravou, rms_0, motivo = fk.tenta_gravar(objetos, cena, KINECT_SIZE,
                                            str(TESTE_MAU), descartes_max=0)
    assert gravou is False and not TESTE_MAU.exists(), (gravou, motivo)
    assert rms_0 > sc.KINECT_INTRINSICS_MAX_RMS, rms_0
    assert "refaca SO'" in motivo and "#3" in motivo and "#10" in motivo, motivo
    # --- descartes ligados: grava SEM as duas poses ruins --------------------
    gravou, rms_ok, motivo = fk.tenta_gravar(objetos, cena, KINECT_SIZE,
                                            str(TESTE_MAU), descartes_max=4)
    assert gravou is True, motivo
    assert rms_ok <= sc.KINECT_INTRINSICS_MAX_RMS, rms_ok
    assert TESTE_MAU.exists(), "gravou e o arquivo NAO existe"
    matriz, _, info = sc.carrega_intrinsecas_kinect(TESTE_MAU)
    assert matriz is not None and info["poses"] == len(cena) - 2, info
    assert "SEM as poses" in motivo and "#3" in motivo and "#10" in motivo, motivo
    # --- e pelo LOOP inteiro (o caminho da bancada), com os descartes ligados --
    TESTE_MAU.unlink(missing_ok=True)
    assert len(cena) >= 14, ("a grade precisa de 14 poses para o descarte sobrar "
                             "12: %d" % len(cena))
    gravou_loop, rms_loop, motivo_loop = _roda_coleta(cena, TESTE_MAU, 3.0,
                                                      descartes=fk.DESCARTES_MAX)
    assert gravou_loop is True, motivo_loop
    assert TESTE_MAU.exists() and rms_loop <= sc.KINECT_INTRINSICS_MAX_RMS, \
        (rms_loop, motivo_loop)
    _, _, info_loop = sc.carrega_intrinsecas_kinect(TESTE_MAU)
    assert info_loop["poses"] >= sc.KINECT_INTRINSICS_MIN_POSES, info_loop
    print("RMS com pose borrada: %d poses -> %.3f px (reprovado) | sem as 2 "
          "piores: RMS %.3f px e %d poses gravadas | no LOOP: %d poses, RMS "
          "%.3f px | veredito e '#N' conferidos"
          % (len(cena), rms, rms_ok, info["poses"], info_loop["poses"], rms_loop))


def _coleta_com_fakekinect():
    """7+8+9+10) `coleta_poses` roda AO VIVO (sem hardware) nos quatro rumos."""
    try:
        _coleta_grava_de_verdade()
        _coleta_recusa_honestamente()
        _coleta_derruba_a_mediana()
        _descarte_recupera()
    finally:
        TESTE_MAU.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())

