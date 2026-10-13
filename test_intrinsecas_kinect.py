"""Prova sintetica de que CONGELAR o modelo do Kinect melhora o overlay.

Contexto (medido na bancada em 30/09/2026): o 6x4 x 57 mm ocupa ~10-20% da
diagonal do quadro do Kinect, entao o modelo do Kinect estimado por sessao
(fx/fy/cx/cy + k1/k2/k3) sai DIFERENTE a cada sessao -- tres arquivos da MESMA
bancada divergiam de 24 a 50 px na projecao do mesmo ponto 3D, e a tecla L media
bias constante (+32,+26 px) crescendo com a excentricidade. O RMS de cada arquivo
continuava otimo (0,44-0,80 px) porque ele so' mede a coerencia INTERNA do npz.

O que degenerava a bancada de 30/09 era o tabuleiro PEQUENO no quadro (10-20% da
diagonal). A cena deste teste e' um FIXTURE daquela medicao -- 5x3 cantos @ 57 mm,
ver CHECKERBOARD_CENA -- e nao segue a placa impressa: MEDIDO em 01/10/2026, com a
placa de hoje (7x5 @ 45 mm, 35 cantos) na MESMA fracao de quadro, a divergencia do
modelo por sessao cai de 14,4 para 1,5 px, porque o que condiciona o modelo nao e'
so' o span: e' tambem o NUMERO de cantos. A ponte com a placa de hoje e' a guarda
de MIGRACAO do fim do arquivo (e nada aqui depende dela para rodar).

Este teste reproduz aquela bancada (mesma fracao de quadro, mesma geometria de
auxiliar e ruido de pixel) e resolve das DUAS maneiras:

  1. `run_stereo(refine_intrinsics=True)` -- comportamento antigo: o
     stereoCalibrate re-estima TAMBEM o Kinect;
  2. `run_stereo(ref_kinect=(K, D) verdadeiros)` -- novo: o modelo do Kinect vem
     CONGELADO e o stereo resolve so' R/T.

Afirma que (2) tem R/T mais perto da verdade e -- o que o usuario ve -- menor
erro em PIXEL da projecao auxiliar -> RGB do Kinect, medido contra o pixel REAL
do ponto na imagem do Kinect. Afirma tambem que o modelo estimado na sessao
divergia do verdadeiro (o erro que o caminho antigo levava para o overlay) e faz
o round-trip salva/carrega do modelo congelado: AVISO por placa diferente (a
placa NAO recusa mais -- as intrinsecas sao da CAMERA e trocar o tabuleiro
impresso nao muda fx/fy/cx/cy) e RECUSA pelo que decide de verdade -- cobertura
(span), poses de menos e RMS alto.

    .venv\\Scripts\\python.exe test_intrinsecas_kinect.py
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import stereo_calibration as sc

KINECT_SIZE = (1920, 1080)
AUX_SIZE = (1280, 720)
#: Intrinsecas VERDADEIRAS do Kinect RGB 1080p (o que o arquivo congelado tem de
#: conter): valores conhecidos do modo 1920x1080 do Kinect v2.
K1_GT = np.array([[1081.37, 0.0, 959.5], [0.0, 1081.37, 539.7], [0.0, 0.0, 1.0]])
D1_GT = np.array([-0.104, 0.062, 0.0005, -0.0005, 0.0])
#: Auxiliar da bancada: 1280x720. Distorcao MODERADA de webcam (barrilete leve):
#: NAO usamos aqui a distorcao gravada no npz real da Trust (k2=-1.61, k3=+2.53):
#: aquele modelo nao e' monotonico e `projectPoints` explode nas bordas do quadro
#: (valores de milhares de px) -- e' overfit da MESMA familia de problema que este
#: teste denuncia no Kinect, mas do lado da auxiliar.
K2_GT = np.array([[998.5, 0.0, 649.8], [0.0, 994.9, 322.4], [0.0, 0.0, 1.0]])
D2_GT = np.array([-0.28, 0.09, 0.001, -0.001, 0.0])
#: Posicao FISICA da auxiliar no referencial do Kinect (m): lado direito, 0,64 m
#: em X, 0,80 m do Kinect -- a geometria da bancada real.
C_GT = np.array([0.642, -0.098, 0.459])
POSES = 13
MODELO_TESTE = Path(__file__).with_name("_kinect_intrinsics_test.npz")
# --- A PLACA DA CENA E' UM FIXTURE (a da medicao de 30/09/2026) --------------
# O tabuleiro da cena e' o 5x3 cantos @ 57 mm medido em 30/09, fixado aqui mesmo
# quando a placa impressa troca. Motivo MEDIDO em 01/10/2026, com a placa de hoje
# (7x5 @ 45 mm, 1,27x maior): mantendo a MESMA fracao de quadro (11-12%, o span da
# medicao) a armadilha quase desaparece -- o modelo do Kinect estimado na sessao
# fica 1,5 px do verdadeiro em vez de 14,4 px -- porque o que condiciona o modelo
# nao e' so' o span, e' tambem o NUMERO DE CANTOS (35 contra 15: 70 equacoes
# contra 30, com o mesmo ruido de 0,5 px por canto). Ou seja: a cena reproduz a
# MEDICAO (uma bancada fraca), nao a placa impressa -- se o fixture fosse trocado
# pela placa de hoje, este teste passaria a medir quase nada.
# A ponte com a placa de HOJE e' a guarda de MIGRACAO do fim do arquivo, que
# compara a placa declarada nos dois modulos.
CHECKERBOARD_CENA = (5, 3)
SQUARE_SIZE_CENA_M = 0.057
#: Placa REAL (a que `stereo_calibration` declara hoje) -- guardada antes de
#: ajustar as globais para o fixture, porque `canonical_order`/`symmetry_indices`/
#: `board_object_points` leem exatamente essas globais.
PLACA_REAL = (tuple(sc.CHECKERBOARD_SIZE), float(sc.SQUARE_SIZE_M))
sc.CHECKERBOARD_COLS, sc.CHECKERBOARD_ROWS = CHECKERBOARD_CENA
sc.CHECKERBOARD_SIZE = CHECKERBOARD_CENA
sc.SQUARE_SIZE_M = SQUARE_SIZE_CENA_M


def rotacao(eixo, graus):
    """Rodrigues: matriz 3x3 de `graus` em torno de `eixo`."""
    eixo = np.asarray(eixo, np.float64)
    return cv2.Rodrigues(np.radians(graus) * eixo / np.linalg.norm(eixo))[0]


def olhando_para(direcao, cima=(0.0, 1.0, 0.0)):
    """R de uma camera posta em T_GT que aponta o eixo Z para `direcao`.

    X_aux = R X_kin + T => as colunas de R^T sao os eixos da auxiliar no
    referencial do Kinect. Reproduz o que a bancada faz com a webcam: apontada
    para o volume onde a mao trabalha.
    """
    z = np.asarray(direcao, np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross(np.asarray(cima, np.float64), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]).T


#: A auxiliar olha para o centro do volume de trabalho (0, 0, 0.85 m).
R_GT = olhando_para(np.array([0.0, 0.0, 0.85]) - C_GT)
#: ARMADILHA da convencao do projeto: X_aux = R X_kin + T, entao a POSICAO da
#: camera no referencial do Kinect e' -R^T T, nao T. Quem posiciona a camera pelo
#: T (como eu tentei primeiro) monta uma geometria que NAO ve o volume de
#: trabalho e o gerador de poses nao acha uma pose sequer.
T_GT = -R_GT @ C_GT


def _visivel(pontos, tamanho, margem=10.0):
    """Cantos dentro do quadro com margem E tabuleiro grande o bastante."""
    largura, altura = tamanho
    xy = np.asarray(pontos, np.float64).reshape(-1, 2)
    return bool(
        np.all(xy[:, 0] > margem) and np.all(xy[:, 0] < largura - margem)
        and np.all(xy[:, 1] > margem) and np.all(xy[:, 1] < altura - margem)
        and sc.board_span(xy) >= sc.MIN_BOARD_SPAN_FRACTION * np.hypot(largura,
                                                                      altura))


def gera_poses(quantidade, rng):
    """Poses do tabuleiro visiveis nas DUAS cameras, na regiao de trabalho."""
    tabuleiro = sc.board_object_points()
    poses = []
    tentativas = 0
    while len(poses) < quantidade and tentativas < 4000:
        tentativas += 1
        centro = np.array([rng.uniform(-0.20, 0.20), rng.uniform(-0.25, 0.25),
                           rng.uniform(0.75, 1.45)])
        giro = (rotacao((1, 0, 0), rng.uniform(-25, 25))
                @ rotacao((0, 1, 0), rng.uniform(-25, 25))
                @ rotacao((0, 0, 1), rng.uniform(0, 360)))
        rvec, _ = cv2.Rodrigues(giro)
        tvec = centro.reshape(3, 1)
        px_kinect = cv2.projectPoints(tabuleiro, rvec, tvec, K1_GT, D1_GT)[0]
        rvec_aux, _ = cv2.Rodrigues(R_GT @ giro)
        tvec_aux = (R_GT @ tvec).reshape(3, 1) + T_GT.reshape(3, 1)
        px_aux = cv2.projectPoints(tabuleiro, rvec_aux, tvec_aux, K2_GT,
                                   D2_GT)[0]
        if not _visivel(px_kinect, KINECT_SIZE):
            continue
        if not _visivel(px_aux, AUX_SIZE):
            continue
        poses.append((rvec, tvec, rvec_aux, tvec_aux))
    if len(poses) < quantidade:
        raise RuntimeError(f"gerou apenas {len(poses)} poses validas")
    return poses


def observacoes(poses, rng):
    """(obj, kin, aux) como o detector os entrega: ordem canonica + ruido."""
    tabuleiro = sc.board_object_points()
    obj, kin, aux = [], [], []
    for rvec, tvec, rvec_aux, tvec_aux in poses:
        px_kinect = cv2.projectPoints(tabuleiro, rvec, tvec, K1_GT, D1_GT)[0]
        px_aux = cv2.projectPoints(tabuleiro, rvec_aux, tvec_aux, K2_GT,
                                   D2_GT)[0]
        px_kinect = sc.canonical_order(px_kinect.reshape(-1, 2))[1]
        px_aux = sc.canonical_order(px_aux.reshape(-1, 2))[1]
        # 0,5 px no Kinect (tabuleiro pequeno no quadro: e' o ruido medido na
        # bancada) e 0,3 px na auxiliar.
        px_kinect = px_kinect + rng.normal(0.0, 0.5, px_kinect.shape)
        px_aux = px_aux + rng.normal(0.0, 0.3, px_aux.shape)
        obj.append(tabuleiro.copy())
        kin.append(px_kinect.reshape(-1, 1, 2))
        aux.append(px_aux.reshape(-1, 1, 2))
    return obj, kin, aux


def erro_de_pose(final):
    """(graus, mm) de erro de R/T contra a verdade + desvio do modelo do Kinect."""
    R_est = np.asarray(final["rotation"], np.float64).reshape(3, 3)
    T_est = np.asarray(final["translation"], np.float64).reshape(3)
    cosseno = (np.trace(R_GT.T @ R_est) - 1.0) / 2.0
    angulo = float(np.degrees(np.arccos(np.clip(cosseno, -1.0, 1.0))))
    translacao = float(np.linalg.norm(T_GT.ravel() - T_est) * 1000.0)
    desvio = sc.compara_modelos_kinect(final["kinect_matrix"],
                                       final["kinect_dist"], K1_GT, D1_GT)
    return angulo, translacao, desvio


def projeta_pela_profundidade(final, px_aux, target_z):
    """Equivalente em numpy de `AuxCameraView.to_kinect_stereo` (um ponto).

    E' a matematica do overlay: pixel da auxiliar -> raio -> corte no plano Z ->
    reprojecao no RGB do Kinect. Reescrita aqui so' com o que o npz guarda
    (matrizes/distorcoes/R/T), como test_stereo_solve.py faz com a sua.
    """
    raio = cv2.undistortPoints(
        np.asarray(px_aux, np.float64).reshape(1, 1, 2).astype(np.float32),
        np.asarray(final["auxiliary_matrix"], np.float64),
        np.asarray(final["auxiliary_dist"], np.float64),
    ).reshape(2)
    R = np.asarray(final["rotation"], np.float64).reshape(3, 3)
    T = np.asarray(final["translation"], np.float64).reshape(3)
    direcao = R.T @ np.array([raio[0], raio[1], 1.0])
    origem = -R.T @ T
    escala = (float(target_z) - origem[2]) / direcao[2]
    ponto = R.T @ (escala * np.array([raio[0], raio[1], 1.0]) - T)
    px, _ = cv2.projectPoints(
        ponto.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
        np.asarray(final["kinect_matrix"], np.float64),
        np.asarray(final["kinect_dist"], np.float64))
    return px.reshape(2)


def erro_do_overlay(final):
    """(media, max, n) px do que o esqueleto auxiliar erra NA IMAGEM do Kinect.

    Mede exatamente o que o usuario ve: o ponto 3D verdadeiro (onde a mao esta)
    projetado na imagem do Kinect pelo modelo VERDADEIRO contra o mesmo ponto
    reconstruido pelo caminho da auxiliar com o modelo estimado.
    """
    pontos = []
    for profundidade in (0.70, 0.85, 1.05):
        for deslocamento in (-0.25, 0.0, 0.25):
            pontos.append(np.array([deslocamento, -0.05, profundidade]))
    erros = []
    for ponto in pontos:
        ponto_aux = R_GT @ ponto + T_GT
        px_aux = cv2.projectPoints(
            ponto_aux.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K2_GT,
            D2_GT)[0].reshape(2)
        if not (0 < px_aux[0] < AUX_SIZE[0] and 0 < px_aux[1] < AUX_SIZE[1]):
            continue
        desenhado = projeta_pela_profundidade(final, px_aux, ponto[2])
        real = cv2.projectPoints(
            ponto.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K1_GT,
            D1_GT)[0].reshape(2)
        erros.append(float(np.linalg.norm(desenhado - real)))
    return float(np.mean(erros)), float(np.max(erros)), len(erros)


def main():
    rng = np.random.default_rng(7)
    poses = gera_poses(POSES, rng)
    obj, kin, aux = observacoes(poses, rng)
    span = float(np.median([sc.board_span(np.asarray(pontos).reshape(-1, 2))
                            for pontos in kin])) / np.hypot(*KINECT_SIZE)
    print(f"Poses: {len(poses)} | tabuleiro da cena {CHECKERBOARD_CENA[0]}x"
          f"{CHECKERBOARD_CENA[1]} @ {SQUARE_SIZE_CENA_M * 1000:g} mm (o da "
          f"medicao de 30/09) a {span:.0%} da diagonal do quadro do Kinect "
          "(a fraqueza da bancada real)"
          + ("" if (CHECKERBOARD_CENA == PLACA_REAL[0]
                    and abs(SQUARE_SIZE_CENA_M - PLACA_REAL[1]) < 1e-9) else
             f" | placa impressa hoje: {PLACA_REAL[0][0]}x{PLACA_REAL[0][1]} @ "
             f"{PLACA_REAL[1] * 1000:g} mm"))

    # Os DOIS caminhos passam pelo pipeline completo (solve_with_repair), que e'
    # quem corrige a ordem dos cantos: chamar run_stereo direto compararia as
    # duas solucoes COM o bug de correspondencia que aquele solver conserta.
    antigo = sc.solve_with_repair(obj, kin, aux, KINECT_SIZE, AUX_SIZE,
                                  verbose=False, ref_kinect=None)[0]
    novo = sc.solve_with_repair(obj, kin, aux, KINECT_SIZE, AUX_SIZE,
                                verbose=False,
                                ref_kinect=(K1_GT, D1_GT))[0]

    resultado = {}
    for nome, final in (("antigo (sem congelar)", antigo),
                        ("CONGELADO (arquivo)", novo)):
        angulo, milimetros, desvio = erro_de_pose(final)
        media, pior, amostras = erro_do_overlay(final)
        resultado[nome] = {"final": final, "angulo": angulo, "mm": milimetros,
                           "desvio": desvio, "media": media, "pior": pior}
        print("%-22s | RMS %6.3f px | R %5.2f graus | T %6.1f mm | modelo do "
              "Kinect %5.1f px fora do verdadeiro | OVERLAY media %6.1f px "
              "(pior %6.1f, n=%d)"
              % (nome, final["stereo_rms"], angulo, milimetros, desvio[0], media,
                 pior, amostras))

    # --- o que o teste AFIRMA ---------------------------------------------
    antigo, novo = resultado["antigo (sem congelar)"], resultado["CONGELADO (arquivo)"]
    # 1) o modo congelado usa MESMO o modelo do arquivo e nao o estima
    assert novo["final"]["kinect_frozen"] is True, "modo congelado nao foi usado"
    assert antigo["final"]["kinect_frozen"] is False
    assert np.allclose(novo["final"]["kinect_matrix"], K1_GT)
    assert np.allclose(novo["final"]["kinect_dist"].reshape(-1), D1_GT)
    assert novo["desvio"][0] == 0.0, novo["desvio"]
    # 2) a armadilha existe TAMBEM nos dados sinteticos: sem congelar, o modelo
    #    do Kinect da sessao sai errado (na bancada real foram 24-50 px)
    assert antigo["desvio"][0] > 5.0, (
        "o modelo do Kinect da sessao deveria divergir do verdadeiro: %.1f px"
        % antigo["desvio"][0])
    # 3) o que o USUARIO ve: o overlay do modo congelado e' bem menor. Repare que
    #    o RMS e o proprio R/T do modo antigo podem sair MELHORES -- o modelo
    #    errado absorve erro de R/T; quem paga e' o pixel na tela (por isso o RMS
    #    nao serve para julgar isto).
    assert novo["media"] < 0.6 * antigo["media"], (
        "overlay congelado %.1f px vs antigo %.1f px"
        % (novo["media"], antigo["media"]))
    assert novo["media"] < 8.0, "overlay congelado alto: %.1f px" % novo["media"]
    assert novo["angulo"] < 2.0 and novo["mm"] < 20.0, (novo["angulo"], novo["mm"])

    # --- round-trip do arquivo do modelo congelado ------------------------
    sc.KINECT_INTRINSICS_FILE = MODELO_TESTE
    try:
        sc.salva_intrinsecas_kinect(K1_GT, D1_GT, KINECT_SIZE, 0.31, 12, 0.42)
        matriz, distorcao, info = sc.carrega_intrinsecas_kinect()
        assert np.allclose(matriz, K1_GT) and np.allclose(distorcao, D1_GT)
        assert abs(info["rms"] - 0.31) < 1e-9 and info["poses"] == 12
        assert info["aviso_placa"] == "", info["aviso_placa"]
        # npz gravado com OUTRA PLACA e' ACEITO, com AVISO: as intrinsecas sao
        # propriedade da CAMERA e nao mudam quando a placa impressa troca.
        # (Ate' 01/10/2026 isto era RECUSA, e a recusa transformava uma troca de
        # placa -- 5x3/57 mm -> 7x5/45 mm -- em "toda sessao volta a re-estimar o
        # Kinect": MEDIDO, dois npz da MESMA bancada a 68,8 px um do outro.)
        np.savez(MODELO_TESTE, kinect_matrix=K1_GT, kinect_dist=D1_GT,
                 checkerboard_size=np.array([11, 7]), square_size_m=0.026,
                 poses=12, span=0.42, rms=0.31)
        matriz_outra, _, info_outra = sc.carrega_intrinsecas_kinect()
        assert matriz_outra is not None, info_outra
        assert "11x7" in info_outra["aviso_placa"], info_outra["aviso_placa"]
        assert "26 mm" in info_outra["aviso_placa"], info_outra["aviso_placa"]
        assert "segue valendo" in info_outra["aviso_placa"], info_outra["aviso_placa"]
        # --- e quem REPROVA de verdade e' a COBERTURA (o span) ---------------
        # 1) arquivo gravado por um caminho que nao mede a cobertura: e' a CARA do
        #    paliativo de 30/09/2026, que deixou a bancada sem modelo nenhum.
        np.savez(MODELO_TESTE, kinect_matrix=K1_GT, kinect_dist=D1_GT,
                 checkerboard_size=np.array([11, 7]), square_size_m=0.026,
                 rms=0.31)
        sem_span, _, info_sem_span = sc.carrega_intrinsecas_kinect()
        assert sem_span is None and "span" in info_sem_span["motivo"], info_sem_span
        # 2) span abaixo do portao: recusa dizendo o numero MEDIDO
        np.savez(MODELO_TESTE, kinect_matrix=K1_GT, kinect_dist=D1_GT,
                 poses=12, span=0.12, rms=0.31)
        fraco, _, info_fraco = sc.carrega_intrinsecas_kinect()
        assert fraco is None and "12%" in info_fraco["motivo"], info_fraco
        # 3) poses de menos: idem
        np.savez(MODELO_TESTE, kinect_matrix=K1_GT, kinect_dist=D1_GT,
                 poses=6, span=0.42, rms=0.31)
        poucas, _, info_poucas = sc.carrega_intrinsecas_kinect()
        assert poucas is None and "6 poses" in info_poucas["motivo"], info_poucas
        # 4) RMS acima do portao: idem
        np.savez(MODELO_TESTE, kinect_matrix=K1_GT, kinect_dist=D1_GT,
                 poses=12, span=0.42, rms=1.9)
        rms_alto, _, info_rms = sc.carrega_intrinsecas_kinect()
        assert rms_alto is None and "RMS" in info_rms["motivo"], info_rms
        # 5) e sem arquivo nenhum a recusa tem de ser limpa (nunca levantar)
        np.savez(MODELO_TESTE, outra_coisa=1.0)
        matriz_ruim, _, info_ruim = sc.carrega_intrinsecas_kinect()
        assert matriz_ruim is None and "kinect_matrix" in info_ruim["motivo"]
    finally:
        MODELO_TESTE.unlink(missing_ok=True)

    # --- guarda de MIGRACAO: a placa do tabuleiro tem de casar nos dois modulos -
    # MEDIDO em 01/10/2026: `stereo_calibration` ja' calibrava com a placa nova
    # (7x5 @ 45 mm) enquanto `kinect_imu_groundtruth` ainda validava 5x3 @ 57 mm.
    # Sintoma: os DOIS npz recem-gravados voltavam `rejeitada` no `stereo_status`
    # ("tabuleiro gravado 7x5 != 5x3 cantos da placa atual") e o
    # `kinect_intrinsics.npz` deixou de ser aceito -- a sessao voltou a RE-ESTIMAR
    # o Kinect, que e' o bug que o modelo congelado existe para matar. A duplicata
    # das constantes e' deliberada (importar `stereo_calibration` aqui dentro da
    # biblioteca traria pykinect2 para quem so' quer o Kinect); a trava contra a
    # divergencia e' esta. Import local: so' esta guarda precisa da biblioteca.
    import kinect_imu_groundtruth as gt
    assert tuple(gt.CHECKERBOARD_SIZE) == tuple(PLACA_REAL[0]), (
        "placa divergente entre os modulos: kinect_imu_groundtruth %s x "
        "stereo_calibration %s" % (gt.CHECKERBOARD_SIZE, PLACA_REAL[0]))
    assert abs(gt.SQUARE_SIZE_M - PLACA_REAL[1]) < 1e-9, (
        "quadrado divergente entre os modulos: %g mm x %g mm"
        % (gt.SQUARE_SIZE_M * 1000.0, PLACA_REAL[1] * 1000.0))
    # a declaracao da CENA tambem tem de ser coerente consigo mesma (COLS/ROWS x
    # CHECKERBOARD_SIZE), senao a grade da cena e a maquinaria de simetria divergem
    assert tuple(sc.CHECKERBOARD_SIZE) == (sc.CHECKERBOARD_COLS,
                                           sc.CHECKERBOARD_ROWS)
    # devolve a placa REAL ao `stereo_calibration` (o teste ajustou as globais para
    # o fixture da cena; restaurar evita contaminar quem importar depois)
    sc.CHECKERBOARD_COLS, sc.CHECKERBOARD_ROWS = PLACA_REAL[0]
    sc.CHECKERBOARD_SIZE = PLACA_REAL[0]
    sc.SQUARE_SIZE_M = PLACA_REAL[1]

    print("INTRINSECAS_KINECT_OK: 2 caminhos comparados | overlay congelado "
          "%.1f px (pior %.1f) contra %.1f px (pior %.1f) do caminho antigo, com "
          "modelo do Kinect %.1f px fora do verdadeiro; cena = fixture da medição "
          "de 30/09 (%dx%d @ %g mm) e placa REAL de hoje (%dx%d @ %g mm) casando "
          "entre kinect_imu_groundtruth e stereo_calibration"
          % (novo["media"], novo["pior"], antigo["media"], antigo["pior"],
             antigo["desvio"][0], CHECKERBOARD_CENA[0], CHECKERBOARD_CENA[1],
             SQUARE_SIZE_CENA_M * 1000, PLACA_REAL[0][0], PLACA_REAL[0][1],
             PLACA_REAL[1] * 1000))


if __name__ == "__main__":
    main()
    sys.exit(0)
