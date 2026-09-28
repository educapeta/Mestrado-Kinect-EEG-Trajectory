"""Teste do suporte a MULTIPLAS webcams auxiliares (Kinect + laptop + USB).

Sem hardware: cria arquivos de calibracao sinteticos para a camera auxiliar
de indice 1 (stereo_calibration_aux1.npz + hand_landmark_calibration_aux1.npz),
monkeypatcheia os resolvers de arquivo do modulo e verifica:

1. CameraAlignment registra as cameras 0 e 1 em aux_models.
2. aux_ready/aux_usable respondem por indice (0 usa o arquivo legado real).
3. set_active_aux troca os campos vivos (matriz/rotação/landmarks) e o
   recalculo de runtime por camera fica guardado por indice.
4. Indice nao registrado -> KeyError.
"""
import sys
from pathlib import Path

import numpy as np

import kinect_imu_groundtruth as gt

TMP_DIR = Path(__file__).with_name("_aux_test")
TMP_DIR.mkdir(exist_ok=True)
STF = TMP_DIR / "stereo_calibration_aux1.npz"
HCF = TMP_DIR / "hand_landmark_calibration_aux1.npz"
#: Stereo da camera 5 = o MESMO par Kinect x auxiliar, mas gravado na placa
#: ANTERIOR (quadrado de 52 mm). Serve para testar o motivo da recusa por
#: incompatibilidade de placa (caso 5b).
STF_ANTIGA = TMP_DIR / "stereo_calibration_aux5.npz"

# --- arquivos sinteticos da camera 1 ---
matrix = np.array([[900.0, 0.0, 630.0], [0.0, 895.0, 360.0],
                   [0.0, 0.0, 1.0]])
np.savez(
    STF,
    kinect_matrix=np.eye(3) * 1000.0 + np.array([[0, 0, 959], [0, 0, 539],
                                                 [0, 0, 1]]) * 0.0,
    kinect_dist=np.zeros(5),
    auxiliary_matrix=matrix,
    auxiliary_dist=np.zeros(5),
    rotation=np.eye(3),
    translation=np.array([-0.22, 0.0, 0.01]),
    # Derivado das constantes do modulo (nao hardcoded): se o tabuleiro mudar,
    # o teste acompanha sozinho e nao fica validando um stereo obsoleto.
    checkerboard_size=np.array(gt.CHECKERBOARD_SIZE),
    square_size_m=gt.SQUARE_SIZE_M,
    rms=0.4,
    auxiliary_size=np.array([1280, 720]),
    kinect_size=np.array([1920, 1080]),
)
np.savez(
    HCF,
    version=gt.HAND_CALIBRATION_VERSION,
    homography=np.eye(3),
    residuals=np.zeros((21, 2)),
    rms=0.01,
    scale_depth_params=np.array([1.5]),
    reference_scale=0.3,
    node_z_offset=np.zeros(21),
)

original_stereo = gt.aux_stereo_file
original_hand = gt.aux_hand_file


def fake_stereo(index):
    if int(index) == 1:
        return STF
    if int(index) == 5:
        return STF_ANTIGA
    return original_stereo(int(index))


def fake_hand(index):
    if int(index) == 1:
        return HCF
    return original_hand(int(index))


gt.aux_stereo_file = fake_stereo
gt.aux_hand_file = fake_hand

try:
    # (0) O PADRAO nao pode trazer a webcam do LAPTOP: ela fica no lugar do
    # Kinect (sem baseline). Nesta maquina (23/09/2026) a do laptop e' o indice
    # 1 ("Integrated Camera"); o padrao e' (2, 0) = as DUAS externas.
    padrao = gt.CameraAlignment()
    indices_padrao = padrao.aux_indices()
    # Ordem = a pedida (a PRIMEIRA e' a camera ativa): 2 = externa da bancada,
    # que e' a vista que ve' as duas maos e tem stereo_calibration_aux2.npz.
    assert indices_padrao == [2, 0], indices_padrao
    assert 1 not in indices_padrao, \
        "a webcam do laptop nao deve entrar no par estereo"

    # (1) maquina POR INDICE: usa uma lista explicita (o arquivo sintetico e' do
    # indice 1), que e' o caso de quem fixa as cameras na linha de comando.
    alignment = gt.CameraAlignment(aux_indices=(1, 2))
    indices = alignment.aux_indices()
    assert indices == [1, 2], indices

    # 2) prontidão por indice
    assert alignment.aux_ready(1), "camera 1 deveria estar pronta"
    assert alignment.aux_usable(1), "camera 1 deveria ter stereo utilizavel"

    # 3) troca de ativa
    alignment.set_active_aux(1)
    assert alignment.active_aux == 1
    assert np.allclose(alignment.auxiliary_matrix, matrix)
    assert alignment.calibrated_auxiliary_size == (1280, 720)
    assert alignment.hand_landmark_homography is not None
    # omita o stereo para camara 0 (legado) nao existe? verifica so a chave
    # a troca para a SEGUNDA auxiliar tambem funciona, e volta para a primeira
    alignment.set_active_aux(2)
    assert alignment.active_aux == 2
    alignment.set_active_aux(1)
    assert alignment.active_aux == 1

    # runtime por camera: tamanho diferente da calibracao -> matriz escalada
    alignment.set_active_aux(1)
    small = np.zeros((400, 640, 3), np.uint8)      # 640x400 != 1280x720
    alignment.set_auxiliary_size(small)
    assert alignment.auxiliary_matrix_runtime is not None
    assert alignment.aux_models[1]["runtime_matrix"] is not None
    assert abs(alignment.auxiliary_matrix_runtime[0, 0]
               - matrix[0, 0] * (640 / 1280)) < 1e-6
    alignment.set_active_aux(1)                     # runtime restaurado
    assert alignment.auxiliary_matrix_runtime is not None

    # 4) indice invalido
    try:
        alignment.set_active_aux(99)
        raise AssertionError("indice 99 nao registrado deveria falhar")
    except KeyError:
        pass

    # 5) lista personalizada explicita: reabilitar a webcam do LAPTOP junto com
    # a USB (montagem antiga, em que ela tambem entrava no par)
    custom = gt.CameraAlignment(aux_indices=(0, 1))
    assert custom.aux_indices() == [0, 1], custom.aux_indices()
    assert custom.aux_ready(1), "camera 1 deveria carregar do arquivo _aux1"
    custom.set_active_aux(0)
    assert custom.active_aux == 0

    # 5b) NPZ DE PLACA DIFERENTE: o motivo da recusa tem de ser dito por extenso.
    # Este e' o caso real de 23/09/2026: a placa foi trocada (6x4 quadrados de
    # 57 mm) e os npz das auxiliares foram gravados na placa anterior (52 mm).
    # Antes, a unica mensagem era "RMS alto", o que mandaria o usuario melhorar
    # luz/postura quando a acao e' RECALIBRAR -- e a escala do que estiver salvo
    # fica ~9% errada, mesmo com a sobreposicao ligada.
    antiga = STF_ANTIGA
    np.savez(
        antiga,
        kinect_matrix=np.eye(3) * 1000.0,
        kinect_dist=np.zeros(5),
        auxiliary_matrix=matrix,
        auxiliary_dist=np.zeros(5),
        rotation=np.eye(3),
        translation=np.array([-0.22, 0.0, 0.01]),
        checkerboard_size=np.array([5, 3]),
        square_size_m=0.052,                 # quadrado da placa ANTIGA
        rms=0.4,                             # RMS BOM: a recusa nao e' de qualidade
        auxiliary_size=np.array([1280, 720]),
        kinect_size=np.array([1920, 1080]),
    )
    so_antiga = gt.CameraAlignment(aux_indices=(5,))
    estado, motivo = so_antiga.stereo_status(5)
    assert estado == "rejeitada", (estado, motivo)
    assert "RECALIBRE" in motivo, motivo
    assert "52" in motivo and "57" in motivo, motivo
    assert not so_antiga.aux_ready(5), "placa diferente nao pode passar por pronta"
    assert so_antiga.aux_usable(5), (
        "o overlay segue desenhando (stereo_usable) -- o que muda e' a escala")
    # e a placa do proprio arquivo e' o que denuncia: nada de "RMS alto"
    assert "RMS" not in motivo, motivo
    antiga.unlink(missing_ok=True)

    # 5c) DESTINO da calibracao da MAO por camera (tecla H). Com a camera ativa em
    # 9 o arquivo tem de ser hand_landmark_calibration_aux9.npz; gravar no NOME
    # LEGADO (o bug antigo) fazia a captura nunca ser carregada -- `aux_hand_file
    # (9)` procura _aux9.npz e a tecla H parecia nao fazer nada.
    original_hand_output = gt.aux_hand_output_file
    original_hand_legado = gt.HAND_CALIBRATION_FILE
    destino_mao = TMP_DIR / "hand_landmark_calibration_aux9.npz"
    legado_em_tmp = TMP_DIR / "hand_landmark_calibration.npz"
    gt.HAND_CALIBRATION_FILE = legado_em_tmp
    gt.aux_hand_output_file = lambda indice: (
        destino_mao if int(indice) == 9 else original_hand_output(int(indice)))
    gt.aux_hand_file = lambda indice: (
        (destino_mao if destino_mao.exists() else None)
        if int(indice) == 9 else fake_hand(indice))
    try:
        alinhamento_mao = gt.CameraAlignment(aux_indices=(9,))
        assert alinhamento_mao.active_aux == 9, alinhamento_mao.active_aux
        assert not alinhamento_mao.hand_landmark_ready, "ja' havia calibracao"
        rng = np.random.default_rng(3)
        base = rng.random((21, 2))
        amostras_aux = [base + rng.normal(0.0, 0.01, (21, 2))
                        for _ in range(8)]
        amostras_kin = [s * 1.10 + np.array([0.02, -0.01]) for s in amostras_aux]
        assert alinhamento_mao.solve_hand_landmark_calibration(
            amostras_aux, amostras_kin, verbose=False), "a homografia falhou"
        assert destino_mao.exists(), "a calibracao da mao nao foi para _aux9"
        assert not legado_em_tmp.exists(), (
            "gravou no nome LEGADO com a camera ativa em 9 -- a captura da "
            "tecla H se perde")
        recarregado = gt.CameraAlignment(aux_indices=(9,))
        assert recarregado.hand_landmark_ready, "nao foi carregada de volta"
    finally:
        gt.aux_hand_output_file = original_hand_output
        gt.HAND_CALIBRATION_FILE = original_hand_legado
        gt.aux_hand_file = fake_hand
        destino_mao.unlink(missing_ok=True)
        legado_em_tmp.unlink(missing_ok=True)

    # 6) escolha POR NOME (--aux-cameras auto): injeta uma enumeracao do Windows
    # sem hardware. O mapa injetado tem a webcam do laptop no indice 1 e uma
    # externa SEM calibracao no indice 3 -- ela deve ficar por ULTIMO (a primeira
    # da lista e' a camera ativa e precisa ter stereo).
    assert gt.camera_names is not None, "camera_names deveria estar importado"
    gt.camera_names._CACHE = ["SIGMA-W780M", "Integrated Camera",
                              "Trust USB Camera", "SIGMA-W780M"]
    automaticos = gt.indices_auxiliares_automaticos()
    assert automaticos == (0, 2, 3), automaticos
    assert 1 not in automaticos, "a webcam do laptop ficou na lista auto"
    assert gt.nome_auxiliar(1) == "Integrated Camera", gt.nome_auxiliar(1)
    assert gt.aviso_camera_do_laptop(1) is not None
    assert gt.aviso_camera_do_laptop(0) is None
    gt.camera_names._CACHE = None          # devolve a enumeracao real ao modulo

    print("CAMERAS_MULTI_OK: 8/8 (padrao = as DUAS externas (2, 0), SEM a do "
          "laptop; lista personalizada, escolha por NOME, o MOTIVO da recusa de "
          "calibracao de placa diferente e o DESTINO por camera da calibracao "
          "da mao)")
finally:
    gt.aux_stereo_file = original_stereo
    gt.aux_hand_file = original_hand
    STF.unlink(missing_ok=True)
    HCF.unlink(missing_ok=True)
    STF_ANTIGA.unlink(missing_ok=True)
    TMP_DIR.rmdir()
sys.exit(0)