"""Teste do overlay MULTICAMERA no RGB do Kinect (sem hardware).

Objetivo do projeto (bancada, 23/09/2026): na imagem do Kinect tem de aparecer o
esqueleto do KINECT (verde) MAIS o de CADA webcam auxiliar (uma de cada lado,
~45 graus), cada um na sua cor. Sem isso nao se ve', num relance, se as duas
auxiliares enxergam a mao -- e sem as duas enxergando nao existe a substituicao
do Kinect pelo par de auxiliares.

Sintetico: duas vistas rigidas conhecidas, uma mao 3D conhecida, um frame preto
de 1920x1080 e a checagem das cores/posicoes desenhadas e do desvio medido
contra o esqueleto verde.

    .venv\\Scripts\\python.exe test_overlay_auxiliares.py
"""
import sys

import numpy as np

import kinect_imu_groundtruth as gt

LARGURA, ALTURA = 1920, 1080


class _LM:
    """Landmark minimo do MediaPipe (x e y normalizados)."""

    def __init__(self, x, y):
        self.x, self.y = x, y


def rotacao(x_deg=0.0, y_deg=0.0, z_deg=0.0):
    ax, ay, az = np.deg2rad([x_deg, y_deg, z_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def vista(indice, rot, trans, tamanho=(1280, 720), focal=900.0):
    matriz = np.array([[focal, 0.0, tamanho[0] / 2.0],
                       [0.0, focal, tamanho[1] / 2.0],
                       [0.0, 0.0, 1.0]])
    return gt.AuxCameraView(
        index=indice,
        kinect_matrix=np.array([[1081.37, 0.0, 959.5],
                                [0.0, 1081.37, 539.5],
                                [0.0, 0.0, 1.0]]),
        kinect_dist=np.zeros(5),
        matrix=matriz,
        dist=np.zeros(5),
        rotation=rot,
        translation=trans.reshape(3, 1),
        rms=0.9,
        stereo_usable=True,
        runtime_size=tamanho,
        calibrated_size=tamanho,
    )


def projeta(matriz, R, T, pontos):
    """Projeta pontos 3D no referencial da camera (X_cam = R X + T)."""
    cam = (R @ np.asarray(pontos, np.float64).T).T + np.asarray(T).reshape(1, 3)
    pixels = (np.asarray(matriz, np.float64) @ cam.T).T
    return pixels[:, :2] / pixels[:, 2:3]


def landmarks_normalizados(pixels, tamanho):
    largura, altura = tamanho
    return [_LM(x / largura, y / altura) for x, y in np.asarray(pixels)]


class _AlinhamentoFake:
    """Dobre do CameraAlignment so' com as vistas (sem arquivos)."""

    def __init__(self, vistas):
        self.vistas = {v.index: v for v in vistas}
        self.active_aux = vistas[0].index if vistas else None
        self.overlay_trim = np.zeros(2)

    def aux_view(self, indice):
        return self.vistas.get(int(indice))

    def active_view(self):
        # O tracado multicamera usa os CAMPOS VIVOS na camera ativa; no dobre os
        # dois caminhos apontam para a mesma vista deterministica.
        return self.vistas.get(int(self.active_aux))


def tracker_sem_kinect():
    """KinectHandTracker sem abrir o Kinect (so' os campos usados no desenho)."""
    tracker = gt.KinectHandTracker.__new__(gt.KinectHandTracker)
    tracker.color_width = LARGURA
    tracker.color_height = ALTURA
    tracker.last_kinect_hands = []
    return tracker


def main():
    rng = np.random.default_rng(5)
    esquerda = vista(3, rotacao(x_deg=-6.0, y_deg=-42.0),
                     np.array([-0.58, 0.16, 0.12]))
    direita = vista(7, rotacao(x_deg=-6.0, y_deg=42.0),
                    np.array([0.58, 0.16, 0.12]))
    alinhamento = _AlinhamentoFake([esquerda, direita])
    tracker = tracker_sem_kinect()

    verdade = (np.array([0.02, -0.05, 1.05])
               + rng.normal(0.0, 0.05, size=(21, 3)))
    verdade[0] = [0.02, -0.05, 1.05]

    def mao_em(v, pontos):
        return landmarks_normalizados(
            projeta(v.matrix, np.asarray(v.rotation).reshape(3, 3),
                    np.asarray(v.translation).reshape(3), pontos),
            v.runtime_size)

    # Cada auxiliar enxerga a mao DO SEU LADO (bancada real): se as duas vissem a
    # MESMA mao, as duas projecoes cairiam no mesmo pixel do Kinect e uma
    # esconderia a outra -- o que tambem e' a prova de que a calibracao esta'
    # coerente entre elas.
    verdade_esq = verdade + np.array([-0.20, 0.0, 0.0])
    verdade_dir = verdade + np.array([0.20, 0.0, 0.0])
    maos = {esquerda.index: [mao_em(esquerda, verdade_esq)],
            direita.index: [mao_em(direita, verdade_dir)]}
    # O esqueleto VERDE (o que o MediaPipe do Kinect detectou) e' a referencia do
    # desvio: aqui ele e' exatamente a projecao da mao da ESQUERDA.
    px_verde_kinect = projeta(gt.KINECT_V2_INTRINSICS_1080P, np.eye(3),
                              np.zeros(3), verdade_esq)
    tracker.last_kinect_hands = [landmarks_normalizados(px_verde_kinect,
                                                        (LARGURA, ALTURA))]

    # --- 1) os DOIS esqueletos auxiliares sao desenhados, em cores diferentes -
    frame = np.zeros((ALTURA, LARGURA, 3), np.uint8)
    resultado = tracker.draw_auxiliary_cameras_on_color(
        frame, maos, alinhamento,
        triangulados={esquerda.index: verdade_esq,
                      direita.index: verdade_dir})
    assert set(resultado) == {esquerda.index, direita.index}, resultado.keys()
    for indice in resultado:
        assert np.isfinite(resultado[indice][3]).all(axis=1).sum() >= 19, (
            f"cam {indice} nao projetou os 21 nos")
    # A mao da ESQUERDA (idx 3) tem o verde por cima -> desvio medido e ~0;
    # a da direita esta' 0,4 m ao lado -> o esqueleto dela fica longe do verde.
    assert resultado[esquerda.index][0] is not None
    assert resultado[esquerda.index][0] < 1.0, resultado[esquerda.index][0]
    assert resultado[direita.index][0] is None or \
        resultado[direita.index][0] > 5.0

    cores = set(map(tuple, frame.reshape(-1, 3)))
    assert (0, 0, 255) in cores, "falta o esqueleto vermelho da primeira auxiliar"
    assert (255, 0, 255) in cores, "falta o esqueleto magenta da segunda auxiliar"

    # --- 2) com `triangulados` a projecao e' EXATA: desvio ~0 contra o verde --
    frame = np.zeros((ALTURA, LARGURA, 3), np.uint8)
    resultado = tracker.draw_auxiliary_cameras_on_color(
        frame, {esquerda.index: [mao_em(esquerda, verdade_esq)]}, alinhamento,
        triangulados={esquerda.index: verdade_esq})
    assert resultado[esquerda.index][0] < 1.0, resultado[esquerda.index][0]
    assert resultado[esquerda.index][4] is not None
    assert resultado[esquerda.index][4].shape == (21, 2)

    # --- 3) sem triangulacao cai no stereo com profundidade RIGIDA -----------
    frame = np.zeros((ALTURA, LARGURA, 3), np.uint8)
    resultado = tracker.draw_auxiliary_cameras_on_color(
        frame, {esquerda.index: [mao_em(esquerda, verdade_esq)]}, alinhamento,
        depths={esquerda.index: [1.05]})
    assert resultado[esquerda.index][0] is not None
    assert resultado[esquerda.index][0] > 0.5, (
        "com profundidade RIGIDA o esqueleto tem de ficar diferente do verde "
        "(os nos estao em Z diferente); desvio ~0 significaria profundidade por "
        "no, contra a regra de desenho")

    # --- 4) camera sem stereo utilizavel e' ignorada (nao desenha lixo) ------
    sem_stereo = gt.AuxCameraView(**{**esquerda.__dict__, "index": 9,
                                     "stereo_usable": False})
    alinhamento_quebrado = _AlinhamentoFake([sem_stereo, direita])
    frame = np.zeros((ALTURA, LARGURA, 3), np.uint8)
    resultado = tracker.draw_auxiliary_cameras_on_color(
        frame, {9: [mao_em(esquerda, verdade_esq)], direita.index: []},
        alinhamento_quebrado,
        triangulados={9: verdade_esq, direita.index: verdade_dir})
    assert 9 not in resultado and not frame.any(), resultado.keys()

    # --- 5) mao fora do volume: nada desenhado (gating de borda) -------------
    fora = verdade_esq + np.array([12.0, 0.0, 0.0])
    px_fora = projeta(esquerda.matrix,
                      np.asarray(esquerda.rotation).reshape(3, 3),
                      np.asarray(esquerda.translation).reshape(3), fora)
    frame = np.zeros((ALTURA, LARGURA, 3), np.uint8)
    resultado = tracker.draw_auxiliary_cameras_on_color(
        frame, {esquerda.index: [landmarks_normalizados(
            px_fora, esquerda.runtime_size)]},
        alinhamento, depths={esquerda.index: [1.05]})
    assert resultado == {}, "mao fora do quadro nao pode gerar overlay"
    assert not frame.any()

    # --- 6) prioridade da triangulacao sobre a profundidade informada --------
    px_tri = tracker._projetar_mao_no_kinect(esquerda,
                                             mao_em(esquerda, verdade_esq),
                                             [9.5], verdade_esq)
    esperado = projeta(gt.KINECT_V2_INTRINSICS_1080P, np.eye(3), np.zeros(3),
                       verdade_esq)
    assert np.allclose(px_tri, esperado, atol=1e-6), (
        "com `triangulado` a profundidade informada nao pode influir")

    # --- 7) helpers de borda/tracado sao compartilhados ----------------------
    mascara = gt.mascara_dentro_do_frame(
        np.array([[10.0, 10.0], [-1000.0, 5.0], [LARGURA, ALTURA]]),
        LARGURA, ALTURA)
    assert list(mascara) == [True, False, True], mascara
    vazio = np.zeros((ALTURA, LARGURA, 3), np.uint8)
    fora_do_frame = np.tile(np.array([[9000.0, 100.0]]), (21, 1))
    assert gt.desenhar_esqueleto_projetado(vazio, fora_do_frame).sum() == 0
    assert not vazio.any(), "nada pode ser desenhado fora do frame"
    dentro = gt.desenhar_esqueleto_projetado(vazio, px_verde_kinect)
    assert dentro.sum() >= 19, "os 21 nos estao dentro do frame"
    assert vazio.any(), "o tracado dentro do frame tem de aparecer"

    print("OVERLAY_AUXILIARES_OK: 7/7 (dois esqueletos auxiliares em cores "
          "distintas, desvio contra o verde, profundidade rigida, camera sem "
          "stereo ignorada, gating de borda e prioridade da triangulacao)")
    return 0


if __name__ == "__main__":
    sys.exit(main())