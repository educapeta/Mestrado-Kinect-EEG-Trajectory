"""Teste da triangulacao com APENAS AS DUAS webcams auxiliares (sem Kinect).

Objetivo do projeto (bancada, 23/09/2026): quando o Kinect perde o video e a
projecao do esqueleto da mao, as DUAS auxiliares (uma de cada lado, ~45 graus)
tem de dizer SOZINHAS onde a mao esta nos 3 eixos do espaco. A geometria entre
elas nao e' calibrada em separado: sai da composicao das duas calibracoes com o
Kinect (X_i = R_i X_kin + T_i) -- ver `relative_transform` no modulo.

Sintetico e sem hardware: duas vistas rigidas conhecidas (~45 graus, baseline de
~1,1 m), uma mao 3D conhecida projetada nas duas com ruido de deteccao de 0.4 px
(tipico do MediaPipe) e a checagem de que a triangulacao RECUPERA a mao no
referencial do KINECT com erro sub-centimetrico.

    .venv\\Scripts\\python.exe test_aux_pair_stereo.py
"""
import sys

import numpy as np

from kinect_imu_groundtruth import (AuxCameraView, erro_reprojecao_px,
                                    relative_transform, triangulate_aux_pairs,
                                    triangulate_pair_points)


def rotacao(x_deg=0.0, y_deg=0.0, z_deg=0.0):
    """Rotacao R = Rz @ Ry @ Rx (graus)."""
    ax, ay, az = np.deg2rad([x_deg, y_deg, z_deg])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def cria_vista(indice, rot, trans, tamanho=(1280, 720), focal=900.0):
    """AuxCameraView sintetica: X_aux = rot X_kin + trans (mesma convencao)."""
    matriz = np.array([[focal, 0.0, tamanho[0] / 2.0],
                       [0.0, focal, tamanho[1] / 2.0],
                       [0.0, 0.0, 1.0]])
    return AuxCameraView(
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
        runtime_matrix=None,
        runtime_size=tamanho,
        calibrated_size=tamanho,
    )


def projeta(vista, pontos_kin, rng=None, ruido_px=0.0):
    """Projeta pontos 3D (frame do Kinect) na vista; devolve px (N,2)."""
    pontos = np.asarray(pontos_kin, np.float64).reshape(-1, 3)
    R = np.asarray(vista.rotation, np.float64).reshape(3, 3)
    T = np.asarray(vista.translation, np.float64).reshape(3)
    cam = (R @ pontos.T).T + T.reshape(1, 3)
    pixels = (np.asarray(vista.matrix, np.float64) @ cam.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    if rng is not None and ruido_px > 0:
        pixels = pixels + rng.normal(0.0, ruido_px, pixels.shape)
    return pixels


def mao_sintetica(rng, palma=(0.02, -0.05, 1.05), espalhamento=0.09):
    pontos = (np.asarray(palma, np.float64)
              + rng.normal(0.0, espalhamento * 0.5, size=(21, 3)))
    pontos[0] = palma
    return pontos


class _LM:
    """Landmark minimo (o que o MediaPipe expoe: x e y normalizados)."""

    def __init__(self, x, y):
        self.x, self.y = x, y


class _AlignmentFake:
    """Dobre do CameraAlignment so' com as vistas (sem arquivos de calibracao)."""

    def __init__(self, vistas, ativa):
        self.vistas = {v.index: v for v in vistas}
        self.active_aux = ativa

    def aux_view(self, indice):
        return self.vistas.get(int(indice))


def deteccao(vista, pixels):
    """Landmarks normalizados que reproduzem EXATAMENTE estes pixels."""
    largura, altura = vista.runtime_size
    return [_LM(x / largura, y / altura)
            for x, y in np.asarray(pixels, np.float64).reshape(-1, 2)]


def main():
    rng = np.random.default_rng(11)
    # Bancada: Kinect na origem; auxiliar esquerda a -42 graus e direita a +42
    # graus no plano XZ, 0,58 m para cada lado e 0,16 m acima.
    esquerda = cria_vista(0, rotacao(x_deg=-6.0, y_deg=-42.0),
                          np.array([-0.58, 0.16, 0.12]))
    direita = cria_vista(1, rotacao(x_deg=-6.0, y_deg=42.0),
                         np.array([0.58, 0.16, 0.12]))

    # --- 1) relacao entre as duas cameras (composicao das duas com o Kinect) --
    R_ab, T_ab = relative_transform(esquerda, direita)
    for ponto in mao_sintetica(rng)[:6]:
        cam_e = (np.asarray(esquerda.rotation, np.float64).reshape(3, 3) @ ponto
                 + np.asarray(esquerda.translation, np.float64).reshape(3))
        cam_d = (np.asarray(direita.rotation, np.float64).reshape(3, 3) @ ponto
                 + np.asarray(direita.translation, np.float64).reshape(3))
        assert np.allclose(R_ab @ cam_e + T_ab, cam_d, atol=1e-9), \
            "relative_transform nao reproduz X_direita = R_ab X_esquerda + T_ab"

    # --- 2) pixels(): normalizado do MediaPipe -> px da CAMERA certa ---------
    marcas = [_LM(0.25, 0.75), _LM(0.5, 0.5)]
    assert np.allclose(esquerda.pixels(marcas), [[320.0, 540.0], [640.0, 360.0]])
    outra = cria_vista(2, rotacao(), np.array([0.0, 0.0, 0.0]),
                       tamanho=(640, 480))
    assert np.allclose(outra.pixels(marcas), [[160.0, 360.0], [320.0, 240.0]]), \
        "a conversao normalizado->pixel tem de usar o tamanho de CADA camera"

    # --- 3) triangulacao do par SEM o Kinect ---------------------------------
    verdade = mao_sintetica(rng)
    px_e = projeta(esquerda, verdade, rng, ruido_px=0.4)
    px_d = projeta(direita, verdade, rng, ruido_px=0.4)
    pontos, erro = triangulate_pair_points(esquerda, direita, px_e, px_d)
    assert pontos is not None, "a triangulacao das duas auxiliares falhou"
    assert pontos.shape == (21, 3), pontos.shape
    validos = np.isfinite(pontos).all(axis=1)
    assert validos.sum() >= 19, f"poucos nos validos: {validos.sum()}/21"
    desvio = np.linalg.norm(pontos[validos] - verdade[validos], axis=1)
    print(f"erro 3D mediano (par de auxiliares): "
          f"{np.median(desvio) * 1000:.2f} mm | max: {desvio.max() * 1000:.2f} mm"
          f" | reproj media: {erro:.2f} px")
    assert np.median(desvio) < 0.005, "erro 3D > 5 mm com ruido de 0.4 px"
    assert erro < 1.5, f"reprojecao media alta: {erro:.2f} px"
    assert abs(float(pontos[9][2]) - float(verdade[9][2])) < 0.010, \
        "Z da palma errado em > 1 cm (o eixo que o Kinect costuma perder)"

    # --- 4) erro de reprojecao tambem MEDE a qualidade do resultado ---------
    # `pontos` esta' no referencial do KINECT; a metrica reprojeta no referencial
    # da PROPRIA camera, entao convertemos antes (X_esq = R X_kin + T).
    pontos_esq = ((np.asarray(esquerda.rotation, np.float64).reshape(3, 3)
                   @ np.nan_to_num(pontos).T).T
                  + np.asarray(esquerda.translation, np.float64).reshape(3))
    assert erro_reprojecao_px(pontos_esq, px_e, esquerda.matrix,
                              esquerda.dist) < 1.5
    assert np.isnan(erro_reprojecao_px(np.full((3, 3), np.nan),
                                       np.zeros((3, 2)), esquerda.matrix,
                                       esquerda.dist))

    # --- 5) recusas: vista sem stereo, par com a MESMA camera, sem pixels ----
    quebrada = cria_vista(3, rotacao(), np.array([0.0, 0.0, 0.0]))
    quebrada = AuxCameraView(**{**quebrada.__dict__, "stereo_usable": False})
    assert triangulate_pair_points(quebrada, esquerda, px_e,
                                   px_e) == (None, None)
    assert triangulate_pair_points(esquerda, esquerda, px_e,
                                   px_e) == (None, None)
    assert triangulate_pair_points(esquerda, direita, px_e[:4],
                                   px_d) == (None, None)

    # --- 6) escolha automatica do MELHOR par (uma camera mal calibrada) -----
    # A terceira vista tem rotacao ERRADA (calibracao ruim): triangulando com ela
    # a reprojecao explode, entao o par (0, 1) tem de vencer -- e' assim que o
    # programa decide quem entra na substituicao do Kinect.
    errada = cria_vista(2, rotacao(y_deg=70.0), np.array([0.10, 0.0, 0.55]))
    px_errada = projeta(errada, verdade, rng, ruido_px=0.4)
    deteccoes = {0: deteccao(esquerda, px_e), 1: deteccao(direita, px_d),
                 2: deteccao(errada, px_errada)}
    alinhamento = _AlignmentFake([esquerda, direita, errada], ativa=1)
    pontos_par, info = triangulate_aux_pairs(alinhamento, deteccoes)
    assert pontos_par is not None, "o escolhedor de par nao triangulou nada"
    assert info["indices"] == (0, 1), (
        f"par escolhido {info['indices']} deveria ser (0, 1): a camera 2 esta' "
        "mal calibrada e reprojeta longe")
    assert info["erro_mediano_px"] < 1.5, info["erro_mediano_px"]
    finitos = np.isfinite(pontos_par).all(axis=1)
    desvio_par = np.linalg.norm(pontos_par[finitos] - verdade[finitos], axis=1)
    assert np.median(desvio_par) < 0.005, "par escolhido com erro 3D alto"

    # --- 7) sem DOIS lados detectados nao ha' par (cai para o Kinect) -------
    assert triangulate_aux_pairs(alinhamento, {0: deteccoes[0]}) == (None, None)
    assert triangulate_aux_pairs(alinhamento, {}) == (None, None)

    print("AUX_PAIR_STEREO_OK: 7/7 (relacao entre as duas auxiliares, pixels por "
          "camera, triangulacao SEM Kinect sub-cm, recusas e escolha do melhor "
          "par com uma camera mal calibrada)")
    return 0


if __name__ == "__main__":
    sys.exit(main())