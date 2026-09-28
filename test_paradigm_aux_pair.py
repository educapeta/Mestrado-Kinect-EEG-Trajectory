"""Teste da FONTE da medida 3D no paradigma com o Kinect cego (sem hardware).

O objetivo do experimento (bancada, 23/09/2026) tem duas metades e esta e' a que
fecha a segunda: quando o Kinect perde o video e a projecao do esqueleto, as DUAS
webcams auxiliares (uma de cada lado, ~45 graus) tem de dizer SOZINHAS onde a mao
esta' nos 3 eixos. Aqui se testa o DESVIO DE FONTE do `_resolve_wrist3d`:

  Kinect cego + 2 auxiliares ......... `triangulado_aux_<i>+<j>` (o par assume)
  Kinect cego + 1 auxiliar .......... nao ha' par -> cai no SDK -> desiste
  2 auxiliares, uma sem calibracao ... nao ha' par -> cai no SDK -> desiste
  Kinect COM a mao .................. prioridade e' a triangulacao Kinect+aux
  Kinect com a mao, sem auxiliares ... `kinect_depth` (regressao: nao muda)

Sintetico: duas vistas rigidas conhecidas e uma mao 3D conhecida; o fake do
tracker nunca acha nada (nem Kinect nem esqueleto SDK), entao QUALQUER ponto nao
nulo so' pode ter vindo da triangulacao.

    .venv\\Scripts\\python.exe test_paradigm_aux_pair.py
"""
import sys

import numpy as np

import eeg_motor_paradigm as p
import kinect_imu_groundtruth as gt

LARGURA, ALTURA = 1920, 1080
TAMANHO_AUX = (1280, 720)


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


def vista(indice, rot, trans, tamanho=TAMANHO_AUX, focal=900.0):
    matriz = np.array([[focal, 0.0, tamanho[0] / 2.0],
                       [0.0, focal, tamanho[1] / 2.0],
                       [0.0, 0.0, 1.0]])
    return gt.AuxCameraView(
        index=indice,
        kinect_matrix=gt.KINECT_V2_INTRINSICS_1080P,
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
    """Dobre do CameraAlignment: so' o que `_resolve_wrist3d` consulta.

    `tri3d` simula o resultado do stereo Kinect+auxiliar (o caminho que tem
    PRIORIDADE quando o Kinect enxerga a mao).
    """

    def __init__(self, vistas, tri3d=None):
        self.vistas = {v.index: v for v in vistas}
        self.active_aux = vistas[0].index if vistas else None
        self.auxiliary_width, self.auxiliary_height = TAMANHO_AUX
        self.overlay_trim = np.zeros(2)
        self.last_triangulated_3d = None
        self.last_triangulated_3d_time = None
        self.tri3d = tri3d
        self.ativados = []

    def aux_view(self, indice):
        return self.vistas.get(int(indice))

    def aux_usable(self, indice):
        return self.aux_view(indice) is not None

    def set_active_aux(self, indice):
        self.ativados.append(int(indice))
        self.active_aux = int(indice)

    def triangulate_hand_points(self, aux_px, kin_px):
        if self.tri3d is None:
            return None, None
        return self.tri3d, np.array([0.4])       # erro de reprojecao sintetico


class _TrackerFake:
    """Kinect sem NADA: nem mao, nem esqueleto do SDK."""

    def __init__(self):
        self.color_width, self.color_height = LARGURA, ALTURA
        self.chamadas_anchor = 0

    def body_hand_anchor(self, side=None):
        self.chamadas_anchor += 1
        return None


def thread_sem_hardware(vistas, tri3d=None):
    """TrackingThread minimo, com so' o que `_resolve_wrist3d` usa."""
    thread = p.TrackingThread.__new__(p.TrackingThread)
    thread.gt = gt
    thread.alignment = _AlinhamentoFake(vistas, tri3d)
    thread.tracker = _TrackerFake()
    thread._triangulated = None
    return thread
def main():
    rng = np.random.default_rng(11)
    esquerda = vista(3, rotacao(x_deg=-6.0, y_deg=-42.0),
                     np.array([-0.58, 0.16, 0.12]))
    direita = vista(7, rotacao(x_deg=-6.0, y_deg=42.0),
                    np.array([0.58, 0.16, 0.12]))
    verdade = np.array([0.02, -0.05, 1.05]) + rng.normal(0.0, 0.05, (21, 3))
    verdade[0] = [0.02, -0.05, 1.05]

    def mao(v, pontos):
        return landmarks_normalizados(
            projeta(v.matrix, np.asarray(v.rotation).reshape(3, 3),
                    np.asarray(v.translation).reshape(3), pontos),
            v.runtime_size)

    px_kinect = projeta(gt.KINECT_V2_INTRINSICS_1080P, np.eye(3), np.zeros(3),
                        verdade)
    kinect_lms = landmarks_normalizados(px_kinect, (LARGURA, ALTURA))

    # --- 1) Kinect CEGO com as DUAS auxiliares: o par assume a medida --------
    thread = thread_sem_hardware([esquerda, direita])
    ponto, fonte, aux_usada, maos = thread._resolve_wrist3d(
        None, None, {esquerda.index: [mao(esquerda, verdade)],
                     direita.index: [mao(direita, verdade)]})
    assert fonte == "triangulado_aux_3+7", fonte
    assert aux_usada in (esquerda.index, direita.index), aux_usada
    assert len(maos) == 21, "as maos usadas tem de voltar para o chamador"
    erro = float(np.linalg.norm(np.asarray(ponto) - verdade[9]))
    assert erro < 0.005, f"erro {erro * 1000:.2f} mm (sintetico perfeito)"
    # Os consumidores (desenho, CSV, fusao) leem o 3D do alignment: tem de
    # ficar fresco, senao a sobreposicao nao usa a medida do par.
    assert thread._triangulated is not None
    assert thread.alignment.last_triangulated_3d is not None
    assert thread.alignment.last_triangulated_3d_time is not None
    assert thread.tracker.chamadas_anchor == 0, (
        "com o par resolvendo, o esqueleto do SDK nao deve ser consultado")

    # --- 2) Kinect cego com UMA auxiliar: nao ha' par -> desiste -----------
    thread = thread_sem_hardware([esquerda])
    ponto, fonte, aux_usada, maos = thread._resolve_wrist3d(
        None, None, {esquerda.index: [mao(esquerda, verdade)]})
    assert ponto is None and fonte == "" and not maos, (ponto, fonte, maos)
    assert thread.tracker.chamadas_anchor == 1, "ainda tem de tentar o SDK"

    # --- 3) Duas auxiliares, mas uma SEM calibracao stereo utilizavel ------
    sem_stereo = gt.AuxCameraView(**{**esquerda.__dict__,
                                     "stereo_usable": False})
    thread = thread_sem_hardware([sem_stereo, direita])
    ponto, fonte, _, _ = thread._resolve_wrist3d(
        None, None, {esquerda.index: [mao(esquerda, verdade)],
                     direita.index: [mao(direita, verdade)]})
    assert ponto is None and fonte == "", (ponto, fonte)

    # --- 4) Kinect COM a mao: Kinect+auxiliar mantem a prioridade ----------
    tri_fake = verdade + 0.002
    thread = thread_sem_hardware([esquerda, direita], tri3d=tri_fake)
    ponto, fonte, aux_usada, maos = thread._resolve_wrist3d(
        None, kinect_lms, {esquerda.index: [mao(esquerda, verdade)],
                           direita.index: [mao(direita, verdade)]})
    assert not fonte.startswith("triangulado_aux_"), fonte
    assert fonte.startswith("triangulado_"), fonte
    assert aux_usada == esquerda.index, aux_usada
    assert np.allclose(ponto, tri_fake[9], atol=1e-9)

    # --- 5) Kinect com a mao e SEM auxiliares: depth do Kinect (regressao) --
    thread = thread_sem_hardware([esquerda, direita])
    palma = np.array([0.01, -0.02, 0.98])
    ponto, fonte, aux_usada, maos = thread._resolve_wrist3d(
        palma, kinect_lms, {})
    assert fonte == "kinect_depth" and np.allclose(ponto, palma), (fonte, ponto)
    assert maos == [] and aux_usada is None

    print("PARADIGM_AUX_PAIR_OK: 5/5 (o par de auxiliares mede a mao com o "
          "Kinect cego, uma auxiliar so' nao substitui, calibracao faltando "
          "recusa, e os caminhos do Kinect seguem com prioridade)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
