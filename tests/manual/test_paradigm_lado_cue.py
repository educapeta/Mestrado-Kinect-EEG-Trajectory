"""Teste da MAO ATIVA DO CUE (lateralidade do tracking) -- 21/10/2026.

Bug relatado pelo usuario: "o Kinect algumas vezes confundiu a mao que esta'
fazendo o tracking: dava a seta a' esquerda, eu mexia a mao esquerda, mas logo
ele estava fazendo o tracking da mao DIREITA durante o video".

Causa: a mao ATIVA era reescolhida A CADA QUADRO pelo rotulo de lateralidade do
MediaPipe -- uma classificacao de APARENCIA do quadro, que oscila com as duas
maos visiveis. Como o `KT_hand` publicado vinha do MESMO rotulo usado para
escolher, o log dava o lado do cue por construcao e nao mostrava a troca: o
unico rastro era o video do priming (esqueleto verde na mao errada).

Casos:
  1. `lado_publicado`: o CUE manda; rotulo oposto = flag (nunca troca o lado);
     sem cue (repouso/origem) o rotulo decide, como antes;
  2. codigo 797 (flag da trial) livre, unico e com legenda no JSON;
  3. LATCH do lado da trial: o lado so' muda quando muda a trial; a ancora e os
     contadores zeram junto; cue vazio esquece a trial;
  4. a ancora (palma do quadro anterior) e' a identidade FISICA da mao: ela
     atualiza a partir das landmarks e VENCE o timeout;
  5. discordancia no meio da trial: o lado publicado continua o do cue, o
     marcador 797 e' emitido UMA vez por trial (com `lado_cue`,
     `lado_detector` e `quadros`) e NAO se repete nos quadros seguintes;
  6. `_select_arm`: com lado definido pelo cue NUNCA publica o braco do outro
     lado (ARM_valid=0 em vez de geometria da mao errada);
  7. gate da triangulacao: erro de reprojecao grosseiro (mao errada casada entre
     as cameras) e' DESCARTADO e a medida cai no depth do Kinect, que e' do
     mesmo quadro.

    .venv\\Scripts\\python.exe test_paradigm_lado_cue.py
"""
import os
import queue
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eeg_motor_paradigm as p                                    # noqa: E402

# =============================================================================
# 1) Politica: o lado publicado e' o do CUE
# =============================================================================
assert p.lado_publicado("left", "right") == ("left", 1)
assert p.lado_publicado("right", "left") == ("right", 1)
assert p.lado_publicado("left", "left") == ("left", 0)
assert p.lado_publicado("right", "right") == ("right", 0)
# Sem rotulo do detector (nao viu nenhuma mao) nao ha' discordancia: o lado do
# cue continua sendo publicado (a medida 3D e' que fica NaN).
assert p.lado_publicado("left", "") == ("left", 0)
assert p.lado_publicado("left", None) == ("left", 0)
# Trial SEM lado (home, origem, calibracao): o rotulo decide -- comportamento
# antigo preservado.
assert p.lado_publicado("", "right") == ("right", 0)
assert p.lado_publicado("", "") == ("", 0)
assert p.lado_publicado(None, None) == ("", 0)
# Um "lado" invalido nunca vira flag nem vira publicacao.
assert p.lado_publicado("both", "left") == ("left", 0)
print("1) OK: lado publicado = lado do CUE (rotulo oposto so' levanta a flag)")

# =============================================================================
# 2) Codigo 797: livre, unico e com legenda
# =============================================================================
assert p.CODE_HAND_SIDE_FLAG == 797
outros = {p.CODE_SESSION, p.CODE_SESSION_END, p.CODE_EYES_OPEN,
          p.CODE_EYES_CLOSED, p.CODE_BASELINE_REST, p.CODE_TRIAL,
          p.CODE_ORIGIN_START, p.CODE_ORIGIN_END, p.CODE_ME_START,
          p.CODE_ME_END, p.CODE_MI_START, p.CODE_MI_END, p.CODE_BLOCK_START,
          p.CODE_BLOCK_END, p.CODE_PAUSE_START, p.CODE_PAUSE_END,
          p.CODE_SLOWMO_START, p.CODE_MOVE_ONSET, p.CODE_TRIAL_CANCEL,
          p.CODE_TRIAL_PRACTICE, p.CODE_LINK_LOST, p.CODE_LINK_OK,
          *p.CODE_HAND.values(), *p.CODE_CONDITION.values()}
assert p.CODE_HAND_SIDE_FLAG not in outros, "797 colide com outro codigo"
assert p.CODE_HAND_SIDE_FLAG in p.CODE_NAMES, "797 sem legenda no JSON"
print("2) OK: codigo 797 (flag da trial) unico e legendado")


# =============================================================================
# Apoio: thread de tracking SEM hardware (so' o estado e as landmarks falsas)
# =============================================================================
class LM:
    """Landmark minimo do MediaPipe (x e y normalizados)."""

    def __init__(self, x, y):
        self.x, self.y = x, y


def mao(x, y=0.5):
    """21 landmarks com a PALMA (no' 9) em (x, y) normalizados."""
    return [LM(x, y) for _ in range(21)]


class TrackerFalso:
    """O que a thread de tracking le' do KinectHandTracker."""

    color_width = 1920
    color_height = 1080

    def __init__(self, lado=""):
        self.last_hand_side = lado

    def body_hand_anchor(self, prefer_side=None):
        """Sem esqueleto do SDK (o caso real das sessoes gravadas): None."""
        return None


def thread_nova(lado_detector="", fila=None):
    """TrackingThread com estado/dummies e sem abrir Kinect/IMU."""
    t = p.TrackingThread(p.SharedState(), object(), with_kinect=False,
                         use_onset=False, marker_queue=fila, aux_indices=())
    t.tracker = TrackerFalso(lado_detector)
    return t


# =============================================================================
# 3) LATCH: o lado so' muda quando muda a trial
# =============================================================================
t = thread_nova("left")
assert t._lado_da_trial("left") == "left"
t._ancora_px, t._ancora_ts = (960.0, 540.0), time.monotonic()
t._discord_frames = 3
t._discord_flag_emitida = True
# Mesma trial: nada e' zerado, mesmo chamando dezenas de vezes.
for _ in range(10):
    assert t._lado_da_trial("left") == "left"
assert t._ancora_px == (960.0, 540.0) and t._discord_frames == 3
assert t._discord_flag_emitida is True
# OUTRA trial (o cue mudou): lado novo, ancora e contadores zerados.
assert t._lado_da_trial("right") == "right"
assert t._lado_trial == "right"
assert t._ancora_px is None and t._ancora_ts == 0.0
assert t._discord_frames == 0 and t._discord_flag_emitida is False
# Sem lado (home/repouso/origem): esquece a trial -- o detector volta a mandar.
assert t._lado_da_trial("") == ""
assert t._lado_trial == "" and t._ancora_px is None
print("3) OK: latch do lado da trial (troca so' quando o cue muda)")

# =============================================================================
# 4) Ancora: identidade FISICA da mao (palma em px) e timeout
# =============================================================================
t = thread_nova("left")
t._lado_da_trial("left")
assert t._ancora_valida() is None                     # ancora ainda nao existe
assert t._mao_ativa(mao(0.25, 0.40), "left") == "left"
assert t._ancora_px == (0.25 * 1920, 0.40 * 1080), t._ancora_px
assert t._ancora_valida() == t._ancora_px             # recem-medida: vale
# Sem landmarks (quadro sem deteccao) a ancora NAO e' apagada: ela e' o que
# mantem a mao quando o detector perde um quadro -- mas o timeout a vence.
assert t._mao_ativa(None, "left") == "left"
assert t._ancora_valida() is not None
t._ancora_ts = time.monotonic() - (p.ANCORA_MAO_TIMEOUT_S + 0.1)
assert t._ancora_valida() is None, "ancora velha tem de vencer"
# Mao perdida e reaparecida longe: a ancora velha nao prende o rastreamento.
t._mao_ativa(mao(0.80, 0.5), "left")
assert t._ancora_px == (0.80 * 1920, 540.0)
assert p.ANCORA_MAO_RAIO_PX < (0.80 - 0.25) * 1920, \
    "o raio tem de ser menor que a separacao tipica entre as maos"
# Sem cue (repouso): nenhuma ancora e' mantida.
assert t._mao_ativa(mao(0.25, 0.40), "") == ""
assert t._ancora_px is None
print("4) OK: ancora = palma do quadro anterior (identidade fisica da mao)")

# =============================================================================
# 5) Discordancia: mantem o lado do cue e marca a trial UMA vez
# =============================================================================
fila = queue.Queue()
t = thread_nova("right", fila)
t._lado_da_trial("left")
# O detector viu a mao DIREITA na trial da ESQUERDA: o bug relatado.
assert t._mao_ativa(mao(0.30, 0.5), "left") == "left", \
    "a mao publicada tem de continuar sendo a do cue"
assert t._discord_frames == 1
assert not fila.empty(), "a discordancia tem de virar marcador"
evento = fila.get_nowait()
assert evento["code"] == 797, evento
assert evento["lado_cue"] == "left" and evento["lado_detector"] == "right", evento
assert evento["origem"] == "tracking" and evento["monotonic_s"] > 0, evento
assert evento["quadros"] >= 1, evento
assert evento["nome"] == p.CODE_NAMES[p.CODE_HAND_SIDE_FLAG], evento
# Quadros SEGUINTES da mesma trial: conta, mas NAO repete o marcador (o CSV
# ficaria poluido e a analise trata a trial uma unica vez).
for _ in range(4):
    assert t._mao_ativa(mao(0.30, 0.5), "left") == "left"
assert t._discord_frames == 5, t._discord_frames
assert fila.empty(), "o marcador 797 e' UM por trial"
# Concordou de novo: o lado continua o do cue e nada e' emitido.
t.tracker.last_hand_side = "left"
assert t._mao_ativa(mao(0.70, 0.5), "left") == "left"
assert t._discord_frames == 5 and fila.empty()
# Trial nova: a flag rearma e o primeiro quadro discordante marca de novo.
t._lado_da_trial("right")
t.tracker.last_hand_side = "left"
assert t._mao_ativa(mao(0.70, 0.5), "right") == "right"
assert t._discord_frames == 1
assert not fila.empty()
assert fila.get_nowait()["lado_cue"] == "right"
print("5) OK: discordancia mantem o lado do cue e marca a trial (797, 1x)")

# =============================================================================
# 6) Braco: nunca publica o lado OPOSTO quando a trial define o lado
# =============================================================================
t = thread_nova()
so_direita = {"right": {"side": "right", "wrist": np.array([0.20, -0.10, 1.2])}}
assert t._select_arm(so_direita, None, "left") is None, \
    "sem o braco do cue, ARM_valid=0 (nunca o braco da outra mao)"
ambos = {"right": {"side": "right", "wrist": np.array([0.20, -0.10, 1.2])},
         "left": {"side": "left", "wrist": np.array([-0.20, -0.10, 1.2])}}
assert t._select_arm(ambos, None, "left")["side"] == "left"
assert t._select_arm(ambos, None, "right")["side"] == "right"
# Sem lado no cue o fallback antigo continua valendo (mao mais proxima do punho
# medido; depois a mais rapida; por ultimo a direita).
assert t._select_arm(ambos, np.array([0.21, -0.10, 1.2]), "")["side"] == "right"
assert t._select_arm(ambos, np.array([-0.19, -0.10, 1.2]), "")["side"] == "left"
assert t._select_arm(so_direita, None, "")["side"] == "right"
assert t._select_arm({}, None, "left") is None
print("6) OK: braco do lado do cue (ARM_valid=0 em vez do braco oposto)")


# =============================================================================
# 7) Gate da triangulacao: mao errada casada entre cameras e' DESCARTADA
# =============================================================================
class AlinhamentoFalso:
    """CameraAlignment minimo: triangula e devolve o erro que o teste mandar."""

    auxiliary_width = 1280
    auxiliary_height = 720

    def __init__(self, erro_px):
        self.erro_px = erro_px
        self.last_triangulated_3d = None
        self.last_triangulated_3d_time = None
        self.chamadas = 0

    def aux_usable(self, indice):
        return True

    def set_active_aux(self, indice):
        return None

    def triangulate_hand_points(self, aux_px, kin_px):
        self.chamadas += 1
        pontos = np.tile(np.array([0.10, -0.20, 1.10]), (21, 1))
        return pontos, np.full(21, float(self.erro_px))


def thread_com_alinhamento(erro_px):
    t = thread_nova()
    t.tracker = TrackerFalso()
    t.alignment = AlinhamentoFalso(erro_px)
    t.gt = types.SimpleNamespace(
        AUX_CAMERA_NAMES={},
        triangulate_aux_pairs=lambda *_a, **_k: (None, None))
    return t


landmarks = mao(0.40)
outras = [mao(0.55)]
duas_maos = {0: outras}
# (a) erro grosseiro (~300 px: as cameras casaram maos DIFERENTES) -> descarta e
#     a medida cai no depth do Kinect, que e' do MESMO quadro.
t = thread_com_alinhamento(300.0)
posicao, fonte, aux_usado, _maos = t._resolve_wrist3d(
    np.array([0.10, -0.20, 1.10]), landmarks, duas_maos, "left")
assert t.alignment.chamadas == 1                  # tentou triangular
assert fonte == "kinect_depth", fonte             # e foi descartada
assert aux_usado is None
assert np.allclose(posicao, [0.10, -0.20, 1.10])
# (b) erro bom (~1 px): a triangulacao VALE (comportamento preservado).
t = thread_com_alinhamento(1.0)
_, fonte, aux_usado, _maos = t._resolve_wrist3d(
    np.array([0.10, -0.20, 1.10]), landmarks, duas_maos, "left")
assert fonte == "triangulado_aux0", fonte
assert aux_usado == 0
# (c) o limiar e' o documentado e separa os dois casos.
assert 1.0 < p.TRI_MAX_ERRO_PX < 300.0, p.TRI_MAX_ERRO_PX
# (d) sem punho do Kinect e sem triangulacao: NaN (nunca um 3D inventado).
t = thread_com_alinhamento(300.0)
posicao, fonte, _aux, _maos = t._resolve_wrist3d(None, None, {}, "left")
assert posicao is None and fonte == ""
print("7) OK: gate da triangulacao (mao errada entre cameras nao vira KT_*)")

print()
print("LADO_CUE_OK: 7/7 blocos (cue manda no lado publicado, latch da trial, "
      "continuidade por ancora, marcador 797 1x por trial, braco so' do cue, "
      "gate da triangulacao)")
