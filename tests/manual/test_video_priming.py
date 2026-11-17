"""Regressao do video de priming da pausa (0,5x) -- "video travado".

Sintomas reais (bancada Eduardo-PC, 30/09/2026, sessao `--tela-cheia` com
g.Nautilus de verdade):

1. o clipe passava na tela em 0,2-0,4 s e a janela CONGELAVA no ultimo quadro
   ate' o fim da pausa de 2 s: o buffer do Kinect capta ~3-6 fps (o laço do
   tracking roda 3x MediaPipe por iteracao) e o `_video_tick` antigo consumia
   1 quadro por tick de 33 ms, devolvendo None da 2a iteracao em diante;
2. a MI inteira ficava com o mesmo quadro CONGELADO: `_start_mi` nunca
   restaurava o cue da janela (o canvas ficava em modo "video").

Harness: relogio fake + janela/controlador falsos (mesmo do
test_paradigm_abort.py). Nenhum event loop do Qt em execucao, nenhum Kinect,
nenhum hardware.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

import eeg_motor_paradigm as p

#: Item 72: `_start_mi` toca um BIPE numa thread (winsound) -- o teste nao
#: pode emitir SOM nem depender de alto-falante; neutraliza o beep.
p._beep_async = lambda *args, **kwargs: None

TRIALS = [{"objeto": obj, "mao": mao, "condicao": f"{obj}_{mao}"}
          for obj, mao in (("garrafa", "direita"), ("bola", "esquerda"))]


class _RelogioFake:
    """time.monotonic() controlado pelo teste."""

    def __init__(self, agora=1000.0):
        self.agora = float(agora)

    def monotonic(self):
        return self.agora

    def avanca(self, segundos):
        self.agora += float(segundos)


class _JanelaFake:
    """Registra o que o controlador mandou mostrar (sem QWidget)."""

    def __init__(self):
        self.chamadas = []

    def show_home(self):
        self.chamadas.append(("home",))

    def show_condition(self, objeto, mao, **kwargs):
        #: **kwargs: o paradigma passa `pulsante=` (cue MI pulsante, 14/10) --
        #: o fake aceita e IGNORA; a tupla gravada fica igual para as assercoes.
        self.chamadas.append(("cue", objeto, mao))

    def show_mi(self, pulsante=False):
        #: MI (19/10): tela neutra "AGORA IMAGINE" -- sem simbolo, sem video.
        self.chamadas.append(("mi",))

    def show_message(self, texto):
        self.chamadas.append(("msg", texto))

    def show_countdown(self, numero):
        self.chamadas.append(("contagem",))

    def show_video_frame(self, quadro):
        # `quadro` e' o que `_to_qimage` devolveu; no teste e' um int (ver
        # `_identifica_quadro`): o alvo aqui e' o TIMING (qual quadro esteve
        # na tela e quando), nao o formato do QImage.
        self.chamadas.append(("video", quadro))

    def show_video_aviso(self, texto):
        #: Aviso "VIDEO INDISPONIVEL" (sem clipe do Kinect) -- recurso que
        #: entrou no mesmo arquivo por outra sessao; o fake acompanha.
        self.chamadas.append(("aviso", texto))

    def show_trial_progress(self, feito, total):
        pass

    @property
    def videos(self):
        return [c for c in self.chamadas if c[0] == "video"]

    @property
    def ultima(self):
        return self.chamadas[-1] if self.chamadas else None


class _ControladorVideo(p._ParadigmController):
    """Sem QTimer: `_schedule` captura o proximo tick, chamado NA MAO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eventos = []
        self._tick = None
        self._delay_ms = 0

    def _schedule(self, delay_ms, fn):
        self._tick = fn
        self._delay_ms = int(delay_ms)

    def _emit(self, code, **extra):
        self.eventos.append(code)


def _novo(video_sec=2.0, mi_sec=1.0):
    state = p.SharedState()
    janela = _JanelaFake()
    controlador = _ControladorVideo(
        window=janela, marker_queue=None, state=state,
        config={"trials": TRIALS, "home_sec": 0.0, "me_sec": 0.0,
                "video_sec": video_sec, "mi_sec": mi_sec,
                "trials_por_bloco": 4, "blocos": 1,
                "pausa_bloco_sec": 0.0, "contagem": 0,
                "olhos_abertos": 0.0, "olhos_fechados": 0.0,
                "repouso_ativo": 0.0, "preparacao": False,
                "priming_dir": "", "salvar_priming": False,
                "on_session_start": None})
    return controlador, janela, state


def _quadros(n):
    """n quadros DISTINTOS: o valor do 1o pixel identifica cada um."""
    return [np.full((4, 4, 3), i, np.uint8) for i in range(n)]


def _identifica_quadro(frame):
    return int(np.asarray(frame).flat[0])


def _roda_pausa(controlador, janela, relogio, passo=0.033):
    """Ticks de `passo` s ate' a deadline; devolve os instantes em que um
    quadro entrou na tela."""
    mostrados = []
    while relogio.agora < controlador._video_deadline:
        relogio.avanca(passo)
        antes = len(janela.videos)
        controlador._tick()
        if len(janela.videos) > antes:
            mostrados.append(relogio.agora)
    return mostrados


# =============================================================================
# 1) clipe "saudavel" (1 s de movimento na pausa de 2 s = 0,5x): toca INTEIRO,
#    termina exatamente na deadline e a MI comeca com a tela AGORA IMAGINE
# =============================================================================
relogio = _RelogioFake()
original_time = p.time
p.time = relogio
try:
    # 1a) `_to_qimage` real continua de pé (QImage nao exige QApplication)
    imagem = p._ParadigmController._to_qimage(np.zeros((4, 4, 3), np.uint8))
    assert imagem.width() == 4 and imagem.height() == 4

    # 1b) playback cronometrado: sem congelar antes da deadline
    controlador, janela, state = _novo(video_sec=2.0)
    controlador._to_qimage = _identifica_quadro      # timing > formato aqui
    controlador._start_video_pause()
    assert state.phase == p.PHASE_VIDEO
    state.publish_video(_quadros(60), duracao_real=1.0)   # tracking publica
    mostrados = _roda_pausa(controlador, janela, relogio)
    assert len(mostrados) >= 50, f"so' {len(mostrados)} quadros na tela"
    # a pose FINAL do movimento (ultimo quadro) chegou a aparecer
    assert janela.videos[-1][1] == 59, janela.videos[-1]
    # e' ela que esta na tela quando a pausa termina (ultimo quadro so' fica
    # preso pelo tempo do SEU passo, nao pelo resto da pausa)
    folga = controlador._video_deadline - mostrados[-1]
    assert folga <= controlador._video_passo + 0.05, f"congelou {folga:.2f} s"
    # clipe cheio: passo ~= 0,5x (1 s / 60 quadros / 0,5 = 0,0333 s)
    assert abs(controlador._video_passo - 1.0 / 60 / 0.5) < 0.005, \
        controlador._video_passo
    # fim da pausa -> MI COM a tela de imaginacao (sem quadro congelado)
    assert state.phase == p.PHASE_MI, state.phase
    assert janela.ultima == ("mi",), janela.ultima
    assert not state.video_frames, "clear_video nao rodou"

    # =========================================================================
    # 2) clipe CURTO (bug real: 4 quadros / 0,3 s de movimento): espacados
    #    para preencher a pausa INTEIRA em vez de acabar em 0,6 s e congelar
    # =========================================================================
    controlador, janela, state = _novo(video_sec=2.0)
    controlador._to_qimage = _identifica_quadro
    controlador._start_video_pause()
    state.publish_video(_quadros(4), duracao_real=0.3)
    mostrados = _roda_pausa(controlador, janela, relogio)
    assert janela.videos, "clipe curto nao apareceu"
    assert janela.videos[-1][1] == 3, janela.videos[-1]   # clipe inteiro
    folga = controlador._video_deadline - mostrados[-1]
    assert folga <= controlador._video_passo + 0.05, f"congelou {folga:.2f} s"
    assert state.phase == p.PHASE_MI, state.phase
    assert janela.ultima == ("mi",), janela.ultima

    # =========================================================================
    # 3) clipe ATRASADO (chega 1,5 s depois, tracking lento): encaixa no tempo
    #    restante SEM congelar, AVISA que ficou > 0,5x e a MI comeca com a tela
    #    (nada de quadro "stale" antes da chegada do clipe)
    # =========================================================================
    controlador, janela, state = _novo(video_sec=2.0)
    controlador._to_qimage = _identifica_quadro
    controlador._start_video_pause()
    atraso_ate = controlador._video_deadline - 0.5
    while relogio.agora < atraso_ate:
        relogio.avanca(0.033)
        controlador._tick()
    assert not janela.videos, "sem clipe nao pode haver quadro (nada de stale)"
    state.publish_video(_quadros(60), duracao_real=1.0)
    aviso = io.StringIO()
    with contextlib.redirect_stdout(aviso):
        mostrados = _roda_pausa(controlador, janela, relogio)
    assert "AVISO" in aviso.getvalue(), aviso.getvalue()
    assert len(mostrados) >= 5, f"clipe atrasado so' apareceu {len(mostrados)}x"
    assert janela.videos[-1][1] >= 30, janela.videos[-1]  # encaixou quase tudo
    assert janela.ultima == ("mi",), janela.ultima

    # =========================================================================
    # 4) clipe NUNCA chega (sem Kinect / tracking morto): MI segue com o cue
    # =========================================================================
    controlador, janela, state = _novo(video_sec=0.5)
    controlador._to_qimage = _identifica_quadro
    controlador._start_video_pause()
    _roda_pausa(controlador, janela, relogio)
    assert not janela.videos, "sem clipe nao pode haver quadro"
    assert state.phase == p.PHASE_MI, state.phase
    assert janela.ultima == ("mi",), janela.ultima
finally:
    p.time = original_time

# =============================================================================
# 5) `video_snapshot` NAO consome o clipe (e a API legada continua de pé)
# =============================================================================
state = p.SharedState()
state.publish_video(_quadros(3), duracao_real=0.6)
a, da = state.video_snapshot()
b, db = state.video_snapshot()
assert len(a) == len(b) == 3 and da == db == 0.6
assert state.take_video_frame() is not None, "API legada sumiu"
assert len(state.video_snapshot()[0]) == 3, "snapshot nao pode consumir"
state.clear_video()
assert state.video_snapshot() == ([], 0.0)

# =============================================================================
# 6) `_publish_slowmo` mede a duracao REAL e imprime a taxa de captura (fps)
# =============================================================================
state = p.SharedState()
tracking = p.TrackingThread(state, p.ImuReceiver(4210), with_kinect=False,
                            aux_indices=(), show_window=False)
tracking.onset_time = 100.0                      # onset -> janela [100, 101]
for i in range(6):                    # 5 fps: 1 quadro a cada 0,2 s
    tracking._frame_buffer.append((100.0 + 0.2 * i,
                                   np.full((4, 4, 3), i, np.uint8)))
captura = io.StringIO()
with contextlib.redirect_stdout(captura):
    tracking._publish_slowmo((101.0, None))
quadros, dur = state.video_snapshot()
assert abs(dur - 1.0) < 1e-9, dur                # span real 101.0 - 100.0
assert len(quadros) == 12, len(quadros)          # 6 reais x repeticoes (0,5x)
texto = captura.getvalue()
assert "fps de captura" in texto and "5.0 fps" in texto, texto

# =============================================================================
# 7) ANCORA DO CLIPE NA TRAJETORIA DA ME: o video comeca no MOVIMENTO.
#    Bug real de 21/10/2026: o detector disparava com a MAO PARADA (5 mm/0,03
#    m/s esta' no chao de ruido do punho 3D) e o clipe comecava 0,2-0,8 s antes
#    do movimento; em 29 clipes, 16 comecavam com a mao parada e 6 nao tinham
#    movimento nenhum. Estes casos falham sem o patch.
# =============================================================================
def _trajetoria_me(inicio_s=1.2, total_s=4.0, fs=30.0, desloc_m=0.18,
                   jitter_mm=2.0, semente=5, t_me=100.10):
    """[(t, pos)] de uma ME: mao parada (jitter) + um alcance no meio."""
    rng = np.random.default_rng(semente)
    pontos = []
    for i in range(int(total_s * fs)):
        t = t_me + i / fs
        s = np.clip((t - t_me - inicio_s) / 0.6, 0.0, 1.0)
        suave = 3 * s ** 2 - 2 * s ** 3
        pontos.append((t, np.array([0.0, 0.0, -desloc_m * suave])
                       + rng.normal(0.0, jitter_mm / 1000.0, 3)))
    return pontos


state = p.SharedState()
tracking = p.TrackingThread(state, p.ImuReceiver(4210), with_kinect=False,
                            aux_indices=(), show_window=False)

# 7a) alcance em 1,2 s: ancora no INICIO do movimento, nao no jitter do comeco
tracking._me_track = _trajetoria_me(inicio_s=1.2)
t0, t1 = tracking._clip_window(104.0)
inicio_mov = 100.10 + 1.2
assert abs((t1 - t0) - p.SLOWMO_SPAN_SEC) < 1e-9, (t0, t1)
assert inicio_mov - 0.12 <= t0 <= inicio_mov + 0.30, (t0, inicio_mov)
assert t0 > 100.10 + 0.6, "ancorou no jitter do inicio da ME e nao no movimento"
assert t1 > inicio_mov + 0.35, "o clipe nao chega a mostrar o movimento"

# 7b) mao PARADA a ME inteira -> fallback antigo (fim da ME), com aviso
tracking._me_track = _trajetoria_me(desloc_m=0.0)
aviso = io.StringIO()
with contextlib.redirect_stdout(aviso):
    t0, t1 = tracking._clip_window(104.0)
assert "SEM movimento confirmado" in aviso.getvalue(), aviso.getvalue()
assert abs(t1 - (104.0 - p.SLOWMO_END_LAG_SEC)) < 1e-9, (t0, t1)

# 7c) sem trajetoria (--sem-onset / Kinect mudo): comportamento antigo com a
#     deteccao em tempo real
tracking._me_track = []
tracking.onset_time = 101.0
assert tracking._clip_window(104.0) == (101.0, 101.0 + p.SLOWMO_SPAN_SEC)

# 7d) sem trajetoria e sem deteccao: recuo fixo do fim da ME
tracking.onset_time = None
t0, t1 = tracking._clip_window(104.0)
assert abs(t1 - (104.0 - p.SLOWMO_END_LAG_SEC)) < 1e-9, t1
assert abs((t1 - t0) - p.SLOWMO_SPAN_SEC) < 1e-9, (t0, t1)

# 7e) movimento MUITO tarde (3,6 s de uma ME de 4 s): o fim da janela e'
#     limitado ao pedido, senao os ultimos quadros ainda nao existiriam
tracking._me_track = _trajetoria_me(inicio_s=3.6, total_s=4.0)
aviso = io.StringIO()
with contextlib.redirect_stdout(aviso):
    t0, t1 = tracking._clip_window(104.0)
assert t1 <= 104.0 - 0.02 + 1e-9, t1
assert 0.0 < t1 - t0 < p.SLOWMO_SPAN_SEC, (t0, t1)
assert "comecou tarde" in aviso.getvalue(), aviso.getvalue()

# =============================================================================
# 8) O CLIPE PUBLICADO e' o do MOVIMENTO: `_publish_slowmo` recorta os quadros
#    do intervalo ancorado (e nao os do fim da ME, como no comportamento antigo)
# =============================================================================
state = p.SharedState()
tracking = p.TrackingThread(state, p.ImuReceiver(4210), with_kinect=False,
                            aux_indices=(), show_window=False)
tracking._me_track = _trajetoria_me(inicio_s=1.2)
for i in range(120):                          # buffer de 4 s a 30 fps
    tracking._frame_buffer.append(
        (100.0 + i / 30.0, np.full((4, 4, 3), i, np.uint8)))
t0, t1 = tracking._clip_window(104.0)
with contextlib.redirect_stdout(io.StringIO()):
    tracking._publish_slowmo((104.0, None))
quadros, dur = state.video_snapshot()
assert quadros, "clipe vazio"
primeiro = _identifica_quadro(quadros[0])
esperado = int(round((t0 - 100.0) * 30.0))     # 1o quadro do intervalo [t0, t1]
assert abs(primeiro - esperado) <= 3, (primeiro, esperado, t0)
assert dur <= p.SLOWMO_SPAN_SEC + 1e-6, dur

print("test_video_priming_OK (playback cronometrado sem congelar | MI com "
      "tela AGORA IMAGINE | snapshot nao consome | fps de captura medido | "
      "clipe ancorado no INICIO DO MOVIMENTO)")
