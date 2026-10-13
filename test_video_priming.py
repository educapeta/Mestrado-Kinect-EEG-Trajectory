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

    def show_condition(self, objeto, mao):
        self.chamadas.append(("cue", objeto, mao))

    def show_message(self, texto):
        self.chamadas.append(("msg", texto))

    def show_countdown(self, numero):
        self.chamadas.append(("contagem",))

    def show_video_frame(self, quadro):
        # `quadro` e' o que `_to_qimage` devolveu; no teste e' um int (ver
        # `_identifica_quadro`): o alvo aqui e' o TIMING (qual quadro esteve
        # na tela e quando), nao o formato do QImage.
        self.chamadas.append(("video", quadro))

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
#    termina exatamente na deadline e a MI comeca com o cue
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
    # fim da pausa -> MI COM o cue restaurado (sem quadro congelado)
    assert state.phase == p.PHASE_MI, state.phase
    assert janela.ultima == ("cue", "garrafa", "direita"), janela.ultima
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
    assert janela.ultima == ("cue", "garrafa", "direita"), janela.ultima

    # =========================================================================
    # 3) clipe ATRASADO (chega 1,5 s depois, tracking lento): encaixa no tempo
    #    restante SEM congelar, AVISA que ficou > 0,5x e a MI comeca com o cue
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
    assert janela.ultima == ("cue", "garrafa", "direita"), janela.ultima

    # =========================================================================
    # 4) clipe NUNCA chega (sem Kinect / tracking morto): MI segue com o cue
    # =========================================================================
    controlador, janela, state = _novo(video_sec=0.5)
    controlador._to_qimage = _identifica_quadro
    controlador._start_video_pause()
    _roda_pausa(controlador, janela, relogio)
    assert not janela.videos, "sem clipe nao pode haver quadro"
    assert state.phase == p.PHASE_MI, state.phase
    assert janela.ultima == ("cue", "garrafa", "direita"), janela.ultima
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

print("test_video_priming_OK (playback cronometrado sem congelar | MI com "
      "cue | snapshot nao consome | fps de captura medido)")
