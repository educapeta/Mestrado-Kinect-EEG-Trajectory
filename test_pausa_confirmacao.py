"""Regressao: a PAUSA ENTRE BLOCOS nao inicia o proximo bloco sozinha.

Relato do trial real de 21/10/2026 (sessao `gravacao_MEMI_20261021_135730`):

    "Quando o bloco chega ao fim, o timer de 2 minutos e' puramente
    ilustrativo; quando chega a 0 segundos NAO e' para comecar os novos trials
    sem a pessoa ter confirmado, clicando ESPACO para continuar."

Causa (antes desta correcao): `_pause_tick` chamava o `on_end` assim que
`remaining <= 0` (ou quando `_pause_skip`), e o `on_end` da pausa de bloco e'
`_finish_block_pause`, que emitia o 793 e chamava `_start_block` -- ou seja, o
fim do timer iniciava os trials seguintes sem ninguem confirmar.

Semantica implementada (e' o que este teste trava):

1. a contagem e' o DESCANSO do participante; zerada, a tela PARA em
   "PRESSIONE ESPACO PARA CONTINUAR" e NADA comeca por tempo;
2. ESPACO durante a contagem ENCURTA os minutos restantes e cai na MESMA tela
   de confirmacao: ainda precisa de um ESPACO NOVO para o bloco iniciar;
3. o ESPACO que encerra a contagem NAO vale como confirmacao (um toque FISICO
   chega ao controlador duas vezes: atalho do Qt + vigia global de hotkeys.py),
   por isso existe `PAUSE_CONFIRM_DWELL_SEC`;
4. a preparacao e o baseline continuam sozinhos ao zerar a contagem (sao
   trechos cronometrados do protocolo; o ESPACO so' os encurta, como antes);
5. 792/793 continuam delimitando a espera REAL (contagem + confirmacao) e o
   evento 793 leva os tempos no JSON (`duracao_s`, `contagem_s`,
   `espera_confirmacao_s`, `contagem_encerrada_por`).

Harness: relogio fake (patch de `p.time`, mesmo padrao do
test_video_priming.py), janela falsa e `_schedule` capturado -- cada tick e'
rodado NA MAO. Nenhum event loop do Qt, nenhum QTimer, nenhum hardware.
"""
from __future__ import annotations

import re
import time as _time_real

import eeg_motor_paradigm as p

TRIALS = [{"objeto": obj, "mao": mao, "condicao": f"{obj}_{mao}"}
          for obj, mao in (("garrafa", "direita"), ("bola", "esquerda"),
                           ("caneta", "direita"), ("bola", "direita"))]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class _RelogioFake:
    """time.monotonic() controlado pelo teste (p.time = relogio)."""

    def __init__(self, agora=1000.0):
        self.agora = float(agora)

    def monotonic(self):
        return self.agora

    def avanca(self, segundos):
        self.agora += float(segundos)


RELOGIO = _RelogioFake()


class _JanelaFake:
    """Registra o que o controlador mandou mostrar (sem QWidget)."""

    def __init__(self):
        self.chamadas = []
        self.progresso = []

    def show_home(self):
        self.chamadas.append(("home",))

    def show_condition(self, objeto, mao, **kwargs):
        #: **kwargs: o paradigma passa `pulsante=` -- o fake aceita e ignora.
        self.chamadas.append(("cue", objeto, mao))

    def show_mi(self, pulsante=False):
        self.chamadas.append(("mi",))

    def show_message(self, texto):
        self.chamadas.append(("msg", texto))

    def show_countdown(self, numero):
        self.chamadas.append(("contagem",))

    def show_video_frame(self, quadro):
        self.chamadas.append(("video",))

    def show_video_aviso(self, texto):
        self.chamadas.append(("aviso", texto))

    def show_trial_progress(self, feito, total):
        self.progresso.append((feito, total))

    @property
    def mensagens(self):
        return [c[1] for c in self.chamadas if c[0] == "msg"]

    @property
    def ultima(self):
        return self.mensagens[-1] if self.mensagens else ""


class _ControladorPausa(p._ParadigmController):
    """Sem QTimer: `_schedule` guarda o proximo tick, rodado na mao."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eventos = []
        self._tick = None
        self._delay_ms = 0

    def _schedule(self, delay_ms, fn):
        self._tick = fn
        self._delay_ms = int(delay_ms)

    def _emit(self, code, **extra):
        self.eventos.append((code, extra))

    # --- utilidades do teste ---
    def codigos(self):
        return [code for code, _extra in self.eventos]

    def evento(self, code):
        return dict(self.eventos).get(code, {})

    def roda_tick(self, segundos=0.0):
        """Avanca o relogio e executa O tick agendado (uma vez)."""
        RELOGIO.avanca(segundos)
        fn = self._tick
        self._tick = None
        if fn is None:
            return False
        fn()
        return True

    def roda_ate_fim_da_contagem(self, limite_s=600.0, passo_s=1.0):
        """Ticks de `passo_s` ate' a contagem terminar (ou o limite)."""
        gasto = 0.0
        while self._tick is not None and not self._pause_esperando:
            self.roda_tick(passo_s)
            gasto += passo_s
            assert gasto <= limite_s, "a contagem nao terminou"
        return gasto


def nova_pausa(pausa_bloco_sec=120.0, por_bloco=2, blocos=2):
    """Controlador no FIM do bloco 1, com a pausa entre blocos ja' armada."""
    RELOGIO.agora = 1000.0
    janela = _JanelaFake()
    controlador = _ControladorPausa(
        window=janela, marker_queue=None, state=p.SharedState(),
        config={"trials": TRIALS, "home_sec": 0.0, "me_sec": 0.0,
                "video_sec": 0.0, "mi_sec": 0.0,
                "trials_por_bloco": por_bloco, "blocos": blocos,
                "pausa_bloco_sec": pausa_bloco_sec, "contagem": 0,
                "olhos_abertos": 0.0, "olhos_fechados": 0.0,
                "repouso_ativo": 0.0, "preparacao": False,
                "priming_dir": "", "salvar_priming": False,
                "on_session_start": None})
    controlador._run_trial(por_bloco - 1)      # ultima trial do bloco 1
    controlador._tick = None                   # descarta o tick da trial
    controlador._end_block(por_bloco)          # fecha o bloco -> pausa
    return controlador, janela


p.time = RELOGIO                # `_pause_tick` le o relogio do modulo
try:
    # =====================================================================
    # 1) `_end_block` arma a pausa COM confirmacao (792 = espera real)
    # =====================================================================
    controlador, janela = nova_pausa()
    assert p.CODE_PAUSE_START in controlador.codigos(), controlador.codigos()
    assert controlador.codigos()[-1] == p.CODE_PAUSE_START, controlador.codigos()
    assert controlador._pause_confirmar is True
    assert controlador._pause_on_end is not None
    assert controlador._pause_esperando is False, "a contagem comecou parada"
    inicio = controlador.evento(p.CODE_PAUSE_START)
    assert inicio.get("bloco") == 1, inicio
    assert abs(inicio.get("duracao_prevista_s", -1) - 120.0) < 1e-9, inicio
    # tela da contagem: conta o tempo e diz o que o ESPACO faz
    assert re.search(r"\d\d:\d\d", janela.ultima), janela.ultima
    assert p.PAUSE_SKIP_HINT in janela.ultima, janela.ultima
    assert "PRESSIONE" not in janela.ultima, janela.ultima
    # a pausa mantem o tick de 250 ms da contagem
    assert controlador._delay_ms == 250, controlador._delay_ms

    # =====================================================================
    # 2) O TIMER ZERA E NADA INICIA (o bug do trial de 21/10)
    # =====================================================================
    controlador.roda_ate_fim_da_contagem()      # 120 ticks de 1 s
    assert controlador._pause_esperando is True, "a contagem nao parou no zero"
    assert controlador._trial_index == 1, controlador._trial_index
    assert p.CODE_PAUSE_END not in controlador.codigos(), \
        "o 793 saiu sem o experimentador confirmar"
    assert p.CODE_BLOCK_START not in controlador.codigos(), \
        "o proximo bloco comecou por tempo (auto-continue)"
    assert "PRESSIONE ESPACO PARA CONTINUAR" in janela.ultima, janela.ultima
    assert "bloco 2" in janela.ultima, janela.ultima
    # a tela de confirmacao NAO conta tempo (nao ha' MM:SS)
    assert not re.search(r"\d\d:\d\d", janela.ultima), janela.ultima
    # o tick da espera e' mais curto (resposta rapida ao ESPACO)
    assert controlador._delay_ms == 150, controlador._delay_ms
    # o painel do experimentador mostra o que fazer
    assert "ESPACO" in controlador.info(), controlador.info()
    # 50 s depois (bem alem do timer) continua parado: NADA automatico
    for _ in range(50):
        controlador.roda_tick(1.0)
    assert controlador._pause_esperando is True
    assert controlador._trial_index == 1
    assert p.CODE_PAUSE_END not in controlador.codigos()
    assert p.CODE_BLOCK_START not in controlador.codigos()

    # =====================================================================
    # 3) ESPACO CONFIRMA E INICIA O BLOCO (tempos reais no 793)
    # =====================================================================
    assert controlador.skip() is None                    # ESPACO do painel
    assert controlador.roda_tick(0.0) is True
    assert controlador._pause_esperando is False
    assert controlador._pause_on_end is None
    codigos = controlador.codigos()
    assert p.CODE_PAUSE_END in codigos, codigos
    assert codigos.index(p.CODE_PAUSE_END) < codigos.index(p.CODE_BLOCK_START)
    bloco_ini = controlador.evento(p.CODE_BLOCK_START)
    assert bloco_ini.get("bloco") == 2, bloco_ini
    assert controlador._trial_index == 2, controlador._trial_index
    assert janela.progresso[-1] == (3, 4), janela.progresso
    fim = controlador.evento(p.CODE_PAUSE_END)
    assert fim.get("contagem_encerrada_por") == "tempo", fim
    assert abs(fim.get("contagem_s", -1) - 120.0) < 0.01, fim
    # 120 s de contagem + 50 s de espera pela confirmacao
    assert abs(fim.get("espera_confirmacao_s", -1) - 50.0) < 0.01, fim
    assert abs(fim.get("duracao_s", -1) - 170.0) < 0.01, fim

    # =====================================================================
    # 4) ESPACO DURANTE A CONTAGEM ENCURTA, MAS NAO INICIA SOZINHO
    # =====================================================================
    controlador, janela = nova_pausa()
    controlador.roda_tick(5.0)                   # 5 s de contagem
    assert controlador._pause_esperando is False
    assert "PRESSIONE" not in janela.ultima, janela.ultima
    controlador.skip()                           # 1o ESPACO: encurta
    controlador.roda_tick(0.0)
    assert controlador._pause_esperando is True, "o ESPACO iniciou o bloco"
    assert controlador._pause_motivo == "espaco", controlador._pause_motivo
    assert controlador._trial_index == 1, controlador._trial_index
    assert p.CODE_PAUSE_END not in controlador.codigos()
    assert p.CODE_BLOCK_START not in controlador.codigos()
    assert "PRESSIONE ESPACO" in janela.ultima, janela.ultima
    # 30 s de tela de confirmacao: continua parado (nao "vaza" para o bloco)
    for _ in range(30):
        controlador.roda_tick(1.0)
    assert controlador._trial_index == 1
    assert p.CODE_BLOCK_START not in controlador.codigos()
    # 2o ESPACO (novo, com a tela ja' visivel): agora sim
    controlador.skip()
    controlador.roda_tick(0.0)
    assert p.CODE_PAUSE_END in controlador.codigos()
    assert controlador._trial_index == 2, controlador._trial_index
    fim = controlador.evento(p.CODE_PAUSE_END)
    assert fim.get("contagem_encerrada_por") == "espaco", fim
    assert abs(fim.get("contagem_s", -1) - 5.0) < 0.01, fim
    assert abs(fim.get("espera_confirmacao_s", -1) - 30.0) < 0.01, fim

    # =====================================================================
    # 5) O ESPACO QUE ENCERRA A CONTAGEM NAO VALE COMO CONFIRMACAO
    #    (um toque FISICO chega 2x: atalho do Qt + vigia global)
    # =====================================================================
    controlador, janela = nova_pausa()
    controlador.roda_tick(5.0)
    controlador.skip()                           # toque fisico (1a entrega)
    controlador.roda_tick(0.0)                   # -> tela de confirmacao
    assert controlador._pause_esperando is True
    controlador.skip()                           # o MESMO toque (2a entrega)
    controlador.roda_tick(0.0)                   # dentro do dwell: ignorado
    assert controlador._pause_esperando is True, \
        "o mesmo ESPACO encerrou a contagem E confirmou o bloco"
    assert p.CODE_BLOCK_START not in controlador.codigos()
    RELOGIO.avanca(p.PAUSE_CONFIRM_DWELL_SEC + 1.0)
    controlador.roda_tick(0.0)                   # sem pedido novo: segue parado
    assert controlador._pause_esperando is True
    assert p.CODE_BLOCK_START not in controlador.codigos()
    controlador.skip()                           # ESPACO novo, sim
    controlador.roda_tick(0.0)
    assert controlador._trial_index == 2, controlador._trial_index

    # =====================================================================
    # 6) PAUSA COM 0 s TAMBEM ESPERA CONFIRMACAO (`--pausa-bloco-sec 0`)
    # =====================================================================
    controlador, janela = nova_pausa(pausa_bloco_sec=0.0)
    controlador.roda_tick(0.0)
    assert controlador._pause_esperando is True, \
        "com 0 s o bloco comecou sozinho (auto-continue de volta)"
    assert controlador._trial_index == 1
    assert "PRESSIONE ESPACO" in janela.ultima, janela.ultima
    RELOGIO.avanca(p.PAUSE_CONFIRM_DWELL_SEC + 0.1)   # tempo de VER a tela
    controlador.skip()
    controlador.roda_tick(0.0)
    assert controlador._trial_index == 2, controlador._trial_index

    # =====================================================================
    # 7) O C (cancel_trial) NAO BURLA A CONFIRMACAO
    # =====================================================================
    controlador, janela = nova_pausa()
    controlador.roda_tick(5.0)
    controlador.skip()
    controlador.roda_tick(0.0)                    # tela de confirmacao
    antes = len(controlador.eventos)
    controlador.cancel_trial()                    # C durante a pausa: ignorado
    assert len(controlador.eventos) == antes, controlador.eventos[antes:]
    assert p.CODE_TRIAL_CANCEL not in controlador.codigos()
    assert controlador._pause_on_end is not None
    assert controlador._trial_index == 1, "o C iniciou a trial seguinte"
    assert controlador._pause_esperando is True
    RELOGIO.avanca(1.0)
    controlador.skip()
    controlador.roda_tick(0.0)
    assert controlador._trial_index == 2, controlador._trial_index

    # =====================================================================
    # 8) PREPARACAO E BASELINE CONTINUAM SOZINHOS (sem `confirmar`)
    # =====================================================================
    RELOGIO.agora = 1000.0
    chamadas = []
    controlador = _ControladorPausa(
        window=_JanelaFake(), marker_queue=None, state=p.SharedState(),
        config={"trials": TRIALS, "home_sec": 0.0, "me_sec": 0.0,
                "video_sec": 0.0, "mi_sec": 0.0, "trials_por_bloco": 2,
                "blocos": 2, "pausa_bloco_sec": 120.0, "contagem": 0,
                "olhos_abertos": 0.0, "olhos_fechados": 0.0,
                "repouso_ativo": 0.0, "preparacao": False,
                "priming_dir": "", "salvar_priming": False,
                "on_session_start": None})
    controlador._start_pause_phase(60.0, "BASELINE", lambda: chamadas.append("fim"))
    assert controlador._pause_confirmar is False
    controlador.roda_tick(61.0)                   # zerou a contagem
    assert chamadas == ["fim"], chamadas
    assert controlador._pause_esperando is False
    # ... e o ESPACO continua encurtando o baseline de imediato
    chamadas.clear()
    controlador._start_pause_phase(60.0, "BASELINE", lambda: chamadas.append("2"))
    controlador.roda_tick(3.0)
    controlador.skip()
    controlador.roda_tick(0.0)
    assert chamadas == ["2"], chamadas

    # =====================================================================
    # 9) TEXTO/API documentados (o operador sabe o que apertar)
    # =====================================================================
    assert "ESPACO" in p.PAUSE_CONFIRM_TEXT, p.PAUSE_CONFIRM_TEXT
    assert p.PAUSE_CONFIRM_DWELL_SEC > 0, p.PAUSE_CONFIRM_DWELL_SEC
    assert "ESPACO" in p.PAUSE_SKIP_HINT, p.PAUSE_SKIP_HINT
    assert p.CODE_PAUSE_START == 792 and p.CODE_PAUSE_END == 793
    assinatura = repr(__import__("inspect").signature(
        p._ParadigmController._start_pause_phase))
    assert "confirmar" in assinatura, assinatura
    # o cabecalho do paradigma explica os DOIS ESPACOS da pausa de bloco
    texto = open("eeg_motor_paradigm.py", encoding="utf-8").read()
    assert "DOIS ESPACOS" in texto.upper(), "cabecalho sem a regra da pausa"
    # segurar a tecla nao pode confirmar sozinho (auto-repeat do Qt)
    assert "setAutoRepeat(False)" in texto, "atalho do Qt com auto-repeat"
    assert "isAutoRepeat()" in texto, "keyPressEvent aceita auto-repeat"
finally:
    p.time = _time_real

print("PAUSA_CONFIRMACAO_OK",
      "timer zera e NAO inicia | ESPACO confirma e inicia | ESPACO durante a "
      "contagem encurta mas nao inicia sozinho | mesmo toque nao confirma "
      f"(dwell {p.PAUSE_CONFIRM_DWELL_SEC:g} s) | baseline segue sozinho")
