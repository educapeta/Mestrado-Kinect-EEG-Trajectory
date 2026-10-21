"""Regressao: BASELINE com tela de instrucao e BIP de inicio/fim por etapa.

Pedido do experimentador no trial real de 21/10/2026:

    "A seguir faremos a calibracao de olhos fechados e depois abertos. Quando
    clicar ESPACO vamos comecar. O bip ira indicar o inicio e depois o final."

Semantica (e' o que este teste trava):

1. antes do baseline aparece uma TELA DE INSTRUCAO (texto `BASELINE_INSTRUCAO`)
   e NADA do baseline comeca ate' o ESPACO;
2. ordem das etapas: olhos FECHADOS -> olhos ABERTOS -> repouso ativo
   (antes era ABERTOS -> FECHADOS);
3. cada etapa emite o seu marcador na ORDEM (277, 276, 784) e tem UM bip de
   INICIO (880 Hz) e UM bip de FIM (660 Hz);
4. entre o bip de fim e o inicio da proxima etapa ha' `BASELINE_GAP_SEC`;
5. o ESPACO durante uma etapa ENCURTA (mesmo comportamento cronometrado de
   antes: baseline nao tem confirmacao) e dispara o bip de fim + proxima etapa;
6. terminadas as etapas, o fluxo segue para `_origin_calibration` (aqui
   substituido por um marcador, sem hardware).

Harness: relogio fake (`p.time`), janela fake e `_schedule` capturado -- cada
tick e' rodado NA MAO. Nenhum event loop do Qt, nenhum QTimer, sem hardware.
"""
from __future__ import annotations

import time as _time_real

import eeg_motor_paradigm as p

TRIALS = [{"objeto": obj, "mao": mao, "condicao": f"{obj}_{mao}"}
          for obj, mao in (("garrafa", "direita"), ("bola", "esquerda"),
                           ("caneta", "direita"), ("bola", "direita"))]


# ---------------------------------------------------------------------------
# Harness (mesmo padrao de test_pausa_confirmacao.py)
# ---------------------------------------------------------------------------
class _RelogioFake:
    def __init__(self, agora=1000.0):
        self.agora = float(agora)

    def monotonic(self):
        return self.agora

    def avanca(self, segundos):
        self.agora += float(segundos)


RELOGIO = _RelogioFake()


class _JanelaFake:
    def __init__(self):
        self.chamadas = []

    def show_home(self):
        self.chamadas.append(("home",))

    def show_condition(self, objeto, mao, **kwargs):
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
        self.chamadas.append(("progresso", feito, total))

    @property
    def mensagens(self):
        return [c[1] for c in self.chamadas if c[0] == "msg"]

    @property
    def ultima(self):
        return self.mensagens[-1] if self.mensagens else ""


class _ControladorBaseline(p._ParadigmController):
    """Sem QTimer: `_schedule` guarda o proximo tick, rodado na mao."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eventos = []
        self._tick = None
        self._delay_ms = 0
        self.origem_chamada = 0

    def _schedule(self, delay_ms, fn):
        self._tick = fn
        self._delay_ms = int(delay_ms)

    def _emit(self, code, **extra):
        self.eventos.append((code, extra))

    def _origin_calibration(self):
        self.origem_chamada += 1

    # --- utilidades ---
    def codigos(self):
        return [code for code, _extra in self.eventos]

    def roda_tick(self, segundos=0.0):
        RELOGIO.avanca(segundos)
        fn = self._tick
        self._tick = None
        if fn is None:
            return False
        fn()
        return True

    def roda_ate(self, condicao, limite_s=1000.0, passo_s=1.0):
        """Roda ticks de `passo_s` ate' `condicao()` ou o limite."""
        gasto = 0.0
        while not condicao() and self._tick is not None:
            self.roda_tick(passo_s)
            gasto += passo_s
            assert gasto <= limite_s, "a condicao nao foi atingida"
        return gasto


def novo_baseline(olhos_fechados=120.0, olhos_abertos=120.0,
                  repouso_ativo=60.0, preparacao=False):
    RELOGIO.agora = 1000.0
    janela = _JanelaFake()
    controlador = _ControladorBaseline(
        window=janela, marker_queue=None, state=p.SharedState(),
        config={"trials": TRIALS, "home_sec": 0.0, "me_sec": 0.0,
                "video_sec": 0.0, "mi_sec": 0.0, "trials_por_bloco": 4,
                "blocos": 1, "pausa_bloco_sec": 120.0, "contagem": 0,
                "olhos_fechados": olhos_fechados,
                "olhos_abertos": olhos_abertos,
                "repouso_ativo": repouso_ativo, "preparacao": preparacao,
                "priming_dir": "", "salvar_priming": False,
                "on_session_start": None})
    return controlador, janela


BEEPS = []


def _beep_fake(freq, ms):
    BEEPS.append((int(freq), int(ms)))


p.time = RELOGIO
p._beep_async = _beep_fake          # nunca toca o buzzer de verdade
try:
    # =====================================================================
    # 1) TELA DE INSTRUCAO: nada do baseline comeca antes do ESPACO
    # =====================================================================
    controlador, janela = novo_baseline()
    BEEPS.clear()
    controlador._begin_session()
    assert controlador.codigos() == [p.CODE_SESSION], controlador.codigos()
    assert not controlador._baseline_steps, "montou o baseline antes do ESPACO"
    texto = janela.ultima
    assert texto, "nenhuma tela de instrucao apareceu"
    assert p.BASELINE_INSTRUCAO.split("\n")[0] in texto, texto
    assert "ESPAÇO" in texto, texto
    assert "FECHADOS" in texto.upper() and "ABERTOS" in texto.upper(), texto
    assert BEEPS == [], "tocou bip antes do ESPACO"
    assert controlador._tick is not None, "a tela nao espera o ESPACO"

    # =====================================================================
    # 2) ESPACO inicia a PRIMEIRA etapa = olhos FECHADOS, com bip de INICIO
    # =====================================================================
    controlador.skip()                     # ESPACO do experimentador
    controlador.roda_tick(0.0)
    codigos = controlador.codigos()
    assert codigos == [p.CODE_SESSION, p.CODE_EYES_CLOSED], codigos
    assert BEEPS == [(p.BASELINE_BEEP_INICIO_HZ, p.BASELINE_BEEP_MS)], BEEPS
    assert "FECHADOS" in janela.ultima.upper(), janela.ultima
    assert "BIP" in janela.ultima.upper(), janela.ultima
    # a tela de etapa NAO e' a de confirmacao: o baseline continua sozinho
    assert controlador._pause_confirmar is False
    assert controlador._delay_ms == 250, controlador._delay_ms

    # =====================================================================
    # 3) Fim dos 120 s -> bip de FIM (660 Hz) -> espera -> proxima etapa =
    #    ABERTOS (a ordem FECHADOS -> ABERTOS e' o que o experimentador pediu)
    # =====================================================================
    BEEPS.clear()
    fim = (p.BASELINE_BEEP_FIM_HZ, p.BASELINE_BEEP_MS)
    inicio = (p.BASELINE_BEEP_INICIO_HZ, p.BASELINE_BEEP_MS)
    controlador.roda_ate(lambda: fim in BEEPS)
    assert p.CODE_EYES_OPEN not in controlador.codigos(), \
        "a proxima etapa comecou ANTES de tocar o bip de fim"
    # houve a espera `BASELINE_GAP_SEC` entre o bip de fim e o inicio da proxima
    assert controlador._delay_ms == p.BASELINE_GAP_SEC * 1000, \
        controlador._delay_ms
    controlador.roda_tick(0.0)             # fim da espera -> proxima etapa
    assert controlador.codigos() == [
        p.CODE_SESSION, p.CODE_EYES_CLOSED, p.CODE_EYES_OPEN], \
        controlador.codigos()
    assert BEEPS[-1] == inicio, BEEPS
    assert "ABERTOS" in janela.ultima.upper(), janela.ultima

    # =====================================================================
    # 4) ABERTOS -> repouso ATIVO -> terminou: chama a origem (sem hardware)
    # =====================================================================
    BEEPS.clear()
    controlador.roda_ate(lambda: p.CODE_BASELINE_REST in controlador.codigos())
    assert controlador.codigos()[-1] == p.CODE_BASELINE_REST, \
        controlador.codigos()
    assert "REPOUSO" in janela.ultima.upper(), janela.ultima
    assert (p.BASELINE_BEEP_FIM_HZ, p.BASELINE_BEEP_MS) in BEEPS
    assert (p.BASELINE_BEEP_INICIO_HZ, p.BASELINE_BEEP_MS) in BEEPS
    controlador.roda_ate(lambda: controlador.origem_chamada == 1)
    assert controlador.origem_chamada == 1, "nao seguiu para a calibracao"
    assert BEEPS[-1] == (p.BASELINE_BEEP_FIM_HZ, p.BASELINE_BEEP_MS), \
        "a ultima etapa nao teve bip de fim"

    # =====================================================================
    # 5) ESPACO durante uma etapa ENCURTA (e dispara o fim + proxima)
    # =====================================================================
    controlador, janela = novo_baseline()
    BEEPS.clear()
    controlador._begin_session()
    controlador.skip()
    controlador.roda_tick(0.0)             # etapa FECHADOS comecou
    controlador.roda_tick(3.0)             # 3 s de 120 s
    BEEPS.clear()
    controlador.skip()                     # ESPACO encurta
    controlador.roda_tick(0.0)             # encerra a etapa (bip de fim)
    assert (p.BASELINE_BEEP_FIM_HZ, p.BASELINE_BEEP_MS) in BEEPS, BEEPS
    controlador.roda_tick(0.0)             # espera do gap -> proxima etapa
    assert p.CODE_EYES_OPEN in controlador.codigos(), \
        "o ESPACO nao encurtou a etapa do baseline"
    assert BEEPS[-1] == (p.BASELINE_BEEP_INICIO_HZ, p.BASELINE_BEEP_MS), BEEPS

    # =====================================================================
    # 6) Texto/constantes: o operador e o participante sabem o que esperar
    # =====================================================================
    assert p.CODE_EYES_OPEN == 276 and p.CODE_EYES_CLOSED == 277, "marcadores"
    assert p.CODE_BASELINE_REST == 784
    instr = p.BASELINE_INSTRUCAO
    for chave in ("ESPAÇO", "BIP", "INÍCIO", "FINAL", "FECHADOS", "ABERTOS"):
        assert chave in instr.upper() or chave in instr, (chave, instr)
    assert p.BASELINE_BEEP_INICIO_HZ != p.BASELINE_BEEP_FIM_HZ, \
        "inicio e fim com o mesmo tom: o participante nao distingue"
    assert p.BASELINE_GAP_SEC > 0, p.BASELINE_GAP_SEC
finally:
    p.time = _time_real

print("BASELINE_INSTRUCOES_OK",
      "instrucao antes de comecar (ESPAÇO) | ordem FECHADOS->ABERTOS->repouso "
      "| bip de INICIO (880) e FIM (660) em cada etapa | ESPACO encurta | "
      "marcadores 277/276/784 na ordem | segue para a origem")
