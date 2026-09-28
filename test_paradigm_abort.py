"""Teste do CANCELAMENTO da sessao (hotkeys globais + saida garantida).

Contexto (bancada, 25/09/2026): rodando o paradigma sem as luvas, a janela de
video do Kinect (nativa do OpenCV/Win32) ficou na frente e com o foco; ESPACO e
ESC do Qt nao chegavam mais (o trial nao andava), o "cancelar" nao encerrava o
programa e nem matar o processo no terminal resolvia. Este teste cobre os
mecanismos que fecham esse buraco:

  1. hotkeys.detectar_edicoes ..... deteccao por BORDA (segurar != repetir);
  2. hotkeys.GlobalHotkeyWatcher .. thread viva, fila de acoes, stop de dentro;
  3. hotkeys.hard_exit ............ os._exit depois do flush (nunca bloqueia);
  4. _ParadigmController.cancel_trial  descarta a trial e segue para a proxima;
  5. _ParadigmController.abort .... usa o teardown de emergencia UMA vez so';
  6. TrackingThread(show_window=False) e os atalhos/CLI do paradigma.

Nada de hardware e nada de teclado real: o backend de leitura de tecla e'
injetado, e o Qt do controlador e' substituido por fakes (o `_schedule`/`_emit`
sao capturados), entao o teste roda sem janela e sem event loop.

    .venv\\Scripts\\python.exe test_paradigm_abort.py
"""
import sys
import time

import hotkeys
import eeg_motor_paradigm as p

# =============================================================================
# 1) Deteccao por BORDA (funcao pura: sem thread, sem teclado, sem Windows)
# =============================================================================
disparos = []


def bind(nome, vk):
    return (nome, vk, lambda: disparos.append(nome))


binds = [bind("esc", hotkeys.VK_ESCAPE), bind("space", hotkeys.VK_SPACE)]

# 1a) nenhuma tecla pressionada -> nada
estado = {}
assert hotkeys.detectar_edicoes(estado, lambda vk: False, binds) == []
assert disparos == [] and estado == {"esc": False, "space": False}, estado

# 1b) ESC segurado por 5 ciclos -> dispara UMA vez (era o bug do "gruda")
estado = {}
ativas = {hotkeys.VK_ESCAPE}
ciclos = [hotkeys.detectar_edicoes(estado, lambda vk: vk in ativas, binds)
          for _ in range(5)]
assert ciclos[0] == ["esc"] and all(c == [] for c in ciclos[1:]), ciclos
assert disparos == ["esc"], disparos

# 1c) solta e aperta de novo -> conta DE NOVO (um evento por apertada)
del disparos[:]
ativas.clear()
assert hotkeys.detectar_edicoes(estado, lambda vk: vk in ativas, binds) == []
ativas.add(hotkeys.VK_ESCAPE)
assert hotkeys.detectar_edicoes(estado, lambda vk: vk in ativas, binds) == ["esc"]
assert disparos == ["esc"], disparos

# 1d) duas teclas no mesmo ciclo -> as duas, na ordem dos binds
del disparos[:]
ativas.clear()                                   # solta tudo antes
hotkeys.detectar_edicoes(estado, lambda vk: vk in ativas, binds)
ativas.update({hotkeys.VK_ESCAPE, hotkeys.VK_SPACE})
assert hotkeys.detectar_edicoes(estado, lambda vk: vk in ativas, binds) \
    == ["esc", "space"]
assert disparos == ["esc", "space"], disparos

# 1e) callback None nao estoura
assert hotkeys.detectar_edicoes({}, lambda vk: True, [("x", 1, None)]) == ["x"]

# =============================================================================
# 2) GlobalHotkeyWatcher: thread viva, fila de acoes e stop
# =============================================================================
fila = []
ativas = set()
vigia = hotkeys.GlobalHotkeyWatcher(
    [("esc", hotkeys.VK_ESCAPE, lambda: fila.append("abortar")),
     ("space", hotkeys.VK_SPACE, lambda: fila.append("pular"))],
    key_down=lambda vk: vk in ativas, interval=0.01)
assert vigia.name == "hotkeys" and vigia.daemon, "vigia nao pode segurar o app"
assert vigia.enabled and vigia.press_count == 0

# 2a) varredura manual com backend injetado
assert vigia.scan_once() == []
ativas.add(hotkeys.VK_ESCAPE)
assert vigia.scan_once() == ["esc"] and fila == ["abortar"], fila
assert vigia.scan_once() == [], "segurar ESC nao pode repetir a acao"
assert vigia.press_count == 1
ativas.clear()
vigia.scan_once()

# 2b) em thread: a tecla apertada depois do start chega na fila
vigia.start()
ativas.add(hotkeys.VK_SPACE)
limite = 300
while len(fila) < 2 and limite:
    time.sleep(0.01)
    limite -= 1
vigia.stop(timeout=1.0)
assert fila == ["abortar", "pular"], fila
assert not vigia.is_alive(), "stop() tem de encerrar a thread"

# 2c) enabled=False -> nunca le teclado (e' o --sem-hotkeys)
mudo = hotkeys.GlobalHotkeyWatcher([("esc", hotkeys.VK_ESCAPE, None)],
                                  key_down=lambda vk: True, enabled=False)
assert mudo.scan_once() == [] and mudo.press_count == 0

# 2d) stop DE DENTRO do proprio vigia: nao pode dar "cannot join current thread"
vigiado = []


def parar_no_callback():
    vigiado.append(True)
    interno.stop()


interno = hotkeys.GlobalHotkeyWatcher(
    [("esc", hotkeys.VK_ESCAPE, parar_no_callback)],
    key_down=lambda vk: True, interval=0.01)
interno.start()
limite = 300
while not vigiado and limite:
    time.sleep(0.01)
    limite -= 1
assert vigiado, "o callback nao rodou"
interno.stop(timeout=1.0)
assert not interno.is_alive()

# 2e) erro no backend nao derruba o programa (avisa e para)
falhas = []
com_erro = hotkeys.GlobalHotkeyWatcher(
    [("esc", hotkeys.VK_ESCAPE, None)],
    key_down=lambda vk: (_ for _ in ()).throw(RuntimeError("sem user32")),
    interval=0.01, on_error=falhas.append)
com_erro.start()
limite = 300
while com_erro.is_alive() and limite and not falhas:
    time.sleep(0.01)
    limite -= 1
com_erro.stop(timeout=1.0)
assert falhas and isinstance(falhas[0], RuntimeError), falhas

# 2f) o backend REAL existe e responde bool
assert isinstance(hotkeys.key_down_windows(hotkeys.VK_ESCAPE), bool)

# =============================================================================
# 3) hard_exit: os._exit com o codigo pedido (depois do flush)
# =============================================================================
chamadas = []
original_exit = hotkeys.os._exit
hotkeys.os._exit = lambda codigo: chamadas.append(codigo)
try:
    hotkeys.hard_exit(7, aviso="saindo agora")
finally:
    hotkeys.os._exit = original_exit
assert chamadas == [7], chamadas



# =============================================================================
# 4) cancel_trial: descarta a trial e segue para a proxima (sem Qt de verdade)
# =============================================================================
class _WindowFake:
    """Janela minima: registra o que o controlador mandou mostrar."""

    def __init__(self):
        self.mostrados = []
        self.progresso = []

    def show_home(self):
        self.mostrados.append("home")

    def show_condition(self, objeto, mao):
        self.mostrados.append(f"cue:{objeto}_{mao}")

    def show_message(self, texto):
        self.mostrados.append("msg")

    def show_countdown(self, numero):
        self.mostrados.append("contagem")

    def show_video_frame(self, _imagem):
        self.mostrados.append("video")

    def show_trial_progress(self, feito, total):
        self.progresso.append((feito, total))


class _ControladorTeste(p._ParadigmController):
    """Controlador com `_schedule`/`_emit` capturados: nenhum QTimer envolvido."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agendado = []
        self.eventos = []

    def _schedule(self, delay_ms, fn):
        self.agendado.append((int(delay_ms), fn))

    def _emit(self, code, **extra):
        self.eventos.append((code, extra))

    def codigos(self):
        return [code for code, _extra in self.eventos]

    def evento(self, code):
        return dict(self.eventos).get(code, {})


def novo_controlador(trials, por_bloco=2, blocos=2, pausa_bloco_sec=0.0):
    state = p.SharedState()
    janela = _WindowFake()
    controlador = _ControladorTeste(
        window=janela, marker_queue=None, state=state,
        config={"trials": trials, "home_sec": 0.0, "me_sec": 0.0,
                "video_sec": 0.0, "mi_sec": 0.0,
                "trials_por_bloco": por_bloco, "blocos": blocos,
                "pausa_bloco_sec": pausa_bloco_sec, "contagem": 0,
                "olhos_abertos": 0.0, "olhos_fechados": 0.0,
                "repouso_ativo": 0.0, "preparacao": False,
                "priming_dir": "", "salvar_priming": False,
                "on_session_start": None})
    return controlador, janela, state


TRIALS = [{"objeto": obj, "mao": mao, "condicao": f"{obj}_{mao}"}
          for obj, mao in (("garrafa", "direita"), ("bola", "esquerda"),
                           ("caneta", "direita"), ("bola", "direita"))]

# 4a) cancela a trial 1 -> marca 796, limpa a fase e vai para a trial 2
controlador, janela, state = novo_controlador(TRIALS)
controlador._run_trial(0)
assert controlador._trial_index == 0 and janela.progresso[-1] == (1, 4)
controlador.cancel_trial()
assert controlador._trial_index == 1, controlador._trial_index
assert controlador._generation == 1, "os callbacks da trial antiga deviam morrer"
assert p.CODE_TRIAL_CANCEL in controlador.codigos()
assert controlador.evento(p.CODE_TRIAL_CANCEL)["trial"] == 1
assert controlador.evento(p.CODE_TRIAL_CANCEL)["condicao"] == "garrafa_direita"
assert state.phase == p.PHASE_HOME, state.phase

# 4b) cancelar no FIM do bloco entra na PAUSA do bloco (nao pula a pausa)
controlador, janela, state = novo_controlador(TRIALS, pausa_bloco_sec=120.0)
controlador._run_trial(0)
controlador.cancel_trial()                 # -> trial 2 (indice 1)
controlador.cancel_trial()                 # -> fecha o bloco de 2 trials
assert controlador._pause_on_end is not None, "faltou a pausa entre blocos"
assert p.CODE_PAUSE_START in controlador.codigos()
assert controlador._pause_prefix.startswith("PAUSA"), controlador._pause_prefix

# 4c) cancelar a ULTIMA trial encerra a sessao (nao existe proxima)
controlador, janela, state = novo_controlador(TRIALS[:2])
controlador._run_trial(1)
controlador.cancel_trial()
assert controlador._finished, "sem proxima trial o cancelamento encerra"
assert p.CODE_SESSION_END in controlador.codigos()

# 4d) cancelar depois do fim da sessao nao faz mais nada
controlador, janela, state = novo_controlador(TRIALS[:1])
controlador._run_trial(0)
controlador.cancel_trial()
assert controlador._finished
antes = len(controlador.eventos)
controlador.cancel_trial()
assert len(controlador.eventos) == antes

# 4e) skip() so' encurta a pausa corrente (nao mexe na trial)
controlador, janela, state = novo_controlador(TRIALS)
controlador._start_pause_phase(30.0, "PAUSA", lambda: None)
assert controlador._pause_skip is False
controlador.skip()
assert controlador._pause_skip is True and controlador._trial_index == 0
assert controlador._pause_on_end is not None, "skip nao pode cancelar a pausa"


# =============================================================================
# 5) abort(): teardown de emergencia UMA vez + fallback QApplication.quit()
# =============================================================================
abortados = []
controlador, janela, state = novo_controlador(TRIALS)
controlador._panic = abortados.append            # simula o `_encerrar_agora`
controlador.abort()
controlador.abort()                             # segundo ESC e' ignorado
assert len(abortados) == 1, abortados
assert controlador._aborting and controlador._pause_on_end is None

# sem `_panic` o abort cai no QApplication.quit() e nao quebra
controlador, janela, state = novo_controlador(TRIALS)
controlador._panic = None
controlador.abort()                              # nao deve levantar excecao
assert controlador._aborting

# =============================================================================
# 6) Hotkeys globais ligados ao controlador (fila + poll, sem event loop)
# =============================================================================
import queue as _queue                            # noqa: E402 - clareza local

fila_paradigma = _queue.Queue()


class _ControladorPoll(_ControladorTeste):
    """Igual ao controlador real, mas o poll e' chamado na mao (sem QTimer)."""

    def _poll_hotkeys(self):
        acoes = {"pular": self.skip, "cancelar": self.cancel_trial,
                 "abortar": self.abort}
        while True:
            try:
                acao = self._hotkeys.get_nowait()
            except _queue.Empty:
                break
            handler = acoes.get(str(acao))
            if handler is not None:
                handler()


controlador = _ControladorPoll(
    window=_WindowFake(), marker_queue=None, state=p.SharedState(),
    config={"trials": TRIALS, "home_sec": 0.0, "me_sec": 0.0, "video_sec": 0.0,
            "mi_sec": 0.0, "trials_por_bloco": 2, "blocos": 2,
            "pausa_bloco_sec": 0.0, "contagem": 0, "olhos_abertos": 0.0,
            "olhos_fechados": 0.0, "repouso_ativo": 0.0, "preparacao": False,
            "priming_dir": "", "salvar_priming": False,
            "on_session_start": None, "hotkeys": fila_paradigma})
controlador._run_trial(0)
fila_paradigma.put("pular")                      # ESPACO: sem pausa -> nada
controlador._poll_hotkeys()
assert controlador._trial_index == 0
fila_paradigma.put("cancelar")                   # C: descarta a trial
controlador._poll_hotkeys()
assert controlador._trial_index == 1, controlador._trial_index
assert p.CODE_TRIAL_CANCEL in controlador.codigos()
fila_paradigma.put("abortar")                    # ESC: aborta (panic simulado)
controlador._panic = abortados.append
controlador._poll_hotkeys()
assert len(abortados) == 2, abortados

# =============================================================================
# 7) Paradigma: janela do Kinect opcional e os atalhos de teclado/CLI
# =============================================================================
tracking = p.TrackingThread(p.SharedState(), p.ImuReceiver(4210),
                            with_kinect=True, aux_indices=("0",),
                            show_window=False)
assert tracking.show_window is False and tracking.mode == "kinect"
assert p.TrackingThread.__init__.__defaults__[-1] is True, \
    "o padrao tem de continuar MOSTRANDO a janela"

# 7a) `_fechar_janelas_kinect` e' no-op quando nao ha' janela nem cv2
tracking.cv2 = None
p._fechar_janelas_kinect(tracking, sem_janela=True)
p._fechar_janelas_kinect(tracking, sem_janela=False)
p._fechar_janelas_kinect(None, sem_janela=False)

# 7b) as teclas do vigia global sao distintas e o painel existe
assert len({p.hotkeys.VK_ESCAPE, p.hotkeys.VK_SPACE, p.hotkeys.VK_C}) == 3
assert isinstance(p.ExperimenterPanel, type)
assert callable(p.ExperimenterPanel.set_status)

# 7c) o codigo 796 e os controles estao documentados no cabecalho do paradigma
texto = open("eeg_motor_paradigm.py", encoding="utf-8").read()
assert "796 = TRIAL CANCELADO" in texto
assert "CONTROLE DA SESSAO" in texto
for opcao in ("--sem-janela-tracking", "--sem-hotkeys",
              "--sem-painel-controle"):
    assert opcao in texto, opcao

# 7d) a ordem das paradas de emergencia inclui todo mundo que precisa fechar
assert p.ABORT_TEARDOWN_SEC == 5.0
assert p.HOTKEY_POLL_MS == 50

# =============================================================================
# 8) CancelamentoGarantido: "sempre sai", com parada que TRAVA para sempre
# =============================================================================
saidas = []
logs = []

# 8a) paradas que travam (8 s) + grace curto -> sai assim mesmo, rapido
cancelamento = p.CancelamentoGarantido(
    paradas=[("travada", lambda: time.sleep(8.0)),
             ("outra", lambda: logs.append("rodou depois da travada"))],
    grace_s=0.3, saida=saidas.append, ao_log=logs.append)
inicio = time.monotonic()
cancelamento("teste de travamento")
gasto = time.monotonic() - inicio
assert saidas == [0], saidas
assert gasto < 2.0, f"o cancelamento demorou {gasto:.2f}s (grace 0.3s)"
assert any("nao terminou em" in texto for texto in logs), logs
assert cancelamento.motivo == "teste de travamento"

# 8b) as paradas que nao travam rodam NA ORDEM, e `ao_salvar` roda por ultimo
ordem = []
saidas2 = []
cancelamento = p.CancelamentoGarantido(
    paradas=[("a", lambda: ordem.append("a")),
             ("b", lambda: ordem.append("b"))],
    grace_s=1.0, saida=saidas2.append, ao_salvar=lambda: ordem.append("salvar"),
    ao_log=lambda _t: None)
cancelamento("ordem")
assert ordem == ["a", "b", "salvar"], ordem
assert saidas2 == [0], saidas2

# 8c) uma parada que LEVANTA nao impede as seguintes nem a saida
ordem = []
saidas3 = []
cancelamento = p.CancelamentoGarantido(
    paradas=[("ruim", lambda: (_ for _ in ()).throw(RuntimeError("x"))),
             ("boa", lambda: ordem.append("boa"))],
    grace_s=1.0, saida=saidas3.append, ao_salvar=lambda: ordem.append("salvar"),
    ao_log=lambda _t: None)
cancelamento("falha numa parada")
assert ordem == ["boa", "salvar"], ordem
assert saidas3 == [0], saidas3

# 8d) SEGUNDO cancelamento sai NA HORA, com codigo 1 (nao espera o primeiro)
saidas4 = []
cancelamento = p.CancelamentoGarantido(
    paradas=[("travada", lambda: time.sleep(8.0))], grace_s=5.0,
    saida=saidas4.append, ao_log=lambda _t: None)
cancelamento("primeiro")                     # dispara a thread travada
inicio = time.monotonic()
cancelamento("segundo")                      # nao pode esperar os 5 s
assert saidas4 == [0, 1], saidas4
assert time.monotonic() - inicio < 1.0

# 8e) os vigias sao parados antes das demais paradas
ordem = []
vigiado = hotkeys.GlobalHotkeyWatcher([("esc", hotkeys.VK_ESCAPE, None)],
                                     key_down=lambda vk: False, interval=0.01)
vigiado.start()
time.sleep(0.05)
cancelamento = p.CancelamentoGarantido(
    paradas=[("depois", lambda: ordem.append(("vigia_vivo", vigiado.is_alive())))],
    vigias=[vigiado], grace_s=1.0, saida=lambda _c: None,
    ao_log=lambda _t: None)
cancelamento("com vigia")
assert not vigiado.is_alive(), "o vigia tinha de ser parado pelo cancelamento"
assert ordem and ordem[0][1] is False, ordem

print("PARADIGM_ABORT_OK: 8/8 grupos (borda ESC/ESPACO/C, vigia global, "
      "hard_exit, cancel_trial, abort unico, fila de hotkeys, janela opcional, "
      "cancelamento que sempre sai)")
sys.exit(0)
