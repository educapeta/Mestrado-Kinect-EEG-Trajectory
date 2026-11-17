"""Cancelamento GLOBAL da sessao por teclado + saida GARANTIDA do processo.

PROBLEMA MEDIDO (bancada, 25/09/2026)
  Rodando o paradigma ME/MI sem as luvas, a janela de video do Kinect (a de
  tracking, criada pelo OpenCV/Win32 em OUTRA thread) ficou NA FRENTE e com o
  foco; a partir dai ESPACO e ESC nao faziam mais nada (o trial nao andava) e,
  ao tentar cancelar, o processo NAO morria -- nem com o comando do terminal.

  Causa: as teclas do paradigma sao tratadas pelo `keyPressEvent` do Qt
  (`_StimulusCanvas`), que so' recebe evento quando a JANELA DO QT tem o foco.
  A janela do OpenCV e' uma janela NATIVA do Win32 (o build do opencv-python
  desta maquina usa WIN32UI, nao Qt): ela rouba o foco e o Qt deixa de ver as
  teclas. E o Ctrl+C nao interrompe codigo nativo (cv2.waitKey/pykinect2/gds),
  entao o `finally` do programa pode ficar preso mesmo depois do Ctrl+C.

SOLUCAO (este modulo)
  * `GlobalHotkeyWatcher`: le o estado das teclas no WINDOWS INTEIRO pelo
    `GetAsyncKeyState` -- funciona com QUALQUER janela em foco, inclusive a do
    OpenCV e a tela cheia do participante. Deteccao por BORDA (solto ->
    pressionado): segurar a tecla NAO repete, uma tecla = um evento.
    Nao usa dependencia nova (ctypes e' da stdlib; pynput/keyboard nao entram no
    `requirements.txt`).
  * `hard_exit()`: saida IMEDIATA do processo (`os._exit`) depois do flush, sem
    `join` de thread nenhuma -- e' o que garante que "cancelar" realmente
    termine, mesmo com a thread de tracking presa no Kinect/OpenCV.

Uso (no paradigma):
    watcher = GlobalHotkeyWatcher([
        ("esc", VK_ESCAPE, lambda: fila.put("abortar")),
        ("space", VK_SPACE, lambda: fila.put("pular")),
    ])
    watcher.start()
    ...
    watcher.stop()
    hard_exit(0)

IMPORTANTE: os callbacks rodam na thread do vigia, NAO na thread do Qt. Nada de
tocar em Qt/store do paradigma direto no callback: publique num `queue.Queue` e
consuma num QTimer da thread principal (foi o que o paradigma fez).
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading

# Codigos de tecla virtual do Windows (VK_*). Sao os que o `GetAsyncKeyState`
# aceita; os nomes seguem a tabela da Microsoft.
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_Q = 0x51
VK_C = 0x43

#: Valor de retorno do `GetAsyncKeyState` que indica "pressionada AGORA"
#: (bit alto). O bit baixo ("pressionada desde a ultima chamada") e' ignorado:
#: a deteccao por borda do Python ja' faz esse papel sem depender de quem mais
#: chamou a API no meio do caminho.
_DOWN_MASK = 0x8000


def _user32():
    """Handle do user32 do Windows (None fora do Windows -> vigia inerte)."""
    if sys.platform != "win32":
        return None
    try:
        return ctypes.windll.user32
    except Exception:                       # noqa: BLE001 - ambiente sem Win32
        return None


def key_down_windows(vk):
    """True se a tecla `vk` estiver PRESSIONADA agora (Windows inteiro).

    Servidor de X / macOS: devolve False (o vigia nao faz nada e o programa
    segue com as teclas do Qt).
    """
    user32 = _user32()
    if user32 is None:
        return False
    return bool(int(user32.GetAsyncKeyState(int(vk))) & _DOWN_MASK)


def detectar_edicoes(estado_anterior, key_down, binds):
    """Dispara os callbacks das teclas que passaram de SOLTA para PRESSIONADA.

    Funcao PURA (testavel sem teclado, sem thread e sem Windows):
      * `key_down(vk) -> bool`  diz o estado da tecla neste ciclo;
      * `estado_anterior` e' o dict nome->bool do ciclo anterior, ATUALIZADO
        aqui (mutado no lugar, para o chamador nao precisar recriar);
      * `binds` e' a lista de `(nome, vk, callback)`.

    Devolve a lista de nomes que dispararam neste ciclo. Deteccao por BORDA:
    manter a tecla pressionada conta UMA vez; soltar e apertar de novo conta de
    novo (era o que faltava: sem isso, ESC/ESPACO "grudavam" e disparavam em
    rajada).
    """
    disparados = []
    for nome, vk, callback in binds:
        pressionada = bool(key_down(vk))
        antes = bool(estado_anterior.get(nome, False))
        estado_anterior[nome] = pressionada
        if pressionada and not antes:
            disparados.append(nome)
            if callback is not None:
                callback()
    return disparados


class GlobalHotkeyWatcher(threading.Thread):
    """Vigia teclas GLOBAIS em thread daemon propria.

    Args:
        binds: lista de `(nome, vk, callback)`.
        key_down: backend de leitura; padrao `key_down_windows`. Injetavel para
            os testes (sem teclado).
        interval: periodo da varredura em segundos (padrao 20 ms: ESC/ESPACO
            respondem antes do proximo quadro de video).
        enabled: False mantem a thread viva mas SEM ler teclado (usado pelo
            `--sem-hotkeys` quando outro programa usa ESC/ESPACO).
        on_error: callable(excecao) chamado se a varredura falhar; sem ele a
            falha marca a thread para parar em vez de girar em erro.
    """

    def __init__(self, binds, key_down=None, interval=0.02, enabled=True,
                 on_error=None, name="hotkeys"):
        super().__init__(daemon=True, name=name)
        self.binds = [(str(nome), int(vk), callback)
                      for nome, vk, callback in binds]
        self._key_down = key_down or key_down_windows
        self.interval = max(0.005, float(interval))
        self.enabled = bool(enabled)
        self.on_error = on_error
        self.press_count = 0               # quantos eventos ja' dispararam
        self._estado = {}
        self._running = threading.Event()
        self._running.set()

    # ------------------------------------------------------------------ API
    def scan_once(self):
        """Um ciclo de varredura; devolve os nomes disparados (testes usam)."""
        if not self.enabled:
            return []
        disparados = detectar_edicoes(self._estado, self._key_down, self.binds)
        self.press_count += len(disparados)
        return disparados

    def run(self):
        while self._running.is_set():
            try:
                self.scan_once()
            except Exception as exc:         # noqa: BLE001 - nunca derruba a app
                if self.on_error is not None:
                    self.on_error(exc)
                self._running.clear()
            self._running.wait(self.interval)

    def stop(self, timeout=1.0):
        """Para a varredura (bounded: nao trava o encerramento da sessao).

        Se o `stop` vier de DENTRO do proprio vigia (um callback que decidiu
        parar), o join e' pulado: juntar a thread atual seria erro em Python
        ("cannot join current thread").
        """
        self._running.clear()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=timeout)


def hard_exit(codigo=0, flush=True, aviso=""):
    """Encerra o processo IMEDIATAMENTE, sem esperar thread/`join` nenhum.

    Motivo: quando a thread de tracking esta' presa dentro de codigo NATIVO
    (cv2.waitKey, pykinect2, gds) nenhum `join`/`finally` volta a rodar. Aqui o
    unico trabalho antes de sair e' dar flush nos buffers (o CSV/JSON ja' foram
    fechados por quem chamou) e a saida e' `os._exit` -- que NAO roda atexit,
    nao espera thread daemon e nao pode ser bloqueada.
    """
    if flush:
        if aviso:
            try:
                sys.stdout.write(aviso + "\n")
            except Exception:               # noqa: BLE001 - stdout ja fechado
                pass
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:               # noqa: BLE001 - stdout ja fechado
                pass
    os._exit(int(codigo))

