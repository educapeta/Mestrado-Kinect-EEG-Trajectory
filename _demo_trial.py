"""Demo OFFLINE do Programa 1 (sem EEG e sem Kinect) para ver a tela do trial
e o recurso de VIDEO EM CAMERA LENTA usando a WEBCAM no lugar do Kinect.

O que roda de verdade neste demo:
  * a MESMA janela de estimulos (StimulusWindow/_StimulusCanvas: cruz, marca de
    home, garrafa/bola/caneta desenhadas, seta do lado da mao) e o MESMO
    controlador (_ParadigmController) do eeg_motor_paradigm, com as fases,
    tempos e marcadores identicos;
  * a MESMA logica do clipe de priming: buffer circular de quadros e, no fim da
    ME, o trecho central do movimento repetido (0,5x = cada quadro 2x);
  * os marcadores sao impressos no console (como o muxer faz na aquisicao),
    com o indice de amostra estimado a 500 Hz.

O que muda em relacao a aquisicao real:
  * o video vem de cv2.VideoCapture(--camera), nao do Kinect;
  * nao ha EEG/pipeline/muxer (nada e gravado em CSV);
  * a calibracao de origem e' simulada em (0,0,0).

Sequencia de cada trial (12 s com os tempos reais):
  REPOUSO 2 s (cruz + marca home) -> CUE+ME 4 s (objeto + seta; MOVIMENTE-SE!)
  -> PAUSA+VIDEO 2 s (clipe do seu movimento em 0,5x) -> MI 4 s (mesmo cue).

Quando o clipe e' gravado: ele mostra --span segundos (padrao 1 s) terminando
--lag segundos (padrao 1 s) antes do FIM da ME - ou seja, com os tempos reais
o trecho aproveitado e' ~2-3 s da ME. Mexa a mao/corpo nesse intervalo.
Para um clipe mais longo, aumente --span E --video-sec: a relacao para 0,5x e'
video_sec = 2 x span.

Teclas: ESPACO inicia/pula pausas | ESC encerra. Feche a janela para sair.

Uso:
    python _demo_trial.py                     # 6 trials (1 por condicao)
    python _demo_trial.py --rapido            # tempos curtos (~4 s/trial)
    python _demo_trial.py --camera 1 --sem-preview
    python _demo_trial.py --trials-por-condicao 2 --seed 3
    python _demo_trial.py --sem-video         # so as cues, sem camera lenta
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import eeg_motor_paradigm as p


class WebcamClip(threading.Thread):
    """Captura a webcam e monta o clipe em camera lenta (0,5x) sob demanda.

    Reproduz o comportamento da thread de tracking do Programa 1: buffer
    circular dos ultimos quadros + clipe com os quadros duplicados quando o
    controlador pede (state.request_slowmo(), no fim da ME).
    """

    JANELA = "webcam - o que esta sendo capturado (experimentador)"

    def __init__(self, state, camera=0, fps=p.SLOWMO_CAPTURE_FPS,
                 buffer_sec=p.SLOWMO_BUFFER_SEC, span=p.SLOWMO_SPAN_SEC,
                 lag=p.SLOWMO_END_LAG_SEC, preview=True, mirror=True,
                 destaque=2, clip_size=(640, 360)):
        super().__init__(daemon=True)
        self.state = state
        self.camera = camera
        self.intervalo = 1.0 / max(1.0, float(fps))
        self.span = float(span)
        self.lag = float(lag)
        self.preview = preview
        self.mirror = mirror
        self.destaque = max(1, int(destaque))
        self.clip_size = tuple(clip_size)
        self.running = True
        self.ok = False
        self.quadros = 0
        self.clipes = 0
        self._buffer = deque(maxlen=int(float(fps) * float(buffer_sec)))

    # ------------------------------------------------------------------ run
    def run(self):
        cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW)
        self.ok = cap.isOpened()
        if not self.ok:
            print(f"[webcam] NAO consegui abrir a camera {self.camera}: o demo "
                  "segue rodando, mas a pausa fica sem video.", flush=True)
            return
        print(f"[webcam] camera {self.camera} aberta | buffer de "
              f"{self._buffer.maxlen} quadros | clipe = {self.span:g} s reais "
              f"em 0,5x | preview={'sim' if self.preview else 'nao'}",
              flush=True)
        proximo_preview = 0.0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            agora = time.monotonic()
            pequeno = cv2.resize(frame, self.clip_size,
                                 interpolation=cv2.INTER_AREA)
            self._buffer.append((agora, pequeno))
            self.quadros += 1
            pedido = self.state.take_slowmo_request()
            if pedido is not None:
                self._monta_clipe(pedido)
            if self.preview and agora >= proximo_preview:
                proximo_preview = agora + 0.1        # preview a 10 fps
                self._mostra_preview(frame)
            sobra = self.intervalo - (time.monotonic() - agora)
            if sobra > 0:
                time.sleep(sobra)
        cap.release()
        if self.preview:
            try:
                cv2.destroyWindow(self.JANELA)
            except cv2.error:
                pass

    def _mostra_preview(self, frame):
        """Janela pequena para o experimentador ver o que esta sendo capturado."""
        vista = cv2.resize(frame, (360, 270), interpolation=cv2.INTER_AREA)
        if self.mirror:
            vista = cv2.flip(vista, 1)
        cv2.rectangle(vista, (0, 0), (359, 269), (0, 0, 255), 3)
        cv2.putText(vista, self.state.phase or "aguardando", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        na_me = self.state.phase == p.PHASE_ME
        dica = ("MOVIMENTE-SE (o clipe sai do fim da ME)" if na_me
                else "o clipe em camera lenta aparece na pausa")
        cv2.putText(vista, dica, (10, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255) if na_me else (190, 190, 190), 1)
        cv2.imshow(self.JANELA, vista)
        cv2.waitKey(1)

    # --------------------------------------------------------------- clipe
    def _janela(self, quando):
        """Intervalo (t0, t1) do movimento que entra no clipe de priming."""
        t1 = quando - self.lag
        return t1 - self.span, t1

    def _monta_clipe(self, quando):
        t0, t1 = self._janela(quando)
        quadros = [(t, f) for t, f in self._buffer if t0 <= t <= t1]
        clipe = []
        for t, quadro in quadros:
            copia = quadro.copy()
            self._rotula(copia, t - t0, max(1e-6, t1 - t0))
            clipe.extend([copia] * self.destaque)
        self.state.publish_video(clipe)
        self.clipes += 1
        print(f"[webcam] clipe {self.clipes}: {len(quadros)} quadros reais -> "
              f"{len(clipe)} em 0,5x "
              f"({len(clipe) / p.SLOWMO_CAPTURE_FPS:.1f} s de video)",
              flush=True)
        if not quadros:
            print("[webcam] AVISO: nenhum quadro no intervalo do clipe; "
                  "mova-se mais perto do fim da ME ou aumente --buffer-sec",
                  flush=True)

    @staticmethod
    def _rotula(quadro, offset, total):
        """Marca o quadro do clipe: tempo real, 0,5x e barra de progresso."""
        altura, largura = quadro.shape[:2]
        cv2.rectangle(quadro, (0, 0), (largura - 1, altura - 1),
                      (0, 255, 255), 3)
        cv2.putText(quadro, f"{offset:0.2f} s", (14, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.putText(quadro, "0,5x", (largura - 116, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        fracao = min(1.0, max(0.0, offset / total))
        cv2.rectangle(quadro, (14, altura - 28),
                      (14 + int((largura - 28) * fracao), altura - 14),
                      (0, 255, 255), -1)


def parse_args():
    """Opcoes do demo (os defaults reproduzem o protocolo final de 12 s)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0,
                        help="Indice do cv2.VideoCapture da webcam")
    parser.add_argument("--trials-por-condicao", type=int, default=1,
                        help="Repeticoes de cada uma das 6 condicoes")
    parser.add_argument("--home-sec", type=float, default=p.HOME_SEC)
    parser.add_argument("--me-sec", type=float, default=p.CUE_ME_SEC)
    parser.add_argument("--video-sec", type=float, default=p.PAUSE_VIDEO_SEC)
    parser.add_argument("--mi-sec", type=float, default=p.MI_SEC)
    parser.add_argument("--span", type=float, default=p.SLOWMO_SPAN_SEC,
                        help="Quanto do movimento entra no clipe (s reais)")
    parser.add_argument("--lag", type=float, default=p.SLOWMO_END_LAG_SEC,
                        help="Quanto antes do fim da ME termina o clipe (s)")
    parser.add_argument("--buffer-sec", type=float,
                        default=p.SLOWMO_BUFFER_SEC,
                        help="Buffer circular de quadros (s)")
    parser.add_argument("--origem-sec", type=float, default=3.0,
                        help="Tempo da tela de origem simulada (s; 0 pula)")
    parser.add_argument("--rapido", action="store_true",
                        help="Tempos curtos (~4 s por trial) para teste rapido")
    parser.add_argument("--sem-video", action="store_true",
                        help="Nao usa a webcam no clipe (so as cues)")
    parser.add_argument("--sem-preview", action="store_true",
                        help="Nao abre a janela de preview da webcam")
    parser.add_argument("--sem-espelho", action="store_true",
                        help="Preview sem espelhamento horizontal")
    parser.add_argument("--tela-cheia", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--apenas", type=int, default=0,
                        help="Roda apenas os N primeiros trials (teste rapido)")
    args = parser.parse_args()
    if args.rapido:
        # tempos curtos e coerentes: video_sec = 2 x span (para 0,5x)
        args.home_sec, args.me_sec, args.mi_sec = 0.6, 1.4, 0.6
        args.span, args.lag, args.video_sec = 0.4, 0.2, 0.8
        args.origem_sec = 0.0
    if args.sem_video:
        args.video_sec = 0.0
    if args.buffer_sec < args.span + args.lag:
        args.buffer_sec = args.span + args.lag + 0.3
        print(f"[demo] buffer ajustado para {args.buffer_sec:g} s "
              "(precisa cobrir span + lag)")
    return args


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    state = p.SharedState()
    state.arm_ik_enabled = False      # sem Kinect: nao procura modelo de elos
    marker_queue = queue.Queue()

    trials = []
    for _ in range(max(1, args.trials_por_condicao)):
        for objeto in p.OBJECTS:
            for mao in p.HANDS:
                trials.append({"objeto": objeto, "mao": mao,
                               "condicao": f"{objeto}_{mao}"})
    if args.apenas > 0:
        trials = trials[:args.apenas]
    duracao_trial = (args.home_sec + args.me_sec + args.video_sec
                     + args.mi_sec)
    print("=" * 78)
    print("DEMO OFFLINE do paradigma ME/MI (sem EEG, sem Kinect: webcam no")
    print("lugar do Kinect apenas para gerar o clipe em camera lenta).")
    print(f"  trials: {len(trials)} ({args.trials_por_condicao} por condicao)"
          f" | trial de {duracao_trial:g} s"
          f" (home {args.home_sec:g} + ME {args.me_sec:g} + "
          f"video {args.video_sec:g} + MI {args.mi_sec:g})")
    print(f"  clipe: {args.span:g} s reais do fim da ME, terminando "
          f"{args.lag:g} s antes do fim, em 0,5x"
          f" ({args.span * 2:.1f} s de video) | buffer {args.buffer_sec:g} s")
    print("  >>> na fase ME (objeto + seta na tela) MOVIMENTE-SE perto do fim")
    print("  ESPACO pula pausas | ESC encerra (os marcadores aparecem aqui)")
    print("=" * 78, flush=True)

    cfg = {
        "trials": trials,
        "home_sec": args.home_sec,
        "me_sec": args.me_sec,
        "video_sec": args.video_sec,
        "mi_sec": args.mi_sec,
        "trials_por_bloco": len(trials),
        "blocos": 1,
        "pausa_bloco_sec": 0.0,
        "contagem": 0,
        "olhos_abertos": 0.0,
        "olhos_fechados": 0.0,
        "repouso_ativo": 0.0,
        "preparacao": False,
    }
    window = p.StimulusWindow(full_screen=args.tela_cheia,
                              on_close=lambda: app.quit())
    controller = p._ParadigmController(window=window,
                                       marker_queue=marker_queue,
                                       state=state, config=cfg)
    window.canvas.on_skip = controller.skip
    window.canvas.on_abort = controller.abort

    def origem_simulada():
        """O demo nao tem Kinect: define a origem como (0,0,0) e segue."""
        state.origin_xyz = (0.0, 0.0, 0.0)
        state.origin_rpy = (0.0, 0.0, 0.0)
        state.origin_ready = True
        print("[demo] calibracao de origem simulada (0,0,0)", flush=True)
        if args.origem_sec > 0:
            controller._window.show_message(
                "ORIGEM (simulada no demo)\n\n"
                f"{args.origem_sec:g} s\n\nmao parada na mesa")
            controller._schedule(args.origem_sec * 1000, controller._countdown)
        else:
            controller._countdown()

    controller._origin_calibration = origem_simulada

    webcam = WebcamClip(state, camera=args.camera,
                        buffer_sec=args.buffer_sec, span=args.span,
                        lag=args.lag, preview=not args.sem_preview,
                        mirror=not args.sem_espelho)
    webcam.start()

    inicio = time.monotonic()

    def drena_marcadores():
        """Imprime os eventos como o muxer faz (indice de amostra ~500 Hz)."""
        while True:
            try:
                evento = marker_queue.get_nowait()
            except queue.Empty:
                return
            amostra = int((evento["monotonic_s"] - inicio) * p.FS)
            extra = " ".join(f"{chave}={evento[chave]}"
                             for chave in ("bloco", "trial", "condicao")
                             if chave in evento)
            print(f"[marca] {evento['code']} ({evento['nome']}) "
                  f"@ amostra {amostra} {extra}".rstrip(), flush=True)

    timer = QTimer()
    timer.timeout.connect(drena_marcadores)
    timer.start(50)

    window.show()
    controller.start(300)
    try:
        app.exec()
    finally:
        timer.stop()
        controller.stop()
        webcam.running = False
        webcam.join(timeout=2.0)
        print(f"[demo] fim | quadros capturados: {webcam.quadros} | "
              f"clipes em camera lenta: {webcam.clipes}", flush=True)


if __name__ == "__main__":
    main()
