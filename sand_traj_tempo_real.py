"""Previsao de TRAJETORIA da mao em tempo real (EEG -> SAND -> overlay).

Consome o checkpoint de sand_traj_treino.py (SANDTrajectory: janela 2 s de
EEG -> sequencia 3D da mao em metros, relativa a origem do Programa 1) e
desenha a trajetoria PREVISTA no overlay do Kinect, junto com a mao real.

O alvo do modelo e' o KT_* da gravacao do Programa 1: landmark 9 do MediaPipe
Hands (centro da palma) triangulado para 3D, em metros no camera space do
Kinect.

Pipeline (g.Pype):
    GNautilus/Generator -> Bandpass(1-30 Hz) -> SANDTrajWindowNode
O no mantem um buffer circular dos ultimos --window-sec segundos, classifica
a cada --interval segundos em THREAD propria (a inferencia nunca atrasa a
amostragem) e publica o resultado no SharedState compartilhado com a thread
de tracking (reusada de eeg_motor_paradigm.py), que desenha:

    trilha MAGENTA  = trajetoria prevista (projetada do camera space 3D);
    circulo VERDE   = punho real (Kinect);
    circulo CIANO   = ponto final previsto (alvo).

A origem de translacao usada no overlay e a MESMA do treino (media das
sessoes gravada no checkpoint, ou --origin). As previsoes seguem em --log-csv.

Uso:
    python sand_traj_tempo_real.py --model sand_traj_model.pt
    python sand_traj_tempo_real.py --model sand_traj_model.pt --sem-kinect
    python sand_traj_tempo_real.py --headless-test 6   # teste sem GUI
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from collections import deque

import numpy as np
import gpype as gp
from gpype.backend.core.i_node import INode
from gpype.backend.core.i_port import IPort

import sand_trajectory_model as st
import eeg_motor_paradigm as emp
from camera_mpu_fusion import ImuReceiver

PORT_IN = gp.Constants.Defaults.PORT_IN
PORT_OUT = gp.Constants.Defaults.PORT_OUT


# =============================================================================
# Logger de previsoes (thread-safe)
# =============================================================================
class TrajPredictionLogger:
    """Grava cada previsao em CSV (resumo + primeiro/ultimo ponto)."""

    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        self._handle = None

    def log(self, movement, span, trajectory, origin, elapsed_s=None):
        if not self._path:
            return
        with self._lock:
            try:
                if self._handle is None:
                    self._handle = open(self._path, "a", encoding="utf-8",
                                        newline="")
                    if self._handle.tell() == 0:
                        self._handle.write(
                            "tempo,movimento_m,extensao_m,n_pontos,"
                            "p0_x,p0_y,p0_z,pn_x,pn_y,pn_z,"
                            "origem_x,origem_y,origem_z,latencia_ms\n")
                first = trajectory[0]
                last = trajectory[-1]
                row = [time.strftime("%Y-%m-%d %H:%M:%S"),
                       f"{movement:.4f}", f"{span:.4f}", str(len(trajectory))]
                row += [f"{value:.4f}" for value in np.asarray(first).reshape(-1)]
                row += [f"{value:.4f}" for value in np.asarray(last).reshape(-1)]
                row += [f"{value:.4f}" for value in np.asarray(origin).reshape(-1)]
                row += ["" if elapsed_s is None else f"{elapsed_s * 1e3:.1f}"]
                self._handle.write(",".join(row) + "\n")
                self._handle.flush()
            except OSError as exc:
                print(f"[logger] falha ao gravar CSV: {exc}", flush=True)

    def close(self):
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


# =============================================================================
# Node de classificacao continua (janela deslizante)
# =============================================================================
class SANDTrajWindowNode(INode):
    """Buffer circular + inferencia assincrona da trajetoria prevista.

    A predicao roda num worker thread: a inferencia (20-300 ms) NUNCA bloqueia
    o fluxo de amostras do g.Pype (que precisa devolver < 2 ms @ 500 Hz).
    """

    def __init__(self, sand, state, interval_s=0.5, logger=None,
                 tag="SAND-TRAJ", **kwargs):
        super().__init__(input_ports=[IPort.Configuration()], **kwargs)
        self._sand = sand
        self._state = state
        self._interval_s = float(interval_s)
        self._logger = logger
        self._tag = tag
        self._window = int(sand.n_timesteps)
        self._buffer = None
        self._counter = 0
        self._next_at = None
        self._queue = queue.Queue(maxsize=4)
        self._worker = None
        self._stop_worker = threading.Event()

    def setup(self, data, port_context_in):
        ctx = port_context_in[PORT_IN]
        fs = float(ctx[gp.Constants.Keys.SAMPLING_RATE])
        channel_count = int(ctx[gp.Constants.Keys.CHANNEL_COUNT])
        needed = int(self._sand.channel_map.max()) + 1
        if channel_count < needed:
            raise ValueError(f"A fonte tem {channel_count} canais e o modelo "
                             f"precisa de pelo menos {needed}.")
        if abs(fs - self._sand.fs) > 1e-6:
            print(f"[{self._tag}] AVISO: fs da fonte ({fs:g} Hz) difere do "
                  f"modelo ({self._sand.fs:g} Hz).", flush=True)
        self._interval = max(1, int(round(self._interval_s * fs)))
        self._buffer = np.zeros((self._window, channel_count), np.float32)
        self._counter = 0
        self._next_at = self._window
        self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                        name="sand-traj-infer")
        self._worker.start()
        print(f"[{self._tag}] janela {self._window} amostras "
              f"({self._window / fs:.2f} s) a cada {self._interval_s:g} s "
              "(inferencia assincrona)", flush=True)
        return super().setup(data, port_context_in)

    def step(self, data):
        block = data.get(PORT_IN)
        if block is None or len(block) == 0:
            return None
        n = block.shape[0]
        self._counter += n
        if n >= self._window:
            self._buffer[:] = block[-self._window:]
        else:
            self._buffer[:-n] = self._buffer[n:]
            self._buffer[-n:] = block
        if self._counter >= self._next_at:
            self._next_at = self._counter + self._interval
            try:
                self._queue.put_nowait(self._buffer.copy())
            except queue.Full:
                pass                       # previsao mais antiga e descartada
        return None

    # ------------------------------------------------------------- previsao
    def _worker_loop(self):
        while not self._stop_worker.is_set():
            try:
                window = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                trajectory, movement = self._sand.predict_window(window)
            except Exception as exc:
                print(f"[{self._tag}] erro de inferencia: {exc}", flush=True)
                self._state.publish_motion(
                    traj_valid=0, traj=None, traj_move=np.nan,
                    traj_time=time.monotonic())
                continue
            elapsed = time.perf_counter() - started
            span = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
            self._state.publish_motion(
                traj=np.asarray(trajectory, np.float64),
                traj_valid=1, traj_move=float(movement),
                traj_span=span, traj_time=time.monotonic())
            origin = self._origin_vector()
            print(f"[{self._tag}] mov={movement:.3f} m extensao={span:.3f} m "
                  f"| p0=({trajectory[0][0]:+.2f},{trajectory[0][1]:+.2f},"
                  f"{trajectory[0][2]:+.2f}) pN=({trajectory[-1][0]:+.2f},"
                  f"{trajectory[-1][1]:+.2f},{trajectory[-1][2]:+.2f}) "
                  f"[{elapsed * 1e3:.0f} ms]", flush=True)
            if self._logger:
                self._logger.log(movement, span, trajectory, origin, elapsed)

    def _origin_vector(self):
        with self._state.lock:
            if self._state.origin_ready and self._state.origin_xyz is not None:
                return np.asarray(self._state.origin_xyz, np.float64)
        return np.zeros(3, np.float64)

    def stop(self):
        self._stop_worker.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        return super().stop()


# =============================================================================
# Overlay: trajetoria prevista sobre o RGB do Kinect
# =============================================================================
def make_extra_draw(tracking, origin):
    """Cria o callback (frame, state) usado pelo hook TrackingThread.extra_draw."""
    cv2 = __import__("cv2")

    def draw(frame, state):
        motion, *_phase = state.snapshot()
        alignment = tracking.alignment
        if alignment is None or tracking.tracker is None:
            return
        size = (tracking.tracker.color_width, tracking.tracker.color_height)
        # punho real (verde): state.motion["x/y/z"] sao ABSOLUTOS (camera space)
        wrist = np.array([motion["x"], motion["y"], motion["z"]], np.float64)
        if np.isfinite(wrist).all() and motion["valid"]:
            wpx = alignment.project_camera_points(
                wrist.reshape(1, 3), size).reshape(-1)
            if np.isfinite(wpx).all():
                center = tuple(np.round(wpx).astype(int))
                cv2.circle(frame, center, 9, (0, 255, 0), 2)
        # trajetoria prevista (magenta): metros RELATIVOS a origem + origin
        trajectory = motion.get("traj")
        if trajectory is None or not motion.get("traj_valid"):
            return
        points_abs = np.asarray(trajectory, np.float64) + np.asarray(
            origin, np.float64).reshape(1, 3)
        pixels = alignment.project_camera_points(points_abs, size)
        if not np.isfinite(pixels).all():
            return
        pixels = np.round(pixels).astype(int)
        for index in range(1, len(pixels)):
            cv2.line(frame, tuple(pixels[index - 1]), tuple(pixels[index]),
                     (255, 0, 255), 2)
        for index, point in enumerate(pixels):
            shade = 120 + 135 * index // max(1, len(pixels) - 1)
            cv2.circle(frame, tuple(point), 4, (shade, 0, shade), -1)
        cv2.circle(frame, tuple(pixels[-1]), 8, (255, 255, 0), -1)
        move = motion.get("traj_move", np.nan)
        span = motion.get("traj_span", np.nan)
        label = (f"previsto mov={move:.3f} m extensao={span:.2f} m | "
                 f"origem=({origin[0]:+.2f},{origin[1]:+.2f},{origin[2]:+.2f})")
        cv2.putText(frame, label, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 255), 2)
        if motion.get("zero_lock"):
            cv2.putText(frame, "ZERAGEM IMU ATIVA (bias+ancora)",
                        (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

    return draw


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="sand_traj_model.pt",
                        help="Checkpoint gerado pelo sand_traj_treino.py")
    parser.add_argument("--source", choices=["generator", "gnautilus"],
                        default="generator")
    parser.add_argument("--fs", type=float, default=500.0)
    parser.add_argument("--channel-count", type=int, default=32)
    parser.add_argument("--frame-size", type=int, default=1)
    parser.add_argument("--channel-map", default=None,
                        help="Lista JSON 1-based (ou .json) ligando cada canal "
                             "da montagem de treino ao canal fisico do "
                             "dispositivo")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Segundos entre previsoes da janela deslizante")
    parser.add_argument("--origin", default=None,
                        help="Origem absoluta 'x,y,z' (m) para projetar a "
                             "trajetoria; padrao: origem media gravada no "
                             "checkpoint")
    parser.add_argument("--sem-kinect", action="store_true",
                        help="Roda sem Kinect (apenas pipeline de previsao)")
    parser.add_argument("--aux-cameras", default="0,1",
                        help="Indices das webcams auxiliares (padrao '0,1')")
    parser.add_argument("--sem-zeroing", action="store_true",
                        help="Desativa a zeragem do IMU pelas cameras")
    parser.add_argument("--log-csv", default=None,
                        help="CSV para registrar todas as previsoes")
    parser.add_argument("--headless-test", type=float, default=0.0,
                        help="Roda N segundos sem GUI (teste automatizado)")
    return parser.parse_args()


def apply_channel_map(sand, spec):
    if not spec:
        return
    if os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as handle:
            spec = handle.read()
    mapping = json.loads(spec)
    if not isinstance(mapping, list) or len(mapping) != len(sand.channel_map):
        raise ValueError(f"--channel-map deve ter {len(sand.channel_map)} "
                         "valores (um por canal da montagem de treino).")
    sand.set_channel_map(mapping)
    print("Mapa de canais personalizado aplicado.")


def resolve_origin(sand, spec):
    """Origem absoluta (m): --origin > meta do checkpoint > zeros."""
    if spec:
        parts = [float(value) for value in spec.replace(";", ",").split(",")]
        if len(parts) != 3:
            raise ValueError("--origin precisa de 3 valores 'x,y,z' (m).")
        origin = np.asarray(parts, np.float64)
        print(f"[origem] informada pelo usuario: {np.round(origin, 3)} m")
        return origin
    meta_origin = sand.cfg.get("meta_origin_m")
    if meta_origin:
        origin = np.asarray(meta_origin, np.float64)
        print(f"[origem] media das sessoes do treino: "
              f"{np.round(origin, 3)} m")
        return origin
    print("[origem] sem referencia: usando (0,0,0) (a trajetoria prevista "
          "aparecera relativa a origem do treino)", flush=True)
    return np.zeros(3, np.float64)


def build_source(args, sand):
    channel_count = args.channel_count or int(sand.channel_map.max()) + 1
    if args.source == "gnautilus":
        # Mesma checagem do Programa 1: evita rodar com a Python do PATH
        # (gtec_gds 1.4.0 sem headers) em vez do .venv do projeto.
        emp._check_gtec_environment()
        print(f"g.Nautilus: {channel_count} canais, frame_size={args.frame_size}",
              flush=True)
        return gp.GNautilus(sampling_rate=args.fs, channel_count=channel_count,
                            frame_size=args.frame_size)
    print(f"Generator sintetico: {channel_count} canais (teste sem hardware)",
          flush=True)
    return gp.Generator(sampling_rate=args.fs, channel_count=channel_count,
                        signal_frequency=10.0, signal_amplitude=10.0,
                        noise_amplitude=5.0)


def main():
    args = parse_args()
    np.random.seed(0)
    sand = st.SandTrajectoryBCI(args.model)
    apply_channel_map(sand, args.channel_map)
    print(sand.describe(), flush=True)
    origin = resolve_origin(sand, args.origin)
    logger = TrajPredictionLogger(args.log_csv) if args.log_csv else None
    headless = args.headless_test > 0

    state = emp.SharedState()
    state.origin_xyz = origin
    state.origin_ready = True

    pipeline = gp.Pipeline()
    source = build_source(args, sand)
    bandpass = gp.Bandpass(
        f_lo=float(sand.cfg.get("f_lo", st.DEFAULT_F_LO)),
        f_hi=float(sand.cfg.get("f_hi", st.DEFAULT_F_HI)))
    pipeline.connect(source, bandpass)
    pipeline.connect(bandpass, SANDTrajWindowNode(
        sand, state, interval_s=args.interval, logger=logger))

    tracking = None
    if not headless:
        tracking = emp.TrackingThread(
            state, ImuReceiver(4210), with_kinect=not args.sem_kinect,
            use_ik=False,
            aux_indices=tuple(
                int(index) for index in
                str(args.aux_cameras).split(",")
                if index.strip().lstrip("-").isdigit()),
            use_zeroing=not args.sem_zeroing)
        tracking.extra_draw = make_extra_draw(tracking, origin)
        tracking.start()

    pipeline.start()
    try:
        if headless:
            print(f"Pipeline ativo por {args.headless_test:g} s "
                  "(headless)...", flush=True)
            time.sleep(args.headless_test)
        else:
            print("Overlay ativo. 'q' na janela do Kinect encerra.",
                  flush=True)
            while not state.quit_flag:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nEncerrando...", flush=True)
    finally:
        pipeline.stop()
        if tracking is not None:
            tracking.running = False
            state.request_quit()
            tracking.join(timeout=5.0)
        if logger:
            logger.close()
    print("Sessao encerrada.", flush=True)


if __name__ == "__main__":
    main()