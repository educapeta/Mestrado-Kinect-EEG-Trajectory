#!/usr/bin/env python
"""Grafico ao vivo dos valores crus do IMU na serial do ESP32.

Layout: duas colunas (direita=1, esquerda=2) x tres paineis:
  - roll/pitch/yaw (graus)
  - accel ax/ay/az (g)
  - gyro gx/gy/gz (dps)

Uso:
    .venv\\Scripts\\python.exe tools\\monitor_imu_grafico.py --porta COM4
    .venv\\Scripts\\python.exe tools\\monitor_imu_grafico.py --simular

Nao reseta a placa. ESC/Q fecha. Ctrl+C tambem.
"""
from __future__ import annotations

import argparse
import math
import threading
import time
from collections import deque
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grafico ao vivo do IMU serial")
    p.add_argument("--porta", default="COM4")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--janela", type=float, default=10.0,
                   help="janela de tempo visivel em segundos (padrao 10)")
    p.add_argument("--simular", action="store_true",
                   help="gera dados simulados (sem hardware)")
    p.add_argument("--sem-janela", action="store_true",
                   help="smoke headless (nao abre janela)")
    p.add_argument("--segundos", type=float, default=0.0,
                   help="0=infinito; >0 encerra depois de N s")
    p.add_argument("--salvar-png", type=str, default=None,
                   help="salva o ultimo frame do grafico em PNG")
    p.add_argument("--max-pontos", type=int, default=300,
                   help="maximo de pontos desenhados por linha (padrao 300)")
    return p


class LeitorSerial(threading.Thread):
    def __init__(self, porta: str, baud: int, dados, lock,
                 simular: bool = False):
        super().__init__(daemon=True)
        self.porta = porta
        self.baud = baud
        self.dados = dados
        self.lock = lock
        self.simular = simular
        self.running = True
        self.erro = None
        self.serial = None
        self._offset = {}
        self.backlog = 0

    def _append(self, lado: int, t_us: int, rpy, acc, gyr) -> None:
        agora = time.monotonic()
        with self.lock:
            d = self.dados[lado]
            if self._offset.get(lado) is None:
                # offset estimado na primeira amostra; a latencia mostrada e'
                # RELATIVA a esse primeiro frame (nao absoluta).
                self._offset[lado] = agora - (float(t_us) / 1e6)
            d["t"].append(agora)
            d["t_us"].append(int(t_us))
            d["lat"].append(max(0.0, (agora - (float(t_us) / 1e6
                                               + self._offset[lado])) * 1000.0))
            d["rpy"].append(rpy)
            d["acc"].append(acc)
            d["gyr"].append(gyr)

    def run(self) -> None:
        if self.simular:
            t0 = time.monotonic()
            while self.running:
                t = time.monotonic() - t0
                for lado in (1, 2):
                    ph = 0.0 if lado == 1 else 0.7
                    rpy = (35.0 * math.sin(0.8 * t + ph),
                           -20.0 * math.sin(0.5 * t + ph),
                           90.0 * math.sin(0.25 * t + ph))
                    acc = (0.1 * math.sin(1.0 * t + ph),
                           -0.1 * math.cos(0.7 * t + ph), 1.0)
                    gyr = (20.0 * math.sin(0.8 * t + ph),
                           -15.0 * math.cos(0.6 * t + ph),
                           10.0 * math.sin(0.3 * t + ph))
                    self._append(lado, int((time.monotonic()) * 1e6),
                                 rpy, acc, gyr)
                time.sleep(0.02)
            return

        try:
            import serial
            self.serial = serial.Serial(
                self.porta, self.baud, timeout=0.2,
                dsrdtr=False, rtscts=False)
            self.serial.dtr = False
            self.serial.rts = False
            self.serial.reset_input_buffer()
        except Exception as exc:  # noqa: BLE001
            self.erro = f"{type(exc).__name__}: {exc}"
            self.running = False
            return

        try:
            while self.running:
                try:
                    self.backlog = int(self.serial.in_waiting)
                except Exception:
                    self.backlog = 0
                raw = self.serial.readline()
                if not raw:
                    continue
                texto = raw.decode("utf-8", errors="replace").strip()
                campos = texto.split(",")
                if len(campos) != 11 or campos[0] not in ("1", "2"):
                    continue
                try:
                    vals = [float(x) for x in campos[1:]]
                except ValueError:
                    continue
                self._append(int(campos[0]), int(vals[0]),
                             vals[1:4], vals[4:7], vals[7:10])
        finally:
            if self.serial is not None:
                self.serial.close()

    def stop(self) -> None:
        self.running = False


def _rect(canvas, x, y, w, h, cor=(35, 35, 35)):
    import cv2
    cv2.rectangle(canvas, (x, y), (x + w, y + h), cor, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)


def _painel(canvas, x, y, w, h, titulo, series, t_ref, janela,
            unidade, y_fixo=None):
    import cv2
    _rect(canvas, x, y, w, h)
    cv2.putText(canvas, titulo, (x + 8, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                cv2.LINE_AA)
    # eixos
    cv2.line(canvas, (x + 40, y + 32), (x + 40, y + h - 25),
             (150, 150, 150), 1)
    cv2.line(canvas, (x + 40, y + h - 25), (x + w - 10, y + h - 25),
             (150, 150, 150), 1)
    # escala
    todos = []
    for _nome, cor, valores in series:
        todos.extend(v for _t, v in valores)
    if y_fixo is not None:
        ymin, ymax = y_fixo
    elif todos:
        ymin, ymax = min(todos), max(todos)
        if ymax - ymin < 1e-6:
            ymin, ymax = ymin - 1, ymax + 1
        pad = 0.10 * (ymax - ymin)
        ymin, ymax = ymin - pad, ymax + pad
    else:
        ymin, ymax = -1, 1
    span = max(ymax - ymin, 1e-6)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = int(y + h - 25 - frac * (h - 60))
        cv2.line(canvas, (x + 40, yy), (x + w - 10, yy),
                 (60, 60, 60), 1)
        v = ymin + frac * span
        cv2.putText(canvas, f"{v:+.2f}", (x + 2, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    # linhas
    for indice, (nome, cor, valores) in enumerate(series):
        pts = []
        for t, v in valores:
            tx = t_ref - janela
            if t < tx:
                continue
            px = int(x + 40 + (t - tx) / max(janela, 1e-6) * (w - 55))
            py = int(y + h - 25 - (v - ymin) / span * (h - 60))
            pts.append((px, py))
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, np.int32)],
                          False, cor, 2)
        cv2.putText(canvas, nome, (x + w - 65, y + 22 + indice * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, cor, 1)
    cv2.putText(canvas, unidade, (x + w - 70, y + h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import cv2

    dados = {1: {k: deque(maxlen=5000)
                 for k in ("t", "t_us", "lat", "rpy", "acc", "gyr")},
             2: {k: deque(maxlen=5000)
                 for k in ("t", "t_us", "lat", "rpy", "acc", "gyr")}}
    lock = threading.Lock()
    leitor = LeitorSerial(args.porta, args.baud, dados, lock,
                          simular=args.simular)
    leitor.start()
    if leitor.erro:
        print(f"[ERRO] {leitor.erro}")
        print("       Feche a bridge/fusion_live/protocolo que usa a porta.")
        return 2
    if not args.sem_janela:
        cv2.namedWindow("IMU raw - RPY | accel | gyro", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("IMU raw - RPY | accel | gyro", 1500, 950)
        try:
            cv2.setWindowProperty("IMU raw - RPY | accel | gyro",
                                  cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    t0 = time.monotonic()
    ultimo = None
    try:
        while leitor.is_alive() or args.simular:
            if leitor.erro:
                print(f"[ERRO] {leitor.erro}")
                break
            if args.segundos > 0 and time.monotonic() - t0 >= args.segundos:
                break
            with lock:
                snap = {lado: {k: list(v) for k, v in d.items()}
                        for lado, d in dados.items()}
            agora = time.monotonic()
            canvas = np.full((900, 1500, 3), 18, dtype=np.uint8)
            cv2.putText(canvas, "IMU RAW - serial ESP32", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
            largura = 700
            altura = 250
            def _dec(vals):
                if len(vals) <= max(2, args.max_pontos):
                    return vals
                passo = max(1, len(vals) // args.max_pontos)
                return vals[::passo]

            for col, lado in enumerate((1, 2)):
                x = 25 + col * (largura + 40)
                nome = "DIREITA (1)" if lado == 1 else "ESQUERDA (2)"
                d = snap[lado]
                ts = _dec(d["t"])
                rpy = _dec(d["rpy"])
                acc = _dec(d["acc"])
                gyr = _dec(d["gyr"])
                s_rpy = [
                    ("roll", (80, 80, 255), [(t, v[0]) for t, v in zip(ts, rpy)]),
                    ("pitch", (80, 255, 80), [(t, v[1]) for t, v in zip(ts, rpy)]),
                    ("yaw", (255, 180, 80), [(t, v[2]) for t, v in zip(ts, rpy)]),
                ]
                s_acc = [
                    ("ax", (80, 80, 255), [(t, v[0]) for t, v in zip(ts, acc)]),
                    ("ay", (80, 255, 80), [(t, v[1]) for t, v in zip(ts, acc)]),
                    ("az", (255, 180, 80), [(t, v[2]) for t, v in zip(ts, acc)]),
                ]
                s_gyr = [
                    ("gx", (80, 80, 255), [(t, v[0]) for t, v in zip(ts, gyr)]),
                    ("gy", (80, 255, 80), [(t, v[1]) for t, v in zip(ts, gyr)]),
                    ("gz", (255, 180, 80), [(t, v[2]) for t, v in zip(ts, gyr)]),
                ]
                _painel(canvas, x, 55, largura, altura, f"{nome} RPY",
                        s_rpy, agora, args.janela, "graus")
                _painel(canvas, x, 320, largura, altura, f"{nome} ACCEL",
                        s_acc, agora, args.janela, "g", (-1.5, 1.5))
                _painel(canvas, x, 585, largura, altura, f"{nome} GYRO",
                        s_gyr, agora, args.janela, "dps")
            lats = []
            for lado in (1, 2):
                if snap[lado]["lat"]:
                    lats.append(snap[lado]["lat"][-1])
            lat_ms = float(np.median(lats)) if lats else float("nan")
            try:
                backlog = int(leitor.backlog)
            except Exception:
                backlog = 0
            cv2.putText(
                canvas,
                f"ESC/Q sair | janela {args.janela:.0f}s | "
                f"lat_rel={lat_ms:.0f}ms | backlog={backlog}B | "
                f"pontos/linha<={args.max_pontos}",
                (20, 880), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1)
            ultimo = canvas
            if not args.sem_janela:
                cv2.imshow("IMU raw - RPY | accel | gyro", canvas)
                if (cv2.waitKey(20) & 0xFF) in (27, ord("q")):
                    break
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        leitor.stop()
        leitor.join(timeout=1.0)
        if ultimo is not None and args.salvar_png:
            cv2.imwrite(args.salvar_png, ultimo)
            print(f"[grafico] salvo em {args.salvar_png}")
        if not args.sem_janela:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
