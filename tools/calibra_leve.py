#!/usr/bin/env python
"""Calibracao compacta Y, leve: so' Kinect + MediaPipe + IMU.

Por que existe: o groundtruth tool roda 2 cameras auxiliares + MediaPipe por
auxiliar + 4 janelas por frame; na bancada isso derruba o FPS e deixa a janela
pequena. Este programa faz SO' o necessario para a tecla Y:

  - Kinect color/depth + MediaPipe (21 landmarks);
  - normal do PLANO DO DORSO pelo depth (a mesma verdade do Y do groundtruth);
  - IMU pela serial/bridge ou UDP 4210/4211;
  - uma janela grande do Kinect;
  - Y inicia a captura (~24 s), girando o punho devagar;
  - no fim resolve T/f por lado, faz backup do JSON anterior (.json.bak) e grava
    imu_palm_alignment_<lado>.json no formato oficial.

Nao substitui o groundtruth para X/O/D/V; e' so' a calibracao Y rapida.

Uso:
  python tools/calibra_leve.py --ponte-serial COM4
  python tools/calibra_leve.py --sem-janela --duracao 30   # smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imu import (  # noqa: E402
    PALM_ALIGNMENT_MAX_RESIDUAL_DEG,
    PALM_ALIGNMENT_MIN_SAMPLES,
    angle_between,
    rotation_matrix,
    solve_palm_alignment_pesos,
)

# Mesmos valores do groundtruth tool (PALM_ALIGNMENT_SAMPLE_INTERVAL /
# PALM_ALIGNMENT_MIN_STEP_DEG), definidos aqui para o script ser autonomo.
SAMPLE_INTERVAL_S = 0.10
MIN_STEP_DEG = 4.0

SIDES = ("right", "left")
BANK_SIDE = {"right": 1, "left": 2}
CAPTURE_SECONDS = 24.0
MIN_SAMPLES = max(5, int(PALM_ALIGNMENT_MIN_SAMPLES))
WINDOW = "Calibracao leve - Kinect (Y=iniciar  ESC=sair)"


class CapturaYLeve:
    """Coleta (rotacao do IMU, normal do plano do Kinect) por lado."""

    def __init__(self, segundos: float = CAPTURE_SECONDS):
        self.duracao = float(segundos)
        self.ativa = False
        self.fim = 0.0
        self.amostras: Dict[str, list] = {lado: [] for lado in SIDES}
        self._ultimo_tempo = {lado: 0.0 for lado in SIDES}
        self._ultima_normal = {lado: None for lado in SIDES}

    def iniciar(self, agora: float) -> None:
        self.ativa = True
        self.fim = agora + self.duracao
        self.amostras = {lado: [] for lado in SIDES}
        self._ultimo_tempo = {lado: 0.0 for lado in SIDES}
        self._ultima_normal = {lado: None for lado in SIDES}

    def parar(self) -> None:
        self.ativa = False

    def restante_s(self) -> float:
        return max(0.0, self.fim - time.monotonic())

    def contar(self, lado: str) -> int:
        return len(self.amostras.get(lado, ()))

    def coletar(self, lado: str, agora: float, sample: Any,
                normal_camera: Optional[np.ndarray]) -> bool:
        if not self.ativa or sample is None or normal_camera is None:
            return False
        if agora - self._ultimo_tempo[lado] < SAMPLE_INTERVAL_S:
            return False
        normal = np.asarray(normal_camera, dtype=np.float64).reshape(3)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
            return False
        normal = normal / np.linalg.norm(normal)
        anterior = self._ultima_normal[lado]
        if anterior is not None and angle_between(anterior, normal) < MIN_STEP_DEG:
            return False
        rotacao = rotation_matrix(float(sample.roll_deg), float(sample.pitch_deg),
                                  float(sample.yaw_deg))
        self.amostras[lado].append((rotacao, normal))
        self._ultima_normal[lado] = normal
        self._ultimo_tempo[lado] = agora
        return True

    def resolver(self, lado: str):
        amostras = self.amostras.get(lado, ())
        if len(amostras) < MIN_SAMPLES:
            return None
        return solve_palm_alignment_pesos(
            [r for r, _n in amostras], [n for _r, n in amostras])


def _caminho_alinhamento(lado: str) -> Path:
    return Path(f"imu_palm_alignment_{lado}.json")


def salvar_alinhamento(lado: str, rotacao: np.ndarray, corpo: np.ndarray,
                       residuo: float) -> bool:
    """Grava o JSON oficial, com .json.bak do anterior."""
    caminho = _caminho_alinhamento(lado)
    anterior = None
    if caminho.exists():
        try:
            anterior = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            anterior = None
    dados = {
        "rotation": np.asarray(rotacao, np.float64).tolist(),
        "body": np.asarray(corpo, np.float64).tolist(),
        "residual_deg": float(residuo),
        "yaw_boot_delta_deg": 0.0,
        "source": "calibrado",
    }
    try:
        caminho.write_text(json.dumps(dados, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[ERRO] nao gravou {caminho.name}: {exc}")
        return False
    if anterior is not None:
        try:
            caminho.with_suffix(".json.bak").write_text(
                json.dumps(anterior, indent=2), encoding="utf-8")
        except OSError:
            pass
    print(f"  {lado}: gravado em {caminho.name} (residuo {residuo:.2f} graus)")
    return True


def _seta_normal(frame: np.ndarray, landmarks: Any, normal: np.ndarray,
                 cor=(255, 0, 255), comprimento: int = 100) -> None:
    import cv2
    try:
        lm = landmarks[9]
        h, w = frame.shape[:2]
        x, y = int(float(lm.x) * w), int(float(lm.y) * h)
    except (IndexError, AttributeError, TypeError, ValueError):
        return
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    norma = float(np.linalg.norm(n))
    if not np.isfinite(norma) or norma < 1e-9:
        return
    n = n / norma
    h, w = frame.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    sx, sy = float(n[0]), float(-n[1])
    if math.hypot(sx, sy) < 0.22:
        raio = int(max(10, min(30, 16 + 16 * abs(float(n[2])))))
        cv2.circle(frame, (x, y), raio, cor, 2)
        cv2.putText(frame, f"{n[2]:+.2f}", (x + raio + 4, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 1, cv2.LINE_AA)
    else:
        fim = (int(x + sx * comprimento), int(y + sy * comprimento))
        cv2.arrowedLine(frame, (x, y), fim, (0, 0, 0), 6, tipLength=0.25)
        cv2.arrowedLine(frame, (x, y), fim, cor, 3, tipLength=0.25)


def _texto(frame: np.ndarray, linhas: list, cor=(255, 255, 255)) -> None:
    import cv2
    y = 30
    for linha in linhas:
        cv2.putText(frame, str(linha), (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, str(linha), (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 1, cv2.LINE_AA)
        y += 25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibracao Y leve: Kinect + MediaPipe + IMU, uma janela.")
    parser.add_argument("--duracao", type=float, default=90.0,
                        help="duracao total do programa em segundos (0=sem limite)")
    parser.add_argument("--tempo-captura", type=float, default=CAPTURE_SECONDS,
                        help=f"duracao da captura Y (padrao {CAPTURE_SECONDS:g} s)")
    parser.add_argument("--auto-iniciar", action="store_true",
                        help="inicia a captura Y automaticamente ao abrir")
    parser.add_argument("--lado", choices=["ambos", "right", "left"],
                        default="ambos",
                        help="calibrar so' um lado (padrao ambos)")
    parser.add_argument("--sem-janela", action="store_true",
                        help="nao abre janela OpenCV (smoke headless)")
    parser.add_argument("--ponte-serial", type=str, default=None, metavar="PORTA",
                        help="sobe a ponte serial (ex.: COM4) para UDP 4210/4211")
    parser.add_argument("--imu-portas", type=str, default="4210,4211",
                        help="portas UDP do IMU (padrao 4210,4211)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    import cv2
    import kinect_compat
    kinect_compat.aplicar()
    from kinect_imu_groundtruth import KinectHandTracker
    from imu import ImuBank

    bridge = None
    if args.ponte_serial:
        try:
            from tools.imu_serial_bridge import ImuSerialBridge
            bridge = ImuSerialBridge(args.ponte_serial)
            bridge.start()
            print(f"[bridge] {args.ponte_serial} -> UDP 4210/4211 ativa")
        except Exception as exc:
            print(f"[AVISO] ponte serial nao iniciou: {exc}")
            bridge = None

    banco = ImuBank(args.imu_portas)
    banco.start()
    tracker = KinectHandTracker(aux_indices=())
    captura = CapturaYLeve(args.tempo_captura)
    if args.sem_janela:
        print("[modo headless] sem janela; use Y nao e' possivel, mas roda smoke.")
    else:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1600, 900)

    if args.auto_iniciar:
        captura.iniciar(time.monotonic())
        print(f"Captura Y iniciada por {args.tempo_captura:g} s.")
    else:
        print("Aperte Y na janela para iniciar a captura (ou --auto-iniciar).")

    inicio = time.monotonic()
    ultimo_aviso = 0.0
    try:
        while True:
            agora = time.monotonic()
            if args.duracao > 0 and agora - inicio >= args.duracao:
                break
            color, depth = tracker.frames()
            display = None
            maos = []
            if color is not None and depth is not None:
                display, maos = tracker.detect_hands_all(color, depth)
            frame = None
            if display is not None:
                frame = cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)

            amostras = banco
            lados_alvo = SIDES if args.lado == "ambos" else (args.lado,)
            for mao in maos or []:
                lado = mao.get("side")
                if lado not in lados_alvo:
                    continue
                landmarks = mao.get("landmarks")
                if landmarks is None:
                    continue
                sample = amostras.amostra(BANK_SIDE[lado])
                idade = None if sample is None else (agora - sample.received_at)
                normal = None
                try:
                    normal, _rms = tracker.palm_plane_camera(landmarks, lado)
                except Exception:
                    normal = None
                if captura.ativa and sample is not None and idade is not None \
                        and idade <= 0.5 and normal is not None:
                    captura.coletar(lado, agora, sample, normal)
                if frame is not None and normal is not None:
                    _seta_normal(frame, landmarks, normal)

            if captura.ativa and captura.restante_s() <= 0.0:
                captura.parar()
                print("Captura Y terminada; resolvendo T/f...")
                for lado in (SIDES if args.lado == "ambos" else (args.lado,)):
                    if captura.contar(lado) < MIN_SAMPLES:
                        print(f"  {lado}: amostras insuficientes "
                              f"({captura.contar(lado)}/{MIN_SAMPLES}); JSON NAO alterado")
                        continue
                    resolvido = captura.resolver(lado)
                    if resolvido is None:
                        print(f"  {lado}: refit falhou (dados degenerados); JSON NAO alterado")
                        continue
                    t_lado, f_lado, residuo = resolvido
                    if residuo > PALM_ALIGNMENT_MAX_RESIDUAL_DEG:
                        print(f"  {lado}: residuo {residuo:.2f} > "
                              f"{PALM_ALIGNMENT_MAX_RESIDUAL_DEG:g}; JSON NAO alterado")
                        continue
                    salvar_alinhamento(lado, t_lado, f_lado, residuo)

            if frame is not None and not args.sem_janela:
                linhas = ["Y = iniciar captura  ESC/Q = sair"]
                if captura.ativa:
                    linhas.append(f"CAPTURANDO: {captura.restante_s():.1f}s")
                for lado in (SIDES if args.lado == "ambos" else (args.lado,)):
                    linhas.append(f"{lado}: {captura.contar(lado)} amostras")
                _texto(frame, linhas)
                cv2.imshow(WINDOW, frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (27, ord("q")):
                    break
                if tecla == ord("y"):
                    captura.iniciar(agora)
                    print(f"Captura Y iniciada por {args.tempo_captura:g} s.")
            else:
                time.sleep(0.01)

            if args.sem_janela:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nInterrompido.")
    finally:
        try:
            tracker.kinect.close()
        except Exception:
            pass
        try:
            banco.stop()
        except Exception:
            pass
        if bridge is not None:
            try:
                bridge.stop()
            except Exception:
                pass
        if not args.sem_janela:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
