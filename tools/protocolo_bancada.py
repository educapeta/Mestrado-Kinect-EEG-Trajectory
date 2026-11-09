#!/usr/bin/env python
"""Protocolo de bancada com CUES na janela: parado -> movimento -> fim.

Motivo: rodar `fusion_live --sem-janela` e pedir para o usuario mover a mao
nao funciona: ele esta' no teclado. Este programa abre uma janela grande,
mostra PREPARACAO com contagem, depois "MAO PARADA" e "MOVA DEVAGAR", e grava
um CSV separado por fase.

Uso:
    python tools/protocolo_bancada.py --ponte-serial COM4

Nao exige tecla durante a coleta. ESC aborta.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Protocolo com janela: preparacao, mao parada, movimento.")
    p.add_argument("--multiview", action="store_true",
                   help="usa MultiViewSource (Kinect+auxiliar) em vez de KinectDepth")
    p.add_argument("--aux-index", type=int, default=0,
                   help="indice da camera auxiliar (padrao 0)")
    p.add_argument("--preparacao", type=float, default=8.0,
                   help="segundos de preparacao antes de gravar (padrao 8)")
    p.add_argument("--parado", type=float, default=10.0,
                   help="segundos de mao parada (padrao 10)")
    p.add_argument("--movimento", type=float, default=10.0,
                   help="segundos de movimento lento (padrao 10)")
    p.add_argument("--ponte-serial", type=str, default=None, metavar="PORTA",
                   help="sobe a ponte serial (ex.: COM4) se necessario")
    p.add_argument("--calib-dir", type=str, default=".",
                   help="diretorio das calibracoes JSON")
    p.add_argument("--saida-prefixo", type=str, default="protocolo_hoje",
                   help="prefixo dos CSVs de saida")
    p.add_argument("--sem-janela", action="store_true",
                   help="smoke sem janela (NAO use para medir movimento)")
    p.add_argument("--kp-yaw", type=float, default=0.5,
                   help="ganho proporcional do yaw (padrao 0.5)")
    p.add_argument("--kp-grav", type=float, default=2.0,
                   help="ganho proporcional da gravidade (padrao 2.0)")
    p.add_argument("--max-yaw-error", type=float, default=0.3,
                   help="gate de erro de yaw em rad (padrao 0.3 ~17 deg)")
    p.add_argument("--sem-snap", action="store_true",
                   help="desabilita snap quando o erro de yaw passa do gate")
    p.add_argument("--snap-conf", type=float, default=0.7,
                   help="confianca minima para snap (padrao 0.7)")
    return p


def _beep(freq: int = 880, ms: int = 150) -> None:
    try:
        import winsound
        winsound.Beep(int(freq), int(ms))
    except Exception:
        print("\a", end="", flush=True)


def _desenha(frame: np.ndarray, titulo: str, subtitulo: str,
             restante: float, cor) -> np.ndarray:
    import cv2
    saida = frame.copy()
    h, w = saida.shape[:2]
    faixa = np.zeros((150, w, 3), dtype=np.uint8)
    faixa[:] = (25, 25, 25)
    cv2.putText(faixa, titulo, (24, 58), cv2.FONT_HERSHEY_SIMPLEX,
                1.35, cor, 4, cv2.LINE_AA)
    cv2.putText(faixa, subtitulo, (24, 105), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(faixa, f"{max(0.0, restante):.1f}s", (w - 150, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3, cv2.LINE_AA)
    return np.vstack([faixa, saida])


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import cv2
    import kinect_compat
    kinect_compat.aplicar()

    from fusion.adapters import (
        KinectMediaPipeSource, MultiViewSource, ImuBankSource, load_calibrations,
    )
    from fusion.live import FusionEngine
    from fusion.complementary import ComplementaryConfig
    from fusion.logging import FusionCsvLogger
    from fusion.metrics import summarize_fusion_log, print_summary

    calibrations = load_calibrations(args.calib_dir)
    faltando = [h for h in ("right", "left") if h not in calibrations]
    if faltando:
        print(f"[ERRO] calibracoes ausentes: {faltando}. Rode o Y primeiro.")
        return 1

    bridge = None
    if args.ponte_serial:
        try:
            from tools.imu_serial_bridge import ImuSerialBridge
            bridge = ImuSerialBridge(args.ponte_serial)
            bridge.start()
            print(f"[bridge] {args.ponte_serial} -> UDP 4210/4211")
        except Exception as exc:
            print(f"[AVISO] bridge nao iniciou: {exc}; usando UDP existente")
            bridge = None

    if args.multiview:
        kinect = MultiViewSource(aux_index=int(args.aux_index))
        print(f"[protocolo] visao multi-view (aux {args.aux_index})")
    else:
        kinect = KinectMediaPipeSource()
    imu = ImuBankSource()
    configs = {}
    for hand, cal in calibrations.items():
        gravity_cam = cal.T_cam_world @ np.array([0.0, 0.0, -1.0])
        configs[hand] = ComplementaryConfig(
            gravity_cam=gravity_cam,
            kp_yaw=float(args.kp_yaw),
            kp_grav=float(args.kp_grav),
            max_yaw_error=float(args.max_yaw_error),
            snap_on_large_yaw_error=not bool(args.sem_snap),
            snap_confidence_threshold=float(args.snap_conf),
        )
    engine = FusionEngine(calibrations, configs=configs)
    kinect.start()
    imu.start()

    if not args.sem_janela:
        cv2.namedWindow("Protocolo de bancada", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Protocolo de bancada", 1600, 900)
        try:
            cv2.setWindowProperty("Protocolo de bancada",
                                  cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    def fase(nome: str, duracao: float, csv_path: str | None,
             titulo: str, subtitulo: str, cor, aguardar: bool = False) -> str:
        logger = FusionCsvLogger(csv_path) if csv_path else None
        if logger:
            logger.open()
        inicio = time.monotonic()
        frame_atual = None
        # No inicio do protocolo NAO comeca sozinho: espera o usuario ver a
        # janela e apertar ESPACO/ENTER. Sem isso, quem esta' no teclado perde
        # a preparacao e os logs de "parado/movimento" saem invalidos.
        if aguardar and not args.sem_janela:
            print("[protocolo] janela aberta: aperte ESPACO/ENTER quando estiver pronto "
                  "(ESC aborta).", flush=True)
            while True:
                try:
                    _ts, frame, _obs = kinect.poll()
                    if frame is not None:
                        frame_atual = frame
                except Exception:
                    pass
                base = frame_atual if frame_atual is not None else np.zeros(
                    (720, 1280, 3), dtype=np.uint8)
                vis = _desenha(base, "PRESSIONE ESPACO",
                               "Quando estiver pronto, aperte ESPACO para comecar",
                               999.0, (0, 220, 255))
                cv2.imshow("Protocolo de bancada", vis)
                key = cv2.waitKey(30) & 0xFF
                if key in (13, 32):
                    _beep(880, 120)
                    break
                if key == 27:
                    return "ABORTADO"
            inicio = time.monotonic()
        elif aguardar:
            print("[protocolo] modo sem janela: iniciando sem espera.")
        try:
            while True:
                agora = time.monotonic()
                restante = duracao - (agora - inicio)
                if restante <= 0:
                    break
                try:
                    _ts, frame, obs = kinect.poll()
                    if frame is not None:
                        frame_atual = frame
                    amostras = imu.poll(agora)
                    agora = time.monotonic()
                    estados = engine.update(agora, obs, amostras)
                except Exception:
                    obs, amostras, estados = {}, {}, {}
                if logger:
                    for hand, estado in estados.items():
                        try:
                            logger.write_state(agora, hand, estado,
                                               obs.get(hand),
                                               amostras.get(hand),
                                               engine.calibration(hand))
                        except Exception:
                            pass
                if frame_atual is not None and not args.sem_janela:
                    vis = _desenha(frame_atual, titulo, subtitulo,
                                   restante, cor)
                    cv2.imshow("Protocolo de bancada", vis)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        return "ABORTADO"
                else:
                    time.sleep(0.01)
        finally:
            if logger:
                logger.close()
        return "OK"

    try:
        etapa = fase("prep", args.preparacao, None,
                     "PREPARACAO", "Solte o teclado; mantenha as DUAS maos visiveis",
                     (0, 220, 255), aguardar=True)
        _beep(660, 120)
        if etapa == "ABORTADO":
            return 0
        parado_path = f"{args.saida_prefixo}_parado.csv"
        mov_path = f"{args.saida_prefixo}_movimento.csv"
        etapa = fase("parado", args.parado, parado_path,
                     "MAO PARADA", "Nao mova as maos; olhe a janela",
                     (0, 255, 0))
        _beep(990, 150)
        if etapa == "ABORTADO":
            return 0
        etapa = fase("movimento", args.movimento, mov_path,
                     "MOVA DEVAGAR", "Gire os punhos devagar, sem pressa",
                     (0, 165, 255))
        _beep(1320, 180)
        if etapa == "ABORTADO":
            return 0
        fase("fim", 2.0, None, "FIM", "Pode voltar ao teclado", (200, 200, 200))
    finally:
        try:
            kinect.close()
        except Exception:
            pass
        try:
            imu.stop()
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

    for path in (parado_path, mov_path):
        print(f"\n===== {path} =====")
        try:
            print_summary(summarize_fusion_log(path))
        except Exception as exc:
            print(f"[AVISO] resumo falhou: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
