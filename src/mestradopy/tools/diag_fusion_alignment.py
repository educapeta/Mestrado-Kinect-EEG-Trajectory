#!/usr/bin/env python
"""
Diagnóstico de alinhamento da fusão (Kinect + MediaPipe + IMU).

Captura dados ao vivo e imprime por lado, por ~0.5 s:
  n_vis    = normal da visão (Kinect/MP, camera frame)
  n_cal    = normal calculada pela calibração (T_cam_world @ R_imu @ f_body)
  n_fus    = normal do filtro fundido
  ang_vis_cal = ângulo entre n_vis e n_cal (graus)
  ang_fus_vis = ângulo entre n_fus e n_vis (graus)
  q_imu    = quaternion do IMU (body->world, do firmware Mahony)
  q_filtro = quaternion do filtro (body->camera)
  rpy/gyro = roll/pitch/yaw + gyro do IMU
  fonte    = fonte ativa do SensorManager

Funciona com bridge já aberta (UDP 4210/4211) ou com --ponte-serial COM4.
NÃO abre janela OpenCV.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Garante que o diretório raiz do projeto está no sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnóstico de alinhamento da fusão (sem janela)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--duracao', type=float, default=0.5,
        help='Duração da captura em segundos (padrão: 0.5)'
    )
    parser.add_argument(
        '--calib-dir', type=str, default='.',
        help='Diretório com arquivos de calibração (padrão: diretório atual)'
    )
    parser.add_argument(
        '--imu-portas', type=str, default='4210,4211',
        help='Portas IMU separadas por vírgula (padrão: 4210,4211)'
    )
    parser.add_argument(
        '--ponte-serial', type=str, default=None, metavar='PORTA',
        help='Se informado (ex.: COM4), inicia tools/imu_serial_bridge em '
             'background e encaminha a serial do ESP32 para UDP 4210/4211'
    )
    parser.add_argument(
        '--sem-kinect', action='store_true',
        help='Desabilita Kinect (apenas IMU + calibração)'
    )
    parser.add_argument(
        '--intervalo', type=float, default=0.05,
        help='Intervalo entre impressões em segundos (padrão: 0.05 = 20 Hz)'
    )
    parser.add_argument(
        '--use-imu-prior', action='store_true',
        help='Ativa modo use_imu_attitude_prior no filtro (testa hipótese do gyro)'
    )
    parser.add_argument(
        '--snap-yaw', action='store_true',
        help='Ativa snap_on_large_yaw_error no filtro'
    )
    return parser


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Ângulo entre dois vetores em graus."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float('nan')
    dot = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return np.rad2deg(np.arccos(dot))


def format_vec(v: Optional[np.ndarray]) -> str:
    if v is None:
        return "       None       "
    v = np.asarray(v, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(v)):
        return "      NaN        "
    return f"[{v[0]:+7.4f} {v[1]:+7.4f} {v[2]:+7.4f}]"


def format_quat(q: Optional[np.ndarray]) -> str:
    if q is None:
        return "                None                "
    q = np.asarray(q, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(q)):
        return "               NaN                "
    return f"[{q[0]:+7.4f} {q[1]:+7.4f} {q[2]:+7.4f} {q[3]:+7.4f}]"


def format_rpy_gyro(imu: Optional[Any]) -> str:
    if imu is None or not imu.valid:
        return "  rpy: (None)  gyro: (None)"
    return (f"  rpy: [{imu.roll_deg:+6.1f} {imu.pitch_deg:+6.1f} {imu.yaw_deg:+6.1f}] "
            f"gyro: [{imu.gyro_dps[0]:+6.1f} {imu.gyro_dps[1]:+6.1f} {imu.gyro_dps[2]:+6.1f}]")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Carrega calibrações
    from fusion.adapters import load_calibrations
    calibrations = load_calibrations(args.calib_dir)

    if not calibrations:
        print("[ERRO] Nenhuma calibração válida encontrada. Execute a calibração primeiro.", file=sys.stderr)
        print("       python tools/calibra_palma_plano.py --lado right", file=sys.stderr)
        print("       python tools/calibra_palma_plano.py --lado left", file=sys.stderr)
        return 1

    # Configura filtro
    from fusion.complementary import ComplementaryConfig
    configs = {}
    for hand, cal in calibrations.items():
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        gravity_cam = cal.T_cam_world @ gravity_world
        config = ComplementaryConfig(
            gravity_cam=gravity_cam,
            use_imu_attitude_prior=args.use_imu_prior,
            snap_on_large_yaw_error=args.snap_yaw or True,  # habilitado por padrão
            snap_confidence_threshold=0.7,
            max_yaw_error=0.3,  # ~17 deg
            kp_grav=2.0,
            kp_yaw=0.5,
        )
        configs[hand] = config

    # Cria engine
    from fusion.live import FusionEngine
    engine = FusionEngine(calibrations, configs=configs)

    # Fontes
    kinect_source = None
    imu_source = None
    bridge = None

    if args.ponte_serial and not args.sem_kinect:
        try:
            from tools.imu_serial_bridge import ImuSerialBridge
            bridge = ImuSerialBridge(args.ponte_serial)
            bridge.start()
            print(f"[INFO] Ponte serial {args.ponte_serial} -> UDP 4210/4211 ativa", flush=True)
            time.sleep(0.5)  # dá tempo da ponte encher a fila
        except Exception as exc:
            print(f"[AVISO] Não foi possível iniciar ponte serial: {exc}", file=sys.stderr)
            bridge = None

    if not args.sem_kinect:
        try:
            from fusion.adapters import KinectMediaPipeSource
            kinect_source = KinectMediaPipeSource()
            print("[INFO] Fonte Kinect+MediaPipe criada", flush=True)
        except Exception as e:
            print(f"[AVISO] Falha ao criar fonte Kinect: {e}", file=sys.stderr)
            kinect_source = None

    try:
        from fusion.adapters import ImuBankSource
        imu_source = ImuBankSource(portas=args.imu_portas)
        print(f"[INFO] Fonte IMU criada (portas: {args.imu_portas})", flush=True)
    except Exception as e:
        print(f"[AVISO] Falha ao criar fonte IMU: {e}", file=sys.stderr)
        imu_source = None

    if kinect_source is None and imu_source is None:
        print("[ERRO] Nenhuma fonte de dados disponível", file=sys.stderr)
        return 1

    # Inicia fontes
    if kinect_source is not None:
        try:
            kinect_source.start()
        except Exception:
            pass
    if imu_source is not None:
        try:
            imu_source.start()
        except Exception:
            pass

    print("\n" + "=" * 140)
    print(f"{'t':>6}  {'hand':<5}  {'n_vis':>22}  {'n_cal':>22}  {'n_fus':>22}  "
          f"{'vis-cal':>7}  {'fus-vis':>7}  {'q_imu':>32}  {'q_filtro':>32}  {'rpy/gyro':<40}  {'fonte'}")
    print("=" * 140)

    start_time = time.monotonic()
    last_print = 0.0

    try:
        while time.monotonic() - start_time < args.duracao:
            now = time.monotonic()

            # Poll Kinect
            kinect_obs: Dict[str, Any] = {}
            if kinect_source is not None:
                try:
                    ts, frame, obs = kinect_source.poll()
                    kinect_obs = obs
                except Exception:
                    pass

            # Poll IMU
            imu_samples: Dict[str, Any] = {}
            if imu_source is not None:
                try:
                    imu_samples = imu_source.poll(now)
                except Exception:
                    pass

            # Update engine
            fused_states = engine.update(now, kinect_obs, imu_samples)

            # Print at interval
            if now - last_print >= args.intervalo:
                last_print = now
                for hand in ('right', 'left'):
                    obs = kinect_obs.get(hand)
                    imu = imu_samples.get(hand)
                    state = fused_states.get(hand)
                    cal = calibrations.get(hand)

                    # n_vis from Kinect
                    n_vis = None
                    if obs is not None and obs.valid and obs.palm_normal_cam is not None:
                        n_vis = np.asarray(obs.palm_normal_cam, dtype=np.float64).reshape(3)
                        if np.linalg.norm(n_vis) < 1e-9:
                            n_vis = None

                    # n_cal from calibration + IMU
                    n_cal = None
                    if cal is not None and imu is not None and imu.valid:
                        try:
                            n_cal = cal.normal_cam_from_imu(imu)
                            if np.linalg.norm(n_cal) < 1e-9:
                                n_cal = None
                        except Exception:
                            pass

                    # n_fus from fused state
                    n_fus = None
                    if state is not None and state.valid and state.normal_cam is not None:
                        n_fus = np.asarray(state.normal_cam, dtype=np.float64).reshape(3)
                        if np.linalg.norm(n_fus) < 1e-9:
                            n_fus = None

                    # Angles
                    ang_vis_cal = angle_between(n_vis, n_cal) if n_vis is not None and n_cal is not None else float('nan')
                    ang_fus_vis = angle_between(n_fus, n_vis) if n_fus is not None and n_vis is not None else float('nan')

                    # q_imu from IMU sample
                    q_imu = None
                    if imu is not None and imu.valid:
                        from fusion.quaternion import from_euler_zyx_deg
                        q_imu = from_euler_zyx_deg(imu.roll_deg, imu.pitch_deg, imu.yaw_deg)

                    # q_filtro from state
                    q_filtro = None
                    if state is not None and state.valid:
                        q_filtro = np.asarray(state.q_body_cam, dtype=np.float64).reshape(4)

                    # Source
                    fonte = state.source if state is not None else "none"

                    t_rel = now - start_time
                    print(f"{t_rel:6.3f}  {hand:<5}  "
                          f"{format_vec(n_vis):>22}  {format_vec(n_cal):>22}  {format_vec(n_fus):>22}  "
                          f"{ang_vis_cal:7.2f}  {ang_fus_vis:7.2f}  "
                          f"{format_quat(q_imu):>32}  {format_quat(q_filtro):>32}  "
                          f"{format_rpy_gyro(imu):<40}  {fonte}")

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário", flush=True)
    finally:
        if kinect_source is not None:
            try:
                kinect_source.close()
            except Exception:
                pass
        if imu_source is not None:
            try:
                imu_source.stop()
            except Exception:
                pass
        if bridge is not None:
            try:
                bridge.stop()
            except Exception:
                pass

    print("=" * 140)
    print("[OK] Diagnóstico concluído")
    return 0


if __name__ == '__main__':
    sys.exit(main())