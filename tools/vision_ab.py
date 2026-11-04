#!/usr/bin/env python
"""
Ferramenta A/B comparando KinectDepthBackend vs MultiViewBackend nas MESMAS amostras.

Uso:
  python tools/vision_ab.py --synthetic                    # Teste sem hardware
  python tools/vision_ab.py --duracao 10 --aux-index 0     # Hardware real
  python tools/vision_ab.py --duracao 10 --ponte-serial COM4 --log-csv saida.csv

Opcoes:
  --duracao N        Segundos de aquisicao (default: 10)
  --sem-janela       Nao abre janelas de video
  --ponte-serial P   Porta serial do ESP32 (default: COM4)
  --aux-index N      Indice da camera auxiliar OpenCV (default: 0)
  --synthetic        Gera pontos 3D sinteticos com ruido/outliers, sem hardware
  --log-csv FILE     Salva CSV por frame: normal/pos/conf/sigma/edge_on/angulo/latencia
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

# Adiciona raiz do projeto ao path para imports funcionarem quando rodado como script
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Lazy imports - so carregados quando necessario
def _import_hardware():
    """Importa modulos de hardware sob demanda."""
    import kinect_compat
    kinect_compat.aplicar()
    import kinect_imu_groundtruth as kinect_gt
    from fusion.adapters import KinectMediaPipeSource, ImuBankSource
    from fusion.aux_vision import AuxStereoSource, compare_frame
    from fusion.vision_backends import VisionNormal, KinectDepthBackend, MultiViewBackend
    return kinect_gt, KinectMediaPipeSource, ImuBankSource, AuxStereoSource, compare_frame, VisionNormal, KinectDepthBackend, MultiViewBackend


@dataclass
class FrameResult:
    """Resultado de um frame para CSV."""
    timestamp: float
    hand: str
    # KinectDepth
    kd_normal_x: float
    kd_normal_y: float
    kd_normal_z: float
    kd_pos_x: float
    kd_pos_y: float
    kd_pos_z: float
    kd_confidence: float
    kd_sigma_deg: float
    kd_rms_m: float
    kd_edge_on: bool
    kd_valid: bool
    # MultiView
    mv_normal_x: float
    mv_normal_y: float
    mv_normal_z: float
    mv_pos_x: float
    mv_pos_y: float
    mv_pos_z: float
    mv_confidence: float
    mv_sigma_deg: float
    mv_rms_m: float
    mv_edge_on: bool
    mv_valid: bool
    # Comparacao
    angle_deg: float
    agree: bool
    # Latencias
    kd_latency_ms: float
    mv_latency_ms: float


def _vn_to_dict(vn: 'VisionNormal', prefix: str) -> Dict[str, Any]:
    """Converte VisionNormal para dict com prefixo."""
    d = {}
    if vn.position_cam is not None:
        d[f"{prefix}_pos_x"] = float(vn.position_cam[0])
        d[f"{prefix}_pos_y"] = float(vn.position_cam[1])
        d[f"{prefix}_pos_z"] = float(vn.position_cam[2])
    else:
        d[f"{prefix}_pos_x"] = float('nan')
        d[f"{prefix}_pos_y"] = float('nan')
        d[f"{prefix}_pos_z"] = float('nan')
    d[f"{prefix}_normal_x"] = float(vn.normal_cam[0])
    d[f"{prefix}_normal_y"] = float(vn.normal_cam[1])
    d[f"{prefix}_normal_z"] = float(vn.normal_cam[2])
    d[f"{prefix}_confidence"] = float(vn.confidence)
    d[f"{prefix}_sigma_deg"] = float(vn.sigma_deg)
    d[f"{prefix}_rms_m"] = float(vn.rms_m)
    d[f"{prefix}_edge_on"] = bool(vn.edge_on)
    d[f"{prefix}_valid"] = bool(vn.valid)
    return d


def run_synthetic(
    duration: float = 10.0,
    log_csv: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executa comparacao A/B sintetica (sem hardware)."""
    from fusion.aux_vision import synthetic_compare_run
    from fusion.vision_backends import VisionNormal

    print(f"[synthetic] Rodando comparacao A/B por {duration:.1f}s (seed={seed})...")

    results = synthetic_compare_run(duration=duration, seed=seed)

    # Estatisticas
    n = len(results)
    angles = [r["angle_deg"] for r in results if r["kinect_depth"].valid and r["multiview"].valid]
    edge_on_kd = sum(1 for r in results if r["kinect_depth"].edge_on)
    edge_on_mv = sum(1 for r in results if r["multiview"].edge_on)

    summary = {
        "n_frames": n,
        "n_valid": len(angles),
        "angle_mean_deg": float(np.mean(angles)) if angles else float('nan'),
        "angle_std_deg": float(np.std(angles)) if angles else float('nan'),
        "angle_median_deg": float(np.median(angles)) if angles else float('nan'),
        "angle_p90_deg": float(np.percentile(angles, 90)) if angles else float('nan'),
        "edge_on_kd_pct": 100.0 * edge_on_kd / n if n > 0 else 0.0,
        "edge_on_mv_pct": 100.0 * edge_on_mv / n if n > 0 else 0.0,
        "agree_pct": 100.0 * sum(1 for r in results if r["agree"]) / n if n > 0 else 0.0,
    }

    # CSV
    if log_csv:
        with open(log_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'hand',
                'kd_normal_x', 'kd_normal_y', 'kd_normal_z',
                'kd_pos_x', 'kd_pos_y', 'kd_pos_z',
                'kd_confidence', 'kd_sigma_deg', 'kd_rms_m', 'kd_edge_on', 'kd_valid',
                'mv_normal_x', 'mv_normal_y', 'mv_normal_z',
                'mv_pos_x', 'mv_pos_y', 'mv_pos_z',
                'mv_confidence', 'mv_sigma_deg', 'mv_rms_m', 'mv_edge_on', 'mv_valid',
                'angle_deg', 'agree',
                'kd_latency_ms', 'mv_latency_ms',
            ])
            for r in results:
                ts = r["kinect_depth"].timestamp
                hand = r["kinect_depth"].hand
                row = [ts, hand]
                row += [
                    r["kinect_depth"].normal_cam[0], r["kinect_depth"].normal_cam[1], r["kinect_depth"].normal_cam[2],
                    r["kinect_depth"].position_cam[0] if r["kinect_depth"].position_cam is not None else float('nan'),
                    r["kinect_depth"].position_cam[1] if r["kinect_depth"].position_cam is not None else float('nan'),
                    r["kinect_depth"].position_cam[2] if r["kinect_depth"].position_cam is not None else float('nan'),
                    r["kinect_depth"].confidence, r["kinect_depth"].sigma_deg, r["kinect_depth"].rms_m,
                    r["kinect_depth"].edge_on, r["kinect_depth"].valid,
                    r["multiview"].normal_cam[0], r["multiview"].normal_cam[1], r["multiview"].normal_cam[2],
                    r["multiview"].position_cam[0] if r["multiview"].position_cam is not None else float('nan'),
                    r["multiview"].position_cam[1] if r["multiview"].position_cam is not None else float('nan'),
                    r["multiview"].position_cam[2] if r["multiview"].position_cam is not None else float('nan'),
                    r["multiview"].confidence, r["multiview"].sigma_deg, r["multiview"].rms_m,
                    r["multiview"].edge_on, r["multiview"].valid,
                    r["angle_deg"], r["agree"],
                    0.0, 0.0,  # latencias sinteticas = 0
                ]
                writer.writerow(row)
        print(f"[synthetic] CSV salvo em {log_csv}")

    print(f"[synthetic] Resumo: n={summary['n_frames']}, validos={summary['n_valid']}, "
          f"angulo medio={summary['angle_mean_deg']:.2f} deg, "
          f"edge_on KD={summary['edge_on_kd_pct']:.1f}% MV={summary['edge_on_mv_pct']:.1f}%, "
          f"agree={summary['agree_pct']:.1f}%")
    return summary


def run_hardware(
    duration: float = 10.0,
    aux_index: int = 0,
    serial_port: str = "COM4",
    no_window: bool = False,
    log_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Executa comparacao A/B com hardware real."""
    kinect_gt, KinectMediaPipeSource, ImuBankSource, AuxStereoSource, compare_frame, VisionNormal, KinectDepthBackend, MultiViewBackend = _import_hardware()

    print(f"[hardware] Iniciando: duracao={duration}s, aux_index={aux_index}, serial={serial_port}")

    # Inicializa fontes COMPARTILHANDO o tracker/alignment: dois
    # KinectHandTracker abririam o mesmo Kinect e conflitariam.
    aux_indices = tuple(getattr(kinect_gt, "AUX_CAMERA_INDICES", (aux_index,)))
    tracker = kinect_gt.KinectHandTracker(aux_indices=aux_indices)
    alignment = kinect_gt.CameraAlignment(aux_indices=aux_indices)
    kinect_source = KinectMediaPipeSource(tracker=tracker)
    stereo_source = AuxStereoSource(
        tracker=tracker, alignment=alignment,
        aux_indices=aux_indices, aux_index=aux_index,
    )

    # IMU bridge (opcional, so para contexto)
    imu_bridge = None
    try:
        from tools.imu_serial_bridge import ImuSerialBridge
        imu_bridge = ImuSerialBridge(port=serial_port)
        imu_bridge.start()
        print(f"[hardware] IMU bridge iniciado em {serial_port}")
    except Exception as e:
        print(f"[hardware] AVISO: IMU bridge nao iniciado ({e})")

    try:
        kinect_source.start()
        stereo_source.start()
    except Exception as e:
        print(f"[hardware] ERRO ao iniciar fontes: {e}")
        if imu_bridge:
            imu_bridge.stop()
        return {"error": str(e)}

    # CSV
    csv_file = None
    csv_writer = None
    if log_csv:
        csv_file = open(log_csv, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'timestamp', 'hand',
            'kd_normal_x', 'kd_normal_y', 'kd_normal_z',
            'kd_pos_x', 'kd_pos_y', 'kd_pos_z',
            'kd_confidence', 'kd_sigma_deg', 'kd_rms_m', 'kd_edge_on', 'kd_valid',
            'mv_normal_x', 'mv_normal_y', 'mv_normal_z',
            'mv_pos_x', 'mv_pos_y', 'mv_pos_z',
            'mv_confidence', 'mv_sigma_deg', 'mv_rms_m', 'mv_edge_on', 'mv_valid',
            'angle_deg', 'agree',
            'kd_latency_ms', 'mv_latency_ms',
        ])

    # Estado para continuidade temporal
    normal_anterior = {"right": None, "left": None}

    # Janelas
    if not no_window:
        import cv2
        cv2.namedWindow("Kinect RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Auxiliary", cv2.WINDOW_NORMAL)

    t0 = time.monotonic()
    frame_count = 0
    all_results = []

    try:
        while time.monotonic() - t0 < duration:
            loop_start = time.monotonic()

            # Poll Kinect
            k_ts, k_frame, k_obs = kinect_source.poll()

            # Poll stereo (triangulacao)
            stereo_frames = stereo_source.poll()

            # Para cada mao detectada em AMBAS as fontes
            for hand in ("right", "left"):
                k_ob = k_obs.get(hand)
                s_frame = stereo_frames.get(hand)

                if k_ob is None or not k_ob.valid or s_frame is None or not s_frame.valid:
                    continue

                # KinectDepth: usa landmarks do Kinect projetados no depth
                # Precisa extrair pontos 3D dos landmarks do Kinect
                k_landmarks = k_ob  # KinectObservation nao tem landmarks crus
                # Usamos a posicao do KinectObservation como proxy
                # Para pontos 3D reais, precisariamos do tracker.palm_plane_camera etc
                # Aqui usamos a triangulacao como "ground truth multi-view"
                # e os pontos do Kinect depth como "Kinect depth"
                # Como o KinectObservation so tem wrist/normal, usamos a triangulacao
                # para AMBOS os backends (A/B no mesmo conjunto de pontos)
                # MAS a ideia original e: KinectDepth usa pontos do Kinect depth,
                # MultiView usa pontos triangulados.
                #
                # Workaround: geramos pontos Kinect "sinteticos" em torno da posicao
                # medida, ou usamos a triangulacao para os dois com pesos diferentes.
                #
                # ABORDAGEM CORRETA: O KinectHandTracker tem palm_plane_camera que
                # devolve normal + rms do plano medio do dorso (5 pontos).
                # Podemos usar esses 5 pontos 3D como "Kinect depth points".
                #
                # Mas KinectObservation nao expõe os pontos crus.
                # Entao vamos usar a triangulacao como base e simular a diferenca:
                # - KinectDepth: pontos triangulados SEM pesos (uniforme)
                # - MultiView: pontos triangulados COM pesos de reprojecao
                #
                # Isso nao e ideal mas demonstra a comparacao A/B.
                # A/B REAL: KinectDepth usa os 5 pontos do DEPTH do Kinect;
                # MultiView usa os pontos triangulados.
                if s_frame.kinect_points is None:
                    continue
                kinect_points = s_frame.kinect_points
                multiview_points = s_frame.points_3d
                reproj_errors = s_frame.reproj_errors.copy()

                # Mede latencias
                kd_start = time.monotonic()
                cmp = compare_frame(
                    kinect_points=kinect_points,
                    multiview_points=multiview_points,
                    hand=hand,
                    timestamp=s_frame.timestamp,
                    reproj_errors=reproj_errors,
                    normal_anterior=normal_anterior[hand],
                )
                kd_latency = (time.monotonic() - kd_start) * 1000

                # Atualiza anterior
                if cmp["multiview"].valid:
                    normal_anterior[hand] = cmp["multiview"].normal_cam.copy()

                # CSV
                if csv_writer:
                    row = [s_frame.timestamp, hand]
                    row += [
                        cmp["kinect_depth"].normal_cam[0], cmp["kinect_depth"].normal_cam[1], cmp["kinect_depth"].normal_cam[2],
                        cmp["kinect_depth"].position_cam[0] if cmp["kinect_depth"].position_cam is not None else float('nan'),
                        cmp["kinect_depth"].position_cam[1] if cmp["kinect_depth"].position_cam is not None else float('nan'),
                        cmp["kinect_depth"].position_cam[2] if cmp["kinect_depth"].position_cam is not None else float('nan'),
                        cmp["kinect_depth"].confidence, cmp["kinect_depth"].sigma_deg, cmp["kinect_depth"].rms_m,
                        cmp["kinect_depth"].edge_on, cmp["kinect_depth"].valid,
                        cmp["multiview"].normal_cam[0], cmp["multiview"].normal_cam[1], cmp["multiview"].normal_cam[2],
                        cmp["multiview"].position_cam[0] if cmp["multiview"].position_cam is not None else float('nan'),
                        cmp["multiview"].position_cam[1] if cmp["multiview"].position_cam is not None else float('nan'),
                        cmp["multiview"].position_cam[2] if cmp["multiview"].position_cam is not None else float('nan'),
                        cmp["multiview"].confidence, cmp["multiview"].sigma_deg, cmp["multiview"].rms_m,
                        cmp["multiview"].edge_on, cmp["multiview"].valid,
                        cmp["angle_deg"], cmp["agree"],
                        kd_latency, 0.0,
                    ]
                    csv_writer.writerow(row)

                all_results.append(cmp)

            frame_count += 1

            # Display
            if not no_window and k_frame is not None:
                import cv2
                cv2.imshow("Kinect RGB", k_frame)
                # Aux frame nao acessivel diretamente aqui; stereo_source nao expõe
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            # Rate limit ~30 fps
            elapsed = time.monotonic() - loop_start
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)

    finally:
        kinect_source.close()
        stereo_source.close()
        if imu_bridge:
            imu_bridge.stop()
        if csv_file:
            csv_file.close()
        if not no_window:
            import cv2
            cv2.destroyAllWindows()

    # Resumo
    if all_results:
        n = len(all_results)
        angles = [r["angle_deg"] for r in all_results if r["kinect_depth"].valid and r["multiview"].valid]
        edge_on_kd = sum(1 for r in all_results if r["kinect_depth"].edge_on)
        edge_on_mv = sum(1 for r in all_results if r["multiview"].edge_on)
        agree = sum(1 for r in all_results if r["agree"])

        summary = {
            "n_frames": n,
            "n_valid": len(angles),
            "angle_mean_deg": float(np.mean(angles)) if angles else float('nan'),
            "angle_std_deg": float(np.std(angles)) if angles else float('nan'),
            "angle_median_deg": float(np.median(angles)) if angles else float('nan'),
            "angle_p90_deg": float(np.percentile(angles, 90)) if angles else float('nan'),
            "edge_on_kd_pct": 100.0 * edge_on_kd / n if n > 0 else 0.0,
            "edge_on_mv_pct": 100.0 * edge_on_mv / n if n > 0 else 0.0,
            "agree_pct": 100.0 * agree / n if n > 0 else 0.0,
        }
    else:
        summary = {
            "n_frames": 0,
            "n_valid": 0,
            "angle_mean_deg": float('nan'),
            "angle_std_deg": float('nan'),
            "angle_median_deg": float('nan'),
            "angle_p90_deg": float('nan'),
            "edge_on_kd_pct": 0.0,
            "edge_on_mv_pct": 0.0,
            "agree_pct": 0.0,
        }

    print(f"[hardware] Resumo: n={summary['n_frames']}, validos={summary['n_valid']}, "
          f"angulo medio={summary['angle_mean_deg']:.2f} deg, "
          f"edge_on KD={summary['edge_on_kd_pct']:.1f}% MV={summary['edge_on_mv_pct']:.1f}%, "
          f"agree={summary['agree_pct']:.1f}%")
    if log_csv:
        print(f"[hardware] CSV salvo em {log_csv}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Ferramenta A/B: KinectDepthBackend vs MultiViewBackend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--duracao", type=float, default=10.0,
                        help="Segundos de aquisicao (default: 10)")
    parser.add_argument("--sem-janela", action="store_true",
                        help="Nao abre janelas de video")
    parser.add_argument("--ponte-serial", default="COM4",
                        help="Porta serial do ESP32 (default: COM4)")
    parser.add_argument("--aux-index", type=int, default=0,
                        help="Indice da camera auxiliar OpenCV (default: 0)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Gera pontos 3D sinteticos com ruido/outliers, sem hardware")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Arquivo CSV para log por frame")

    args = parser.parse_args()

    if args.synthetic:
        run_synthetic(duration=args.duracao, log_csv=args.log_csv)
    else:
        run_hardware(
            duration=args.duracao,
            aux_index=args.aux_index,
            serial_port=args.ponte_serial,
            no_window=args.sem_janela,
            log_csv=args.log_csv,
        )


if __name__ == "__main__":
    main()