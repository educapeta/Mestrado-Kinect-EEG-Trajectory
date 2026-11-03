#!/usr/bin/env python
"""
CLI de bancada leve para rodar a V1 ao vivo.

Uso:
  python tools/fusion_live.py --duracao 20
  python tools/fusion_live.py --sem-janela --duracao 10
  python tools/fusion_live.py --sem-imu --duracao 15
  python tools/fusion_live.py --sem-kinect --duracao 15
  python tools/fusion_live.py --synthetic --sem-janela --duracao 1
  python tools/fusion_live.py --calib-dir C:\\mestradopy --duracao 20
  python tools/fusion_live.py --imu-portas 4210,4211 --duracao 20
  python tools/fusion_live.py --sem-calibracao --synthetic --sem-janela --duracao 1

Opcoes:
  --duracao SEGUNDOS     Duracao total da execucao em segundos (padrao: 30)
  --sem-janela           Nao abre janela OpenCV (modo headless/smoke)
  --sem-imu              Desabilita fonte IMU (diagnostico apenas Kinect)
  --sem-kinect           Desabilita fonte Kinect (diagnostico apenas IMU)
  --synthetic            Usa fontes sinteticas (sem hardware)
  --calib-dir DIRETORIO  Diretorio com arquivos de calibracao (padrao: cwd)
  --imu-portas PORTAS    Portas IMU separadas por virgula (padrao: 4210,4211)
  --sem-calibracao       Usa identidade/f=z para smoke; avisa que nao e' calibrado
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Garante que o diretório raiz do projeto está no sys.path para imports do pacote fusion
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos com ajuda em português."""
    parser = argparse.ArgumentParser(
        description="CLI de bancada leve para rodar a V1 de fusão ao vivo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--duracao', type=float, default=30.0,
        help='Duração total da execução em segundos (padrão: 30)'
    )
    parser.add_argument(
        '--sem-janela', action='store_true',
        help='Não abre janela OpenCV (modo headless/smoke)'
    )
    parser.add_argument(
        '--sem-imu', action='store_true',
        help='Desabilita fonte IMU (diagnóstico apenas Kinect)'
    )
    parser.add_argument(
        '--sem-kinect', action='store_true',
        help='Desabilita fonte Kinect (diagnóstico apenas IMU)'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Usa fontes sintéticas (sem hardware)'
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
        '--sem-calibracao', action='store_true',
        help='Usa identidade/f=z para smoke; avisa que não é calibrado'
    )
    return parser


def create_identity_calibration(hand: str):
    """Cria uma calibração identidade (não calibrada) para smoke test."""
    from fusion.contracts import Calibration
    return Calibration(
        hand=hand,
        T_cam_world=np.eye(3, dtype=np.float64),
        f_body=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        yaw_boot_delta_deg=0.0,
        residual_deg=None,
        source='identity_smoke'
    )


def load_calibrations_with_fallback(
    calib_dir: str,
    no_calibration: bool
) -> Dict[str, Any]:
    """
    Carrega calibrações com fallback para identidade se --sem-calibracao.

    Args:
        calib_dir: Diretório com arquivos de calibração
        no_calibration: Se True, usa calibração identidade

    Returns:
        Dict mapeando hand -> Calibration

    Raises:
        SystemExit: Se calibrações ausentes e sem --sem-calibracao
    """
    from fusion.adapters import load_calibrations

    calibrations = load_calibrations(calib_dir)

    if no_calibration:
        print("[AVISO] Usando calibração IDENTIDADE (não calibrado) -- apenas para smoke test")
        for hand in ('right', 'left'):
            if hand not in calibrations:
                calibrations[hand] = create_identity_calibration(hand)
        return calibrations

    # Verifica se temos pelo menos uma calibração válida
    if not calibrations:
        print(f"[ERRO] Nenhuma calibração válida encontrada em '{calib_dir}'", file=sys.stderr)
        print("       Execute a calibração primeiro:", file=sys.stderr)
        print("         python tools/calibra_palma_plano.py --lado right", file=sys.stderr)
        print("         python tools/calibra_palma_plano.py --lado left", file=sys.stderr)
        print("       Ou use --sem-calibracao para smoke test (não calibrado)", file=sys.stderr)
        sys.exit(1)

    missing = [h for h in ('right', 'left') if h not in calibrations]
    if missing:
        print(f"[AVISO] Calibrações faltando para: {missing}", file=sys.stderr)
        print("         Resultados para essas mãos podem ser imprecisos.", file=sys.stderr)

    return calibrations


def create_sources(args, calibrations: Dict[str, Any]):
    """
    Cria as fontes Kinect, IMU ou sintéticas baseado nos argumentos.

    Returns:
        Tuple (kinect_source, imu_source, synthetic_imu_poller)
        synthetic_imu_poller é uma callable para poll IMU quando usando SyntheticSources
    """
    kinect_source = None
    imu_source = None
    synthetic_imu_poller = None

    if args.synthetic:
        # Usa fontes sintéticas - não abre hardware
        from fusion.adapters import SyntheticSources
        synthetic = SyntheticSources()
        synthetic.start()
        kinect_source = synthetic
        synthetic_imu_poller = synthetic.poll_imu
        print("[INFO] Modo sintético: sem hardware Kinect/IMU")
    else:
        # Fontes reais
        if not args.sem_kinect:
            try:
                from fusion.adapters import KinectMediaPipeSource
                kinect_source = KinectMediaPipeSource()
                print("[INFO] Fonte Kinect+MediaPipe criada")
            except Exception as e:
                print(f"[AVISO] Falha ao criar fonte Kinect: {e}", file=sys.stderr)
                kinect_source = None

        if not args.sem_imu:
            try:
                from fusion.adapters import ImuBankSource
                imu_source = ImuBankSource(portas=args.imu_portas)
                print(f"[INFO] Fonte IMU criada (portas: {args.imu_portas})")
            except Exception as e:
                print(f"[AVISO] Falha ao criar fonte IMU: {e}", file=sys.stderr)
                imu_source = None

    return kinect_source, imu_source, synthetic_imu_poller


def run_fusion(
    args,
    calibrations: Dict[str, Any],
    kinect_source,
    imu_source,
    synthetic_imu_poller
) -> Dict[str, Any]:
    """
    Executa o loop de fusão ao vivo.

    Returns:
        Dict com estatísticas de resumo
    """
    from fusion.live import FusionEngine, LiveFusionApp, LiveConfig

    # Cria engine de fusão
    engine = FusionEngine(calibrations)

    # Configuração do app
    config = LiveConfig(
        duration_s=args.duracao,
        show=not args.sem_janela,
        print_every_s=1.0,
        max_dt=0.1
    )

    # Wrapper para fonte IMU sintética se necessário
    class SyntheticImuWrapper:
        def __init__(self, poller):
            self._poller = poller
            self._started = False

        def start(self):
            self._started = True

        def stop(self):
            self._started = False

        def poll(self, now: Optional[float] = None):
            if not self._started:
                return {}
            return self._poller(now)

    # Determina a fonte IMU efetiva
    effective_imu_source = imu_source
    if synthetic_imu_poller is not None:
        effective_imu_source = SyntheticImuWrapper(synthetic_imu_poller)

    # Cria e roda o app
    app = LiveFusionApp(
        engine=engine,
        kinect_source=kinect_source,
        imu_source=effective_imu_source,
        config=config
    )

    print(f"[INFO] Iniciando fusão V1 por {args.duracao:.1f}s...")
    if args.sem_janela:
        print("[INFO] Modo headless (sem janela OpenCV)")

    try:
        summary = app.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário (Ctrl+C)")
        summary = {
            'frames': getattr(app, '_frame_count', 0),
            'states': getattr(app, '_state_count', 0),
            'duration_s': time.monotonic() - getattr(app, '_start_time', time.monotonic()),
            'fps': 0.0
        }
    except Exception as e:
        print(f"[ERRO] Falha durante execução: {e}", file=sys.stderr)
        raise

    return summary


def print_summary(summary: Dict[str, Any], calibrations: Dict[str, Any]) -> None:
    """Imprime resumo final da execução."""
    print("\n" + "=" * 50)
    print("RESUMO DA EXECUÇÃO")
    print("=" * 50)
    print(f"Duração:     {summary.get('duration_s', 0):.2f} s")
    print(f"Frames:      {summary.get('frames', 0)}")
    print(f"Estados:     {summary.get('states', 0)}")
    print(f"FPS médio:   {summary.get('fps', 0):.1f}")
    print(f"Calibrações: {list(calibrations.keys())}")
    print("=" * 50)


def main(argv: Optional[list[str]] = None) -> int:
    """Função principal do CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Carrega calibrações
    calibrations = load_calibrations_with_fallback(args.calib_dir, args.sem_calibracao)

    # Cria fontes
    kinect_source, imu_source, synthetic_imu_poller = create_sources(args, calibrations)

    # Verifica se temos pelo menos uma fonte
    if kinect_source is None and imu_source is None and synthetic_imu_poller is None:
        print("[ERRO] Nenhuma fonte de dados disponível", file=sys.stderr)
        print("       Verifique --sem-kinect, --sem-imu, --synthetic", file=sys.stderr)
        return 1

    # Executa fusão
    try:
        summary = run_fusion(args, calibrations, kinect_source, imu_source, synthetic_imu_poller)
    except Exception as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        return 1

    # Imprime resumo
    print_summary(summary, calibrations)

    return 0


if __name__ == '__main__':
    sys.exit(main())