#!/usr/bin/env python
"""
Testes para o CLI fusion_live.py

Testa:
1. Importação do módulo sem hardware
2. Parser argparse com --help e parse_args das flags
3. Execução subprocess com --synthetic --sem-janela --duracao 1 --sem-calibracao (exit 0 + resumo)
4. Erro de calibração ausente sem abrir hardware (exit != 0 + mensagem clara)
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path


def test_import_without_hardware():
    """Testa que o módulo pode ser importado sem abrir Kinect/IMU."""
    print("1) Testando importação sem hardware...")
    # Importa o módulo principal - não deve abrir hardware
    import tools.fusion_live as fusion_live
    assert hasattr(fusion_live, 'main')
    assert hasattr(fusion_live, 'build_parser')
    assert hasattr(fusion_live, 'load_calibrations_with_fallback')
    assert hasattr(fusion_live, 'create_sources')
    assert hasattr(fusion_live, 'run_fusion')
    print("   OK: Módulo importado sem abrir hardware")


def test_parser_help():
    """Testa que --help funciona e mostra ajuda em português."""
    print("2) Testando parser --help...")
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run([
        sys.executable, 'tools/fusion_live.py', '--help'
    ], capture_output=True, text=True, encoding='utf-8', errors='replace',
       timeout=10, env=env)

    assert result.returncode == 0, f"--help falhou: {result.stderr}"
    assert '--duracao' in result.stdout
    assert '--sem-janela' in result.stdout
    assert '--sem-imu' in result.stdout
    assert '--sem-kinect' in result.stdout
    assert '--synthetic' in result.stdout
    assert '--calib-dir' in result.stdout
    assert '--imu-portas' in result.stdout
    assert '--sem-calibracao' in result.stdout
    # Verifica português
    assert 'Duração' in result.stdout or 'duracao' in result.stdout.lower()
    print("   OK: --help funciona e mostra flags em português")


def test_parser_parse_args():
    """Testa parse_args para todas as flags."""
    print("3) Testando parse_args...")
    import tools.fusion_live as fusion_live

    parser = fusion_live.build_parser()

    # Teste flags booleanas
    args = parser.parse_args(['--sem-janela', '--sem-imu', '--sem-kinect', '--synthetic', '--sem-calibracao'])
    assert args.sem_janela is True
    assert args.sem_imu is True
    assert args.sem_kinect is True
    assert args.synthetic is True
    assert args.sem_calibracao is True

    # Teste com valores
    args = parser.parse_args(['--duracao', '10', '--calib-dir', '/tmp', '--imu-portas', '5000,5001'])
    assert args.duracao == 10.0
    assert args.calib_dir == '/tmp'
    assert args.imu_portas == '5000,5001'

    # Teste defaults
    args = parser.parse_args([])
    assert args.duracao == 30.0
    assert args.calib_dir == '.'
    assert args.imu_portas == '4210,4211'
    assert args.sem_janela is False
    assert args.sem_imu is False
    assert args.sem_kinect is False
    assert args.synthetic is False
    assert args.sem_calibracao is False

    print("   OK: parse_args funciona para todas as flags e defaults")


def test_synthetic_smoke():
    """Testa execução com --synthetic --sem-janela --duracao 1 --sem-calibracao."""
    print("4) Testando execução sintética (smoke)...")

    # Usa diretório temporário vazio para garantir sem calibração
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent)
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run([
            sys.executable, 'tools/fusion_live.py',
            '--synthetic',
            '--sem-janela',
            '--duracao', '1',
            '--sem-calibracao',
            '--calib-dir', tmpdir
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30, env=env, cwd=Path(__file__).parent)

    print(f"   stdout: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
    if result.stderr:
        print(f"   stderr: {result.stderr[-500:] if len(result.stderr) > 500 else result.stderr}")

    assert result.returncode == 0, f"Exit code {result.returncode} != 0"

    # Verifica resumo na saída
    assert 'RESUMO DA EXECUÇÃO' in result.stdout or 'RESUMO' in result.stdout
    assert 'Duração:' in result.stdout
    assert 'Frames:' in result.stdout
    assert 'Estados:' in result.stdout
    assert 'FPS' in result.stdout
    assert 'Calibrações:' in result.stdout

    # Verifica aviso de não calibrado
    assert 'AVISO' in result.stdout and 'não calibrado' in result.stdout.lower()

    print("   OK: Execução sintética exit 0 com resumo e aviso")


def test_missing_calibration_error():
    """Testa erro claro quando calibrações ausentes sem --sem-calibracao."""
    print("5) Testando erro de calibração ausente...")

    # Usa diretório temporário vazio
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent)
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run([
            sys.executable, 'tools/fusion_live.py',
            '--sem-kinect',
            '--sem-imu',
            '--calib-dir', tmpdir,
            '--duracao', '1'
            # SEM --sem-calibracao
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, env=env, cwd=Path(__file__).parent)

    print(f"   stdout: {result.stdout}")
    print(f"   stderr: {result.stderr}")
    print(f"   returncode: {result.returncode}")

    assert result.returncode != 0, f"Deveria falhar (exit != 0), mas retornou {result.returncode}"

    # Verifica mensagem de erro clara
    output = result.stdout + result.stderr
    assert 'ERRO' in output.upper() or 'ERROR' in output.upper()
    assert 'calibração' in output.lower() or 'calibracao' in output.lower()
    assert 'calibra_palma_plano' in output.lower() or 'calibração' in output.lower()

    print("   OK: Erro claro com comando de calibração, exit != 0")


def test_identity_calibration_creation():
    """Testa criação de calibração identidade."""
    print("6) Testando criação de calibração identidade...")
    import tools.fusion_live as fusion_live
    from fusion.contracts import Calibration

    cal_right = fusion_live.create_identity_calibration('right')
    cal_left = fusion_live.create_identity_calibration('left')

    assert isinstance(cal_right, Calibration)
    assert isinstance(cal_left, Calibration)
    assert cal_right.hand == 'right'
    assert cal_left.hand == 'left'
    assert cal_right.source == 'identity_smoke'
    assert cal_left.source == 'identity_smoke'

    # Verifica que é identidade
    assert np.allclose(cal_right.T_cam_world, np.eye(3))
    assert np.allclose(cal_left.T_cam_world, np.eye(3))
    assert np.allclose(cal_right.f_body, [0.0, 0.0, 1.0])
    assert np.allclose(cal_left.f_body, [0.0, 0.0, 1.0])

    print("   OK: Calibração identidade criada corretamente")


def test_load_calibrations_fallback():
    """Testa load_calibrations_with_fallback com e sem --sem-calibracao."""
    print("7) Testando load_calibrations_with_fallback...")

    import tools.fusion_live as fusion_live
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        # Sem calibração e sem --sem-calibracao -> deve falhar (sys.exit)
        try:
            fusion_live.load_calibrations_with_fallback(tmpdir, no_calibration=False)
            assert False, "Deveria ter chamado sys.exit"
        except SystemExit as e:
            assert e.code == 1

        # Com --sem-calibracao -> deve retornar identidade
        calibrations = fusion_live.load_calibrations_with_fallback(tmpdir, no_calibration=True)
        assert 'right' in calibrations
        assert 'left' in calibrations
        assert calibrations['right'].source == 'identity_smoke'
        assert calibrations['left'].source == 'identity_smoke'

    print("   OK: Fallback funciona corretamente")


if __name__ == '__main__':
    # Adiciona o diretório raiz ao path para imports
    sys.path.insert(0, str(Path(__file__).parent))

    import numpy as np

    print("=" * 60)
    print("TESTES DO CLI fusion_live.py")
    print("=" * 60)

    try:
        test_import_without_hardware()
        test_parser_help()
        test_parser_parse_args()
        test_identity_calibration_creation()
        test_load_calibrations_fallback()
        test_synthetic_smoke()
        test_missing_calibration_error()

        print("\n" + "=" * 60)
        print("TODOS OS TESTES PASSARAM!")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n[ERRO] Teste falhou: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)