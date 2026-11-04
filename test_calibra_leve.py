"""Testes do calibra_leve.py -- sem hardware."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import calibra_leve  # noqa: E402


def main():
    parser = calibra_leve.build_parser()
    args = parser.parse_args([])
    assert args.tempo_captura == calibra_leve.CAPTURE_SECONDS
    assert args.ponte_serial is None
    assert args.sem_janela is False
    assert args.auto_iniciar is False
    args2 = parser.parse_args(["--ponte-serial", "COM4", "--auto-iniciar",
                               "--sem-janela", "--duracao", "2"])
    assert args2.ponte_serial == "COM4"
    assert args2.auto_iniciar and args2.sem_janela

    cap = calibra_leve.CapturaYLeve(1.0)
    cap.iniciar(0.0)
    assert cap.ativa
    rng = np.random.default_rng(1)
    t = 0.0
    for i in range(20):
        sample = SimpleNamespace(roll_deg=float(rng.uniform(-60, 60)),
                                 pitch_deg=float(rng.uniform(-60, 60)),
                                 yaw_deg=float(rng.uniform(-180, 180)))
        normal = rng.normal(size=3)
        t += 0.2
        cap.coletar("right", t, sample, normal)
    assert cap.contar("right") >= 1
    cap.parar()
    assert not cap.ativa

    # JSON de calibracao valido, mas sem tocar no arquivo oficial.
    import tempfile, os
    antigo = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            ok = calibra_leve.salvar_alinhamento(
                "right", np.eye(3), np.array([0.0, 0.0, 1.0]), 3.0)
            assert ok
            import json
            d = json.loads(Path("imu_palm_alignment_right.json").read_text(encoding="utf-8"))
            assert d["rotation"] == np.eye(3).tolist()
            assert d["body"] == [0.0, 0.0, 1.0]
            assert d["residual_deg"] == 3.0
        finally:
            os.chdir(antigo)
    print("CALIBRA_LEVE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
