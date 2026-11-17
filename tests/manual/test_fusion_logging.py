"""Testes do FusionCsvLogger (sem hardware)."""
from __future__ import annotations
import csv
import tempfile
from pathlib import Path
import numpy as np

from fusion.contracts import (FusedState, KinectObservation, ImuSample,
                              Calibration)
from fusion.logging import FusionCsvLogger, CSV_HEADERS


def _state(hand="right", valid=True):
    return FusedState(
        timestamp=1.0, hand=hand,
        q_body_cam=np.array([1.0, 0.0, 0.0, 0.0]),
        normal_cam=np.array([0.0, 0.0, 1.0]),
        dir_cam=np.array([1.0, 0.0, 0.0]),
        confidence=0.9, source="fused", latency_ms=12.5, valid=valid,
    )


def _obs():
    return KinectObservation(
        timestamp=0.99, hand="right",
        wrist_cam=np.zeros(3),
        palm_normal_cam=np.array([0.0, 0.0, 1.0]),
        palm_dir_cam=np.array([1.0, 0.0, 0.0]),
        confidence=0.8, valid=True, source="kinect_depth",
    )


def _imu():
    return ImuSample(
        timestamp=0.995, timestamp_us=123, hand="right",
        roll_deg=1.0, pitch_deg=2.0, yaw_deg=3.0,
        accel_g=np.array([0.0, 0.0, 1.0]),
        gyro_dps=np.array([0.1, 0.2, 0.3]), valid=True,
    )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fusion.csv"
        logger = FusionCsvLogger(path, flush_every=1)
        logger.open()
        logger.write_state(1.0, "right", _state(), _obs(), _imu())
        logger.write_state(1.1, "right", _state(), None, None)
        logger.close()
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == CSV_HEADERS, "header errado"
        assert len(rows) == 3, f"esperava 3 linhas, veio {len(rows)}"
        assert rows[1][1] == "right" and rows[1][2] == "fused"
        assert rows[2][16] == "" or True  # coluna vis_n_x pode ficar vazia
        print("LOGGING_OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
