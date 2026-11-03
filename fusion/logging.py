"""
CSV logging for fusion live runs.

Provides FusionCsvLogger to write per-hand/frame rows with all available
fusion state, observations, IMU data, and error metrics.
"""

from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .contracts import (
    FusedState,
    KinectObservation,
    ImuSample,
    Calibration,
    SourceStatus,
)


# CSV column names in order
CSV_HEADERS = [
    't_mono_s',
    'hand',
    'source',
    'confidence',
    'valid',
    'latency_ms',
    'q_w', 'q_x', 'q_y', 'q_z',
    'n_x', 'n_y', 'n_z',
    'dir_x', 'dir_y', 'dir_z',
    'vis_n_x', 'vis_n_y', 'vis_n_z',
    'err_fused_vis_deg',
    'err_cal_vis_deg',
    'imu_age_ms',
    'imu_roll_deg', 'imu_pitch_deg', 'imu_yaw_deg',
    'imu_gx_dps', 'imu_gy_dps', 'imu_gz_dps',
]


class FusionCsvLogger:
    """
    CSV logger for fusion live runs.

    Writes one row per hand per frame with all available data from:
    - FusedState (engine output)
    - KinectObservation (vision)
    - ImuSample (IMU)
    - Calibration (for cal_vs_vis error)

    Opens/closes file with header and flushes periodically.
    """

    def __init__(
        self,
        path: str | Path,
        flush_every: int = 10,
    ):
        """
        Initialize logger.

        Args:
            path: CSV file path to write
            flush_every: Flush to disk every N rows (default: 10)
        """
        self._path = Path(path)
        self._flush_every = flush_every
        self._file = None
        self._writer = None
        self._row_count = 0

    def open(self) -> None:
        """Open CSV file and write header."""
        if self._file is not None:
            return
        self._file = open(self._path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_HEADERS)
        self._row_count = 0

    def close(self) -> None:
        """Close CSV file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> 'FusionCsvLogger':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _angle_deg(self, v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> Optional[float]:
        """Compute angle in degrees between two vectors, or None if invalid."""
        if v1 is None or v2 is None:
            return None
        v1 = np.asarray(v1, dtype=np.float64).ravel()
        v2 = np.asarray(v2, dtype=np.float64).ravel()
        if v1.shape != (3,) or v2.shape != (3,):
            return None
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        dot = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
        return math.degrees(math.acos(dot))

    def write_state(
        self,
        t_mono: float,
        hand: str,
        fused: FusedState,
        kinect_obs: Optional[KinectObservation] = None,
        imu_sample: Optional[ImuSample] = None,
        calibration: Optional[Calibration] = None,
    ) -> None:
        """
        Write one row for a hand at a timestamp.

        Args:
            t_mono: Monotonic timestamp in seconds
            hand: 'right' or 'left'
            fused: FusedState from engine
            kinect_obs: Optional KinectObservation for this hand
            imu_sample: Optional ImuSample for this hand
            calibration: Optional Calibration for this hand
        """
        if self._writer is None:
            raise RuntimeError("Logger not opened. Call open() or use context manager.")

        # Vision normal from Kinect
        vis_normal = None
        if kinect_obs is not None and kinect_obs.valid and kinect_obs.palm_normal_cam is not None:
            n = kinect_obs.palm_normal_cam
            if np.all(np.isfinite(n)) and np.linalg.norm(n) > 1e-6:
                vis_normal = n / np.linalg.norm(n)

        # Fused normal
        fused_normal = fused.normal_cam if fused.valid else None
        fused_dir = fused.dir_cam if fused.valid else None
        fused_q = fused.q_body_cam if fused.valid else None

        # Error: fused vs vision
        err_fused_vis = self._angle_deg(fused_normal, vis_normal)

        # Error: calibration (from IMU) vs vision
        err_cal_vis = None
        if calibration is not None and imu_sample is not None and imu_sample.valid and vis_normal is not None:
            try:
                n_cal = calibration.normal_cam_from_imu(imu_sample)
                if np.all(np.isfinite(n_cal)) and np.linalg.norm(n_cal) > 1e-6:
                    err_cal_vis = self._angle_deg(n_cal, vis_normal)
            except Exception:
                pass

        # IMU age
        imu_age_ms = 0.0
        if imu_sample is not None:
            imu_age_ms = max(0.0, (t_mono - imu_sample.timestamp) * 1000.0)

        # Build row
        row = [
            f'{t_mono:.6f}',
            hand,
            fused.source if fused.valid else 'none',
            f'{fused.confidence:.4f}',
            '1' if fused.valid else '0',
            f'{fused.latency_ms:.2f}',
        ]

        # Quaternion
        if fused_q is not None and np.all(np.isfinite(fused_q)):
            row.extend([f'{fused_q[0]:.6f}', f'{fused_q[1]:.6f}',
                        f'{fused_q[2]:.6f}', f'{fused_q[3]:.6f}'])
        else:
            row.extend(['', '', '', ''])

        # Fused normal
        if fused_normal is not None and np.all(np.isfinite(fused_normal)):
            row.extend([f'{fused_normal[0]:.6f}', f'{fused_normal[1]:.6f}', f'{fused_normal[2]:.6f}'])
        else:
            row.extend(['', '', ''])

        # Fused direction
        if fused_dir is not None and np.all(np.isfinite(fused_dir)):
            row.extend([f'{fused_dir[0]:.6f}', f'{fused_dir[1]:.6f}', f'{fused_dir[2]:.6f}'])
        else:
            row.extend(['', '', ''])

        # Vision normal
        if vis_normal is not None:
            row.extend([f'{vis_normal[0]:.6f}', f'{vis_normal[1]:.6f}', f'{vis_normal[2]:.6f}'])
        else:
            row.extend(['', '', ''])

        # Errors
        row.append(f'{err_fused_vis:.3f}' if err_fused_vis is not None else '')
        row.append(f'{err_cal_vis:.3f}' if err_cal_vis is not None else '')

        # IMU data
        row.append(f'{imu_age_ms:.2f}')

        if imu_sample is not None and imu_sample.valid:
            row.extend([
                f'{imu_sample.roll_deg:.3f}',
                f'{imu_sample.pitch_deg:.3f}',
                f'{imu_sample.yaw_deg:.3f}',
                f'{imu_sample.gyro_dps[0]:.3f}',
                f'{imu_sample.gyro_dps[1]:.3f}',
                f'{imu_sample.gyro_dps[2]:.3f}',
            ])
        else:
            row.extend(['', '', '', '', '', ''])

        self._writer.writerow(row)
        self._row_count += 1

        if self._row_count % self._flush_every == 0:
            self._file.flush()

    def flush(self) -> None:
        """Flush pending writes to disk."""
        if self._file is not None:
            self._file.flush()

    @property
    def path(self) -> Path:
        """Return the CSV file path."""
        return self._path

    @property
    def rows_written(self) -> int:
        """Return number of rows written so far."""
        return self._row_count


def create_logger(path: str | Path, flush_every: int = 10) -> FusionCsvLogger:
    """Factory function to create and open a FusionCsvLogger."""
    logger = FusionCsvLogger(path, flush_every)
    logger.open()
    return logger