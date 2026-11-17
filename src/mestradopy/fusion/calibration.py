"""
Calibration loading, saving, and application for hand-eye transforms.

Reuses fusion.contracts.Calibration dataclass and validation.
Provides JSON round-trip, CalibrationManager for multi-hand handling,
and normal vector computation from quaternion or IMU samples.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .contracts import Calibration, ImuSample
from .quaternion import from_euler_zyx_deg, to_matrix, rotate_vector


def load_calibration(path: str | Path, hand: str) -> Optional[Calibration]:
    """
    Load calibration from JSON file.

    Expected JSON format:
    {
        "rotation": [[...], [...], [...]],  // 3x3 T_cam_world
        "body": [fx, fy, fz],               // f_body unit vector
        "residual_deg": float or null,
        "yaw_boot_delta_deg": float,
        "source": "description string"
    }

    Args:
        path: Path to JSON file
        hand: Hand identifier ("left" or "right")

    Returns:
        Validated Calibration instance, or None if file missing/invalid
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    cal = Calibration.from_dict(data, hand)
    if not cal.validate():
        return None

    return cal


def save_calibration(cal: Calibration, path: str | Path) -> None:
    """
    Save calibration to JSON file with native float types and indentation.

    Args:
        cal: Calibration instance to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = cal.to_dict()

    # Ensure native Python floats (not numpy types) for JSON serialization
    def convert(obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    data = convert(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class CalibrationManager:
    """
    Manages calibrations for multiple hands.

    Provides convenient access and normal vector computation from
    either quaternions or raw IMU samples.
    """

    def __init__(self, calibrations: Optional[Dict[str, Calibration]] = None):
        """
        Initialize with optional pre-loaded calibrations.

        Args:
            calibrations: Dict mapping hand -> Calibration
        """
        self._calibrations: Dict[str, Calibration] = {}
        if calibrations:
            for hand, cal in calibrations.items():
                self.add(cal)

    def add(self, cal: Calibration) -> None:
        """
        Add or replace a calibration.

        Args:
            cal: Calibration instance (must be valid)
        """
        if not cal.validate():
            raise ValueError(f"Invalid calibration for hand '{cal.hand}'")
        self._calibrations[cal.hand] = cal

    def get(self, hand: str) -> Optional[Calibration]:
        """
        Get calibration for a hand.

        Args:
            hand: Hand identifier ("left" or "right")

        Returns:
            Calibration or None if not found
        """
        return self._calibrations.get(hand)

    def normal_from_q(self, hand: str, q_body_cam: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute palm normal in camera frame from body-to-camera quaternion.

        Args:
            hand: Hand identifier
            q_body_cam: Quaternion from body to camera frame (w, x, y, z)

        Returns:
            Palm normal in camera frame (3,) unit vector, or None if no calibration
        """
        cal = self.get(hand)
        if cal is None:
            return None
        return cal.normal_cam_from_q(q_body_cam)

    def normal_from_imu(self, hand: str, sample: ImuSample) -> Optional[np.ndarray]:
        """
        Compute palm normal in camera frame from IMU sample.

        Uses: T_cam_world @ R_imu @ f_body
        where R_imu is from_euler_zyx_deg(roll, pitch, yaw)

        Args:
            hand: Hand identifier
            sample: IMU sample with orientation angles

        Returns:
            Palm normal in camera frame (3,) unit vector, or None if no calibration
        """
        cal = self.get(hand)
        if cal is None:
            return None
        return cal.normal_cam_from_imu(sample)

    def validate_all(self) -> Dict[str, bool]:
        """
        Validate all stored calibrations.

        Returns:
            Dict mapping hand -> validation result (True/False)
        """
        return {hand: cal.validate() for hand, cal in self._calibrations.items()}

    def __contains__(self, hand: str) -> bool:
        return hand in self._calibrations

    def __len__(self) -> int:
        return len(self._calibrations)

    def keys(self):
        return self._calibrations.keys()