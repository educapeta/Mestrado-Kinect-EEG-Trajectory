"""
Data contracts for sensor fusion pipeline.

All dataclasses use default_factory for array fields to avoid mutable defaults.
Classes are serializable to/from dict for CSV/JSON storage.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import numpy as np

from .quaternion import (
    normalize, rotate_vector, from_euler_zyx_deg, to_matrix
)


def _arr3() -> np.ndarray:
    """Default factory for 3-element float64 array."""
    return np.zeros(3, dtype=np.float64)


def _arr4() -> np.ndarray:
    """Default factory for 4-element float64 array (quaternion)."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _arr21_2() -> np.ndarray:
    """Default factory for (21, 2) float64 array."""
    return np.zeros((21, 2), dtype=np.float64)


def _arr21_3() -> np.ndarray:
    """Default factory for (21, 3) float64 array."""
    return np.zeros((21, 3), dtype=np.float64)


def _arr33() -> np.ndarray:
    """Default factory for (3, 3) float64 array."""
    return np.eye(3, dtype=np.float64)


@dataclass
class KinectObservation:
    """
    Observation from Kinect depth camera.

    Attributes:
        timestamp: Monotonic timestamp in seconds
        hand: Hand identifier ("left" or "right")
        wrist_cam: Wrist position in camera frame (3,) meters, or None
        palm_normal_cam: Palm normal in camera frame (3,) unit vector, or None
        palm_dir_cam: Palm direction in camera frame (3,) unit vector, or None
        confidence: Detection confidence [0, 1]
        valid: Whether observation is valid
        source: Source identifier string
    """
    timestamp: float
    hand: str
    wrist_cam: Optional[np.ndarray] = field(default_factory=_arr3)
    palm_normal_cam: Optional[np.ndarray] = field(default_factory=_arr3)
    palm_dir_cam: Optional[np.ndarray] = field(default_factory=_arr3)
    confidence: float = 0.0
    valid: bool = False
    source: str = "kinect_depth"

    def __post_init__(self):
        # Convert None to zero arrays for consistent handling
        if self.wrist_cam is None:
            self.wrist_cam = _arr3()
        if self.palm_normal_cam is None:
            self.palm_normal_cam = _arr3()
        if self.palm_dir_cam is None:
            self.palm_dir_cam = _arr3()


@dataclass
class MediaPipeObservation:
    """
    Observation from MediaPipe hand tracking.

    Attributes:
        timestamp: Monotonic timestamp in seconds
        hand: Hand identifier ("left" or "right")
        landmarks_2d: 21 hand landmarks in pixels (21, 2), or None
        landmarks_3d_cam: 21 hand landmarks in camera frame (21, 3) meters, or None
        handedness: MediaPipe handedness score [0, 1]
        confidence: Detection confidence [0, 1]
        valid: Whether observation is valid
        source: Source identifier string
    """
    timestamp: float
    hand: str
    landmarks_2d: Optional[np.ndarray] = field(default_factory=_arr21_2)
    landmarks_3d_cam: Optional[np.ndarray] = field(default_factory=_arr21_3)
    handedness: float = 0.0
    confidence: float = 0.0
    valid: bool = False
    source: str = "mediapipe"

    def __post_init__(self):
        if self.landmarks_2d is None:
            self.landmarks_2d = _arr21_2()
        if self.landmarks_3d_cam is None:
            self.landmarks_3d_cam = _arr21_3()


@dataclass
class ImuSample:
    """
    IMU sample from glove sensor.

    Attributes:
        timestamp: Monotonic timestamp in seconds
        timestamp_us: Microsecond timestamp from device
        hand: Hand identifier ("left" or "right")
        roll_deg: Roll angle in degrees
        pitch_deg: Pitch angle in degrees
        yaw_deg: Yaw angle in degrees
        accel_g: Accelerometer reading in g (3,)
        gyro_dps: Gyroscope reading in deg/s (3,)
        valid: Whether sample is valid
        source: Source identifier string
    """
    timestamp: float
    timestamp_us: int
    hand: str
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    accel_g: np.ndarray = field(default_factory=_arr3)
    gyro_dps: np.ndarray = field(default_factory=_arr3)
    valid: bool = True
    source: str = "imu_udp"


@dataclass
class Calibration:
    """
    Hand-eye calibration data.

    Transforms from body (IMU) frame to camera frame.
    T_cam_world: Rotation from world to camera frame (3, 3)
    f_body: Palm normal in body frame (3,) unit vector
    yaw_boot_delta_deg: Yaw offset at boot time
    residual_deg: Calibration residual error in degrees
    source: Calibration source description
    """
    hand: str
    T_cam_world: np.ndarray = field(default_factory=_arr33)
    f_body: np.ndarray = field(default_factory=_arr3)
    yaw_boot_delta_deg: float = 0.0
    residual_deg: Optional[float] = None
    source: str = ""

    def validate(self) -> bool:
        """
        Validate calibration data integrity.

        Checks:
        - T_cam_world is 3x3 and approximately SO(3)
        - f_body is 3-element unit vector
        - All values are finite

        Returns:
            True if valid, False otherwise
        """
        # Check shapes
        if self.T_cam_world.shape != (3, 3):
            return False
        if self.f_body.shape != (3,):
            return False

        # Check finite values
        if not np.all(np.isfinite(self.T_cam_world)):
            return False
        if not np.all(np.isfinite(self.f_body)):
            return False
        if not np.isfinite(self.yaw_boot_delta_deg):
            return False
        if self.residual_deg is not None and not np.isfinite(self.residual_deg):
            return False

        # Check T_cam_world is approximately SO(3)
        # Orthogonality: R @ R.T ≈ I
        if not np.allclose(self.T_cam_world @ self.T_cam_world.T, np.eye(3), atol=1e-4):
            return False
        # Determinant ≈ 1
        if not np.isclose(np.linalg.det(self.T_cam_world), 1.0, atol=1e-4):
            return False

        # Check f_body is unit vector
        if not np.isclose(np.linalg.norm(self.f_body), 1.0, atol=1e-4):
            return False

        return True

    def normal_cam_from_q(self, q_body_cam: np.ndarray) -> np.ndarray:
        """
        Compute palm normal in camera frame from body-to-camera quaternion.

        Args:
            q_body_cam: Quaternion from body to camera frame (w, x, y, z)

        Returns:
            Palm normal in camera frame (3,) unit vector
        """
        return rotate_vector(q_body_cam, self.f_body)

    def normal_cam_from_imu(self, sample: ImuSample) -> np.ndarray:
        """
        Compute palm normal in camera frame from IMU sample.

        Uses: T_cam_world @ R_imu @ f_body
        where R_imu is from_euler_zyx_deg(roll, pitch, yaw)

        Args:
            sample: IMU sample with orientation angles

        Returns:
            Palm normal in camera frame (3,) unit vector
        """
        from .quaternion import from_euler_zyx_deg, to_matrix
        q_imu = from_euler_zyx_deg(sample.roll_deg, sample.pitch_deg, sample.yaw_deg)
        R_imu = to_matrix(q_imu)
        # R_imu transforms from body to world
        # T_cam_world transforms from world to camera
        # So: v_cam = T_cam_world @ R_imu @ f_body
        v_world = R_imu @ self.f_body
        return self.T_cam_world @ v_world

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary with native Python types.

        Returns:
            Dictionary suitable for JSON/CSV serialization
        """
        return {
            "hand": self.hand,
            "rotation": self.T_cam_world.tolist(),
            "body": self.f_body.tolist(),
            "residual_deg": self.residual_deg,
            "yaw_boot_delta_deg": self.yaw_boot_delta_deg,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], hand: str) -> Calibration:
        """
        Deserialize from dictionary.

        Args:
            d: Dictionary with keys rotation, body, residual_deg, yaw_boot_delta_deg, source
            hand: Hand identifier ("left" or "right")

        Returns:
            Calibration instance
        """
        return cls(
            hand=hand,
            T_cam_world=np.array(d.get("rotation", np.eye(3)), dtype=np.float64),
            f_body=np.array(d.get("body", [0.0, 0.0, 1.0]), dtype=np.float64),
            residual_deg=d.get("residual_deg"),
            yaw_boot_delta_deg=d.get("yaw_boot_delta_deg", 0.0),
            source=d.get("source", ""),
        )


@dataclass
class FusedState:
    """
    Fused hand state estimate.

    Attributes:
        timestamp: Monotonic timestamp in seconds
        hand: Hand identifier ("left" or "right")
        q_body_cam: Quaternion from body to camera frame (4,)
        normal_cam: Palm normal in camera frame (3,)
        dir_cam: Palm direction in camera frame (3,)
        confidence: Fusion confidence [0, 1]
        source: Source description (e.g., "kinect+imu", "imu_only")
        latency_ms: Processing latency in milliseconds
        valid: Whether state is valid
        schema_ver: Schema version for forward compatibility
    """
    timestamp: float
    hand: str
    q_body_cam: np.ndarray = field(default_factory=_arr4)
    normal_cam: np.ndarray = field(default_factory=_arr3)
    dir_cam: np.ndarray = field(default_factory=_arr3)
    confidence: float = 0.0
    source: str = ""
    latency_ms: float = 0.0
    valid: bool = False
    schema_ver: int = 1

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary with native Python types.

        Returns:
            Dictionary suitable for JSON/CSV serialization
        """
        return {
            "timestamp": float(self.timestamp),
            "hand": self.hand,
            "q_body_cam": self.q_body_cam.tolist(),
            "normal_cam": self.normal_cam.tolist(),
            "dir_cam": self.dir_cam.tolist(),
            "confidence": float(self.confidence),
            "source": self.source,
            "latency_ms": float(self.latency_ms),
            "valid": bool(self.valid),
            "schema_ver": int(self.schema_ver),
        }


@dataclass
class SourceStatus:
    """
    Status of a data source.

    Attributes:
        source: Source identifier string
        age_ms: Age of latest data in milliseconds
        confidence: Current confidence [0, 1]
        valid: Whether source is currently valid
        reason: Human-readable reason if invalid
    """
    source: str
    age_ms: float
    confidence: float
    valid: bool
    reason: str = ""