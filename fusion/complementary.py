"""
Quaternion complementary filter for sensor fusion.

Combines IMU gyroscope prediction with gravity and vision (yaw) corrections
using a complementary filter on SO(3).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .contracts import ImuSample, Calibration, FusedState
from .quaternion import (
    normalize, multiply, conjugate, rotate_vector, exp_so3, angle_between_deg
)


@dataclass
class ComplementaryConfig:
    """Configuration for complementary filter."""
    kp_grav: float = 2.0          # Proportional gain for gravity correction
    kp_yaw: float = 0.5           # Proportional gain for yaw/vision correction
    max_dt: float = 0.1           # Maximum dt for integration (seconds)
    accel_norm_min: float = 0.5   # Minimum acceptable accel norm (g)
    accel_norm_max: float = 1.5   # Maximum acceptable accel norm (g)
    max_grav_error: float = 0.5   # Maximum gravity error (rad) for gating
    max_yaw_error: float = 0.3    # Maximum yaw error (rad) for gating
    gravity_cam: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )  # Gravity vector in camera frame


class QuaternionComplementaryFilter:
    """
    Quaternion-based complementary filter for orientation estimation.

    State: q_body_cam (quaternion from body/IMU frame to camera frame)
    """

    def __init__(
        self,
        config: Optional[ComplementaryConfig] = None,
        q0: Optional[np.ndarray] = None,
        calibration: Optional[Calibration] = None
    ):
        """
        Initialize filter.

        Args:
            config: Filter configuration (uses defaults if None)
            q0: Initial quaternion (w, x, y, z). Defaults to identity.
            calibration: Hand-eye calibration for vision updates.
        """
        self.config = config if config is not None else ComplementaryConfig()
        self.calibration = calibration

        if q0 is not None:
            self._q = normalize(q0)
        else:
            self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def reset(self, q: np.ndarray) -> None:
        """Reset filter state to given quaternion."""
        self._q = normalize(q)

    @property
    def state(self) -> np.ndarray:
        """Return copy of current state quaternion (w, x, y, z)."""
        return self._q.copy()

    def predict(self, gyro_dps: np.ndarray, dt: float) -> None:
        """
        Predict step: integrate gyroscope measurement.

        Args:
            gyro_dps: Angular velocity in body frame (deg/s), shape (3,)
            dt: Time step in seconds
        """
        # Validate dt
        if not (dt > 0.0 and np.isfinite(dt)):
            return
        if dt > self.config.max_dt:
            return

        # Convert gyro to rad/s
        omega_body = np.deg2rad(gyro_dps)

        # Quaternion derivative: q_dot = 0.5 * q ⊗ [0, omega_body]
        # Integration: q <- q ⊗ exp_so3(omega_body * dt)
        # exp_so3 takes rotation vector (axis * angle), where angle = |omega| * dt
        rotvec = omega_body * dt
        dq = exp_so3(rotvec)
        self._q = multiply(self._q, dq)
        self._q = normalize(self._q)

    def update_gravity(self, accel_g: np.ndarray, dt: float) -> bool:
        """
        Correct using gravity vector from accelerometer.

        Args:
            accel_g: Accelerometer reading in body frame (g), shape (3,)
            dt: Time step in seconds

        Returns:
            True if correction applied, False if gated out
        """
        # Validate dt
        if not (dt > 0.0 and np.isfinite(dt)):
            return False
        if dt > self.config.max_dt:
            return False

        # Normalize accelerometer reading (accelerometer measures specific force = -gravity)
        # Use accelerometer directly as reference vector (points UP, opposite to gravity)
        accel_norm = np.linalg.norm(accel_g)
        if not (self.config.accel_norm_min <= accel_norm <= self.config.accel_norm_max):
            return False

        a_body = accel_g / accel_norm  # Accelerometer reading (points UP)

        # Predicted gravity in body frame: g_pred_body = R(q)^T * gravity_cam
        # = rotate_vector(conj(q), gravity_cam) (points DOWN)
        q_conj = conjugate(self._q)
        g_pred_body = rotate_vector(q_conj, self.config.gravity_cam)

        # Error vector: cross product of predicted gravity (DOWN) and accelerometer (UP)
        # e = cross(g_pred_body, a_body) gives correct correction direction for RIGHT multiplication
        e = np.cross(g_pred_body, a_body)
        error_mag = np.linalg.norm(e)

        # Gate on error magnitude
        if error_mag > self.config.max_grav_error:
            return False

        # Correction: q <- q ⊗ exp_so3(kp_grav * e * dt)
        correction = self.config.kp_grav * e * dt
        dq = exp_so3(correction)
        self._q = multiply(self._q, dq)
        self._q = normalize(self._q)

        return True

    def update_yaw(
        self,
        normal_cam: np.ndarray,
        dt: float,
        confidence: float = 1.0
    ) -> bool:
        """
        Correct using vision measurement of palm normal.

        Args:
            normal_cam: Measured palm normal in camera frame, shape (3,)
            dt: Time step in seconds
            confidence: Vision confidence [0, 1]

        Returns:
            True if correction applied, False if gated out or no calibration
        """
        # Validate dt
        if not (dt > 0.0 and np.isfinite(dt)):
            return False
        if dt > self.config.max_dt:
            return False

        # Need calibration for vision update
        if self.calibration is None:
            return False

        # Validate confidence
        if confidence <= 0.0:
            return False

        # Normalize measured normal
        n_vis = normal_cam
        n_vis_norm = np.linalg.norm(n_vis)
        if n_vis_norm == 0.0:
            return False
        n_vis = n_vis / n_vis_norm

        # Predicted normal in camera frame
        n_pred = self.calibration.normal_cam_from_q(self._q)

        # Error vector: cross product
        e = np.cross(n_pred, n_vis)
        error_mag = np.linalg.norm(e)

        # Gate on error magnitude
        if error_mag > self.config.max_yaw_error:
            return False

        # Correction applied on LEFT: q <- exp_so3(kp_yaw * conf * e * dt) ⊗ q
        correction = self.config.kp_yaw * confidence * e * dt
        dq = exp_so3(correction)
        self._q = multiply(dq, self._q)  # LEFT multiplication
        self._q = normalize(self._q)

        return True

    def step(
        self,
        imu: Optional[ImuSample],
        normal_cam: Optional[np.ndarray],
        dt: float,
        vision_confidence: float = 1.0
    ) -> np.ndarray:
        """
        Complete filter step: predict + gravity update + yaw update.

        Args:
            imu: IMU sample (gyro_dps used for predict, accel_g for gravity)
            normal_cam: Vision palm normal in camera frame
            dt: Time step in seconds
            vision_confidence: Confidence for vision update [0, 1]

        Returns:
            Updated state quaternion (copy)
        """
        # Validate dt
        if not (dt > 0.0 and np.isfinite(dt)) or dt > self.config.max_dt:
            return self.state

        # Predict from gyro
        if imu is not None and imu.valid:
            self.predict(imu.gyro_dps, dt)

            # Gravity correction from accelerometer
            self.update_gravity(imu.accel_g, dt)

        # Yaw correction from vision
        if normal_cam is not None:
            self.update_yaw(normal_cam, dt, vision_confidence)

        return self.state

    def normal_cam(self) -> Optional[np.ndarray]:
        """
        Get current palm normal estimate in camera frame.

        Returns:
            Normal vector in camera frame, or None if no calibration
        """
        if self.calibration is None:
            return None
        return self.calibration.normal_cam_from_q(self._q)

    def get_fused_state(
        self,
        timestamp: float,
        hand: str,
        confidence: float = 1.0,
        source: str = "complementary",
        latency_ms: float = 0.0,
        dir_cam: Optional[np.ndarray] = None,
    ) -> FusedState:
        """
        Create FusedState from current filter state.

        Args:
            timestamp: Monotonic timestamp
            hand: Hand identifier
            confidence: Fusion confidence
            source: Source description
            latency_ms: Processing latency

        Returns:
            FusedState instance
        """
        q = self.state
        normal = self.normal_cam()
        valid = (normal is not None)
        if normal is None:
            normal = np.zeros(3, dtype=np.float64)
        # A direcao dos dedos NAO e' derivavel so' da normal/f_body. Nesta V1
        # ela e' opcional; quem tiver o vetor (wrist->middle MCP) passa em
        # dir_cam. Nao invente uma direcao fixa.
        if dir_cam is None:
            dir_vec = np.zeros(3, dtype=np.float64)
        else:
            dir_vec = np.asarray(dir_cam, dtype=np.float64).reshape(3)
            nrm = np.linalg.norm(dir_vec)
            if nrm > 1e-9:
                dir_vec = dir_vec / nrm
            else:
                dir_vec = np.zeros(3, dtype=np.float64)

        return FusedState(
            timestamp=timestamp,
            hand=hand,
            q_body_cam=q,
            normal_cam=normal,
            dir_cam=dir_vec,
            confidence=confidence,
            source=source,
            latency_ms=latency_ms,
            valid=bool(valid)
        )