"""
Sensor fusion decision logic for choosing between vision, IMU, or fused sources.

Provides SensorDecision dataclass and SensorManager class that evaluates
available sources and decides which to use based on validity, confidence, and age.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from .contracts import (
    KinectObservation,
    MediaPipeObservation,
    ImuSample,
)


@dataclass
class SensorDecision:
    """
    Decision output from SensorManager.

    Attributes:
        source: Source identifier string ("kinect", "mediapipe", "imu", "fused", "none")
        confidence: Combined confidence [0, 1]
        reason: Human-readable explanation
        use_vision: Whether to use vision data
        use_imu: Whether to use IMU data
    """
    source: str
    confidence: float
    reason: str
    use_vision: bool
    use_imu: bool


class SensorManager:
    """
    Decides which sensor sources to use for fusion.

    Rules:
    - IMU is valid if sample exists, valid=True, and age <= max_imu_age_ms
    - Vision is valid if observation exists, valid=True, and confidence >= min_vision_confidence
    - If both valid -> "fused" with combined confidence (geometric mean)
    - If only IMU valid -> "imu_only"
    - If only vision valid -> "vision_only" (prefers kinect over mediapipe)
    - If neither valid -> "none"
    """

    def __init__(
        self,
        min_vision_confidence: float = 0.35,
        max_imu_age_ms: float = 200.0,
    ):
        """
        Initialize sensor manager.

        Args:
            min_vision_confidence: Minimum confidence for vision to be considered valid
            max_imu_age_ms: Maximum age in milliseconds for IMU to be considered fresh
        """
        self.min_vision_confidence = min_vision_confidence
        self.max_imu_age_ms = max_imu_age_ms

    def _check_imu(self, imu: Optional[ImuSample], now: float) -> tuple[bool, float, str]:
        """
        Check IMU validity and return (valid, confidence, reason).

        Returns:
            Tuple of (is_valid, confidence, reason_string)
        """
        if imu is None:
            return False, 0.0, "no IMU sample"

        if not getattr(imu, "valid", False):
            return False, 0.0, "IMU sample marked invalid"

        age_ms = (now - imu.timestamp) * 1000.0
        if age_ms > self.max_imu_age_ms:
            return False, 0.0, f"IMU age {age_ms:.1f}ms > max {self.max_imu_age_ms}ms"

        # IMU confidence is binary (valid/fresh) in V1
        return True, 1.0, "IMU valid and fresh"

    def _check_vision(
        self,
        vision: Optional[Union[KinectObservation, MediaPipeObservation]],
    ) -> tuple[bool, float, str, str]:
        """
        Check vision validity and return (valid, confidence, reason, source_name).

        Prefers kinect over mediapipe if both provided (handled by caller).
        """
        if vision is None:
            return False, 0.0, "no vision observation", "none"

        if not getattr(vision, "valid", False):
            return False, 0.0, "vision observation marked invalid", vision.source

        conf = getattr(vision, "confidence", 0.0)
        if conf < self.min_vision_confidence:
            return False, conf, f"vision confidence {conf:.2f} < {self.min_vision_confidence}", vision.source

        return True, conf, "vision valid and confident", vision.source

    def decide(
        self,
        kinect: Optional[KinectObservation],
        vision: Optional[Union[KinectObservation, MediaPipeObservation]],
        imu: Optional[ImuSample],
        now: float,
    ) -> SensorDecision:
        """
        Decide which sources to use based on current observations.

        Args:
            kinect: Kinect depth observation (may be None)
            vision: Either KinectObservation or MediaPipeObservation (may be None)
            imu: IMU sample (may be None)
            now: Current timestamp in seconds

        Returns:
            SensorDecision with source selection and confidence
        """
        # Check IMU
        imu_valid, imu_conf, imu_reason = self._check_imu(imu, now)

        # Check vision (prefer kinect if both passed, but vision arg takes priority here)
        # The caller should pass the preferred vision source in 'vision' parameter
        vis_valid, vis_conf, vis_reason, vis_source = self._check_vision(vision)

        # If vision failed but kinect is available and different from vision, try kinect
        if not vis_valid and kinect is not None and kinect is not vision:
            vis_valid, vis_conf, vis_reason, vis_source = self._check_vision(kinect)

        # Decision logic
        if imu_valid and vis_valid:
            # Both available -> fused
            combined_conf = np.sqrt(imu_conf * vis_conf)  # geometric mean
            return SensorDecision(
                source="fused",
                confidence=float(combined_conf),
                reason=f"both valid: {imu_reason}; {vis_reason}",
                use_vision=True,
                use_imu=True,
            )

        if imu_valid:
            return SensorDecision(
                source="imu_only",
                confidence=imu_conf,
                reason=imu_reason,
                use_vision=False,
                use_imu=True,
            )

        if vis_valid:
            return SensorDecision(
                source=f"vision_only:{vis_source}",
                confidence=vis_conf,
                reason=vis_reason,
                use_vision=True,
                use_imu=False,
            )

        # Neither valid
        reasons = []
        if imu is not None:
            reasons.append(imu_reason)
        if vision is not None:
            reasons.append(vis_reason)
        elif kinect is not None:
            _, _, k_reason, _ = self._check_vision(kinect)
            reasons.append(k_reason)
        else:
            reasons.append("no sensors available")

        return SensorDecision(
            source="none",
            confidence=0.0,
            reason="; ".join(reasons),
            use_vision=False,
            use_imu=False,
        )