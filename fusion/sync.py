"""
Sample buffering and synchronization for multi-sensor fusion.

Provides:
- SampleBuffer: time-bounded buffer for any sample type with .timestamp attribute
- SyncSnapshot: container for synchronized multi-source observations
- SyncBuffer: coordinates Kinect, MediaPipe, and IMU buffers with nearest-time matching
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, TypeVar, Generic
import numpy as np

from .contracts import (
    KinectObservation,
    MediaPipeObservation,
    ImuSample,
    SourceStatus,
)

T = TypeVar("T")


class SampleBuffer(Generic[T]):
    """
    Time-bounded buffer for samples with a .timestamp attribute.

    Automatically removes samples older than max_age_s on each addition.
    Provides nearest-time lookup within a maximum age window.
    """

    def __init__(self, max_age_s: float = 0.25):
        """
        Initialize buffer.

        Args:
            max_age_s: Maximum age of samples to retain (seconds)
        """
        self.max_age_s = max_age_s
        self._samples: list[T] = []

    def add(self, sample: T) -> None:
        """
        Add a sample to the buffer and prune old entries.

        Args:
            sample: Object with a .timestamp attribute (float, seconds)
        """
        if not hasattr(sample, "timestamp"):
            raise ValueError("Sample must have a 'timestamp' attribute")
        self._samples.append(sample)
        self._prune(sample.timestamp)

    def _prune(self, current_time: float) -> None:
        """Remove samples older than max_age_s relative to current_time."""
        cutoff = current_time - self.max_age_s
        self._samples = [s for s in self._samples if s.timestamp >= cutoff]

    def latest(self) -> Optional[T]:
        """
        Return the most recent sample, or None if buffer is empty.

        Returns:
            Sample with highest timestamp, or None
        """
        if not self._samples:
            return None
        return max(self._samples, key=lambda s: s.timestamp)

    def clear(self) -> None:
        """Remove all samples from the buffer."""
        self._samples.clear()

    def nearest(self, timestamp: float, max_age_s: Optional[float] = None) -> Optional[T]:
        """
        Find sample with timestamp closest to the given time.

        Args:
            timestamp: Target timestamp (seconds)
            max_age_s: Maximum allowed age difference. Defaults to buffer's max_age_s.

        Returns:
            Nearest sample within max_age_s, or None if none found
        """
        if not self._samples:
            return None

        if max_age_s is None:
            max_age_s = self.max_age_s

        best: Optional[T] = None
        best_diff = float("inf")

        # Small epsilon for floating point comparison
        eps = 1e-12

        for sample in self._samples:
            diff = abs(sample.timestamp - timestamp)
            if diff <= max_age_s + eps and diff < best_diff:
                best_diff = diff
                best = sample

        return best

    def __len__(self) -> int:
        return len(self._samples)

    def __bool__(self) -> bool:
        return bool(self._samples)


@dataclass
class SyncSnapshot:
    """
    Synchronized multi-source observation at a specific timestamp.

    Attributes:
        timestamp: Target timestamp for synchronization (seconds)
        kinect: Nearest Kinect observation, or None
        mediapipe: Nearest MediaPipe observation, or None
        imu: Nearest IMU sample, or None
        statuses: Per-source status information including age and validity
    """
    timestamp: float
    kinect: Optional[KinectObservation] = None
    mediapipe: Optional[MediaPipeObservation] = None
    imu: Optional[ImuSample] = None
    statuses: list[SourceStatus] = field(default_factory=list)


class SyncBuffer:
    """
    Coordinated buffer for Kinect, MediaPipe, and IMU streams.

    Maintains independent SampleBuffers for each source and produces
    synchronized snapshots by nearest-time matching.
    """

    def __init__(self, max_age_s: float = 0.25):
        """
        Initialize sync buffer.

        Args:
            max_age_s: Maximum age for all source buffers (seconds)
        """
        self.max_age_s = max_age_s
        self._kinect_buf = SampleBuffer[KinectObservation](max_age_s)
        self._mediapipe_buf = SampleBuffer[MediaPipeObservation](max_age_s)
        self._imu_buf = SampleBuffer[ImuSample](max_age_s)

    def add_kinect(self, obs: KinectObservation) -> None:
        """Add a Kinect observation."""
        self._kinect_buf.add(obs)

    def add_mediapipe(self, obs: MediaPipeObservation) -> None:
        """Add a MediaPipe observation."""
        self._mediapipe_buf.add(obs)

    def add_imu(self, sample: ImuSample) -> None:
        """Add an IMU sample."""
        self._imu_buf.add(sample)

    def snapshot(self, timestamp: Optional[float] = None) -> SyncSnapshot:
        """
        Create a synchronized snapshot at the given timestamp.

        If timestamp is None, uses the latest available timestamp across all sources.

        Args:
            timestamp: Target timestamp (seconds). Defaults to latest.

        Returns:
            SyncSnapshot with nearest samples and status info
        """
        if timestamp is None:
            # Use the latest timestamp from any non-empty buffer
            latest_times = []
            for buf in (self._kinect_buf, self._mediapipe_buf, self._imu_buf):
                latest = buf.latest()
                if latest is not None:
                    latest_times.append(latest.timestamp)
            if not latest_times:
                timestamp = 0.0
            else:
                timestamp = max(latest_times)

        # Get nearest samples
        kinect = self._kinect_buf.nearest(timestamp, self.max_age_s)
        mediapipe = self._mediapipe_buf.nearest(timestamp, self.max_age_s)
        imu = self._imu_buf.nearest(timestamp, self.max_age_s)

        # Build status list
        statuses = []
        for src_name, sample in [
            ("kinect", kinect),
            ("mediapipe", mediapipe),
            ("imu", imu),
        ]:
            if sample is not None:
                age_ms = (timestamp - sample.timestamp) * 1000.0
                valid = getattr(sample, "valid", True)
                confidence = getattr(sample, "confidence", 1.0 if valid else 0.0)
                statuses.append(SourceStatus(
                    source=src_name,
                    age_ms=abs(age_ms),
                    confidence=confidence,
                    valid=valid,
                    reason="" if valid else "invalid sample",
                ))
            else:
                statuses.append(SourceStatus(
                    source=src_name,
                    age_ms=float("inf"),
                    confidence=0.0,
                    valid=False,
                    reason="no sample within max_age_s",
                ))

        return SyncSnapshot(
            timestamp=timestamp,
            kinect=kinect,
            mediapipe=mediapipe,
            imu=imu,
            statuses=statuses,
        )

    def clear(self) -> None:
        """Clear all source buffers."""
        self._kinect_buf.clear()
        self._mediapipe_buf.clear()
        self._imu_buf.clear()


def sample_age_ms(sample: Any, now: float) -> float:
    """
    Compute age of a sample in milliseconds.

    Args:
        sample: Object with .timestamp attribute
        now: Current timestamp (seconds)

    Returns:
        Age in milliseconds (positive if sample is in the past)
    """
    if sample is None or not hasattr(sample, "timestamp"):
        return float("inf")
    return (now - sample.timestamp) * 1000.0


def is_within_max_age(sample: Any, now: float, max_age_s: float) -> bool:
    """
    Check if sample is within max_age_s of now.

    Args:
        sample: Object with .timestamp attribute
        now: Current timestamp (seconds)
        max_age_s: Maximum allowed age (seconds)

    Returns:
        True if sample age <= max_age_s (with small epsilon for floating point)
    """
    if sample is None or not hasattr(sample, "timestamp"):
        return False
    return (now - sample.timestamp) <= max_age_s + 1e-12