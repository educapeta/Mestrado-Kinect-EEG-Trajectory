"""
Tests for fusion.sync module.

Run: python test_fusion_sync.py
Exit code 0 = all tests passed, 1 = any failure.
"""

import sys
import numpy as np

from fusion.contracts import KinectObservation, MediaPipeObservation, ImuSample
from fusion.sync import (
    SampleBuffer,
    SyncBuffer,
    SyncSnapshot,
    sample_age_ms,
    is_within_max_age,
)


def test_sample_buffer_basic():
    """Test basic SampleBuffer operations."""
    buf = SampleBuffer[ImuSample](max_age_s=0.1)

    # Empty buffer
    assert buf.latest() is None
    assert buf.nearest(0.0) is None
    assert len(buf) == 0

    # Add samples
    s1 = ImuSample(timestamp=1.0, timestamp_us=1000000, hand="left", roll_deg=0, pitch_deg=0, yaw_deg=0)
    s2 = ImuSample(timestamp=1.05, timestamp_us=1050000, hand="left", roll_deg=10, pitch_deg=0, yaw_deg=0)
    s3 = ImuSample(timestamp=1.1, timestamp_us=1100000, hand="left", roll_deg=20, pitch_deg=0, yaw_deg=0)

    buf.add(s1)
    buf.add(s2)
    buf.add(s3)

    assert len(buf) == 3
    assert buf.latest() is s3

    # Nearest
    nearest = buf.nearest(1.04)
    assert nearest is s2

    nearest = buf.nearest(1.08)
    assert nearest is s3

    # max_age filter
    nearest = buf.nearest(1.04, max_age_s=0.01)
    assert nearest is s2  # s2 is at 1.05, diff=0.01

    nearest = buf.nearest(1.04, max_age_s=0.005)
    assert nearest is None  # s2 diff=0.01 > 0.005

    print("test_sample_buffer_basic: OK")


def test_sample_buffer_pruning():
    """Test automatic pruning of old samples."""
    buf = SampleBuffer[ImuSample](max_age_s=0.1)

    s1 = ImuSample(timestamp=1.0, timestamp_us=1000000, hand="left")
    s2 = ImuSample(timestamp=1.05, timestamp_us=1050000, hand="left")
    s3 = ImuSample(timestamp=1.15, timestamp_us=1150000, hand="left")

    buf.add(s1)
    buf.add(s2)
    # At s3 timestamp (1.15), cutoff is 1.05, so s1 (1.0) should be pruned
    buf.add(s3)

    assert len(buf) == 2
    assert buf.latest() is s3
    # s1 should be gone
    assert all(s.timestamp >= 1.05 for s in buf._samples)

    print("test_sample_buffer_pruning: OK")


def test_sample_buffer_clear():
    """Test clearing the buffer."""
    buf = SampleBuffer[ImuSample](max_age_s=0.1)
    buf.add(ImuSample(timestamp=1.0, timestamp_us=1000000, hand="left"))
    assert len(buf) == 1
    buf.clear()
    assert len(buf) == 0
    assert buf.latest() is None
    print("test_sample_buffer_clear: OK")


def test_sync_buffer_basic():
    """Test SyncBuffer with all three sources."""
    sync = SyncBuffer(max_age_s=0.25)

    # Add samples at different times
    k = KinectObservation(timestamp=1.0, hand="left", confidence=0.9, valid=True)
    m = MediaPipeObservation(timestamp=1.02, hand="left", confidence=0.8, valid=True)
    i = ImuSample(timestamp=1.01, timestamp_us=1010000, hand="left", valid=True)

    sync.add_kinect(k)
    sync.add_mediapipe(m)
    sync.add_imu(i)

    snap = sync.snapshot(1.01)

    assert isinstance(snap, SyncSnapshot)
    assert snap.timestamp == 1.01
    assert snap.kinect is k
    assert snap.mediapipe is m
    assert snap.imu is i
    assert len(snap.statuses) == 3

    # Check statuses
    for status in snap.statuses:
        assert status.valid is True
        assert status.confidence > 0
        assert status.age_ms >= 0

    print("test_sync_buffer_basic: OK")


def test_sync_buffer_partial():
    """Test SyncBuffer with only some sources."""
    sync = SyncBuffer(max_age_s=0.25)

    k = KinectObservation(timestamp=1.0, hand="left", confidence=0.9, valid=True)
    i = ImuSample(timestamp=1.01, timestamp_us=1010000, hand="left", valid=True)

    sync.add_kinect(k)
    sync.add_imu(i)

    snap = sync.snapshot(1.01)

    assert snap.kinect is k
    assert snap.imu is i
    assert snap.mediapipe is None

    # Status for missing mediapipe
    mp_status = next(s for s in snap.statuses if s.source == "mediapipe")
    assert mp_status.valid is False
    assert mp_status.age_ms == float("inf")
    assert mp_status.confidence == 0.0

    print("test_sync_buffer_partial: OK")


def test_sync_buffer_max_age():
    """Test SyncBuffer respects max_age_s."""
    sync = SyncBuffer(max_age_s=0.05)

    k = KinectObservation(timestamp=1.0, hand="left", confidence=0.9, valid=True)
    i = ImuSample(timestamp=1.01, timestamp_us=1010000, hand="left", valid=True)

    sync.add_kinect(k)
    sync.add_imu(i)

    # Request snapshot at 1.2 (well beyond max_age_s=0.05)
    snap = sync.snapshot(1.2)

    # Both should be None or invalid due to age
    k_status = next(s for s in snap.statuses if s.source == "kinect")
    i_status = next(s for s in snap.statuses if s.source == "imu")

    # kinect at 1.0, age = 0.2s = 200ms > 50ms max_age
    assert k_status.valid is False
    assert snap.kinect is None

    # imu at 1.01, age = 0.19s = 190ms > 50ms max_age
    assert i_status.valid is False
    assert snap.imu is None

    print("test_sync_buffer_max_age: OK")


def test_sync_buffer_auto_timestamp():
    """Test SyncBuffer auto-selects latest timestamp when None given."""
    sync = SyncBuffer(max_age_s=0.25)

    k = KinectObservation(timestamp=1.0, hand="left", confidence=0.9, valid=True)
    i = ImuSample(timestamp=1.05, timestamp_us=1050000, hand="left", valid=True)

    sync.add_kinect(k)
    sync.add_imu(i)

    # No timestamp provided -> should use latest (1.05 from IMU)
    snap = sync.snapshot()
    assert snap.timestamp == 1.05

    # Nearest at 1.05: kinect age=0.05, imu age=0
    assert snap.kinect is k
    assert snap.imu is i

    print("test_sync_buffer_auto_timestamp: OK")


def test_sync_buffer_clear():
    """Test clearing all buffers."""
    sync = SyncBuffer(max_age_s=0.25)
    sync.add_kinect(KinectObservation(timestamp=1.0, hand="left"))
    sync.add_imu(ImuSample(timestamp=1.0, timestamp_us=1000000, hand="left"))
    sync.clear()

    snap = sync.snapshot(1.0)
    assert snap.kinect is None
    assert snap.imu is None
    assert snap.mediapipe is None

    print("test_sync_buffer_clear: OK")


def test_sample_age_ms():
    """Test sample_age_ms helper."""
    s = ImuSample(timestamp=1.0, timestamp_us=1000000, hand="left")

    age = sample_age_ms(s, 1.0)
    assert age == 0.0

    age = sample_age_ms(s, 1.001)
    assert abs(age - 1.0) < 0.001  # 1ms

    age = sample_age_ms(s, 1.01)
    assert abs(age - 10.0) < 0.001  # 10ms

    # None sample
    assert sample_age_ms(None, 1.0) == float("inf")

    # Object without timestamp
    class NoTS:
        pass
    assert sample_age_ms(NoTS(), 1.0) == float("inf")

    print("test_sample_age_ms: OK")


def test_is_within_max_age():
    """Test is_within_max_age helper."""
    s = ImuSample(timestamp=1.0, timestamp_us=1000000, hand="left")

    assert is_within_max_age(s, 1.0, 0.1) is True
    assert is_within_max_age(s, 1.05, 0.1) is True
    assert is_within_max_age(s, 1.1, 0.1) is True  # boundary
    assert is_within_max_age(s, 1.11, 0.1) is False
    assert is_within_max_age(s, 1.2, 0.1) is False

    assert is_within_max_age(None, 1.0, 0.1) is False

    print("test_is_within_max_age: OK")


def run_all():
    """Run all tests."""
    tests = [
        test_sample_buffer_basic,
        test_sample_buffer_pruning,
        test_sample_buffer_clear,
        test_sync_buffer_basic,
        test_sync_buffer_partial,
        test_sync_buffer_max_age,
        test_sync_buffer_auto_timestamp,
        test_sync_buffer_clear,
        test_sample_age_ms,
        test_is_within_max_age,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"{test.__name__}: FAILED - {e}")
            failed += 1

    if failed == 0:
        print("\nAll tests PASSED")
        return 0
    else:
        print(f"\n{failed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all())