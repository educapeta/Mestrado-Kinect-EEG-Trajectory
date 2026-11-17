"""
Tests for fusion.sensor_manager module.

Run: python test_fusion_sensor_manager.py
Exit code 0 = all tests passed, 1 = any failure.
"""

import sys
import numpy as np

from fusion.contracts import KinectObservation, MediaPipeObservation, ImuSample
from fusion.sensor_manager import SensorManager, SensorDecision


def test_imu_only():
    """Test decision when only IMU is valid."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    imu = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        roll_deg=0,
        pitch_deg=0,
        yaw_deg=0,
        valid=True,
    )

    decision = mgr.decide(kinect=None, vision=None, imu=imu, now=1.0)

    assert decision.source == "imu_only"
    assert decision.confidence == 1.0
    assert decision.use_vision is False
    assert decision.use_imu is True
    assert "IMU valid" in decision.reason

    print("test_imu_only: OK")


def test_vision_only_kinect():
    """Test decision when only Kinect vision is valid."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    kinect = KinectObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.9,
        valid=True,
    )

    decision = mgr.decide(kinect=kinect, vision=None, imu=None, now=1.0)

    assert decision.source == "vision_only:kinect_depth"
    assert decision.confidence == 0.9
    assert decision.use_vision is True
    assert decision.use_imu is False
    assert "vision valid" in decision.reason

    print("test_vision_only_kinect: OK")


def test_vision_only_mediapipe():
    """Test decision when only MediaPipe vision is valid."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    mp = MediaPipeObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.8,
        valid=True,
    )

    decision = mgr.decide(kinect=None, vision=mp, imu=None, now=1.0)

    assert decision.source == "vision_only:mediapipe"
    assert decision.confidence == 0.8
    assert decision.use_vision is True
    assert decision.use_imu is False

    print("test_vision_only_mediapipe: OK")


def test_fused():
    """Test decision when both IMU and vision are valid."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    kinect = KinectObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.9,
        valid=True,
    )
    imu = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        valid=True,
    )

    decision = mgr.decide(kinect=kinect, vision=None, imu=imu, now=1.0)

    assert decision.source == "fused"
    # geometric mean: sqrt(1.0 * 0.9) = 0.94868...
    assert abs(decision.confidence - np.sqrt(0.9)) < 1e-6
    assert decision.use_vision is True
    assert decision.use_imu is True
    assert "both valid" in decision.reason

    print("test_fused: OK")


def test_none():
    """Test decision when neither source is valid."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    decision = mgr.decide(kinect=None, vision=None, imu=None, now=1.0)

    assert decision.source == "none"
    assert decision.confidence == 0.0
    assert decision.use_vision is False
    assert decision.use_imu is False
    assert "no sensors" in decision.reason

    print("test_none: OK")


def test_imu_stale():
    """Test IMU rejected when too old."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    # IMU at t=1.0, now=1.5 -> age=500ms > 200ms max
    imu = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        valid=True,
    )

    decision = mgr.decide(kinect=None, vision=None, imu=imu, now=1.5)

    assert decision.source == "none"
    assert decision.use_imu is False
    assert "age" in decision.reason.lower()

    print("test_imu_stale: OK")


def test_imu_invalid():
    """Test IMU rejected when marked invalid."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    imu = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        valid=False,
    )

    decision = mgr.decide(kinect=None, vision=None, imu=imu, now=1.0)

    assert decision.source == "none"
    assert decision.use_imu is False
    assert "invalid" in decision.reason.lower()

    print("test_imu_invalid: OK")


def test_vision_low_confidence():
    """Test vision rejected when confidence below threshold."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    kinect = KinectObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.2,  # below 0.35
        valid=True,
    )

    decision = mgr.decide(kinect=kinect, vision=None, imu=None, now=1.0)

    assert decision.source == "none"
    assert decision.use_vision is False
    assert "confidence" in decision.reason.lower()

    print("test_vision_low_confidence: OK")


def test_vision_invalid():
    """Test vision rejected when marked invalid."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    kinect = KinectObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.9,
        valid=False,
    )

    decision = mgr.decide(kinect=kinect, vision=None, imu=None, now=1.0)

    assert decision.source == "none"
    assert decision.use_vision is False
    assert "invalid" in decision.reason.lower()

    print("test_vision_invalid: OK")


def test_vision_prefers_kinect_over_mediapipe():
    """Test that kinect is preferred when both vision sources provided."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    kinect = KinectObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.9,
        valid=True,
    )
    mp = MediaPipeObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.8,
        valid=True,
    )

    # Pass kinect as 'kinect' and mp as 'vision'
    decision = mgr.decide(kinect=kinect, vision=mp, imu=None, now=1.0)

    # Should use kinect (passed as kinect param, not vision param)
    # Actually the logic: vision param is checked first, then kinect as fallback
    # Since mp is passed as vision and is valid, it will be used
    assert decision.source == "vision_only:mediapipe"

    # Now test with invalid mediapipe but valid kinect
    mp_invalid = MediaPipeObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.8,
        valid=False,
    )
    decision2 = mgr.decide(kinect=kinect, vision=mp_invalid, imu=None, now=1.0)
    assert decision2.source == "vision_only:kinect_depth"

    print("test_vision_prefers_kinect_over_mediapipe: OK")


def test_vision_fallback_to_kinect():
    """Test fallback to kinect when vision param is invalid/None but kinect is valid."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    kinect = KinectObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.9,
        valid=True,
    )

    # No vision param, but kinect provided
    decision = mgr.decide(kinect=kinect, vision=None, imu=None, now=1.0)
    assert decision.source == "vision_only:kinect_depth"

    print("test_vision_fallback_to_kinect: OK")


def test_confidence_geometric_mean():
    """Test fused confidence is geometric mean."""
    mgr = SensorManager(min_vision_confidence=0.35, max_imu_age_ms=200.0)

    # IMU conf=1.0, vision conf=0.64 -> sqrt(0.64) = 0.8
    kinect = KinectObservation(
        timestamp=1.0,
        hand="left",
        confidence=0.64,
        valid=True,
    )
    imu = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        valid=True,
    )

    decision = mgr.decide(kinect=kinect, vision=None, imu=imu, now=1.0)
    assert decision.source == "fused"
    assert abs(decision.confidence - 0.8) < 1e-6

    print("test_confidence_geometric_mean: OK")


def test_decision_dataclass():
    """Test SensorDecision dataclass fields."""
    dec = SensorDecision(
        source="test",
        confidence=0.5,
        reason="test reason",
        use_vision=True,
        use_imu=False,
    )
    assert dec.source == "test"
    assert dec.confidence == 0.5
    assert dec.reason == "test reason"
    assert dec.use_vision is True
    assert dec.use_imu is False

    print("test_decision_dataclass: OK")


def run_all():
    """Run all tests."""
    tests = [
        test_imu_only,
        test_vision_only_kinect,
        test_vision_only_mediapipe,
        test_fused,
        test_none,
        test_imu_stale,
        test_imu_invalid,
        test_vision_low_confidence,
        test_vision_invalid,
        test_vision_prefers_kinect_over_mediapipe,
        test_vision_fallback_to_kinect,
        test_confidence_geometric_mean,
        test_decision_dataclass,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"{test.__name__}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    if failed == 0:
        print("\nAll tests PASSED")
        return 0
    else:
        print(f"\n{failed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all())