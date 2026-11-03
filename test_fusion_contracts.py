#!/usr/bin/env python3
"""
Tests for fusion.contracts module.

Run: python test_fusion_contracts.py
Exit code: 0 on success, 1 on failure.
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, r"C:\mestradopy")

from fusion.contracts import (
    KinectObservation,
    MediaPipeObservation,
    ImuSample,
    Calibration,
    FusedState,
    SourceStatus,
)
from fusion.quaternion import from_euler_zyx_deg, to_matrix


def assert_allclose(a, b, atol=1e-10, msg=""):
    """Assert arrays are close with custom message."""
    if not np.allclose(a, b, atol=atol):
        raise AssertionError(f"{msg}: {a} != {b} (atol={atol})")


def assert_raises(exc_type, func, *args, **kwargs):
    """Assert function raises exception."""
    try:
        func(*args, **kwargs)
        raise AssertionError(f"Expected {exc_type.__name__} but none raised")
    except exc_type:
        pass


def test_kinect_observation():
    """Test KinectObservation dataclass."""
    print("Testing KinectObservation...")
    obs = KinectObservation(
        timestamp=1.0,
        hand="left",
        wrist_cam=np.array([0.1, 0.2, 0.3]),
        palm_normal_cam=np.array([0.0, 0.0, 1.0]),
        palm_dir_cam=np.array([1.0, 0.0, 0.0]),
        confidence=0.9,
        valid=True,
    )
    assert obs.timestamp == 1.0
    assert obs.hand == "left"
    assert_allclose(obs.wrist_cam, [0.1, 0.2, 0.3])
    assert_allclose(obs.palm_normal_cam, [0.0, 0.0, 1.0])
    assert_allclose(obs.palm_dir_cam, [1.0, 0.0, 0.0])
    assert obs.confidence == 0.9
    assert obs.valid is True
    assert obs.source == "kinect_depth"

    # Test defaults
    obs2 = KinectObservation(timestamp=2.0, hand="right")
    assert_allclose(obs2.wrist_cam, [0.0, 0.0, 0.0])
    assert_allclose(obs2.palm_normal_cam, [0.0, 0.0, 0.0])
    assert_allclose(obs2.palm_dir_cam, [0.0, 0.0, 0.0])
    assert obs2.confidence == 0.0
    assert obs2.valid is False
    print("  OK")


def test_mediapipe_observation():
    """Test MediaPipeObservation dataclass."""
    print("Testing MediaPipeObservation...")
    obs = MediaPipeObservation(
        timestamp=1.0,
        hand="left",
        landmarks_2d=np.zeros((21, 2)),
        landmarks_3d_cam=np.zeros((21, 3)),
        handedness=0.8,
        confidence=0.9,
        valid=True,
    )
    assert obs.timestamp == 1.0
    assert obs.hand == "left"
    assert obs.landmarks_2d.shape == (21, 2)
    assert obs.landmarks_3d_cam.shape == (21, 3)
    assert obs.handedness == 0.8
    assert obs.confidence == 0.9
    assert obs.valid is True
    assert obs.source == "mediapipe"

    # Test defaults
    obs2 = MediaPipeObservation(timestamp=2.0, hand="right")
    assert obs2.landmarks_2d.shape == (21, 2)
    assert obs2.landmarks_3d_cam.shape == (21, 3)
    print("  OK")


def test_imu_sample():
    """Test ImuSample dataclass."""
    print("Testing ImuSample...")
    sample = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        roll_deg=10.0,
        pitch_deg=20.0,
        yaw_deg=30.0,
        accel_g=np.array([0.0, 0.0, 1.0]),
        gyro_dps=np.array([1.0, 2.0, 3.0]),
        valid=True,
    )
    assert sample.timestamp == 1.0
    assert sample.timestamp_us == 1000000
    assert sample.hand == "left"
    assert sample.roll_deg == 10.0
    assert sample.pitch_deg == 20.0
    assert sample.yaw_deg == 30.0
    assert_allclose(sample.accel_g, [0.0, 0.0, 1.0])
    assert_allclose(sample.gyro_dps, [1.0, 2.0, 3.0])
    assert sample.valid is True
    assert sample.source == "imu_udp"

    # Test defaults
    sample2 = ImuSample(timestamp=2.0, timestamp_us=2000000, hand="right")
    assert sample2.roll_deg == 0.0
    assert_allclose(sample2.accel_g, [0.0, 0.0, 0.0])
    assert_allclose(sample2.gyro_dps, [0.0, 0.0, 0.0])
    print("  OK")


def test_calibration_validate():
    """Test Calibration validation."""
    print("Testing Calibration.validate...")

    # Valid calibration: T=eye, f=z
    cal = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    assert cal.validate() is True, "Valid calibration should pass"

    # Invalid: T not orthogonal
    cal2 = Calibration(
        hand="left",
        T_cam_world=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    assert cal2.validate() is False, "Non-orthogonal T should fail"

    # Invalid: T det != 1
    cal3 = Calibration(
        hand="left",
        T_cam_world=np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    assert cal3.validate() is False, "det=-1 should fail"

    # Invalid: f_body not unit
    cal4 = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([1.0, 1.0, 1.0]),
    )
    assert cal4.validate() is False, "Non-unit f_body should fail"

    # Invalid: NaN in T
    cal5 = Calibration(
        hand="left",
        T_cam_world=np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]]),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    assert cal5.validate() is False, "NaN in T should fail"

    # Invalid: wrong shape T
    cal6 = Calibration(
        hand="left",
        T_cam_world=np.eye(4),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    assert cal6.validate() is False, "Wrong T shape should fail"

    # Invalid: wrong shape f_body
    cal7 = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0]),
    )
    assert cal7.validate() is False, "Wrong f_body shape should fail"

    print("  OK")


def test_calibration_normal_cam_from_q():
    """Test Calibration.normal_cam_from_q."""
    print("Testing Calibration.normal_cam_from_q...")
    from fusion.quaternion import normalize

    cal = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
    )

    # Identity quaternion -> f_body unchanged
    q = np.array([1.0, 0.0, 0.0, 0.0])
    normal = cal.normal_cam_from_q(q)
    assert_allclose(normal, [0.0, 0.0, 1.0], msg="normal from identity")

    # 90 deg around X -> Z becomes -Y
    q = from_euler_zyx_deg(90, 0, 0)
    normal = cal.normal_cam_from_q(q)
    assert_allclose(normal, [0.0, -1.0, 0.0], atol=1e-10, msg="normal from 90 deg X")
    print("  OK")


def test_calibration_normal_cam_from_imu():
    """Test Calibration.normal_cam_from_imu."""
    print("Testing Calibration.normal_cam_from_imu...")

    # T_cam_world = eye, f_body = z
    cal = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
    )

    sample = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        roll_deg=90.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
    )
    normal = cal.normal_cam_from_imu(sample)
    # 90 deg roll around X: Z -> -Y
    assert_allclose(normal, [0.0, -1.0, 0.0], atol=1e-10, msg="normal from IMU roll=90")

    # With non-identity T_cam_world: 90 deg around Z
    cal2 = Calibration(
        hand="left",
        T_cam_world=to_matrix(from_euler_zyx_deg(0, 0, 90)),  # Rz(90)
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    sample2 = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
    )
    normal2 = cal2.normal_cam_from_imu(sample2)
    # Rz(90) @ Z = Z (Z axis unchanged by Z rotation)
    assert_allclose(normal2, [0.0, 0.0, 1.0], atol=1e-10, msg="normal from T_cam_world Rz(90)")

    print("  OK")


def test_calibration_to_from_dict():
    """Test Calibration serialization roundtrip."""
    print("Testing Calibration.to_dict/from_dict...")
    cal = Calibration(
        hand="left",
        T_cam_world=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        f_body=np.array([0.0, 0.0, 1.0]),
        yaw_boot_delta_deg=5.5,
        residual_deg=2.3,
        source="manual_calibration",
    )

    d = cal.to_dict()
    assert d["hand"] == "left"
    assert "rotation" in d
    assert "body" in d
    assert d["yaw_boot_delta_deg"] == 5.5
    assert d["residual_deg"] == 2.3
    assert d["source"] == "manual_calibration"

    # Roundtrip
    cal2 = Calibration.from_dict(d, "left")
    assert cal2.hand == "left"
    assert_allclose(cal2.T_cam_world, cal.T_cam_world)
    assert_allclose(cal2.f_body, cal.f_body)
    assert cal2.yaw_boot_delta_deg == 5.5
    assert cal2.residual_deg == 2.3
    assert cal2.source == "manual_calibration"

    # Test with missing optional fields
    d_minimal = {"rotation": np.eye(3).tolist(), "body": [0.0, 0.0, 1.0]}
    cal3 = Calibration.from_dict(d_minimal, "right")
    assert cal3.hand == "right"
    assert cal3.yaw_boot_delta_deg == 0.0
    assert cal3.residual_deg is None
    assert cal3.source == ""
    print("  OK")


def test_fused_state():
    """Test FusedState dataclass."""
    print("Testing FusedState...")
    from fusion.quaternion import normalize

    q = normalize(np.array([0.7, 0.2, 0.3, 0.6], dtype=np.float64))
    state = FusedState(
        timestamp=1.0,
        hand="left",
        q_body_cam=q,
        normal_cam=np.array([0.0, 0.0, 1.0]),
        dir_cam=np.array([1.0, 0.0, 0.0]),
        confidence=0.95,
        source="kinect+imu",
        latency_ms=12.5,
        valid=True,
    )
    assert state.timestamp == 1.0
    assert state.hand == "left"
    assert_allclose(state.q_body_cam, q)
    assert_allclose(state.normal_cam, [0.0, 0.0, 1.0])
    assert_allclose(state.dir_cam, [1.0, 0.0, 0.0])
    assert state.confidence == 0.95
    assert state.source == "kinect+imu"
    assert state.latency_ms == 12.5
    assert state.valid is True
    assert state.schema_ver == 1

    # Test to_dict
    d = state.to_dict()
    assert d["timestamp"] == 1.0
    assert d["hand"] == "left"
    assert "q_body_cam" in d
    assert d["confidence"] == 0.95
    assert d["source"] == "kinect+imu"
    assert d["latency_ms"] == 12.5
    assert d["valid"] is True
    assert d["schema_ver"] == 1
    print("  OK")


def test_source_status():
    """Test SourceStatus dataclass."""
    print("Testing SourceStatus...")
    status = SourceStatus(
        source="kinect_depth",
        age_ms=15.0,
        confidence=0.8,
        valid=True,
        reason="",
    )
    assert status.source == "kinect_depth"
    assert status.age_ms == 15.0
    assert status.confidence == 0.8
    assert status.valid is True
    assert status.reason == ""

    status2 = SourceStatus(
        source="imu_udp",
        age_ms=500.0,
        confidence=0.0,
        valid=False,
        reason="timeout",
    )
    assert status2.valid is False
    assert status2.reason == "timeout"
    print("  OK")


def test_invalid_shapes():
    """Test that invalid shapes are caught."""
    print("Testing invalid shapes...")

    # KinectObservation with wrong shape arrays
    # Note: dataclass doesn't validate on init, but we can test manually
    # This is just to verify the default factories produce correct shapes
    obs = KinectObservation(timestamp=1.0, hand="left")
    assert obs.wrist_cam.shape == (3,)
    assert obs.palm_normal_cam.shape == (3,)
    assert obs.palm_dir_cam.shape == (3,)

    mp = MediaPipeObservation(timestamp=1.0, hand="left")
    assert mp.landmarks_2d.shape == (21, 2)
    assert mp.landmarks_3d_cam.shape == (21, 3)

    imu = ImuSample(timestamp=1.0, timestamp_us=0, hand="left")
    assert imu.accel_g.shape == (3,)
    assert imu.gyro_dps.shape == (3,)

    cal = Calibration(hand="left")
    assert cal.T_cam_world.shape == (3, 3)
    assert cal.f_body.shape == (3,)

    state = FusedState(timestamp=1.0, hand="left")
    assert state.q_body_cam.shape == (4,)
    assert state.normal_cam.shape == (3,)
    assert state.dir_cam.shape == (3,)

    print("  OK")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running fusion.contracts tests")
    print("=" * 50)

    test_kinect_observation()
    test_mediapipe_observation()
    test_imu_sample()
    test_calibration_validate()
    test_calibration_normal_cam_from_q()
    test_calibration_normal_cam_from_imu()
    test_calibration_to_from_dict()
    test_fused_state()
    test_source_status()
    test_invalid_shapes()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)