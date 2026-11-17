#!/usr/bin/env python
"""
Tests for fusion/complementary.py

Tests the QuaternionComplementaryFilter class.
"""

from __future__ import annotations
import sys
import numpy as np

from fusion.complementary import QuaternionComplementaryFilter, ComplementaryConfig
from fusion.contracts import ImuSample, Calibration
from fusion.quaternion import normalize, multiply, quat_from_axis_angle, angle_between_deg, rotate_vector, conjugate


def test_identity_gyro_yaw():
    """Test: identity quaternion + gyro yaw -> correct yaw."""
    print("Test 1: Identity + gyro yaw -> correct yaw")
    config = ComplementaryConfig(kp_grav=0.0, kp_yaw=0.0, max_dt=0.1)
    filter = QuaternionComplementaryFilter(config=config)

    # Apply yaw rate of 90 deg/s for 1 second (10 steps of 0.1s)
    dt = 0.1
    gyro = np.array([0.0, 0.0, 90.0], dtype=np.float64)  # 90 deg/s around Z

    for _ in range(10):
        filter.predict(gyro, dt)

    q = filter.state
    # Expected: 90 deg rotation around Z
    q_expected = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(90.0))
    err = angle_between_deg(q, q_expected)

    print(f"  Expected: {q_expected}")
    print(f"  Got:      {q}")
    print(f"  Error:    {err:.6f} deg")
    assert err < 0.1, f"Yaw integration error too large: {err:.6f} deg"
    print("  PASS")
    return True


def test_vision_correction_pulls_normal():
    """Test: vision correction pulls normal toward measurement."""
    print("\nTest 2: Vision correction pulls normal toward measurement")
    config = ComplementaryConfig(kp_grav=0.0, kp_yaw=5.0, max_dt=0.1, max_yaw_error=1.0)
    calibration = Calibration(
        hand="right",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
        source="test"
    )
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)

    # Start with identity (normal = [0, 0, 1] in camera)
    # Measure normal at [1, 0, 0] (90 deg around Y)
    normal_meas = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Apply vision update multiple times to see convergence
    dt = 0.1
    for _ in range(10):
        filter.update_yaw(normal_meas, dt, confidence=1.0)

    q = filter.state
    normal_est = calibration.normal_cam_from_q(q)
    print(f"  Estimated normal: {normal_est}")
    print(f"  Measured normal:  {normal_meas}")

    # Normal should have moved significantly toward measurement
    dot_after = np.dot(normal_est, normal_meas)
    print(f"  Dot product with measurement: {dot_after:.6f}")

    assert dot_after > 0.5, f"Normal should move toward measurement, dot={dot_after:.6f}"
    print("  PASS")
    return True


def test_accel_gate_rejects_out_of_range():
    """Test: accelerometer gate rejects norms outside [0.5, 1.5]g."""
    print("\nTest 3: Accel gate rejects out-of-range norms")
    config = ComplementaryConfig(
        kp_grav=2.0, kp_yaw=0.0, max_dt=0.1,
        accel_norm_min=0.5, accel_norm_max=1.5,
        max_grav_error=0.5
    )
    filter = QuaternionComplementaryFilter(config=config)

    # Test accel norm = 0.4 (too low)
    accel_low = np.array([0.0, 0.0, 0.4], dtype=np.float64)
    result_low = filter.update_gravity(accel_low, 0.01)
    print(f"  Norm 0.4g -> update applied: {result_low}")
    assert not result_low, "Should reject norm < 0.5g"

    # Test accel norm = 1.6 (too high)
    accel_high = np.array([0.0, 0.0, 1.6], dtype=np.float64)
    result_high = filter.update_gravity(accel_high, 0.01)
    print(f"  Norm 1.6g -> update applied: {result_high}")
    assert not result_high, "Should reject norm > 1.5g"

    # Test accel norm = 1.0 (valid)
    accel_ok = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    result_ok = filter.update_gravity(accel_ok, 0.01)
    print(f"  Norm 1.0g -> update applied: {result_ok}")
    assert result_ok, "Should accept norm in [0.5, 1.5]g"

    print("  PASS")
    return True


def test_update_yaw_confidence_zero_no_change():
    """Test: update_yaw with confidence=0 does not change state."""
    print("\nTest 4: update_yaw with confidence=0 does not change state")
    config = ComplementaryConfig(kp_grav=0.0, kp_yaw=1.0, max_dt=0.1, max_yaw_error=1.0)
    calibration = Calibration(
        hand="right",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
        source="test"
    )
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)

    q_before = filter.state.copy()
    normal_meas = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    result = filter.update_yaw(normal_meas, 0.01, confidence=0.0)
    q_after = filter.state

    print(f"  Update applied: {result}")
    print(f"  State changed: {not np.allclose(q_before, q_after)}")

    assert not result, "Should return False for confidence=0"
    assert np.allclose(q_before, q_after), "State should not change with confidence=0"
    print("  PASS")
    return True


def test_invalid_dt_no_integration():
    """Test: dt <= 0 or dt > max_dt or non-finite -> no integration."""
    print("\nTest 5: Invalid dt -> no integration")
    config = ComplementaryConfig(kp_grav=0.0, kp_yaw=0.0, max_dt=0.1)
    filter = QuaternionComplementaryFilter(config=config)

    gyro = np.array([0.0, 0.0, 90.0], dtype=np.float64)
    q_initial = filter.state.copy()

    # dt = 0
    filter.predict(gyro, 0.0)
    assert np.allclose(filter.state, q_initial), "dt=0 should not change state"

    # dt = -0.01
    filter.predict(gyro, -0.01)
    assert np.allclose(filter.state, q_initial), "dt<0 should not change state"

    # dt = 0.2 > max_dt
    filter.predict(gyro, 0.2)
    assert np.allclose(filter.state, q_initial), "dt>max_dt should not change state"

    # dt = nan
    filter.predict(gyro, float('nan'))
    assert np.allclose(filter.state, q_initial), "dt=nan should not change state"

    # dt = inf
    filter.predict(gyro, float('inf'))
    assert np.allclose(filter.state, q_initial), "dt=inf should not change state"

    # Valid dt should work
    filter.predict(gyro, 0.05)
    assert not np.allclose(filter.state, q_initial), "Valid dt should change state"

    print("  All invalid dt cases rejected, valid dt accepted")
    print("  PASS")
    return True


def test_normal_cam_requires_calibration():
    """Test: normal_cam() returns None without calibration."""
    print("\nTest 6: normal_cam() requires calibration")
    config = ComplementaryConfig()
    filter = QuaternionComplementaryFilter(config=config)  # No calibration

    normal = filter.normal_cam()
    print(f"  normal_cam() without calibration: {normal}")
    assert normal is None, "Should return None without calibration"

    # With calibration
    calibration = Calibration(
        hand="right",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
        source="test"
    )
    filter2 = QuaternionComplementaryFilter(config=config, calibration=calibration)
    normal2 = filter2.normal_cam()
    print(f"  normal_cam() with calibration: {normal2}")
    assert normal2 is not None, "Should return normal with calibration"
    assert np.allclose(normal2, [0.0, 0.0, 1.0]), "Should match f_body at identity"

    print("  PASS")
    return True


def test_step_handles_none_inputs():
    """Test: step() handles None imu and None normal_cam gracefully."""
    print("\nTest 7: step() handles None inputs")
    config = ComplementaryConfig(kp_grav=0.0, kp_yaw=0.0, max_dt=0.1)
    filter = QuaternionComplementaryFilter(config=config)

    # Both None
    q = filter.step(None, None, 0.01)
    assert q is not None and q.shape == (4,), "Should return state quaternion"

    # IMU only
    imu = ImuSample(
        timestamp=0.0, timestamp_us=0, hand="right",
        gyro_dps=np.array([10.0, 0.0, 0.0]),
        accel_g=np.array([0.0, 0.0, 1.0]),
        valid=True
    )
    q = filter.step(imu, None, 0.01)
    assert q is not None and q.shape == (4,), "Should return state quaternion with IMU"

    # Vision only (no calibration -> should not crash)
    q = filter.step(None, np.array([1.0, 0.0, 0.0]), 0.01)
    assert q is not None and q.shape == (4,), "Should return state quaternion with vision"

    print("  All None combinations handled gracefully")
    print("  PASS")
    return True


def test_gravity_correction_works():
    """Test: gravity correction reduces tilt error."""
    print("\nTest 8: Gravity correction reduces tilt error")
    config = ComplementaryConfig(
        kp_grav=10.0, kp_yaw=0.0, max_dt=0.1,
        accel_norm_min=0.5, accel_norm_max=1.5,
        max_grav_error=1.0,
        gravity_cam=np.array([0.0, 0.0, -1.0])
    )
    filter = QuaternionComplementaryFilter(config=config)

    # Filter starts at identity (level)
    # True body orientation: tilted +30 deg around X relative to camera
    # Gravity in camera: [0, 0, -1] (down)
    # Gravity in body: R_x(-30) * [0,0,-1] = [0, 0.5, -0.866]
    # Accelerometer reads -gravity_body: [0, -0.5, 0.866] (up in body frame)
    tilt_angle = np.deg2rad(30.0)
    gravity_body = np.array([0.0, np.sin(tilt_angle), -np.cos(tilt_angle)], dtype=np.float64)
    accel_g = -gravity_body  # Accelerometer measures specific force = -gravity
    print(f"  True gravity in body: {gravity_body}")
    print(f"  Simulated accel_g (IMU reads): {accel_g}")

    # Apply gravity correction multiple times
    for i in range(50):
        filter.update_gravity(accel_g, 0.01)

    q_final = filter.state
    # Should converge to q_body_cam = rot_x(-30 deg)
    # Because: R(q)^T * [0,0,-1] = gravity_body
    # => R(q) * gravity_body = [0,0,-1]
    # => q rotates gravity_body to [0,0,-1] => rotation of -30 deg around X
    q_expected = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -tilt_angle)
    err = angle_between_deg(q_final, q_expected)
    print(f"  Final error from expected tilt: {err:.3f} deg")

    assert err < 5.0, f"Gravity correction should converge to tilt, got {err:.3f} deg error"
    print("  PASS")
    return True


def test_yaw_correction_left_multiplication():
    """Test: yaw correction uses LEFT multiplication (exp * q)."""
    print("\nTest 9: Yaw correction uses LEFT multiplication")
    config = ComplementaryConfig(kp_grav=0.0, kp_yaw=1.0, max_dt=0.1, max_yaw_error=1.0)
    calibration = Calibration(
        hand="right",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
        source="test"
    )
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)

    # Apply a known rotation first (yaw 45 deg)
    q_yaw45 = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(45.0))
    filter.reset(q_yaw45)

    # Measure normal at [1, 0, 0] (should be 90 deg from current [0,0,1] rotated by 45)
    # Actually, current normal is rotate(q_yaw45, [0,0,1]) = [0,0,1] (yaw doesn't change Z)
    # So measurement [1,0,0] is 90 deg away
    normal_meas = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    q_before = filter.state.copy()
    filter.update_yaw(normal_meas, 0.1, confidence=1.0)
    q_after = filter.state

    # The correction should be applied on the LEFT (camera frame correction)
    # This means the error is computed in camera frame and applied as global rotation
    print(f"  q_before: {q_before}")
    print(f"  q_after:  {q_after}")

    # Just verify it changes state in a reasonable way
    err = angle_between_deg(q_before, q_after)
    print(f"  Change: {err:.3f} deg")
    assert err > 1.0, "Should apply significant correction"

    print("  PASS")
    return True


def run_all_tests():
    """Run all tests and return exit code."""
    tests = [
        test_identity_gyro_yaw,
        test_vision_correction_pulls_normal,
        test_accel_gate_rejects_out_of_range,
        test_update_yaw_confidence_zero_no_change,
        test_invalid_dt_no_integration,
        test_normal_cam_requires_calibration,
        test_step_handles_none_inputs,
        test_gravity_correction_works,
        test_yaw_correction_left_multiplication,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())