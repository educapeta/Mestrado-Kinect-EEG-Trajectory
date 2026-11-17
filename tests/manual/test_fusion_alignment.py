#!/usr/bin/env python
"""
Testes sintéticos para o modo IMU attitude prior e snap/gate de yaw.

Testa:
1. use_imu_attitude_prior: filtro usa atitude do IMU convertida por T como prior
   (sem integrar gyro) - verifica convergência para q_imu calibrado.
2. snap_on_large_yaw_error: quando erro > max_yaw_error mas confiança alta,
   alinha à normal (snap) em vez de gating out - verifica correção de target change.
3. Comportamento padrão (sem prior, sem snap) ainda funciona.
"""

from __future__ import annotations
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, r"C:\mestradopy")

from fusion.complementary import QuaternionComplementaryFilter, ComplementaryConfig
from fusion.contracts import ImuSample, Calibration
from fusion.quaternion import (
    normalize, multiply, conjugate, rotate_vector, exp_so3,
    angle_between_deg, quat_from_axis_angle, from_euler_zyx_deg, to_matrix
)


def assert_close(a, b, atol=1e-6, msg=""):
    if not np.allclose(a, b, atol=atol):
        raise AssertionError(f"{msg}: {a} != {b} (atol={atol})")


def test_imu_attitude_prior_basic():
    """Test: use_imu_attitude_prior uses IMU attitude as prior each frame."""
    print("Test 1: IMU attitude prior basic")
    
    # Calibration: T_cam_world = rotation 90deg around Y, f_body = [0,0,1]
    # So body Z maps to camera X
    T_cam_world = to_matrix(from_euler_zyx_deg(0, 90, 0))  # 90 deg around Y
    f_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    calibration = Calibration(
        hand="right",
        T_cam_world=T_cam_world,
        f_body=f_body,
        source="test"
    )
    
    config = ComplementaryConfig(
        kp_grav=0.0,  # No gravity correction
        kp_yaw=0.0,   # No yaw correction
        use_imu_attitude_prior=True,
        max_dt=0.1
    )
    
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)
    
    # IMU sample: body rotated 30 deg around X relative to world
    imu = ImuSample(
        timestamp=0.0, timestamp_us=0, hand="right",
        roll_deg=30.0, pitch_deg=0.0, yaw_deg=0.0,
        accel_g=np.array([0.0, 0.0, 1.0]),
        gyro_dps=np.array([100.0, 0.0, 0.0]),  # Large gyro - should be IGNORED
        valid=True
    )
    
    # Expected q_body_cam = T_cam_world @ R_imu(body->world)
    q_imu_body_world = from_euler_zyx_deg(30.0, 0.0, 0.0)
    R_imu = to_matrix(q_imu_body_world)
    R_cam_body = T_cam_world @ R_imu
    from fusion.quaternion import from_matrix
    q_expected = from_matrix(R_cam_body)
    
    # Step multiple times - should converge to IMU prior immediately
    for _ in range(5):
        q = filter.step(imu, None, 0.01)
    
    err = angle_between_deg(q, q_expected)
    print(f"  Expected: {q_expected}")
    print(f"  Got:      {q}")
    print(f"  Error:    {err:.6f} deg")
    
    assert err < 0.1, f"IMU prior should match calibrated IMU attitude, got {err:.6f} deg error"
    print("  PASS")
    return True


def test_imu_attitude_prior_with_gravity():
    """Test: IMU prior + gravity correction still works for tilt."""
    print("\nTest 2: IMU prior with gravity correction")
    
    T_cam_world = np.eye(3, dtype=np.float64)
    f_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    calibration = Calibration(
        hand="right",
        T_cam_world=T_cam_world,
        f_body=f_body,
        source="test"
    )
    
    config = ComplementaryConfig(
        kp_grav=5.0,  # Gravity correction enabled
        kp_yaw=0.0,
        use_imu_attitude_prior=True,
        max_dt=0.1,
        gravity_cam=np.array([0.0, 0.0, -1.0]),
        accel_norm_min=0.5,
        accel_norm_max=1.5
    )
    
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)
    
    # IMU says body is level (roll=0), but accelerometer says tilted +20 deg around X
    # Gravity in body frame for +20 deg tilt: [0, sin(20), -cos(20)]
    # Accelerometer reads -gravity: [0, -sin(20), cos(20)]
    tilt = np.deg2rad(20.0)
    gravity_body = np.array([0.0, np.sin(tilt), -np.cos(tilt)], dtype=np.float64)
    accel_g = -gravity_body
    
    imu = ImuSample(
        timestamp=0.0, timestamp_us=0, hand="right",
        roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,  # IMU says level
        accel_g=accel_g,
        gyro_dps=np.array([0.0, 0.0, 0.0]),
        valid=True
    )
    
    # Run filter - prior mode resets to IMU attitude (level) each frame,
    # then applies gravity correction. The gravity correction will pull toward
    # the accelerometer, but next frame the prior resets again.
    # Net effect: filter tracks IMU attitude (level) with small gravity perturbation.
    for _ in range(100):
        filter.step(imu, None, 0.01)
    
    q_final = filter.state
    normal = calibration.normal_cam_from_q(q_final)
    
    # With prior mode active, the filter should stay close to IMU attitude (level)
    # because prior is applied EVERY frame before gravity correction.
    # The normal should be close to [0, 0, 1] (f_body at identity = level)
    print(f"  IMU roll: 0 (level), accel indicates ~20 deg tilt")
    print(f"  Final normal: {normal}")
    print(f"  Normal Z (should be ~1.0 for level): {normal[2]:.4f}")
    
    # Prior mode dominates: should stay near level
    assert normal[2] > 0.9, f"Prior mode should keep filter near IMU attitude (level), got Z={normal[2]:.4f}"
    print("  PASS")
    return True


def test_snap_on_large_yaw_error():
    """Test: snap_on_large_yaw_error aligns to normal when error > gate and confidence high."""
    print("\nTest 3: Snap on large yaw error (high confidence)")
    
    T_cam_world = np.eye(3, dtype=np.float64)
    f_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    calibration = Calibration(
        hand="right",
        T_cam_world=T_cam_world,
        f_body=f_body,
        source="test"
    )
    
    config = ComplementaryConfig(
        kp_grav=0.0,
        kp_yaw=0.5,
        max_yaw_error=0.3,  # ~17 deg gate
        snap_on_large_yaw_error=True,
        snap_confidence_threshold=0.7,
        max_dt=0.1
    )
    
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)
    
    # Start filter with normal at [0, 0, 1] (identity)
    # Vision measures normal at [1, 0, 0] (90 deg around Y) - LARGE error > gate
    # With high confidence (0.9), should SNAP to vision
    normal_meas = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    
    # Step with high confidence
    for _ in range(10):
        filter.step(None, normal_meas, 0.01, vision_confidence=0.9)
    
    q = filter.state
    normal_est = calibration.normal_cam_from_q(q)
    
    print(f"  Initial normal: [0, 0, 1]")
    print(f"  Vision normal:  [1, 0, 0] (90 deg error)")
    print(f"  Gate: 17 deg, confidence: 0.9 > 0.7 threshold")
    print(f"  Final normal: {normal_est}")
    
    # Should have snapped close to vision normal
    dot = np.dot(normal_est, normal_meas)
    print(f"  Dot with vision: {dot:.6f}")
    
    assert dot > 0.99, f"Should snap to vision normal, dot={dot:.6f}"
    print("  PASS")
    return True


def test_no_snap_on_low_confidence():
    """Test: NO snap when confidence is below threshold, even if error > gate."""
    print("\nTest 4: No snap on low confidence")
    
    T_cam_world = np.eye(3, dtype=np.float64)
    f_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    calibration = Calibration(
        hand="right",
        T_cam_world=T_cam_world,
        f_body=f_body,
        source="test"
    )
    
    config = ComplementaryConfig(
        kp_grav=0.0,
        kp_yaw=0.5,
        max_yaw_error=0.3,  # ~17 deg gate
        snap_on_large_yaw_error=True,
        snap_confidence_threshold=0.7,
        max_dt=0.1
    )
    
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)
    
    # Same large error (90 deg), but LOW confidence (0.5 < 0.7)
    normal_meas = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    
    # Step with low confidence - should NOT snap, should gate out
    for _ in range(10):
        filter.step(None, normal_meas, 0.01, vision_confidence=0.5)
    
    q = filter.state
    normal_est = calibration.normal_cam_from_q(q)
    
    print(f"  Vision normal: [1, 0, 0] (90 deg error)")
    print(f"  Confidence: 0.5 < 0.7 threshold")
    print(f"  Final normal: {normal_est}")
    
    # Should stay near identity (no correction applied due to gate)
    dot = np.dot(normal_est, np.array([0.0, 0.0, 1.0]))
    print(f"  Dot with initial [0,0,1]: {dot:.6f}")
    
    assert dot > 0.99, f"Should NOT snap with low confidence, stayed at initial, dot={dot:.6f}"
    print("  PASS")
    return True


def test_snap_disabled_fallback():
    """Test: When snap disabled, large error is gated out (original behavior)."""
    print("\nTest 5: Snap disabled - large error gated out")
    
    T_cam_world = np.eye(3, dtype=np.float64)
    f_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    calibration = Calibration(
        hand="right",
        T_cam_world=T_cam_world,
        f_body=f_body,
        source="test"
    )
    
    config = ComplementaryConfig(
        kp_grav=0.0,
        kp_yaw=0.5,
        max_yaw_error=0.3,  # ~17 deg gate
        snap_on_large_yaw_error=False,  # DISABLED
        max_dt=0.1
    )
    
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)
    
    normal_meas = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # 90 deg error
    
    for _ in range(10):
        filter.step(None, normal_meas, 0.01, vision_confidence=1.0)
    
    q = filter.state
    normal_est = calibration.normal_cam_from_q(q)
    
    print(f"  Snap disabled, error 90 deg > 17 deg gate")
    print(f"  Final normal: {normal_est}")
    
    # Should stay near identity (gated out)
    dot = np.dot(normal_est, np.array([0.0, 0.0, 1.0]))
    print(f"  Dot with initial: {dot:.6f}")
    
    assert dot > 0.99, f"Should gate out when snap disabled, dot={dot:.6f}"
    print("  PASS")
    return True


def test_snap_with_small_error():
    """Test: Small error (< gate) uses normal proportional correction, not snap."""
    print("\nTest 6: Small error uses proportional correction")
    
    T_cam_world = np.eye(3, dtype=np.float64)
    f_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    calibration = Calibration(
        hand="right",
        T_cam_world=T_cam_world,
        f_body=f_body,
        source="test"
    )
    
    config = ComplementaryConfig(
        kp_grav=0.0,
        kp_yaw=2.0,  # Higher gain for faster convergence
        max_yaw_error=0.3,
        snap_on_large_yaw_error=True,
        max_dt=0.1
    )
    
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)
    
    # Small error: 10 deg around Y (within 17 deg gate)
    normal_meas = rotate_vector(from_euler_zyx_deg(0, 10, 0), np.array([0.0, 0.0, 1.0]))
    
    for _ in range(20):
        filter.step(None, normal_meas, 0.01, vision_confidence=1.0)
    
    q = filter.state
    normal_est = calibration.normal_cam_from_q(q)
    
    print(f"  Vision normal: 10 deg from initial (within gate)")
    print(f"  Final normal: {normal_est}")
    
    # Check convergence to vision normal
    dot = np.dot(normal_est, normal_meas)
    print(f"  Dot with vision: {dot:.6f}")
    
    assert dot > 0.95, f"Should converge to vision normal with proportional correction, dot={dot:.6f}"
    print("  PASS")
    return True


def test_prior_vs_gyro_integration():
    """Test: Compare prior mode vs gyro integration when gyro has axis/sign mismatch."""
    print("\nTest 7: Prior mode vs gyro integration (axis mismatch simulation)")
    
    T_cam_world = np.eye(3, dtype=np.float64)
    f_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    calibration = Calibration(
        hand="right",
        T_cam_world=T_cam_world,
        f_body=f_body,
        source="test"
    )
    
    # IMU at 45 deg roll
    imu = ImuSample(
        timestamp=0.0, timestamp_us=0, hand="right",
        roll_deg=45.0, pitch_deg=0.0, yaw_deg=0.0,
        accel_g=np.array([0.0, 0.0, 1.0]),
        gyro_dps=np.array([10.0, 0.0, 0.0]),  # Constant roll rate
        valid=True
    )
    
    q_imu = from_euler_zyx_deg(45.0, 0.0, 0.0)
    
    # Filter A: Prior mode (should track IMU attitude)
    config_prior = ComplementaryConfig(
        kp_grav=0.0, kp_yaw=0.0, use_imu_attitude_prior=True, max_dt=0.1
    )
    filter_prior = QuaternionComplementaryFilter(config=config_prior, calibration=calibration)
    
    # Filter B: Gyro integration (simulating axis mismatch: gyro sign flipped)
    config_gyro = ComplementaryConfig(
        kp_grav=0.0, kp_yaw=0.0, use_imu_attitude_prior=False, max_dt=0.1
    )
    filter_gyro = QuaternionComplementaryFilter(config=config_gyro, calibration=calibration)
    filter_gyro.reset(q_imu)  # Start at same orientation
    
    # Simulate: IMU says 45 deg roll, but gyro integration with WRONG sign
    # would drift away. In prior mode, we ignore gyro and use IMU attitude directly.
    for i in range(200):  # 2 seconds
        dt = 0.01
        # Prior filter: uses IMU attitude (constant 45 deg)
        filter_prior.step(imu, None, dt)
        # Gyro filter: integrates gyro (but we flip sign to simulate mismatch)
        gyro_wrong = np.array([-10.0, 0.0, 0.0])  # Wrong sign!
        filter_gyro.predict(gyro_wrong, dt)
        filter_gyro.update_gravity(imu.accel_g, dt)
    
    q_prior = filter_prior.state
    q_gyro = filter_gyro.state
    
    err_prior = angle_between_deg(q_prior, q_imu)
    err_gyro = angle_between_deg(q_gyro, q_imu)
    
    print(f"  True IMU attitude: 45 deg roll")
    print(f"  Prior mode error:   {err_prior:.3f} deg")
    print(f"  Gyro (wrong sign) error: {err_gyro:.3f} deg")
    
    # Prior should stay accurate, gyro with wrong sign should drift (20 deg in 2s)
    assert err_prior < 1.0, f"Prior mode should track IMU, err={err_prior:.3f}"
    assert err_gyro > 15.0, f"Gyro with wrong sign should drift, err={err_gyro:.3f}"
    print("  PASS - Prior mode immune to gyro axis/sign mismatch")
    return True


def test_default_behavior_unchanged():
    """Test: Default config (no prior, no snap) behaves as before."""
    print("\nTest 8: Default behavior unchanged")
    
    T_cam_world = np.eye(3, dtype=np.float64)
    f_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    calibration = Calibration(
        hand="right",
        T_cam_world=T_cam_world,
        f_body=f_body,
        source="test"
    )
    
    # Default config (no new flags)
    config = ComplementaryConfig()
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)
    
    # Verify defaults
    assert config.use_imu_attitude_prior == False
    assert config.snap_on_large_yaw_error == True  # New default is True
    assert config.snap_confidence_threshold == 0.7
    
    # Test basic gyro integration still works
    gyro = np.array([0.0, 0.0, 90.0], dtype=np.float64)  # 90 deg/s yaw
    dt = 0.01
    for _ in range(100):  # 1 second
        filter.predict(gyro, dt)
    
    q = filter.state
    q_expected = from_euler_zyx_deg(0, 0, 90.0)  # 90 deg yaw
    err = angle_between_deg(q, q_expected)
    
    print(f"  Gyro integration 90 deg/s * 1s = 90 deg yaw")
    print(f"  Error: {err:.3f} deg")
    
    assert err < 1.0, f"Default gyro integration should work, err={err:.3f}"
    print("  PASS")
    return True


def run_all_tests():
    tests = [
        test_imu_attitude_prior_basic,
        test_imu_attitude_prior_with_gravity,
        test_snap_on_large_yaw_error,
        test_no_snap_on_low_confidence,
        test_snap_disabled_fallback,
        test_snap_with_small_error,
        test_prior_vs_gyro_integration,
        test_default_behavior_unchanged,
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