"""
Tests for fusion.calibration module.

Run: python test_fusion_calibration.py
Exit code 0 = all tests passed, 1 = any failure.
"""

import sys
import json
import tempfile
import numpy as np

from fusion.contracts import Calibration, ImuSample
from fusion.calibration import (
    load_calibration,
    save_calibration,
    CalibrationManager,
)
from fusion.quaternion import from_euler_zyx_deg, to_matrix, normalize


def test_calibration_validate():
    """Test Calibration.validate() method."""
    # Valid calibration
    cal = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
        yaw_boot_delta_deg=0.0,
        residual_deg=1.5,
        source="test",
    )
    assert cal.validate() is True

    # Invalid: T_cam_world not 3x3
    cal_bad = Calibration(
        hand="left",
        T_cam_world=np.eye(4),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    assert cal_bad.validate() is False

    # Invalid: f_body not unit
    cal_bad2 = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([1.0, 2.0, 3.0]),
    )
    assert cal_bad2.validate() is False

    # Invalid: T_cam_world not orthogonal
    cal_bad3 = Calibration(
        hand="left",
        T_cam_world=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    assert cal_bad3.validate() is False

    # Invalid: NaN values
    cal_bad4 = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, np.nan]),
    )
    assert cal_bad4.validate() is False

    print("test_calibration_validate: OK")


def test_calibration_to_from_dict():
    """Test Calibration serialization round-trip."""
    cal = Calibration(
        hand="right",
        T_cam_world=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        f_body=np.array([0.0, 0.0, 1.0]),
        yaw_boot_delta_deg=5.5,
        residual_deg=2.3,
        source="manual",
    )

    d = cal.to_dict()
    assert d["hand"] == "right"
    assert d["yaw_boot_delta_deg"] == 5.5
    assert d["residual_deg"] == 2.3
    assert d["source"] == "manual"
    assert len(d["rotation"]) == 3
    assert len(d["body"]) == 3

    cal2 = Calibration.from_dict(d, "right")
    assert cal2.hand == "right"
    assert np.allclose(cal2.T_cam_world, cal.T_cam_world)
    assert np.allclose(cal2.f_body, cal.f_body)
    assert cal2.yaw_boot_delta_deg == 5.5
    assert cal2.residual_deg == 2.3
    assert cal2.source == "manual"

    # Test with missing optional fields
    d_minimal = {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "body": [0, 0, 1]}
    cal3 = Calibration.from_dict(d_minimal, "left")
    assert cal3.hand == "left"
    assert cal3.yaw_boot_delta_deg == 0.0
    assert cal3.residual_deg is None
    assert cal3.source == ""

    print("test_calibration_to_from_dict: OK")


def test_calibration_normal_from_q():
    """Test normal_cam_from_q with identity transform."""
    # T_cam_world = I, f_body = [0, 0, 1]
    cal = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
    )

    # Identity quaternion -> normal should be f_body = [0, 0, 1]
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    normal = cal.normal_cam_from_q(q_id)
    assert np.allclose(normal, [0.0, 0.0, 1.0])

    # 90 deg rotation around Z -> [0, 0, 1] stays [0, 0, 1]
    q_90z = from_euler_zyx_deg(0, 0, 90)
    normal = cal.normal_cam_from_q(q_90z)
    assert np.allclose(normal, [0.0, 0.0, 1.0])

    # 90 deg rotation around Y -> [0, 0, 1] becomes [1, 0, 0]
    q_90y = from_euler_zyx_deg(0, 90, 0)
    normal = cal.normal_cam_from_q(q_90y)
    assert np.allclose(normal, [1.0, 0.0, 0.0], atol=1e-6)

    print("test_calibration_normal_from_q: OK")


def test_calibration_normal_from_imu():
    """Test normal_cam_from_imu with identity transform."""
    cal = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
    )

    # IMU sample with zero angles
    sample = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        valid=True,
    )
    normal = cal.normal_cam_from_imu(sample)
    assert np.allclose(normal, [0.0, 0.0, 1.0])

    # 90 deg pitch (around Y) -> [0, 0, 1] becomes [1, 0, 0]
    sample2 = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        roll_deg=0.0,
        pitch_deg=90.0,
        yaw_deg=0.0,
        valid=True,
    )
    normal2 = cal.normal_cam_from_imu(sample2)
    assert np.allclose(normal2, [1.0, 0.0, 0.0], atol=1e-6)

    # 90 deg roll (around X) -> [0, 0, 1] becomes [0, -1, 0]
    sample3 = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        roll_deg=90.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        valid=True,
    )
    normal3 = cal.normal_cam_from_imu(sample3)
    assert np.allclose(normal3, [0.0, -1.0, 0.0], atol=1e-6)

    # 90 deg yaw (around Z) -> [0, 0, 1] stays [0, 0, 1]
    sample4 = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=90.0,
        valid=True,
    )
    normal4 = cal.normal_cam_from_imu(sample4)
    assert np.allclose(normal4, [0.0, 0.0, 1.0], atol=1e-6)

    print("test_calibration_normal_from_imu: OK")


def test_calibration_with_T_cam_world():
    """Test calibration with non-identity T_cam_world."""
    # T_cam_world rotates world to camera: 90 deg around Z
    T = to_matrix(from_euler_zyx_deg(0, 0, 90))
    cal = Calibration(
        hand="left",
        T_cam_world=T,
        f_body=np.array([0.0, 0.0, 1.0]),
    )

    # Body frame normal [0,0,1] -> world [0,0,1] (identity q) -> camera [0,0,1] (Z rotation doesn't change Z)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    normal = cal.normal_cam_from_q(q_id)
    assert np.allclose(normal, [0.0, 0.0, 1.0])

    # Body-to-camera quaternion: body->world (90 deg pitch) then world->camera (T_cam_world)
    # q_body_cam = T_cam_world * q_body_world
    # For 90 deg pitch in body: q_body_world = from_euler_zyx_deg(0, 90, 0)
    # T_cam_world as quaternion: from_euler_zyx_deg(0, 0, 90)
    # q_body_cam = q_T_cam_world * q_body_world
    from fusion.quaternion import multiply
    q_T = from_euler_zyx_deg(0, 0, 90)
    q_pitch = from_euler_zyx_deg(0, 90, 0)
    q_body_cam = multiply(q_T, q_pitch)
    normal2 = cal.normal_cam_from_q(q_body_cam)
    # [0,0,1] -> body pitch 90 -> [1,0,0] world -> cam yaw 90 -> [0,1,0]
    assert np.allclose(normal2, [0.0, 1.0, 0.0], atol=1e-6)

    print("test_calibration_with_T_cam_world: OK")


def test_load_save_calibration():
    """Test JSON load/save round-trip."""
    cal = Calibration(
        hand="left",
        T_cam_world=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        f_body=np.array([0.0, 0.0, 1.0]),
        yaw_boot_delta_deg=3.14,
        residual_deg=1.23,
        source="test_save_load",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        save_calibration(cal, tmp_path)

        # Check file content
        with open(tmp_path, "r") as f:
            data = json.load(f)
        assert data["hand"] == "left"
        assert data["yaw_boot_delta_deg"] == 3.14
        assert data["residual_deg"] == 1.23
        assert data["source"] == "test_save_load"
        assert data["rotation"] == [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        assert data["body"] == [0.0, 0.0, 1.0]

        # Load back
        cal2 = load_calibration(tmp_path, "left")
        assert cal2 is not None
        assert cal2.hand == "left"
        assert np.allclose(cal2.T_cam_world, cal.T_cam_world)
        assert np.allclose(cal2.f_body, cal.f_body)
        assert cal2.yaw_boot_delta_deg == 3.14
        assert cal2.residual_deg == 1.23
        assert cal2.source == "test_save_load"

        # Load with wrong hand
        cal3 = load_calibration(tmp_path, "right")
        assert cal3 is not None
        assert cal3.hand == "right"  # hand is overridden by argument

        # Load non-existent file
        cal4 = load_calibration("/nonexistent/path.json", "left")
        assert cal4 is None

        # Load invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json")
            bad_path = f.name
        cal5 = load_calibration(bad_path, "left")
        assert cal5 is None

    finally:
        import os
        try:
            os.unlink(tmp_path)
        except:
            pass

    print("test_load_save_calibration: OK")


def test_calibration_manager():
    """Test CalibrationManager."""
    cal_left = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    cal_right = Calibration(
        hand="right",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
    )

    mgr = CalibrationManager({"left": cal_left, "right": cal_right})

    assert len(mgr) == 2
    assert "left" in mgr
    assert "right" in mgr
    assert mgr.get("left") is cal_left
    assert mgr.get("right") is cal_right
    assert mgr.get("nonexistent") is None

    # Add new
    cal_left2 = Calibration(
        hand="left",
        T_cam_world=to_matrix(from_euler_zyx_deg(0, 0, 90)),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    mgr.add(cal_left2)
    assert mgr.get("left") is cal_left2

    # normal_from_q
    q = np.array([1.0, 0.0, 0.0, 0.0])
    normal = mgr.normal_from_q("left", q)
    assert np.allclose(normal, [0.0, 0.0, 1.0])

    normal = mgr.normal_from_q("right", q)
    assert np.allclose(normal, [0.0, 0.0, 1.0])

    normal = mgr.normal_from_q("nonexistent", q)
    assert normal is None

    # normal_from_imu
    sample = ImuSample(
        timestamp=1.0,
        timestamp_us=1000000,
        hand="left",
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        valid=True,
    )
    normal = mgr.normal_from_imu("left", sample)
    assert np.allclose(normal, [0.0, 0.0, 1.0])

    normal = mgr.normal_from_imu("nonexistent", sample)
    assert normal is None

    # validate_all
    results = mgr.validate_all()
    assert results == {"left": True, "right": True}

    print("test_calibration_manager: OK")


def test_calibration_manager_invalid_add():
    """Test CalibrationManager rejects invalid calibration."""
    mgr = CalibrationManager()
    cal_bad = Calibration(
        hand="left",
        T_cam_world=np.eye(4),  # invalid shape
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    try:
        mgr.add(cal_bad)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # expected

    print("test_calibration_manager_invalid_add: OK")


def run_all():
    """Run all tests."""
    tests = [
        test_calibration_validate,
        test_calibration_to_from_dict,
        test_calibration_normal_from_q,
        test_calibration_normal_from_imu,
        test_calibration_with_T_cam_world,
        test_load_save_calibration,
        test_calibration_manager,
        test_calibration_manager_invalid_add,
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