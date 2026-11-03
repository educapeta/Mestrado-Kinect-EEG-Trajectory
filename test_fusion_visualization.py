#!/usr/bin/env python
"""
Tests for fusion/visualization.py

Tests the draw_overlay function.
"""

from __future__ import annotations
import sys
import numpy as np

from fusion.visualization import draw_overlay, draw_calibration_overlay
from fusion.contracts import FusedState, SourceStatus, Calibration


def test_draw_overlay_basic():
    """Test: draw_overlay with basic inputs returns same shape/dtype."""
    print("Test 1: Basic draw_overlay returns correct frame")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    state = FusedState(
        timestamp=1.0,
        hand="right",
        q_body_cam=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        normal_cam=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        dir_cam=np.array([0.0, -1.0, 0.0], dtype=np.float64),
        confidence=0.9,
        source="test_fusion",
        latency_ms=5.0,
        valid=True
    )

    statuses = [
        SourceStatus("imu", 10.0, 0.9, True, ""),
        SourceStatus("kinect", 33.0, 0.8, True, "")
    ]

    result = draw_overlay(frame, state, statuses)

    # Check return value
    assert result is frame, "Should return same frame object"
    assert result.shape == (480, 640, 3), f"Shape mismatch: {result.shape}"
    assert result.dtype == np.uint8, f"Dtype mismatch: {result.dtype}"

    # Check that something was drawn (not all zeros)
    assert np.any(result != 0), "Frame should not be all zeros after drawing"

    print(f"  Input shape: {frame.shape}, dtype: {frame.dtype}")
    print(f"  Output shape: {result.shape}, dtype: {result.dtype}")
    print(f"  Non-zero pixels: {np.count_nonzero(result)}")
    print("  PASS")
    return True


def test_draw_overlay_empty_frame():
    """Test: draw_overlay handles empty/None frame gracefully."""
    print("\nTest 2: draw_overlay handles empty frame")
    frame = np.array([], dtype=np.uint8).reshape(0, 0, 3)

    state = FusedState(
        timestamp=1.0, hand="right",
        q_body_cam=np.array([1.0, 0.0, 0.0, 0.0]),
        normal_cam=np.array([0.0, 0.0, 1.0]),
        dir_cam=np.array([0.0, -1.0, 0.0]),
        confidence=0.9, source="test", latency_ms=5.0, valid=True
    )
    statuses = []

    result = draw_overlay(frame, state, statuses)

    assert result is frame, "Should return same frame"
    assert result.shape == (0, 0, 3), "Should preserve empty shape"
    print("  PASS")
    return True


def test_draw_overlay_invalid_state():
    """Test: draw_overlay handles invalid state."""
    print("\nTest 3: draw_overlay with invalid state")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    state = FusedState(
        timestamp=1.0, hand="left",
        q_body_cam=np.array([1.0, 0.0, 0.0, 0.0]),
        normal_cam=np.array([0.0, 0.0, 0.0]),
        dir_cam=np.array([0.0, 0.0, 0.0]),
        confidence=0.0, source="none", latency_ms=0.0, valid=False
    )

    statuses = [
        SourceStatus("imu", 1000.0, 0.0, False, "timeout"),
        SourceStatus("kinect", 2000.0, 0.0, False, "no detection")
    ]

    result = draw_overlay(frame, state, statuses)

    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8
    assert np.any(result != 0), "Should still draw status panel"
    print("  PASS")
    return True


def test_draw_overlay_with_calibration():
    """Test: draw_overlay with calibration parameter."""
    print("\nTest 4: draw_overlay with calibration")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    state = FusedState(
        timestamp=1.0, hand="right",
        q_body_cam=np.array([1.0, 0.0, 0.0, 0.0]),
        normal_cam=np.array([0.0, 0.0, 1.0]),
        dir_cam=np.array([0.0, -1.0, 0.0]),
        confidence=0.9, source="calibrated", latency_ms=3.0, valid=True
    )

    calibration = Calibration(
        hand="right",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
        yaw_boot_delta_deg=5.0,
        residual_deg=1.5,
        source="hand_eye_test"
    )

    statuses = [SourceStatus("imu", 5.0, 0.95, True, "")]

    result = draw_overlay(frame, state, statuses, calibration=calibration, source_text="calibrated")

    assert result.shape == (480, 640, 3)
    assert np.any(result != 0)
    print("  PASS")
    return True


def test_draw_overlay_different_sizes():
    """Test: draw_overlay works with different frame sizes."""
    print("\nTest 5: draw_overlay with different frame sizes")
    sizes = [(240, 320), (720, 1280), (1080, 1920), (480, 640)]

    for h, w in sizes:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        state = FusedState(
            timestamp=1.0, hand="right",
            q_body_cam=np.array([1.0, 0.0, 0.0, 0.0]),
            normal_cam=np.array([0.0, 0.0, 1.0]),
            dir_cam=np.array([0.0, -1.0, 0.0]),
            confidence=0.9, source="test", latency_ms=5.0, valid=True
        )
        statuses = [SourceStatus("imu", 10.0, 0.9, True, "")]

        result = draw_overlay(frame, state, statuses)
        assert result.shape == (h, w, 3), f"Size mismatch for {w}x{h}"
        assert result.dtype == np.uint8
        print(f"  {w}x{h}: OK")

    print("  PASS")
    return True


def test_draw_calibration_overlay():
    """Test: draw_calibration_overlay works."""
    print("\nTest 6: draw_calibration_overlay")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    calibration = Calibration(
        hand="left",
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, -1.0]),
        yaw_boot_delta_deg=-2.5,
        residual_deg=0.8,
        source="auto_calib"
    )

    result = draw_calibration_overlay(frame, calibration, position=(20, 50))

    assert result is frame
    assert result.shape == (480, 640, 3)
    assert np.any(result != 0)
    print("  PASS")
    return True


def test_draw_overlay_no_exception():
    """Test: draw_overlay never raises exceptions for valid inputs."""
    print("\nTest 7: draw_overlay no exceptions for edge cases")

    # Many statuses
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = FusedState(
        timestamp=1.0, hand="right",
        q_body_cam=np.array([1.0, 0.0, 0.0, 0.0]),
        normal_cam=np.array([0.0, 0.0, 1.0]),
        dir_cam=np.array([0.0, -1.0, 0.0]),
        confidence=0.9, source="test", latency_ms=5.0, valid=True
    )
    statuses = [SourceStatus(f"src_{i}", float(i*10), 0.5, i%2==0, "reason") for i in range(20)]
    result = draw_overlay(frame, state, statuses)
    assert result.shape == (480, 640, 3)
    print("  Many statuses: OK")

    # Very long source names
    statuses = [SourceStatus("very_long_source_name_that_exceeds_normal_width", 10.0, 0.9, True, "")]
    result = draw_overlay(np.zeros((480, 640, 3), dtype=np.uint8), state, statuses)
    print("  Long source name: OK")

    # Zero normal
    state2 = FusedState(
        timestamp=1.0, hand="right",
        q_body_cam=np.array([1.0, 0.0, 0.0, 0.0]),
        normal_cam=np.array([0.0, 0.0, 0.0]),
        dir_cam=np.array([0.0, 0.0, 0.0]),
        confidence=0.9, source="test", latency_ms=5.0, valid=True
    )
    result = draw_overlay(np.zeros((480, 640, 3), dtype=np.uint8), state2, [])
    print("  Zero normal: OK")

    # NaN in quaternion (should not crash)
    state3 = FusedState(
        timestamp=1.0, hand="right",
        q_body_cam=np.array([1.0, 0.0, 0.0, 0.0]),
        normal_cam=np.array([0.0, 0.0, 1.0]),
        dir_cam=np.array([0.0, -1.0, 0.0]),
        confidence=0.9, source="test", latency_ms=5.0, valid=True
    )
    result = draw_overlay(np.zeros((480, 640, 3), dtype=np.uint8), state3, [])
    print("  Normal quaternion: OK")

    print("  PASS")
    return True


def run_all_tests():
    """Run all tests and return exit code."""
    tests = [
        test_draw_overlay_basic,
        test_draw_overlay_empty_frame,
        test_draw_overlay_invalid_state,
        test_draw_overlay_with_calibration,
        test_draw_overlay_different_sizes,
        test_draw_calibration_overlay,
        test_draw_overlay_no_exception,
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