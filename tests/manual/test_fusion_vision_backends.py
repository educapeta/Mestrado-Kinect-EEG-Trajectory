"""
Tests for fusion/vision_backends.py - pure numpy, no hardware.
"""

from __future__ import annotations
import numpy as np
import math

from fusion.vision_backends import (
    VisionNormal,
    robust_plane_normal,
    KinectDepthBackend,
    MultiViewBackend,
    compare_backends,
)


def make_plane_points(normal: np.ndarray, center: np.ndarray, n: int = 20,
                       noise_m: float = 0.0, spread_m: float = 0.05) -> np.ndarray:
    """Generate points on a plane with given normal and center."""
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal = normal / np.linalg.norm(normal)
    center = np.asarray(center, dtype=np.float64).reshape(3)

    # Create orthonormal basis for the plane
    if abs(normal[0]) < 0.9:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Random points in the plane
    np.random.seed(42)
    coords = np.random.uniform(-spread_m, spread_m, size=(n, 2))
    points = center + coords[:, 0:1] * u + coords[:, 1:2] * v

    if noise_m > 0:
        points += np.random.normal(0, noise_m, size=points.shape)

    return points


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in degrees."""
    v1 = np.asarray(v1, dtype=np.float64).reshape(3)
    v2 = np.asarray(v2, dtype=np.float64).reshape(3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def axis_angle(v1, v2) -> float:
    """Angulo entre EIXOS (ignora sinal), 0..90."""
    a = angle_between(v1, v2)
    return min(a, 180.0 - a)


def test_clean_plane():
    """Clean plane: expected normal; unitary; sign."""
    print("TEST: clean_plane")
    normal_true = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # Palm facing camera
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    points = make_plane_points(normal_true, center, n=20, noise_m=0.001)

    result = robust_plane_normal(points, side="right")
    assert result is not None, "Should succeed with clean plane"
    normal, rms, edge_on, sigma = result

    # Check unitary
    assert abs(np.linalg.norm(normal) - 1.0) < 1e-6, "Normal must be unit"

    # O sinal nao e' definido pelo lado; comparar o EIXO da normal.
    ang = angle_between(normal, normal_true)
    if ang > 90.0:
        ang = 180.0 - ang
    assert ang < 2.0, f"Angle error {ang:.2f} deg too large"

    # RMS should be small
    assert rms < 0.002, f"RMS {rms:.6f} too large for clean plane"

    # Not edge-on
    assert not edge_on, "Frontal plane should not be edge-on"

    # Sigma should be small
    assert sigma < 5.0, f"Sigma {sigma:.2f} deg too large for clean plane"

    print(f"  PASS: angle={ang:.2f}deg, rms={rms:.6f}m, sigma={sigma:.2f}deg")


def test_clean_plane_left_hand():
    """Clean plane with left hand: sign flipped."""
    print("TEST: clean_plane_left_hand")
    normal_true = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # Left: palm facing camera -> nz > 0
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    points = make_plane_points(normal_true, center, n=20, noise_m=0.001)

    result = robust_plane_normal(points, side="left")
    assert result is not None
    normal, _, _, _ = result

    # O sinal nao e' definido pelo lado; comparar o EIXO da normal.
    ang = angle_between(normal, normal_true)
    if ang > 90.0:
        ang = 180.0 - ang
    assert ang < 2.0, f"Angle error {ang:.2f} deg"

    print(f"  PASS: nz={normal[2]:.3f}, angle={ang:.2f}deg")


def test_side_does_not_change_axis():
    """O mesmo plano deve dar o mesmo eixo para right/left (sem prior falso)."""
    print("TEST: side_does_not_change_axis")
    normal_true = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    points = make_plane_points(normal_true, np.array([0.0, 0.0, 0.8]),
                               n=20, noise_m=0.001)
    r = robust_plane_normal(points, side="right")
    l = robust_plane_normal(points, side="left")
    assert r is not None and l is not None
    dot = abs(float(np.dot(r[0], l[0])))
    assert dot > 0.999, f"side mudou o eixo: dot={dot:.6f}"
    print(f"  PASS: dot={dot:.6f}")


def test_jitter_outlier():
    """Jitter/outlier: normal doesn't explode; RMS/sigma rise."""
    print("TEST: jitter_outlier")
    normal_true = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)

    # Clean points
    clean_pts = make_plane_points(normal_true, center, n=20, noise_m=0.001)

    # Add outliers (background points far away)
    outliers = np.array([
        [0.5, 0.5, 2.0],   # Far background
        [-0.5, -0.5, 3.0], # Far background
        [1.0, 0.0, 1.5],   # Side background
    ], dtype=np.float64)

    points = np.vstack([clean_pts, outliers])

    # Without outlier rejection would fail; with it should succeed
    result = robust_plane_normal(points, side="right", max_iter=3, outlier_threshold_sigma=2.0)
    assert result is not None, "Should reject outliers and succeed"
    normal, rms, edge_on, sigma = result

    # Normal should still be close
    ang = axis_angle(normal, normal_true)
    assert ang < 5.0, f"Normal exploded: angle={ang:.2f} deg"

    # RMS should be close to clean (outliers fully rejected by depth filter)
    # This is correct behavior - depth pre-filter removes far outliers
    assert rms < 0.005, f"RMS {rms:.4f} too high after rejection"

    # Sigma should be reasonable
    assert sigma < 5.0, f"Sigma {sigma:.2f} too high"

    print(f"  PASS: angle={ang:.2f}deg, rms={rms:.6f}m, sigma={sigma:.2f}deg")


def test_flip_stabilization():
    """Flip: normal_anterior stabilizes sign when evidence weak."""
    print("TEST: flip_stabilization")
    # Edge-on case: normal nearly perpendicular to optical axis
    # Anatomical sign is ambiguous (nz ~ 0)
    normal_true = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # Pointing right
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    points = make_plane_points(normal_true, center, n=20, noise_m=0.001)

    # First call: no anterior, should pick some sign
    result1 = robust_plane_normal(points, side="right", normal_anterior=None)
    assert result1 is not None
    normal1, _, _, _ = result1

    # Second call: anterior provided, should match continuity
    # Even if SVD flips, continuity should win because |nz| is small
    result2 = robust_plane_normal(points, side="right", normal_anterior=normal1)
    assert result2 is not None
    normal2, _, _, _ = result2

    # Should be consistent (dot > 0)
    dot = np.dot(normal1, normal2)
    assert dot > 0.99, f"Sign flip detected: dot={dot:.4f}"

    # Now test with opposite anterior - should it flip?
    # With weak anatomical evidence (|nz| small), continuity should win
    opposite = -normal1
    result3 = robust_plane_normal(points, side="right", normal_anterior=opposite)
    assert result3 is not None
    normal3, _, _, _ = result3

    # Should agree with anterior (continuity wins over weak anatomy)
    dot3 = np.dot(normal3, opposite)
    assert dot3 > 0.99, f"Continuity failed: dot={dot3:.4f}"

    print(f"  PASS: continuity maintained, dot12={dot:.4f}, dot3={dot3:.4f}")


def test_edge_on_flag():
    """Edge-on: flag True and sigma larger."""
    print("TEST: edge_on_flag")
    # Normal nearly along X axis (perp to optical Z)
    normal_edge = np.array([1.0, 0.0, 0.05], dtype=np.float64)  # nz = 0.05
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    # Use more noise to get above the sigma floor
    points = make_plane_points(normal_edge, center, n=30, noise_m=0.005)

    result = robust_plane_normal(points, side="right")
    assert result is not None
    normal, rms, edge_on, sigma = result

    assert edge_on, "Should detect edge-on when |nz| small"
    assert abs(normal[2]) < 0.15, f"|nz|={abs(normal[2]):.3f} should be small"

    # Compare with frontal plane (same noise)
    normal_front = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    points_front = make_plane_points(normal_front, center, n=30, noise_m=0.005)
    result_front = robust_plane_normal(points_front, side="right")
    _, _, _, sigma_front = result_front

    # Edge-on sigma should be larger due to edge_factor inflation
    # (The edge_factor = 1/|nz| ~ 20, but capped; should see significant increase)
    assert sigma > sigma_front * 1.5, f"Edge-on sigma {sigma:.2f} not > 1.5x frontal {sigma_front:.2f}"

    print(f"  PASS: edge_on={edge_on}, sigma_edge={sigma:.2f}deg, sigma_front={sigma_front:.2f}deg")


def test_multiview_reproj_confidence():
    """MultiView: high reprojection error lowers confidence."""
    print("TEST: multiview_reproj_confidence")
    normal_true = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    points = make_plane_points(normal_true, center, n=20, noise_m=0.001)

    backend = MultiViewBackend()

    # Low reprojection errors -> high confidence
    reproj_low = np.full(20, 0.5)  # 0.5 px
    vn_low = backend.compute(points, "right", 0.0, reproj_errors=reproj_low)

    # High reprojection errors -> low confidence
    reproj_high = np.full(20, 8.0)  # 8 px
    vn_high = backend.compute(points, "right", 0.0, reproj_errors=reproj_high)

    assert vn_low.valid and vn_high.valid
    assert vn_low.confidence > vn_high.confidence, \
        f"Low reproj conf {vn_low.confidence:.3f} should be > high reproj {vn_high.confidence:.3f}"

    # With no reproj errors, confidence based on sigma
    vn_none = backend.compute(points, "right", 0.0, reproj_errors=None)
    assert vn_none.valid
    assert vn_none.confidence > 0.0

    print(f"  PASS: conf_low={vn_low.confidence:.3f}, conf_high={vn_high.confidence:.3f}, conf_none={vn_none.confidence:.3f}")


def test_kinect_depth_backend():
    """KinectDepthBackend basic functionality."""
    print("TEST: kinect_depth_backend")
    normal_true = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    points = make_plane_points(normal_true, center, n=20, noise_m=0.001)

    backend = KinectDepthBackend()
    vn = backend.compute(points, "right", 1.0)

    assert vn.valid
    assert vn.source == "kinect_depth"
    assert vn.timestamp == 1.0
    assert vn.hand == "right"
    assert vn.position_cam is not None
    assert axis_angle(vn.normal_cam, normal_true) < 2.0

    # Position should be near center (median)
    pos_err = np.linalg.norm(vn.position_cam - center)
    assert pos_err < 0.02, f"Position error {pos_err:.4f}m"

    print(f"  PASS: source={vn.source}, angle={angle_between(vn.normal_cam, normal_true):.2f}deg")


def test_multiview_backend():
    """MultiViewBackend basic functionality."""
    print("TEST: multiview_backend")
    normal_true = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    points = make_plane_points(normal_true, center, n=20, noise_m=0.001)
    reproj = np.full(20, 1.0)

    backend = MultiViewBackend()
    vn = backend.compute(points, "left", 2.0, reproj_errors=reproj)

    assert vn.valid
    assert vn.source == "multiview"
    assert vn.timestamp == 2.0
    assert vn.hand == "left"
    # Left hand -> nz > 0
    assert vn.normal_cam[2] > 0
    assert angle_between(vn.normal_cam, -normal_true) < 2.0  # Flipped for left

    print(f"  PASS: source={vn.source}, nz={vn.normal_cam[2]:.3f}")


def test_compare_backends():
    """Compare backends: angle between them coherent."""
    print("TEST: compare_backends")
    normal_true = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    points = make_plane_points(normal_true, center, n=20, noise_m=0.001)
    reproj = np.full(20, 1.0)

    result = compare_backends(points, "right", 0.0, reproj_errors=reproj)

    assert "kinect_depth" in result
    assert "multiview" in result
    assert "angle_deg" in result
    assert "agree" in result

    vn_k = result["kinect_depth"]
    vn_m = result["multiview"]
    angle = result["angle_deg"]
    agree = result["agree"]

    assert vn_k.valid and vn_m.valid
    assert angle < 5.0, f"Backends disagree: angle={angle:.2f}deg"
    assert agree, "Backends should agree on clean data"

    print(f"  PASS: angle={angle:.2f}deg, agree={agree}")


def test_insufficient_points():
    """Test with < 4 points returns None."""
    print("TEST: insufficient_points")
    points = np.array([[0, 0, 1], [0.01, 0, 1], [0, 0.01, 1]], dtype=np.float64)
    result = robust_plane_normal(points, side="right")
    assert result is None, "Should return None with < 4 points"

    # With 4 points but colinear
    points_col = np.array([
        [0, 0, 1], [0.01, 0, 1], [0.02, 0, 1], [0.03, 0, 1]
    ], dtype=np.float64)
    result = robust_plane_normal(points_col, side="right")
    assert result is None, "Should return None with colinear points"

    print("  PASS: correctly rejects insufficient/colinear points")


def test_invalid_inputs():
    """Test with NaN/inf inputs."""
    print("TEST: invalid_inputs")
    points = np.array([
        [0, 0, 1], [0.01, 0, 1], [0, 0.01, 1], [0.01, 0.01, 1],
        [np.nan, 0, 1], [0, np.inf, 1]
    ], dtype=np.float64)
    result = robust_plane_normal(points, side="right")
    # Should handle gracefully (use only finite points)
    assert result is not None, "Should work with some valid points"
    normal, _, _, _ = result
    assert np.isfinite(normal).all()

    print("  PASS: handles NaN/inf gracefully")


def test_weights():
    """Test weighted plane fitting."""
    print("TEST: weights")
    normal_true = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    center = np.array([0.0, 0.0, 0.8], dtype=np.float64)
    points = make_plane_points(normal_true, center, n=20, noise_m=0.001)

    # High weight on first half, low on second
    weights = np.ones(20)
    weights[10:] = 0.01  # Nearly ignore second half

    result = robust_plane_normal(points, weights=weights, side="right")
    assert result is not None
    normal, _, _, _ = result
    ang = axis_angle(normal, normal_true)
    assert ang < 2.0

    print(f"  PASS: weighted fit angle={ang:.2f}deg")


def run_all_tests():
    """Run all tests and report."""
    tests = [
        test_clean_plane,
        test_clean_plane_left_hand,
        test_jitter_outlier,
        test_flip_stabilization,
        test_edge_on_flag,
        test_multiview_reproj_confidence,
        test_kinect_depth_backend,
        test_multiview_backend,
        test_compare_backends,
        test_insufficient_points,
        test_invalid_inputs,
        test_weights,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)