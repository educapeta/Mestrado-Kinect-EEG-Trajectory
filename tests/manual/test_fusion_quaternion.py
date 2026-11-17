#!/usr/bin/env python3
"""
Tests for fusion.quaternion module.

Run: python test_fusion_quaternion.py
Exit code: 0 on success, 1 on failure.
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, r"C:\mestradopy")

from fusion.quaternion import (
    normalize,
    multiply,
    conjugate,
    rotate_vector,
    quat_from_axis_angle,
    exp_so3,
    slerp,
    from_euler_zyx_deg,
    to_matrix,
    from_matrix,
    angle_between_deg,
    quat_from_two_vectors,
)


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


def test_normalize():
    """Test quaternion normalization."""
    print("Testing normalize...")
    q = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)
    qn = normalize(q)
    assert_allclose(qn, [1.0, 0.0, 0.0, 0.0], msg="normalize identity")
    assert np.isclose(np.linalg.norm(qn), 1.0), "normalize unit norm"

    # Random quaternion
    q = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    qn = normalize(q)
    assert np.isclose(np.linalg.norm(qn), 1.0), "normalize random"

    # Zero quaternion should raise
    assert_raises(ValueError, normalize, np.zeros(4))
    print("  OK")


def test_multiply():
    """Test Hamilton product."""
    print("Testing multiply...")
    # Identity
    q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    q = normalize(q)
    assert_allclose(multiply(q_id, q), q, msg="multiply identity left")
    assert_allclose(multiply(q, q_id), q, msg="multiply identity right")

    # i * i = -1
    qi = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    result = multiply(qi, qi)
    assert_allclose(result, [-1.0, 0.0, 0.0, 0.0], msg="i*i=-1")

    # j * j = -1
    qj = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    result = multiply(qj, qj)
    assert_allclose(result, [-1.0, 0.0, 0.0, 0.0], msg="j*j=-1")

    # k * k = -1
    qk = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    result = multiply(qk, qk)
    assert_allclose(result, [-1.0, 0.0, 0.0, 0.0], msg="k*k=-1")

    # i * j = k
    result = multiply(qi, qj)
    assert_allclose(result, [0.0, 0.0, 0.0, 1.0], msg="i*j=k")

    # j * k = i
    result = multiply(qj, qk)
    assert_allclose(result, [0.0, 1.0, 0.0, 0.0], msg="j*k=i")

    # k * i = j
    result = multiply(qk, qi)
    assert_allclose(result, [0.0, 0.0, 1.0, 0.0], msg="k*i=j")

    print("  OK")


def test_conjugate():
    """Test quaternion conjugate."""
    print("Testing conjugate...")
    q = np.array([0.5, 0.3, 0.4, 0.6], dtype=np.float64)
    qc = conjugate(q)
    assert_allclose(qc, [0.5, -0.3, -0.4, -0.6], msg="conjugate values")

    # q * q* = |q|^2
    q = normalize(q)
    prod = multiply(q, conjugate(q))
    assert_allclose(prod, [1.0, 0.0, 0.0, 0.0], msg="q*q*=1")
    print("  OK")


def test_rotate_vector():
    """Test vector rotation."""
    print("Testing rotate_vector...")
    # 90 deg around Z: (1,0,0) -> (0,1,0)
    q = from_euler_zyx_deg(0, 0, 90)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v_rot = rotate_vector(q, v)
    assert_allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-10, msg="rotate 90 deg Z")

    # 180 deg around X: (0,1,0) -> (0,-1,0)
    q = from_euler_zyx_deg(180, 0, 0)
    v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v_rot = rotate_vector(q, v)
    assert_allclose(v_rot, [0.0, -1.0, 0.0], atol=1e-10, msg="rotate 180 deg X")

    # Identity rotation
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v_rot = rotate_vector(q, v)
    assert_allclose(v_rot, v, msg="rotate identity")

    # rotate_vector vs matrix multiplication
    q = normalize(np.array([0.7, 0.2, 0.3, 0.6], dtype=np.float64))
    v = np.array([0.5, -0.3, 0.8], dtype=np.float64)
    v_rot_q = rotate_vector(q, v)
    R = to_matrix(q)
    v_rot_m = R @ v
    assert_allclose(v_rot_q, v_rot_m, atol=1e-10, msg="rotate_vector vs matrix")
    print("  OK")


def test_quat_from_axis_angle():
    """Test axis-angle to quaternion."""
    print("Testing quat_from_axis_angle...")
    # 90 deg around Z
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    q = quat_from_axis_angle(axis, np.pi/2)
    expected = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2], dtype=np.float64)
    assert_allclose(q, expected, atol=1e-10, msg="axis-angle 90 deg Z")

    # 180 deg around X
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    q = quat_from_axis_angle(axis, np.pi)
    expected = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    assert_allclose(q, expected, atol=1e-10, msg="axis-angle 180 deg X")

    # Zero axis should raise
    assert_raises(ValueError, quat_from_axis_angle, np.zeros(3), 1.0)
    print("  OK")


def test_exp_so3():
    """Test exponential map from so(3)."""
    print("Testing exp_so3...")
    # Small angle -> identity
    q = exp_so3(np.array([1e-8, 0.0, 0.0], dtype=np.float64))
    assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-7, msg="exp_so3 small angle")

    # 90 deg around Z
    q = exp_so3(np.array([0.0, 0.0, np.pi/2], dtype=np.float64))
    expected = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2], dtype=np.float64)
    assert_allclose(q, expected, atol=1e-10, msg="exp_so3 90 deg Z")

    # Roundtrip: log not implemented, but we can check exp(axis*angle) == axis-angle
    axis = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = 0.7
    q1 = quat_from_axis_angle(axis, angle)
    q2 = exp_so3(axis * angle)
    assert_allclose(q1, q2, atol=1e-10, msg="exp_so3 vs axis-angle")
    print("  OK")


def test_slerp():
    """Test spherical linear interpolation."""
    print("Testing slerp...")
    qa = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    qb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # 180 deg around X

    # t=0 -> qa
    assert_allclose(slerp(qa, qb, 0.0), qa, msg="slerp t=0")
    # t=1 -> qb
    assert_allclose(slerp(qa, qb, 1.0), qb, msg="slerp t=1")
    # t=0.5 -> 90 deg around X
    q_mid = slerp(qa, qb, 0.5)
    expected = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0], dtype=np.float64)
    assert_allclose(q_mid, expected, atol=1e-10, msg="slerp t=0.5")

    # Short path (qa and -qb same rotation)
    qb_neg = -qb
    q_mid = slerp(qa, qb_neg, 0.5)
    # Should still take short path (same rotation as qb, but q or -q both valid)
    # Check that angle between q_mid and expected is 0 or 180 (same rotation)
    angle = angle_between_deg(q_mid, expected)
    assert np.isclose(angle, 0.0, atol=1e-10) or np.isclose(angle, 180.0, atol=1e-10), "slerp short path"

    # Near parallel
    qa = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    qb = np.array([0.9999, 0.01, 0.0, 0.0], dtype=np.float64)
    qb = normalize(qb)
    q_mid = slerp(qa, qb, 0.5)
    assert np.isclose(np.linalg.norm(q_mid), 1.0), "slerp near parallel unit norm"
    print("  OK")


def test_from_euler_zyx_deg():
    """Test Euler ZYX to quaternion."""
    print("Testing from_euler_zyx_deg...")
    # roll=90, pitch=0, yaw=0 -> 90 deg around X
    q = from_euler_zyx_deg(90, 0, 0)
    expected = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0], dtype=np.float64)
    assert_allclose(q, expected, atol=1e-10, msg="euler roll=90")

    # pitch=90 -> 90 deg around Y
    q = from_euler_zyx_deg(0, 90, 0)
    expected = np.array([np.sqrt(2)/2, 0.0, np.sqrt(2)/2, 0.0], dtype=np.float64)
    assert_allclose(q, expected, atol=1e-10, msg="euler pitch=90")

    # yaw=90 -> 90 deg around Z
    q = from_euler_zyx_deg(0, 0, 90)
    expected = np.array([np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2], dtype=np.float64)
    assert_allclose(q, expected, atol=1e-10, msg="euler yaw=90")

    # Known example: roll=90, pitch=0, yaw=0
    q = from_euler_zyx_deg(90, 0, 0)
    R = to_matrix(q)
    v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v_rot = R @ v
    # 90 deg around X: Y -> Z
    assert_allclose(v_rot, [0.0, 0.0, 1.0], atol=1e-10, msg="euler known example")
    print("  OK")


def test_to_from_matrix():
    """Test quaternion <-> matrix roundtrip."""
    print("Testing to_matrix / from_matrix...")

    # Identity
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    R = to_matrix(q)
    assert_allclose(R, np.eye(3), msg="to_matrix identity")
    q2 = from_matrix(R)
    assert_allclose(q2, q, atol=1e-10, msg="from_matrix identity")

    # Random quaternion
    q = normalize(np.array([0.7, 0.2, 0.3, 0.6], dtype=np.float64))
    R = to_matrix(q)
    # Check orthogonality
    assert_allclose(R @ R.T, np.eye(3), atol=1e-10, msg="to_matrix orthogonal")
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10), "to_matrix det=1"
    q2 = from_matrix(R)
    # q and -q represent same rotation
    if np.dot(q, q2) < 0:
        q2 = -q2
    assert_allclose(q2, q, atol=1e-10, msg="roundtrip random")

    # Multiple random tests
    np.random.seed(42)
    for _ in range(10):
        q = normalize(np.random.randn(4))
        R = to_matrix(q)
        q2 = from_matrix(R)
        if np.dot(q, q2) < 0:
            q2 = -q2
        assert_allclose(q2, q, atol=1e-10, msg=f"roundtrip iter {_}")

    # Degenerate matrix should raise
    assert_raises(ValueError, from_matrix, np.zeros((3, 3)))
    assert_raises(ValueError, from_matrix, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))  # det=-1
    assert_raises(ValueError, from_matrix, np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))  # not orthogonal
    print("  OK")


def test_angle_between_deg():
    """Test angle between quaternions."""
    print("Testing angle_between_deg...")
    qa = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    qb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert np.isclose(angle_between_deg(qa, qb), 0.0, atol=1e-10), "angle 0 deg"

    qb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # 180 deg around X
    assert np.isclose(angle_between_deg(qa, qb), 180.0, atol=1e-10), "angle 180 deg"

    qb = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0.0, 0.0], dtype=np.float64)  # 90 deg around X
    assert np.isclose(angle_between_deg(qa, qb), 90.0, atol=1e-10), "angle 90 deg"

    # q and -q should give 0 angle
    qb = -qa
    assert np.isclose(angle_between_deg(qa, qb), 0.0, atol=1e-10), "angle q vs -q"
    print("  OK")


def test_quat_from_two_vectors():
    """Test minimum rotation between two vectors."""
    print("Testing quat_from_two_vectors...")
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    q = quat_from_two_vectors(a, b)
    v_rot = rotate_vector(q, a)
    assert_allclose(v_rot, b, atol=1e-10, msg="two vectors 90 deg")

    # Parallel
    q = quat_from_two_vectors(a, a)
    assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-10, msg="two vectors parallel")

    # Anti-parallel
    q = quat_from_two_vectors(a, -a)
    v_rot = rotate_vector(q, a)
    assert_allclose(v_rot, -a, atol=1e-10, msg="two vectors anti-parallel")

    # Zero vectors should raise
    assert_raises(ValueError, quat_from_two_vectors, np.zeros(3), a)
    assert_raises(ValueError, quat_from_two_vectors, a, np.zeros(3))

    # Non-unit vectors (should normalize)
    a = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 3.0, 0.0], dtype=np.float64)
    q = quat_from_two_vectors(a, b)
    v_rot = rotate_vector(q, a / np.linalg.norm(a))
    assert_allclose(v_rot, b / np.linalg.norm(b), atol=1e-10, msg="two vectors non-unit")
    print("  OK")


def test_shape_validation():
    """Test shape validation errors."""
    print("Testing shape validation...")
    assert_raises(ValueError, normalize, np.array([1.0, 0.0, 0.0]))  # wrong shape
    assert_raises(ValueError, multiply, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert_raises(ValueError, conjugate, np.array([1.0, 0.0, 0.0]))
    assert_raises(ValueError, rotate_vector, np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0]))
    assert_raises(ValueError, from_matrix, np.eye(4))
    print("  OK")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running fusion.quaternion tests")
    print("=" * 50)

    test_normalize()
    test_multiply()
    test_conjugate()
    test_rotate_vector()
    test_quat_from_axis_angle()
    test_exp_so3()
    test_slerp()
    test_from_euler_zyx_deg()
    test_to_from_matrix()
    test_angle_between_deg()
    test_quat_from_two_vectors()
    test_shape_validation()

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