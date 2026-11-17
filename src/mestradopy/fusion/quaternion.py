"""
Quaternion operations for sensor fusion.

Conventions:
- Quaternion format: (w, x, y, z) as numpy float64 array of shape (4,)
- Hamilton multiplication: q1 ⊗ q2
- Rotation: v' = q ⊗ [0, v] ⊗ q*
- All functions are pure (no side effects)
"""

from __future__ import annotations
import numpy as np


def _ensure_float64(arr: np.ndarray) -> np.ndarray:
    """Ensure array is float64 and contiguous."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_quaternion(q: np.ndarray, name: str = "q") -> np.ndarray:
    """Validate quaternion shape and return as float64."""
    q = _ensure_float64(q)
    if q.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {q.shape}")
    return q


def _validate_vector3(v: np.ndarray, name: str = "v") -> np.ndarray:
    """Validate 3D vector shape and return as float64."""
    v = _ensure_float64(v)
    if v.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {v.shape}")
    return v


def _validate_matrix3x3(R: np.ndarray, name: str = "R") -> np.ndarray:
    """Validate 3x3 matrix shape and return as float64."""
    R = _ensure_float64(R)
    if R.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {R.shape}")
    return R


def normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit length.

    Args:
        q: Quaternion (w, x, y, z) of shape (4,)

    Returns:
        Normalized quaternion of shape (4,)
    """
    q = _validate_quaternion(q)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Cannot normalize zero quaternion")
    return q / norm


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions: a ⊗ b.

    Args:
        a: First quaternion (w, x, y, z) of shape (4,)
        b: Second quaternion (w, x, y, z) of shape (4,)

    Returns:
        Product quaternion of shape (4,)
    """
    a = _validate_quaternion(a, "a")
    b = _validate_quaternion(b, "b")

    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return _ensure_float64(np.array([w, x, y, z]))


def conjugate(q: np.ndarray) -> np.ndarray:
    """
    Conjugate of quaternion: q* = (w, -x, -y, -z).

    Args:
        q: Quaternion (w, x, y, z) of shape (4,)

    Returns:
        Conjugated quaternion of shape (4,)
    """
    q = _validate_quaternion(q)
    return _ensure_float64(np.array([q[0], -q[1], -q[2], -q[3]]))


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by quaternion q: v' = q ⊗ [0, v] ⊗ q*.

    Args:
        q: Unit quaternion (w, x, y, z) of shape (4,)
        v: 3D vector of shape (3,)

    Returns:
        Rotated 3D vector of shape (3,)
    """
    q = normalize(q)  # Ensure unit quaternion
    v = _validate_vector3(v)

    # Pure quaternion [0, v]
    v_q = _ensure_float64(np.array([0.0, v[0], v[1], v[2]]))

    # q ⊗ v_q ⊗ q*
    q_conj = conjugate(q)
    rotated = multiply(multiply(q, v_q), q_conj)

    return rotated[1:]  # Return vector part


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Create quaternion from axis-angle representation.

    Args:
        axis: Unit rotation axis of shape (3,)
        angle_rad: Rotation angle in radians

    Returns:
        Quaternion (w, x, y, z) of shape (4,)
    """
    axis = _validate_vector3(axis, "axis")
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        raise ValueError("Axis cannot be zero vector")
    axis = axis / axis_norm

    half_angle = angle_rad * 0.5
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    return _ensure_float64(np.array([
        cos_half,
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half
    ]))


def exp_so3(rotvec_rad: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) (rotation vector) to unit quaternion.

    Args:
        rotvec_rad: Rotation vector (axis * angle) of shape (3,)

    Returns:
        Quaternion (w, x, y, z) of shape (4,)
    """
    rotvec_rad = _validate_vector3(rotvec_rad, "rotvec_rad")
    theta = np.linalg.norm(rotvec_rad)

    if theta < 1e-12:
        # Small angle approximation
        return _ensure_float64(np.array([1.0, 0.0, 0.0, 0.0]))

    axis = rotvec_rad / theta
    return quat_from_axis_angle(axis, theta)


def slerp(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.

    Args:
        qa: Start quaternion (w, x, y, z) of shape (4,)
        qb: End quaternion (w, x, y, z) of shape (4,)
        t: Interpolation parameter in [0, 1]

    Returns:
        Interpolated quaternion of shape (4,)
    """
    qa = normalize(qa)
    qb = normalize(qb)

    if t <= 0.0:
        return qa
    if t >= 1.0:
        return qb

    # Compute cosine of angle between quaternions
    dot = np.dot(qa, qb)

    # Ensure shortest path (q and -q represent same rotation)
    if dot < 0.0:
        qb = -qb
        dot = -dot

    # Clamp for numerical stability
    dot = np.clip(dot, -1.0, 1.0)

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = qa + t * (qb - qa)
        return normalize(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * qa + s1 * qb
    return normalize(result)


def from_euler_zyx_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create quaternion from ZYX Euler angles (yaw, pitch, roll).

    Rotation order: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    This means: first roll around X, then pitch around Y, then yaw around Z.

    Args:
        roll: Roll angle in degrees (rotation around X)
        pitch: Pitch angle in degrees (rotation around Y)
        yaw: Yaw angle in degrees (rotation around Z)

    Returns:
        Quaternion (w, x, y, z) of shape (4,)
    """
    # Convert to radians
    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    y = np.deg2rad(yaw)

    # Half angles
    cr, cp, cy = np.cos(r * 0.5), np.cos(p * 0.5), np.cos(y * 0.5)
    sr, sp, sy = np.sin(r * 0.5), np.sin(p * 0.5), np.sin(y * 0.5)

    # ZYX order: q = qz * qy * qx
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return _ensure_float64(np.array([w, x, y, z]))


def to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion to 3x3 rotation matrix.

    Args:
        q: Unit quaternion (w, x, y, z) of shape (4,)

    Returns:
        3x3 rotation matrix
    """
    q = normalize(q)
    w, x, y, z = q

    # Hamilton convention rotation matrix
    R = _ensure_float64(np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)]
    ]))

    return R


def from_matrix(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to unit quaternion.

    Args:
        R: 3x3 rotation matrix (must be valid SO(3))

    Returns:
        Unit quaternion (w, x, y, z) of shape (4,)

    Raises:
        ValueError: If matrix is degenerate or not a valid rotation
    """
    R = _validate_matrix3x3(R)

    # Check orthogonality and determinant
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise ValueError("Matrix is not orthogonal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("Matrix determinant is not 1 (not a proper rotation)")

    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = _ensure_float64(np.array([w, x, y, z]))
    return normalize(q)


def angle_between_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    """
    Compute angle between two quaternions in degrees.

    Args:
        qa: First quaternion (w, x, y, z) of shape (4,)
        qb: Second quaternion (w, x, y, z) of shape (4,)

    Returns:
        Angle in degrees [0, 180]
    """
    qa = normalize(qa)
    qb = normalize(qb)

    dot = np.abs(np.dot(qa, qb))
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    return np.rad2deg(angle_rad)


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimum rotation quaternion between two unit vectors.

    Computes quaternion that rotates vector a to align with vector b.

    Args:
        a: Source unit vector of shape (3,)
        b: Target unit vector of shape (3,)

    Returns:
        Quaternion (w, x, y, z) of shape (4,)
    """
    a = _validate_vector3(a, "a")
    b = _validate_vector3(b, "b")

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0.0 or b_norm == 0.0:
        raise ValueError("Input vectors cannot be zero")

    a = a / a_norm
    b = b / b_norm

    # Handle parallel/anti-parallel cases
    dot = np.dot(a, b)
    if dot > 1.0 - 1e-12:
        # Vectors are parallel, no rotation needed
        return _ensure_float64(np.array([1.0, 0.0, 0.0, 0.0]))
    if dot < -1.0 + 1e-12:
        # Vectors are anti-parallel, 180 degree rotation around any perpendicular axis
        # Find a perpendicular vector
        if abs(a[0]) < 0.9:
            perp = _ensure_float64(np.array([1.0, 0.0, 0.0]))
        else:
            perp = _ensure_float64(np.array([0.0, 1.0, 0.0]))
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return quat_from_axis_angle(axis, np.pi)

    # General case
    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm

    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return quat_from_axis_angle(axis, angle)