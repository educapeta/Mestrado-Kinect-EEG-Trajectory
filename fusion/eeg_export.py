"""
EEG export scaffolding for FUSED_* columns (schema v1).

This module provides column definitions and row-building utilities for integrating
the fused palm orientation into the EEG motor paradigm CSV output.
It has NO dependency on the EEG paradigm itself - pure data contracts.

Schema v1 (as defined in docs/ARQUITETURA_FUSAO.md §9.2):
- Quaternion body->camera: FUSED_q_w, FUSED_q_x, FUSED_q_y, FUSED_q_z (unit quaternion)
- Fused palm normal in camera frame: FUSED_n_x, FUSED_n_y, FUSED_n_z (unit vector)
- Fused finger direction in camera frame: FUSED_dir_x, FUSED_dir_y, FUSED_dir_z (unit vector)
- Euler angles (deg): FUSED_elev_deg, FUSED_azim_deg, FUSED_pitch_deg, FUSED_yaw_deg
- Confidence: FUSED_confidence [0, 1]
- Latency: FUSED_latency_ms (ms)
- Source: FUSED_source (int enum: 0=fallback, 1=imu_only, 2=vision_only, 3=fused)
- Schema version: FUSED_schema_ver (int, =1 for v1)

Feature flag concept: FUSION_ENABLED (constant, default False for V1).
Integration activates only after Phase 4 validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import FusedState

# =============================================================================
# Schema version and feature flag
# =============================================================================

FUSED_SCHEMA_VERSION = 1
"""Schema version for FUSED_* columns. Increment on breaking changes."""

FUSION_ENABLED = False
"""
Feature flag for EEG integration.
Set to True ONLY after Phase 4 validation passes.
Controlled by --fusion-enabled CLI flag in eeg_motor_paradigm.py (future).
"""

# Source enum mapping (matches ARQUITETURA_FUSAO.md §9.2 table)
FUSED_SOURCE_ENUM = {
    "fallback": 0,
    "imu_only": 1,
    "vision_only": 2,
    "fused": 3,
}
FUSED_SOURCE_ENUM_REV = {v: k for k, v in FUSED_SOURCE_ENUM.items()}


# =============================================================================
# FUSED_COLUMNS v1 - exact order as per architecture document
# =============================================================================

FUSED_COLUMNS: list[str] = [
    # Quaternion body -> camera (w, x, y, z)
    "FUSED_q_w",
    "FUSED_q_x",
    "FUSED_q_y",
    "FUSED_q_z",
    # Fused palm normal in camera frame (unit vector)
    "FUSED_n_x",
    "FUSED_n_y",
    "FUSED_n_z",
    # Fused finger direction in camera frame (unit vector)
    "FUSED_dir_x",
    "FUSED_dir_y",
    "FUSED_dir_z",
    # Euler angles derived from fused quaternion (degrees)
    "FUSED_elev_deg",    # elevation of palm normal (+90 = palm up)
    "FUSED_azim_deg",    # azimuth of palm normal (+90 = normal right)
    "FUSED_pitch_deg",   # pitch of finger direction (+90 = fingers up)
    "FUSED_yaw_deg",     # yaw of finger direction (+90 = fingers right, 0 = forward)
    # Fusion quality metrics
    "FUSED_confidence",      # [0, 1] combined confidence
    "FUSED_latency_ms",      # processing latency in milliseconds
    "FUSED_source",          # int enum: 0=fallback, 1=imu_only, 2=vision_only, 3=fused
    "FUSED_schema_ver",      # schema version (1)
]

# Human-readable documentation for each column (for CSV headers / JSON metadata)
FUSED_COLUMN_DOCS: dict[str, str] = {
    "FUSED_q_w": "Quaternion w component (body->camera frame), unit quaternion",
    "FUSED_q_x": "Quaternion x component (body->camera frame), unit quaternion",
    "FUSED_q_y": "Quaternion y component (body->camera frame), unit quaternion",
    "FUSED_q_z": "Quaternion z component (body->camera frame), unit quaternion",
    "FUSED_n_x": "Fused palm normal X in camera frame (unit vector, meters/meters)",
    "FUSED_n_y": "Fused palm normal Y in camera frame (unit vector, meters/meters)",
    "FUSED_n_z": "Fused palm normal Z in camera frame (unit vector, meters/meters)",
    "FUSED_dir_x": "Fused finger direction X in camera frame (unit vector)",
    "FUSED_dir_y": "FUSED finger direction Y in camera frame (unit vector)",
    "FUSED_dir_z": "FUSED finger direction Z in camera frame (unit vector)",
    "FUSED_elev_deg": "Elevation angle of fused palm normal (degrees, +90=palm up)",
    "FUSED_azim_deg": "Azimuth angle of fused palm normal (degrees, +90=normal right)",
    "FUSED_pitch_deg": "Pitch angle of fused finger direction (degrees, +90=fingers up)",
    "FUSED_yaw_deg": "Yaw angle of fused finger direction (degrees, +90=fingers right)",
    "FUSED_confidence": "Fusion confidence [0, 1] (combined IMU + vision)",
    "FUSED_latency_ms": "Estimated fusion processing latency (milliseconds)",
    "FUSED_source": "Source enum: 0=fallback, 1=imu_only, 2=vision_only, 3=fused",
    "FUSED_schema_ver": "Schema version of FUSED_* columns (1)",
}


def fused_columns() -> list[str]:
    """
    Return the list of FUSED_* column names in canonical order (schema v1).

    Returns:
        List of 19 column names matching FUSED_COLUMNS.
    """
    return list(FUSED_COLUMNS)


def fused_column_docs() -> dict[str, str]:
    """
    Return human-readable documentation for each FUSED_* column.

    Returns:
        Dict mapping column name -> description with units.
    """
    return dict(FUSED_COLUMN_DOCS)


# =============================================================================
# Angle computation helpers (pure functions, no EEG dependency)
# =============================================================================

def _quat_to_elev_azim(q: np.ndarray) -> tuple[float, float]:
    """
    Compute elevation and azimuth of the palm normal from quaternion.

    Assumes quaternion represents body->camera rotation.
    Palm normal in body frame is +Z (0, 0, 1).
    Elevation: angle from XY plane, +90 = pointing up (toward +Y in camera? wait)
    Azimuth: angle in XY plane, +90 = pointing toward +X (right in camera frame)

    Camera frame (Kinect): X right, Y down, Z into scene.
    Palm normal +Z in body -> after rotation, n_cam = q * (0,0,1) * q^-1 = rotate_vector(q, [0,0,1])
    """
    from .quaternion import rotate_vector

    n_cam = rotate_vector(q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    # Elevation: angle from horizontal plane (XY), positive up
    # In camera frame, Y is down, so -n_y is "up"
    elev = np.degrees(np.arcsin(np.clip(-n_cam[1], -1.0, 1.0)))
    # Azimuth: angle in XZ plane (horizontal), 0 = forward (+Z), +90 = right (+X)
    azim = np.degrees(np.arctan2(n_cam[0], n_cam[2]))
    return float(elev), float(azim)


def _quat_to_pitch_yaw_dir(q: np.ndarray, dir_body: np.ndarray | None = None) -> tuple[float, float]:
    """
    Compute pitch and yaw of finger direction from quaternion.

    Finger direction in body frame: +X (1, 0, 0) typically.
    If dir_body not provided, assume +X.
    """
    from .quaternion import rotate_vector

    if dir_body is None:
        dir_body = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    d_cam = rotate_vector(q, dir_body)
    # Pitch: angle from horizontal (XZ plane), positive up
    pitch = np.degrees(np.arcsin(np.clip(-d_cam[1], -1.0, 1.0)))
    # Yaw: angle in XZ plane, 0 = forward (+Z), +90 = right (+X)
    yaw = np.degrees(np.arctan2(d_cam[0], d_cam[2]))
    return float(pitch), float(yaw)


def _normalize_if_finite(arr: np.ndarray | None) -> np.ndarray:
    """Normalize a 3D vector if finite, else return NaN vector."""
    if arr is None:
        return np.full(3, np.nan, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if np.all(np.isfinite(arr)) and np.linalg.norm(arr) > 1e-10:
        return arr / np.linalg.norm(arr)
    return np.full(3, np.nan, dtype=np.float64)


# =============================================================================
# Core row-building function
# =============================================================================

def fused_row(state: FusedState) -> list[float | int]:
    """
    Build a single CSV row (list of values) from a FusedState for schema v1.

    Args:
        state: FusedState from fusion pipeline (contracts.py).

    Returns:
        List of 19 values in FUSED_COLUMNS order, ready for CSV writer.
        NaN for invalid/missing vector components; 0 for invalid confidence/latency.
        schema_ver is always FUSED_SCHEMA_VERSION (1).
    """
    # Quaternion (w, x, y, z)
    q = state.q_body_cam
    if np.all(np.isfinite(q)):
        q_w, q_x, q_y, q_z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    else:
        q_w = q_x = q_y = q_z = float("nan")

    # Palm normal (n_x, n_y, n_z) - unit vector in camera frame
    n = _normalize_if_finite(state.normal_cam)
    n_x, n_y, n_z = float(n[0]), float(n[1]), float(n[2])

    # Finger direction (dir_x, dir_y, dir_z) - unit vector in camera frame
    d = _normalize_if_finite(state.dir_cam)
    dir_x, dir_y, dir_z = float(d[0]), float(d[1]), float(d[2])

    # Euler angles (degrees) - compute from quaternion if vectors are NaN
    if np.isfinite(q).all():
        elev_deg, azim_deg = _quat_to_elev_azim(q)
        pitch_deg, yaw_deg = _quat_to_pitch_yaw_dir(q)
    else:
        elev_deg = azim_deg = pitch_deg = yaw_deg = float("nan")

    # Confidence [0, 1]
    conf = float(state.confidence) if np.isfinite(state.confidence) else 0.0
    conf = max(0.0, min(1.0, conf))

    # Latency (ms)
    latency = float(state.latency_ms) if np.isfinite(state.latency_ms) else 0.0
    latency = max(0.0, latency)

    # Source enum
    source_str = state.source.lower().strip() if state.source else "fallback"
    source_enum = FUSED_SOURCE_ENUM.get(source_str, 0)

    # Schema version
    schema_ver = int(state.schema_ver) if state.schema_ver > 0 else FUSED_SCHEMA_VERSION

    return [
        q_w, q_x, q_y, q_z,
        n_x, n_y, n_z,
        dir_x, dir_y, dir_z,
        elev_deg, azim_deg, pitch_deg, yaw_deg,
        conf, latency, source_enum, schema_ver,
    ]


def fused_row_dict(state: FusedState) -> dict[str, float | int]:
    """
    Build a dict mapping column name -> value from a FusedState.

    Convenience for DictWriter or JSON serialization.

    Args:
        state: FusedState from fusion pipeline.

    Returns:
        Dict with keys = FUSED_COLUMNS, values = fused_row(state).
    """
    values = fused_row(state)
    return dict(zip(FUSED_COLUMNS, values))


# =============================================================================
# CSV header manipulation
# =============================================================================

def append_fused_columns(header: list[str]) -> list[str]:
    """
    Append FUSED_* columns to an existing motion header list.

    Does NOT modify the input list; returns a new list.

    Args:
        header: Existing column header list (e.g., MOTION_COLUMNS from eeg_motor_paradigm.py).

    Returns:
        New list: header + FUSED_COLUMNS.
    """
    return list(header) + fused_columns()


# =============================================================================
# Validation
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating a fused row."""
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_fused_row(row: dict[str, Any] | list[float | int]) -> ValidationResult:
    """
    Validate a fused row (dict or list) against schema v1 expectations.

    Checks:
    - All 19 columns present (if dict) or correct length (if list)
    - Quaternion is unit norm (if finite)
    - Normal and direction are unit vectors (if finite)
    - Confidence in [0, 1]
    - Latency >= 0
    - Source enum in {0, 1, 2, 3}
    - schema_ver == 1

    Args:
        row: Dict mapping column names to values, or list of 19 values in order.

    Returns:
        ValidationResult with ok=True if all checks pass.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Convert to dict for uniform handling
    if isinstance(row, list):
        if len(row) != len(FUSED_COLUMNS):
            errors.append(f"Expected {len(FUSED_COLUMNS)} values, got {len(row)}")
            return ValidationResult(False, errors, warnings)
        row_dict = dict(zip(FUSED_COLUMNS, row))
    else:
        row_dict = dict(row)

    # Check all columns present
    missing = [c for c in FUSED_COLUMNS if c not in row_dict]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Quaternion unit norm check
    q_vals = [row_dict.get(f"FUSED_q_{c}") for c in ("w", "x", "y", "z")]
    if all(v is not None and np.isfinite(v) for v in q_vals):
        q_norm = np.linalg.norm(q_vals)
        if not np.isclose(q_norm, 1.0, atol=1e-3):
            warnings.append(f"Quaternion not unit norm: ||q||={q_norm:.6f} (expected 1.0)")

    # Normal unit vector check
    n_vals = [row_dict.get(f"FUSED_n_{c}") for c in ("x", "y", "z")]
    if all(v is not None and np.isfinite(v) for v in n_vals):
        n_norm = np.linalg.norm(n_vals)
        if not np.isclose(n_norm, 1.0, atol=1e-3):
            warnings.append(f"Palm normal not unit norm: ||n||={n_norm:.6f} (expected 1.0)")

    # Direction unit vector check
    d_vals = [row_dict.get(f"FUSED_dir_{c}") for c in ("x", "y", "z")]
    if all(v is not None and np.isfinite(v) for v in d_vals):
        d_norm = np.linalg.norm(d_vals)
        if not np.isclose(d_norm, 1.0, atol=1e-3):
            warnings.append(f"Finger direction not unit norm: ||d||={d_norm:.6f} (expected 1.0)")

    # Confidence range
    conf = row_dict.get("FUSED_confidence")
    if conf is not None and np.isfinite(conf):
        if not (0.0 <= conf <= 1.0):
            errors.append(f"FUSED_confidence out of range [0,1]: {conf}")

    # Latency non-negative
    lat = row_dict.get("FUSED_latency_ms")
    if lat is not None and np.isfinite(lat):
        if lat < 0:
            errors.append(f"FUSED_latency_ms negative: {lat}")

    # Source enum
    src = row_dict.get("FUSED_source")
    if src is not None:
        try:
            src_int = int(src)
            if src_int not in FUSED_SOURCE_ENUM_REV:
                errors.append(f"FUSED_source invalid enum: {src_int} (expected 0,1,2,3)")
        except (ValueError, TypeError):
            errors.append(f"FUSED_source not integer: {src}")

    # Schema version
    schema = row_dict.get("FUSED_schema_ver")
    if schema is not None:
        try:
            if int(schema) != FUSED_SCHEMA_VERSION:
                warnings.append(f"FUSED_schema_ver={schema} != expected {FUSED_SCHEMA_VERSION}")
        except (ValueError, TypeError):
            errors.append(f"FUSED_schema_ver not integer: {schema}")

    return ValidationResult(len(errors) == 0, errors, warnings)


# =============================================================================
# Convenience: synthetic state for testing
# =============================================================================

def make_synthetic_fused_state(
    hand: str = "right",
    q_body_cam: np.ndarray | None = None,
    normal_cam: np.ndarray | None = None,
    dir_cam: np.ndarray | None = None,
    confidence: float = 0.9,
    latency_ms: float = 15.0,
    source: str = "fused",
    valid: bool = True,
    schema_ver: int = FUSED_SCHEMA_VERSION,
) -> FusedState:
    """
    Create a synthetic FusedState for testing without hardware.

    Args:
        hand: "left" or "right"
        q_body_cam: Quaternion body->camera (w,x,y,z). Default: identity.
        normal_cam: Palm normal in camera frame. Default: computed from q.
        dir_cam: Finger direction in camera frame. Default: computed from q.
        confidence: Fusion confidence [0,1].
        latency_ms: Processing latency in ms.
        source: Source string ("fused", "imu_only", "vision_only", "fallback").
        valid: Whether state is valid.
        schema_ver: Schema version.

    Returns:
        FusedState with sensible defaults.
    """
    from .quaternion import rotate_vector

    if q_body_cam is None:
        q_body_cam = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q_body_cam = np.asarray(q_body_cam, dtype=np.float64)
        q_body_cam = q_body_cam / np.linalg.norm(q_body_cam)

    if normal_cam is None:
        normal_cam = rotate_vector(q_body_cam, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    else:
        normal_cam = np.asarray(normal_cam, dtype=np.float64)

    if dir_cam is None:
        dir_cam = rotate_vector(q_body_cam, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    else:
        dir_cam = np.asarray(dir_cam, dtype=np.float64)

    return FusedState(
        timestamp=time.monotonic(),
        hand=hand,
        q_body_cam=q_body_cam,
        normal_cam=normal_cam,
        dir_cam=dir_cam,
        confidence=confidence,
        source=source,
        latency_ms=latency_ms,
        valid=valid,
        schema_ver=schema_ver,
    )


# Need time for synthetic state
import time