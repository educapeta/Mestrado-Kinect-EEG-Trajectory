"""
Tests for fusion/eeg_export.py - FUSED_* columns schema v1.

Run: python -m pytest test_fusion_eeg_export.py -v
Or:  python test_fusion_eeg_export.py
"""

from __future__ import annotations

import sys
import numpy as np

# Ensure we can import fusion package
sys.path.insert(0, r"C:\mestradopy")

from fusion.eeg_export import (
    FUSED_COLUMNS,
    FUSED_COLUMN_DOCS,
    FUSED_SCHEMA_VERSION,
    FUSION_ENABLED,
    FUSED_SOURCE_ENUM,
    FUSED_SOURCE_ENUM_REV,
    fused_columns,
    fused_column_docs,
    fused_row,
    fused_row_dict,
    append_fused_columns,
    validate_fused_row,
    make_synthetic_fused_state,
)
from fusion.contracts import FusedState


def test_fused_columns_unique_and_ordered():
    """FUSED_COLUMNS has 18 unique entries in canonical order."""
    cols = fused_columns()
    assert len(cols) == 18, f"Expected 18 columns, got {len(cols)}: {cols}"
    assert len(set(cols)) == 18, "Duplicate column names found"
    assert cols == FUSED_COLUMNS, "fused_columns() should return canonical FUSED_COLUMNS"
    # Check exact order matches architecture document
    expected = [
        "FUSED_q_w", "FUSED_q_x", "FUSED_q_y", "FUSED_q_z",
        "FUSED_n_x", "FUSED_n_y", "FUSED_n_z",
        "FUSED_dir_x", "FUSED_dir_y", "FUSED_dir_z",
        "FUSED_elev_deg", "FUSED_azim_deg", "FUSED_pitch_deg", "FUSED_yaw_deg",
        "FUSED_confidence", "FUSED_latency_ms", "FUSED_source", "FUSED_schema_ver",
    ]
    assert cols == expected, f"Column order mismatch:\n  got:      {cols}\n  expected: {expected}"


def test_fused_column_docs_complete():
    """Every column has documentation with units."""
    docs = fused_column_docs()
    assert set(docs.keys()) == set(FUSED_COLUMNS), "Docs keys mismatch columns"
    for col, doc in docs.items():
        assert isinstance(doc, str) and len(doc) > 0, f"Empty doc for {col}"
        # Basic sanity: should mention unit or purpose
        assert any(kw in doc.lower() for kw in ["unit", "degree", "meter", "millisecond", "enum", "version", "confidence", "latency", "quaternion", "normal", "direction", "elevation", "azimuth", "pitch", "yaw"]), f"Doc for {col} lacks unit/description: {doc}"


def test_schema_version_and_flag():
    """Schema version is 1, feature flag is False by default (V1)."""
    assert FUSED_SCHEMA_VERSION == 1
    assert FUSION_ENABLED is False, "FUSION_ENABLED should be False for V1 (enabled after Phase 4 validation)"


def test_source_enum():
    """Source enum matches architecture document."""
    assert FUSED_SOURCE_ENUM == {"fallback": 0, "imu_only": 1, "vision_only": 2, "fused": 3}
    assert FUSED_SOURCE_ENUM_REV == {0: "fallback", 1: "imu_only", 2: "vision_only", 3: "fused"}


def test_fused_row_with_synthetic_state():
    """fused_row produces 19 values from a valid FusedState."""
    state = make_synthetic_fused_state(
        hand="right",
        confidence=0.85,
        latency_ms=12.5,
        source="fused",
    )
    row = fused_row(state)
    assert len(row) == 18
    # All values should be finite for a valid synthetic state
    for i, val in enumerate(row):
        col = FUSED_COLUMNS[i]
        assert np.isfinite(val), f"Column {col} has non-finite value: {val}"
    # Check specific values
    assert row[FUSED_COLUMNS.index("FUSED_confidence")] == 0.85
    assert row[FUSED_COLUMNS.index("FUSED_latency_ms")] == 12.5
    assert row[FUSED_COLUMNS.index("FUSED_source")] == 3  # fused
    assert row[FUSED_COLUMNS.index("FUSED_schema_ver")] == 1


def test_fused_row_quaternion_unit_norm():
    """Quaternion in output row is unit norm."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    q_w = row[FUSED_COLUMNS.index("FUSED_q_w")]
    q_x = row[FUSED_COLUMNS.index("FUSED_q_x")]
    q_y = row[FUSED_COLUMNS.index("FUSED_q_y")]
    q_z = row[FUSED_COLUMNS.index("FUSED_q_z")]
    q_norm = np.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)
    assert np.isclose(q_norm, 1.0, atol=1e-6), f"Quaternion norm={q_norm}"


def test_fused_row_normal_and_dir_unit_norm():
    """Normal and direction vectors are unit norm in output row."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    for prefix in ("n", "dir"):
        x = row[FUSED_COLUMNS.index(f"FUSED_{prefix}_x")]
        y = row[FUSED_COLUMNS.index(f"FUSED_{prefix}_y")]
        z = row[FUSED_COLUMNS.index(f"FUSED_{prefix}_z")]
        norm = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(norm, 1.0, atol=1e-6), f"{prefix} vector norm={norm}"


def test_fused_row_euler_angles_computed():
    """Euler angles are computed from quaternion and are finite."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    for angle_col in ("FUSED_elev_deg", "FUSED_azim_deg", "FUSED_pitch_deg", "FUSED_yaw_deg"):
        val = row[FUSED_COLUMNS.index(angle_col)]
        assert np.isfinite(val), f"{angle_col} is not finite: {val}"
        # Angles should be in reasonable range
        assert -180 <= val <= 180, f"{angle_col} out of range: {val}"


def test_fused_row_with_nan_inputs():
    """fused_row handles NaN in state vectors gracefully (outputs NaN)."""
    # Create state with NaN quaternion
    state = make_synthetic_fused_state()
    state.q_body_cam = np.full(4, np.nan)
    state.normal_cam = np.full(3, np.nan)
    state.dir_cam = np.full(3, np.nan)
    row = fused_row(state)
    # Quaternion components should be NaN
    for c in ("w", "x", "y", "z"):
        assert np.isnan(row[FUSED_COLUMNS.index(f"FUSED_q_{c}")])
    # Normal and dir should be NaN
    for prefix in ("n", "dir"):
        for c in ("x", "y", "z"):
            assert np.isnan(row[FUSED_COLUMNS.index(f"FUSED_{prefix}_{c}")])
    # Angles should be NaN
    for angle_col in ("FUSED_elev_deg", "FUSED_azim_deg", "FUSED_pitch_deg", "FUSED_yaw_deg"):
        assert np.isnan(row[FUSED_COLUMNS.index(angle_col)])
    # But confidence, latency, source, schema_ver should still be valid
    assert row[FUSED_COLUMNS.index("FUSED_confidence")] == 0.9
    assert row[FUSED_COLUMNS.index("FUSED_latency_ms")] == 15.0
    assert row[FUSED_COLUMNS.index("FUSED_source")] == 3
    assert row[FUSED_COLUMNS.index("FUSED_schema_ver")] == 1


def test_fused_row_none_handling():
    """fused_row handles None vectors by producing NaN."""
    state = make_synthetic_fused_state()
    state.normal_cam = None  # type: ignore
    state.dir_cam = None  # type: ignore
    row = fused_row(state)
    for prefix in ("n", "dir"):
        for c in ("x", "y", "z"):
            assert np.isnan(row[FUSED_COLUMNS.index(f"FUSED_{prefix}_{c}")]), f"{prefix}_{c} should be NaN"


def test_fused_row_dict():
    """fused_row_dict returns dict with correct keys and values."""
    state = make_synthetic_fused_state(confidence=0.77, latency_ms=8.0, source="imu_only")
    d = fused_row_dict(state)
    assert set(d.keys()) == set(FUSED_COLUMNS)
    assert d["FUSED_confidence"] == 0.77
    assert d["FUSED_latency_ms"] == 8.0
    assert d["FUSED_source"] == 1  # imu_only
    assert d["FUSED_schema_ver"] == 1


def test_append_fused_columns():
    """append_fused_columns adds FUSED_* to existing header without mutation."""
    original = ["Time", "EEG_Ch01", "KT_x_m"]
    original_copy = list(original)
    new_header = append_fused_columns(original)
    assert original == original_copy, "Original header was mutated"
    assert new_header[:3] == original
    assert new_header[3:] == FUSED_COLUMNS
    assert len(new_header) == 3 + 18


def test_validate_fused_row_valid():
    """validate_fused_row passes for a correct row."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    result = validate_fused_row(row)
    assert result.ok, f"Valid row failed validation: {result.errors}"
    assert len(result.errors) == 0


def test_validate_fused_row_dict():
    """validate_fused_row works with dict input."""
    state = make_synthetic_fused_state()
    row_dict = fused_row_dict(state)
    result = validate_fused_row(row_dict)
    assert result.ok, f"Valid dict failed: {result.errors}"


def test_validate_fused_row_missing_columns():
    """validate_fused_row catches missing columns."""
    state = make_synthetic_fused_state()
    row_dict = fused_row_dict(state)
    del row_dict["FUSED_q_w"]
    result = validate_fused_row(row_dict)
    assert not result.ok
    assert any("Missing columns" in e for e in result.errors)


def test_validate_fused_row_wrong_length():
    """validate_fused_row catches wrong list length."""
    result = validate_fused_row([1.0, 2.0, 3.0])  # only 3 values
    assert not result.ok
    assert any("Expected 18 values" in e for e in result.errors)


def test_validate_fused_row_confidence_out_of_range():
    """validate_fused_row catches confidence outside [0,1]."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    row[FUSED_COLUMNS.index("FUSED_confidence")] = 1.5
    result = validate_fused_row(row)
    assert not result.ok
    assert any("confidence out of range" in e.lower() for e in result.errors)


def test_validate_fused_row_negative_latency():
    """validate_fused_row catches negative latency."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    row[FUSED_COLUMNS.index("FUSED_latency_ms")] = -5.0
    result = validate_fused_row(row)
    assert not result.ok
    assert any("latency" in e.lower() and "negative" in e.lower() for e in result.errors)


def test_validate_fused_row_invalid_source():
    """validate_fused_row catches invalid source enum."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    row[FUSED_COLUMNS.index("FUSED_source")] = 99
    result = validate_fused_row(row)
    assert not result.ok
    assert any("source" in e.lower() and "invalid" in e.lower() for e in result.errors)


def test_validate_fused_row_schema_version_warning():
    """validate_fused_row warns on schema version mismatch."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    row[FUSED_COLUMNS.index("FUSED_schema_ver")] = 99
    result = validate_fused_row(row)
    assert result.ok  # warning, not error
    assert any("schema_ver" in w and "99" in w for w in result.warnings)


def test_validate_fused_row_quaternion_norm_warning():
    """validate_fused_row warns on non-unit quaternion."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    # Make quaternion non-unit (scale by 2)
    for c in ("w", "x", "y", "z"):
        idx = FUSED_COLUMNS.index(f"FUSED_q_{c}")
        row[idx] = row[idx] * 2
    result = validate_fused_row(row)
    assert result.ok  # warning, not error
    assert any("quaternion not unit norm" in w.lower() for w in result.warnings)


def test_validate_fused_row_normal_norm_warning():
    """validate_fused_row warns on non-unit normal."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    for c in ("x", "y", "z"):
        idx = FUSED_COLUMNS.index(f"FUSED_n_{c}")
        row[idx] = row[idx] * 2
    result = validate_fused_row(row)
    assert result.ok
    assert any("palm normal not unit norm" in w.lower() for w in result.warnings)


def test_validate_fused_row_dir_norm_warning():
    """validate_fused_row warns on non-unit direction."""
    state = make_synthetic_fused_state()
    row = fused_row(state)
    for c in ("x", "y", "z"):
        idx = FUSED_COLUMNS.index(f"FUSED_dir_{c}")
        row[idx] = row[idx] * 2
    result = validate_fused_row(row)
    assert result.ok
    assert any("finger direction not unit norm" in w.lower() for w in result.warnings)


def test_make_synthetic_fused_state_defaults():
    """make_synthetic_fused_state produces valid state with defaults."""
    state = make_synthetic_fused_state()
    assert isinstance(state, FusedState)
    assert state.hand == "right"
    assert np.all(np.isfinite(state.q_body_cam))
    assert np.all(np.isfinite(state.normal_cam))
    assert np.all(np.isfinite(state.dir_cam))
    assert state.confidence == 0.9
    assert state.latency_ms == 15.0
    assert state.source == "fused"
    assert state.valid is True
    assert state.schema_ver == 1


def test_make_synthetic_fused_state_custom_quaternion():
    """make_synthetic_fused_state respects custom quaternion."""
    q = np.array([0.7071, 0.0, 0.7071, 0.0])  # 90 deg around Y
    q = q / np.linalg.norm(q)
    state = make_synthetic_fused_state(q_body_cam=q)
    assert np.allclose(state.q_body_cam, q, atol=1e-4)
    # Normal and dir should be rotated accordingly
    from fusion.quaternion import rotate_vector
    expected_n = rotate_vector(q, np.array([0.0, 0.0, 1.0]))
    expected_d = rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(state.normal_cam, expected_n, atol=1e-4)
    assert np.allclose(state.dir_cam, expected_d, atol=1e-4)


def test_all_source_enums_in_row():
    """All four source enum values produce valid rows."""
    for source_str, source_enum in FUSED_SOURCE_ENUM.items():
        state = make_synthetic_fused_state(source=source_str)
        row = fused_row(state)
        assert row[FUSED_COLUMNS.index("FUSED_source")] == source_enum
        result = validate_fused_row(row)
        assert result.ok, f"Source {source_str} failed: {result.errors}"


# =============================================================================
# Main entry point for direct execution
# =============================================================================

if __name__ == "__main__":
    import traceback

    tests = [
        test_fused_columns_unique_and_ordered,
        test_fused_column_docs_complete,
        test_schema_version_and_flag,
        test_source_enum,
        test_fused_row_with_synthetic_state,
        test_fused_row_quaternion_unit_norm,
        test_fused_row_normal_and_dir_unit_norm,
        test_fused_row_euler_angles_computed,
        test_fused_row_with_nan_inputs,
        test_fused_row_none_handling,
        test_fused_row_dict,
        test_append_fused_columns,
        test_validate_fused_row_valid,
        test_validate_fused_row_dict,
        test_validate_fused_row_missing_columns,
        test_validate_fused_row_wrong_length,
        test_validate_fused_row_confidence_out_of_range,
        test_validate_fused_row_negative_latency,
        test_validate_fused_row_invalid_source,
        test_validate_fused_row_schema_version_warning,
        test_validate_fused_row_quaternion_norm_warning,
        test_validate_fused_row_normal_norm_warning,
        test_validate_fused_row_dir_norm_warning,
        test_make_synthetic_fused_state_defaults,
        test_make_synthetic_fused_state_custom_quaternion,
        test_all_source_enums_in_row,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n=== {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)