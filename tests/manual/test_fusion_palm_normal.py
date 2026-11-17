"""Testes do estabilizador de normal da palma (sem hardware)."""
from __future__ import annotations

import math

import numpy as np

from fusion.palm_normal import PalmNormalStabilizer, StabilizerConfig


def _angle(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = float(np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)),
                      -1.0, 1.0))
    return math.degrees(math.acos(d))


def _rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([[math.cos(r), 0.0, math.sin(r)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(r), 0.0, math.cos(r)]], float)


def test_jitter_reduction():
    print("TEST: jitter_reduction")
    est = PalmNormalStabilizer()
    base = np.array([0.0, 0.0, -1.0], float)
    saidas = []
    for i in range(60):
        deg = 6.0 if i % 2 == 0 else -6.0
        n = _rot_y(deg) @ base
        saidas.append(est.update("right", n, 0.002, 10.0 + i * 0.03))
    # O estabilizador so' deve ficar perto da media; a normal crua alterna +-6.
    desvios = [_angle(v.normal_cam, base) for v in saidas[-20:] if v.valid]
    assert desvios, "sem saidas validas"
    assert max(desvios) < 4.0, f"jitter nao reduziu: {max(desvios):.2f}"
    print(f"  PASS: desvio max={max(desvios):.2f} deg")


def test_sign_continuity():
    print("TEST: sign_continuity")
    est = PalmNormalStabilizer()
    n = np.array([0.1, 0.2, 0.97], float)
    est.update("right", n, 0.002, 1.0)
    out = est.update("right", -n, 0.002, 1.03)
    assert out.valid
    assert float(np.dot(out.normal_cam, n)) > 0.95
    print("  PASS: flip alinhado ao anterior")


def test_edge_on_confidence():
    print("TEST: edge_on_confidence")
    est = PalmNormalStabilizer()
    frontal = est.update("right", np.array([0.0, 0.0, -1.0]), 0.002, 1.0)
    edge = est.update("right", np.array([0.99, 0.0, 0.05]), 0.002, 1.03)
    assert not frontal.edge_on
    assert edge.edge_on
    assert edge.confidence < frontal.confidence
    assert edge.sigma_deg > frontal.sigma_deg
    print(f"  PASS: conf {frontal.confidence:.2f} -> {edge.confidence:.2f}; "
          f"sigma {frontal.sigma_deg:.1f} -> {edge.sigma_deg:.1f}")


def test_dropout_hold_and_reset():
    print("TEST: dropout_hold_and_reset")
    est = PalmNormalStabilizer(StabilizerConfig(hold_s=0.10, max_gap_s=0.20))
    n = np.array([0.0, 0.0, -1.0], float)
    est.update("right", n, 0.002, 5.0)
    hold = est.update("right", None, 0.0, 5.05)
    assert hold.valid and hold.held
    perdido = est.update("right", None, 0.0, 6.0)
    assert not perdido.valid
    print("  PASS: hold curto e reset longo")


def test_gap_reset():
    print("TEST: gap_reset")
    est = PalmNormalStabilizer(StabilizerConfig(max_gap_s=0.1))
    n1 = np.array([0.0, 0.0, -1.0], float)
    n2 = np.array([1.0, 0.0, 0.0], float)
    est.update("right", n1, 0.002, 1.0)
    out = est.update("right", n2, 0.002, 2.0)
    assert out.valid
    assert _angle(out.normal_cam, n2) < 1.0
    print("  PASS: gap reseta estado")


def main():
    test_jitter_reduction()
    test_sign_continuity()
    test_edge_on_confidence()
    test_dropout_hold_and_reset()
    test_gap_reset()
    print("\nPALM_NORMAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
