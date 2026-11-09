"""Testes do MultiViewSource com fonte fake (sem hardware)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fusion.adapters import MultiViewSource


class FakeMultiView:
    def __init__(self):
        self.started = False
        self.last_kinect_bgra = None

    def start(self):
        self.started = True

    def close(self):
        self.started = False

    def poll(self):
        # Plano de 5 pontos do dorso, aproximadamente frontal.
        pts = np.array([
            [0.0, 0.0, 1.0],
            [0.05, 0.05, 1.0],
            [0.05, 0.00, 1.0],
            [0.05, -0.05, 1.0],
            [-0.05, 0.0, 1.0],
        ], dtype=np.float64)
        return {"right": SimpleNamespace(
            palm_points_3d=pts,
            palm_reproj_errors=np.full(5, 0.8, dtype=np.float64),
        )}


def main():
    fake = FakeMultiView()
    src = MultiViewSource(aux_index=0, source=fake)
    src.start()
    assert fake.started
    now, frame, obs = src.poll()
    assert isinstance(now, float)
    assert frame is None
    assert "right" in obs
    o = obs["right"]
    assert o.valid
    assert o.source == "multiview"
    assert abs(np.linalg.norm(o.palm_normal_cam) - 1.0) < 1e-6
    assert o.confidence > 0.5
    assert o.wrist_cam.shape == (3,)
    src.close()
    assert not fake.started
    print("MULTIVIEW_SOURCE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
