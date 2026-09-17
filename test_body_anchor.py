"""Teste do body_hand_anchor: selecao de junta, filtros e prioridade.

Mocka o body frame do SDK (PyKinectV2) sem hardware. Verifica:
1. Pulso tracked com Z valido -> retorna (x, y, z, nome).
2. Junta Inferred -> ignora (so Tracked conta).
3. Z fora da faixa (0.3-4.0 m) -> ignora e tenta a proxima junta.
4. Nenhum corpo tracked -> None.
"""
import sys
import types
from types import SimpleNamespace

import kinect_imu_groundtruth as kig


def make_body(joints_spec, tracked=True):
    """joints_spec: dict {JointType: (x, y, z, tracking_state)}"""
    joints = {}
    for jtype, (x, y, z, ts) in joints_spec.items():
        joints[jtype] = SimpleNamespace(
            Position=SimpleNamespace(x=x, y=y, z=z), TrackingState=ts
        )
    return SimpleNamespace(is_tracked=tracked, joints=joints)


def make_bodies(bodies):
    return SimpleNamespace(bodies=bodies)


TS_TRACKED = kig.PyKinectV2.TrackingState_Tracked
TS_INFERRED = kig.PyKinectV2.TrackingState_Inferred
WR = kig.PyKinectV2.JointType_WristRight
WL = kig.PyKinectV2.JointType_WristLeft
HR = kig.PyKinectV2.JointType_HandTipRight

tracker = kig.KinectHandTracker.__new__(kig.KinectHandTracker)  # sem __init__ (sem hardware)

# 1) Pulso direito tracked, Z valido
tracker.last_bodies = make_bodies(
    [make_body({WR: (0.21, -0.10, 1.05, TS_TRACKED)})]
)
r = tracker.body_hand_anchor()
assert r is not None and r[3] == "wrist_right", f"caso 1 falhou: {r}"
assert abs(r[0] - 0.21) < 1e-9 and abs(r[2] - 1.05) < 1e-9, f"valores errados: {r}"

# 2) Pulso Inferred e HandTip tracked -> cai para handtip (reserva)
tracker.last_bodies = make_bodies(
    [make_body({WR: (0.2, 0.0, 1.0, TS_INFERRED), HR: (0.22, -0.11, 1.02, TS_TRACKED)})]
)
r = tracker.body_hand_anchor()
assert r is not None and r[3] == "handtip_right", f"caso 2 falhou: {r}"

# 3) Z fora da faixa no pulso direito -> usa o esquerdo valido
tracker.last_bodies = make_bodies(
    [make_body({WR: (0.2, 0.0, 0.1, TS_TRACKED), WL: (-0.3, -0.1, 0.95, TS_TRACKED)})]
)
r = tracker.body_hand_anchor()
assert r is not None and r[3] == "wrist_left", f"caso 3 falhou: {r}"

# 4) Corpo nao tracked -> None
tracker.last_bodies = make_bodies(
    [make_body({WR: (0.21, -0.10, 1.05, TS_TRACKED)}, tracked=False)]
)
assert tracker.body_hand_anchor() is None, "caso 4 falhou"

# 5) Sem body frame -> None
tracker.last_bodies = None
assert tracker.body_hand_anchor() is None, "caso 5 falhou"

# 6) Dois corpos: so o tracked interessa
tracker.last_bodies = make_bodies(
    [
        make_body({WR: (9.9, 9.9, 9.9, TS_TRACKED)}, tracked=False),
        make_body({WR: (0.15, -0.05, 0.88, TS_TRACKED)}),
    ]
)
r = tracker.body_hand_anchor()
assert r is not None and abs(r[2] - 0.88) < 1e-9, f"caso 6 falhou: {r}"

print("BODY_ANCHOR_OK: 6/6 casos")
sys.exit(0)
