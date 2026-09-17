"""Teste do ArmLinkModel: cinematica inversa de 2 elos com elos rigidos.

Sem hardware (juntas sinteticas no camera space do Kinect). Verifica:
1. Round-trip geometrico: com ombro+punho+comprimentos reais, o cotovelo
   reconstruido coincide com o cotovelo verdadeiro.
2. Otimizacao: cotovelo MEDIDO com elo errado (ruido do esqueleto) e
   corrigido pela IK para a posicao fisicamente coerente.
3. Saturacoes: punho fora do alcance (braco esticado) e perto demais.
4. Angulos: interior do cotovelo, elevacao e azimute do braco.
5. Calibracao + persistencia: mediana das amostras, arm_model.json e filtro
   de plausibilidade.
6. Sem modelo calibrado -> esqueleto cru (ik=0).
7. arm_joints: selecao de lado (cue) e juntas ausentes/inferred.
8. Orientacao da palma: normal/direcao dos dedos e angulos elev/azim/pitch/yaw.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import kinect_imu_groundtruth as kig


def link(base, direction, length):
    """Ponto a `length` metros de `base`, na direcao dada."""
    d = np.asarray(direction, np.float64)
    return np.asarray(base, np.float64) + d / np.linalg.norm(d) * length


def close(a, b, tol=1e-6):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) < tol


# ----------------------------------------------------------------- geometria
L1 = 0.33          # ombro -> cotovelo (m)
L2 = 0.26          # cotovelo -> punho (m)
SHOULDER = np.array([0.20, 1.30, 1.60])
ELBOW = link(SHOULDER, (0.35, -0.60, -0.30), L1)
WRIST = link(ELBOW, (0.20, -0.30, 0.85), L2)
model = kig.ArmLinkModel(path=Path(__file__).with_name("tmp_arm_test.json"))
model.l1, model.l2 = L1, L2
TS_TRACKED = kig.PyKinectV2.TrackingState_Tracked
TS_INFERRED = kig.PyKinectV2.TrackingState_Inferred

# 1) Round-trip: cotovelo medido correto -> IK reproduz o mesmo cotovelo
sol = model.solve_arm(SHOULDER, ELBOW, WRIST)
assert sol is not None, "caso 1: solve_arm devolveu None"
assert close(sol["elbow"], ELBOW, 1e-9), f"caso 1: {sol['elbow']} != {ELBOW}"
assert sol["ik"] == 1 and sol["clamped"] == 0, f"caso 1: ik={sol['ik']}"
assert abs(sol["l1_m"] - L1) < 1e-9 and abs(sol["l2_m"] - L2) < 1e-9

# 2) Otimizacao: cotovelo do esqueleto ERRADO (elo 25% maior) -> IK corrige
bad_elbow = link(SHOULDER, (0.35, -0.60, -0.30), L1 * 1.25)
sol_bad = model.solve_arm(SHOULDER, bad_elbow, WRIST)
assert sol_bad["ik"] == 1, "caso 2: deveria usar IK"
err_before = abs(np.linalg.norm(bad_elbow - SHOULDER) - L1)
err_after = abs(sol_bad["l1_m"] - L1)
assert err_before > 0.05 and err_after < 1e-9, (
    f"caso 2: erro antes={err_before:.4f} depois={err_after:.4f}")
# o cotovelo corrigido fica sobre o circulo de solucoes: a distancia ao
# punho tambem volta a ser L2 (geometria consistente)
assert abs(np.linalg.norm(sol_bad["elbow"] - WRIST) - L2) < 1e-9

# 3) Saturacoes: punho longe demais -> braco esticado e clamped=1
far_wrist = link(SHOULDER, (1.0, 0.0, 0.0), L1 + L2 + 0.15)
sol_far = model.solve_arm(SHOULDER, None, far_wrist)
assert sol_far["clamped"] == 1, "caso 3a: deveria saturar em alcance"
# o cotovelo continua a L1 do ombro (elo rigido respeitado)...
assert abs(sol_far["l1_m"] - L1) < 1e-6, sol_far["l1_m"]
# ...mas o punho medido esta inalcancavel, entao L2 aparente fica maior que
# o elo real (e a distancia medida punho-cotovelo, informativa no CSV).
assert sol_far["l2_m"] > L2, "caso 3a: l2 deveria exceder o elo"
assert np.linalg.norm(sol_far["elbow"] - far_wrist) == sol_far["l2_m"]
assert abs(sol_far["elbow_angle_deg"] - 180.0) < 1e-3, (
    f"caso 3a: angulo={sol_far['elbow_angle_deg']}")
# perto demais -> dobra no maximo, clamped=1
near_wrist = SHOULDER + np.array([0.01, 0.0, 0.0])
sol_near = model.solve_arm(SHOULDER, None, near_wrist)
assert sol_near["clamped"] == 1, "caso 3b: deveria saturar dobrado"

# 4) Angulos: braco apontando para cima -> elevacao +90; para baixo -> -90
up = model.solve_arm(np.zeros(3), np.array([0.0, L1, 0.0]),
                     np.array([0.0, L1 + L2, 0.0]))
assert abs(up["shoulder_elev_deg"] - 90.0) < 1e-6, up["shoulder_elev_deg"]
down = model.solve_arm(np.zeros(3), np.array([0.0, -L1, 0.0]),
                       np.array([0.0, -(L1 + L2), 0.0]))
assert abs(down["shoulder_elev_deg"] + 90.0) < 1e-6, down["shoulder_elev_deg"]
side = model.solve_arm(np.zeros(3), np.array([L1, 0.0, 0.0]),
                       np.array([L1 + L2, 0.0, 0.0]))
assert abs(side["shoulder_elev_deg"]) < 1e-6, side["shoulder_elev_deg"]
assert abs(side["shoulder_azim_deg"] - 90.0) < 1e-6, side["shoulder_azim_deg"]
# angulo interno do cotovelo com braco esticado = 180
straight_elbow = link(SHOULDER, (0.0, 1.0, 0.0), L1)
straight_wrist = link(straight_elbow, (0.0, 1.0, 0.0), L2)
assert abs(model.interior_angle_deg(SHOULDER, straight_elbow,
                                    straight_wrist) - 180.0) < 1e-4
# e com o punho na direcao oposta (dobrado 180) = 0
folded_wrist = link(straight_elbow, (0.0, -1.0, 0.0), L2)
assert abs(model.interior_angle_deg(SHOULDER, straight_elbow,
                                    folded_wrist)) < 1e-4

print("ARM_IK: casos 1-4 OK")

# 5) Calibracao + persistencia: mediana das amostras e filtro de plausibilidade
cal = kig.ArmLinkModel(path=Path(__file__).with_name("tmp_arm_test.json"))
rng = np.random.default_rng(11)
accepted = 0
for i in range(60):
    # ruido de +-2 cm nas juntas (tipo Kinect), direcoes variando
    noise = rng.normal(0.0, 0.01, 3)
    el = link(SHOULDER + noise, (0.3, -0.5, -0.4), L1)
    wr = link(el, (0.1, -0.2, 0.9), L2)
    if cal.add_sample(SHOULDER, el, wr, side="right"):
        accepted += 1
assert accepted == 60, f"caso 5: {accepted} amostras aceitas (esperado 60)"
# amostra implausivel (elo de 5 cm) deve ser rejeitada
bad = link(SHOULDER, (1.0, 0.0, 0.0), 0.05)
assert not cal.add_sample(SHOULDER, bad, link(bad, (1.0, 0.0, 0.0), 0.05)), \
    "caso 5: amostra implausivel deveria ser rejeitada"
assert cal.finish_calibration(min_samples=30), "caso 5: calibracao falhou"
assert abs(cal.l1 - L1) < 0.005 and abs(cal.l2 - L2) < 0.005, (
    f"caso 5: L1={cal.l1:.4f} L2={cal.l2:.4f}")
assert cal.ready and cal.reach_m is not None and cal.side == "right"
# recarregar do disco devolve os mesmos valores
reloaded = kig.ArmLinkModel(path=Path(__file__).with_name("tmp_arm_test.json"))
assert reloaded.ready, "caso 5: modelo nao recarregou"
assert abs(reloaded.l1 - cal.l1) < 1e-9 and abs(reloaded.l2 - cal.l2) < 1e-9
assert reloaded.bend_ref is not None and len(reloaded.bend_ref) == 3
# poucas amostras -> nao calibra (nao sobrescreve o modelo bom com lixo)
few = kig.ArmLinkModel(path=Path(__file__).with_name("tmp_arm_test.json"))
for _ in range(5):
    el = link(SHOULDER, (0.3, -0.5, -0.4), L1)
    few.add_sample(SHOULDER, el, link(el, (0.1, -0.2, 0.9), L2))
assert not few.finish_calibration(min_samples=30), "caso 5: deveria recusar"

# 6) Sem modelo calibrado -> esqueleto cru (ik=0), sem inventar cotovelo
raw = kig.ArmLinkModel(path=Path(__file__).with_name("tmp_arm_test_absent.json"))
assert not raw.ready, "caso 6: modelo inexistente nao deveria carregar"
sol_raw = raw.solve_arm(SHOULDER, ELBOW, WRIST)
assert sol_raw["ik"] == 0, f"caso 6: ik={sol_raw['ik']}"
assert close(sol_raw["elbow"], ELBOW), "caso 6: deve manter o cotovelo medido"
assert abs(sol_raw["elbow_angle_deg"]
           - kig.ArmLinkModel.interior_angle_deg(SHOULDER, ELBOW, WRIST)) < 1e-9

# 7) arm_joints: selecao de lado pelo cue + juntas ausentes
J = kig.PyKinectV2
MOCK = {
    "joints": {
        J.JointType_ShoulderRight: (0.20, 1.30, 1.60, TS_TRACKED),
        J.JointType_ElbowRight: (0.31, 1.10, 1.50, TS_TRACKED),
        J.JointType_WristRight: (0.36, 1.05, 1.80, TS_TRACKED),
        J.JointType_ShoulderLeft: (-0.20, 1.30, 1.60, TS_TRACKED),
        J.JointType_ElbowLeft: (-0.31, 1.10, 1.50, TS_INFERRED),
        J.JointType_WristLeft: (-0.36, 1.05, 1.80, TS_TRACKED),
    }
}
tracker = kig.KinectHandTracker.__new__(kig.KinectHandTracker)
tracker.last_bodies = SimpleNamespace(
    bodies=[SimpleNamespace(is_tracked=True, joints={
        jt: SimpleNamespace(Position=SimpleNamespace(x=v[0], y=v[1], z=v[2]),
                            TrackingState=v[3])
        for jt, v in MOCK["joints"].items()})])
right = tracker.arm_joints()
assert right is not None and right["side"] == "right", f"caso 7: {right}"
assert close(right["shoulder"], [0.20, 1.30, 1.60]), right["shoulder"]
assert right["tracked"] == {"shoulder": True, "elbow": True, "wrist": True}
left = tracker.arm_joints(prefer_side="left")
assert left is not None and left["side"] == "left", f"caso 7: {left}"
assert close(left["elbow"], [-0.31, 1.10, 1.50]), left["elbow"]
assert left["tracked"]["elbow"] is False, "caso 7: Inferred nao e tracked"
# sem o ombro -> lado indisponivel
tracker.last_bodies = SimpleNamespace(
    bodies=[SimpleNamespace(is_tracked=True, joints={
        J.JointType_ElbowRight: SimpleNamespace(
            Position=SimpleNamespace(x=0.3, y=1.1, z=1.5),
            TrackingState=TS_TRACKED)})])
assert tracker.arm_joints() is None, "caso 7: sem ombro deveria devolver None"
tracker.last_bodies = None
assert tracker.arm_joints() is None, "caso 7: sem body frame deveria ser None"

print("ARM_IK: casos 5-7 OK")

# 8) Orientacao da palma: frame (normal + direcao dos dedos) e angulos
W0 = np.zeros(3)                     # punho na origem, para o teste
# palma apoiada na mesa (virada para BAIXO): normal -Y, dedos para frente +Z
normal, direction = kig.palm_frame(W0, np.array([1.0, 0.0, 1.0]),
                                   np.array([-1.0, 0.0, 1.0]),
                                   np.array([0.0, 0.0, 1.0]))
assert close(np.abs(normal), [0.0, 1.0, 0.0]), normal
assert normal[1] < 0, f"palma para baixo deveria ter ny<0: {normal}"
assert close(direction, [0.0, 0.0, 1.0]), direction
angles = kig.palm_angles(normal, direction)
assert abs(angles["elev"] + 90.0) < 1e-6, angles["elev"]    # -90 = p/ baixo
assert abs(angles["pitch"]) < 1e-6 and abs(angles["yaw"]) < 1e-6
assert angles["valid"] == 1

# palma virada para CIMA: indicador e minimo trocados de lado -> normal +Y
normal_up, dir_up = kig.palm_frame(W0, np.array([-1.0, 0.0, 1.0]),
                                   np.array([1.0, 0.0, 1.0]),
                                   np.array([0.0, 0.0, 1.0]))
assert normal_up[1] > 0, normal_up
assert abs(kig.palm_angles(normal_up, dir_up)["elev"] - 90.0) < 1e-6

# normal lateral +X -> azimute +90 (palma virada para a direita)
normal_x, dir_x = kig.palm_frame(W0, np.array([0.0, 1.0, 1.0]),
                                 np.array([0.0, -1.0, 1.0]),
                                 np.array([0.0, 0.0, 1.0]))
assert close(normal_x, [1.0, 0.0, 0.0]), normal_x
assert abs(kig.palm_angles(normal_x, dir_x)["azim"] - 90.0) < 1e-6

# normal calculada virada para a camera (+Z) e INVERTIDA (sinal fixado nz<0)
normal_flip, _ = kig.palm_frame(W0, np.array([1.0, 0.0, 0.0]),
                                np.array([0.0, 1.0, 0.0]),
                                np.array([0.0, 0.0, 1.0]))
assert normal_flip[2] < 0, normal_flip
assert abs(abs(kig.palm_angles(normal_flip, None)["azim"]) - 180.0) < 1e-6

# sem os pontos do plano -> NaN e valid=0 (nunca 0.0 fingindo ser medida)
none_angles = kig.palm_angles(None, None)
assert none_angles["valid"] == 0
assert np.isnan([none_angles["elev"], none_angles["azim"],
                 none_angles["pitch"], none_angles["yaw"]]).all()
# sem a direcao dos dedos: normal ainda vale (elev/azim), pitch/yaw = NaN
partial = kig.palm_angles(np.array([0.0, -1.0, 0.0]), None)
assert partial["valid"] == 0 and abs(partial["elev"] + 90.0) < 1e-9
assert np.isnan(partial["pitch"]) and np.isnan(partial["yaw"])
# dedos saindo da normal (mao "de lado"): direcao fica no plano da palma
normal_side, dir_side = kig.palm_frame(W0, np.array([1.0, 0.0, 1.0]),
                                       np.array([-1.0, 0.0, 1.0]),
                                       np.array([0.0, 1.0, 1.0]))
assert abs(float(np.dot(normal_side, dir_side))) < 1e-9, (normal_side, dir_side)
print("ARM_IK: casos 8 OK")

Path(__file__).with_name("tmp_arm_test.json").unlink(missing_ok=True)
print("ARM_IK_OK: 8/8 casos")
