"""Teste da GRAVACAO das novas colunas (ARM_* posicao/angulo e PALM_*).

Nao usa hardware: monta o dicionario de movimento do SharedState, aplica o
mesmo caminho do RecordingMuxer (_motion_row, que monta a linha do CSV) e
confere coluna a coluna:

1. Ordem/tamanho: MOTION_COLUMNS = KT(3) + IMU(7) + KTT_valid(1) +
   ARM(18) + PALM(11) + KT_hand/KT_src(2) = 42, e o cabecalho do CSV.
2. Posicoes do braco sao RELATIVAS a origem; KT_* e IMU_* tambem.
3. Angulos articulares e versores da palma sao ABSOLUTOS (origem nao muda).
4. Sem medida -> NaN e flags de validade em 0 (nunca 0.0 fingindo ser medido).
5. A linha tem exatamente o numero de colunas anunciado no contexto de saida.
"""
import sys

import numpy as np

import eeg_motor_paradigm as p
import kinect_imu_groundtruth as kig

row_keys = p.MOTION_COLUMNS
assert len(row_keys) == len(set(row_keys)), "colunas duplicadas"
assert p.N_MOTION_COLS == 42, f"N_MOTION_COLS={p.N_MOTION_COLS}"
assert len(p.KT_COLUMNS) == 3 and len(p.IMU_COLUMNS) == 7
assert len(p.ARM_COLUMNS) == 18, len(p.ARM_COLUMNS)
assert len(p.PALM_COLUMNS) == 11, len(p.PALM_COLUMNS)

# O cabecalho do CSV e montado como Time + EEG_ChNN + MOTION_COLUMNS + Marker.
n_eeg = 32
header = (["Time"] + [f"EEG_Ch{i + 1:02d}" for i in range(n_eeg)]
          + list(row_keys) + ["Marker"])
assert header[1:33] == [f"EEG_Ch{i + 1:02d}" for i in range(32)]
assert header[33] == "KT_x_m"
assert header[39] == "IMU_pos_x_m", header[36:44]
assert header[41] == "IMU_pos_z_m"
assert header[42] == "ZERO_lock" and header[43] == "KTT_valid"
assert header[44] == "ARM_shoulder_x_m"
assert header[-4] == "PALM_valid" and header[-3] == "KT_hand"
assert header[-2] == "KT_src" and header[-1] == "Marker"


def motion_dict(arm=True, palm=True):
    """Dicionario de movimento do SharedState, como publish_motion recebe."""
    base = {
        "x": 0.31, "y": 1.02, "z": 1.44,
        "roll": 12.0, "pitch": -3.0, "yaw": 47.0,
        "valid": 1, "src": "triangulado", "esp_ts_us": 123.0, "pc_time": 0.0,
        "imu_pos": p._nan3(), "imu_vel": np.nan, "zero_lock": 0,
        "arm_shoulder": p._nan3(), "arm_elbow": p._nan3(),
        "arm_wrist": p._nan3(),
        "arm_elbow_angle": np.nan, "arm_shoulder_elev": np.nan,
        "arm_shoulder_azim": np.nan, "arm_upper_len": np.nan,
        "arm_fore_len": np.nan, "arm_valid": 0, "arm_ik": 0,
        "arm_clamped": 0, "arm_side": 0,
        "palm_normal": p._nan3(), "palm_dir": p._nan3(),
        "palm_elev": np.nan, "palm_azim": np.nan, "palm_pitch": np.nan,
        "palm_yaw": np.nan, "palm_valid": 0,
    }
    if arm:
        base.update(
            arm_shoulder=np.array([0.60, 1.35, 1.70]),
            arm_elbow=np.array([0.72, 1.18, 1.55]),
            arm_wrist=np.array([0.85, 1.05, 1.90]),
            arm_elbow_angle=97.5, arm_shoulder_elev=-32.0,
            arm_shoulder_azim=21.0, arm_upper_len=0.331, arm_fore_len=0.262,
            arm_valid=1, arm_ik=1, arm_clamped=0, arm_side=1)
    if palm:
        normal = np.array([0.10, -0.95, -0.22])
        normal = normal / np.linalg.norm(normal)
        direction = np.array([0.30, 0.24, 0.92])
        direction = direction / np.linalg.norm(direction)
        angles = kig.palm_angles(normal, direction)
        base.update(
            palm_normal=normal, palm_dir=direction,
            palm_elev=angles["elev"], palm_azim=angles["azim"],
            palm_pitch=angles["pitch"], palm_yaw=angles["yaw"], palm_valid=1)
    return base


# ---------------------------------------------------------------------------
# 1) Linha completa: ordem, posicoes relativas, angulos absolutos
# ---------------------------------------------------------------------------
muxer = p.RecordingMuxer.__new__(p.RecordingMuxer)   # sem a pipeline
motion = motion_dict()
origin = np.array([0.60, 1.30, 1.60])       # origem = mao na mesa
orpy = np.array([0.0, 0.0, 0.0])

row = p.build_motion_row(motion, origin, orpy)
assert isinstance(row, list) and len(row) == p.N_MOTION_COLS, len(row)
col = dict(zip(row_keys, row))

# KT_* e ARM_* relativos a origem
assert abs(col["KT_x_m"] - (0.31 - 0.60)) < 1e-9
assert abs(col["ARM_shoulder_y_m"] - (1.35 - 1.30)) < 1e-9
assert abs(col["ARM_elbow_z_m"] - (1.55 - 1.60)) < 1e-9
assert abs(col["ARM_wrist_x_m"] - (0.85 - 0.60)) < 1e-9
# posicao FUSIONADA do IMU: relativa a origem; zeragem gravada como 0/1
assert np.isnan(col["IMU_pos_x_m"])            # imu_pos NaN -> NaN no CSV
assert col["ZERO_lock"] in (0.0, 1.0)
# angulos articulares e elos efetivos NAO sofrem com a origem
assert col["ARM_elbow_angle_deg"] == 97.5
assert col["ARM_shoulder_elev_deg"] == -32.0
assert abs(col["ARM_upper_len_m"] - 0.331) < 1e-12
assert abs(col["ARM_fore_len_m"] - 0.262) < 1e-12
# flags de diagnostico da IK
assert col["ARM_valid"] == 1 and col["ARM_ik"] == 1
assert col["ARM_clamped"] == 0 and col["ARM_side"] == 1
# versores da palma absolutos (unitarios, sinal apontando para a camera)
normal = np.array([col["PALM_nx"], col["PALM_ny"], col["PALM_nz"]])
direction = np.array([col["PALM_dirx"], col["PALM_diry"], col["PALM_dirz"]])
assert abs(np.linalg.norm(normal) - 1.0) < 1e-9, normal
assert abs(np.linalg.norm(direction) - 1.0) < 1e-9, direction
assert normal[2] < 0, "normal deve apontar para a camera (nz < 0)"
expected = kig.palm_angles(normal, direction)
assert abs(col["PALM_elev_deg"] - expected["elev"]) < 1e-9
assert abs(col["PALM_azim_deg"] - expected["azim"]) < 1e-9
assert abs(col["PALM_pitch_deg"] - expected["pitch"]) < 1e-9
assert abs(col["PALM_yaw_deg"] - expected["yaw"]) < 1e-9
assert col["PALM_valid"] == 1
# Lateralidade e fonte da posicao 3D (KT_hand / KT_src): sempre presentes na
# linha, com 0 quando o dicionario ainda nao traz a informacao.
assert col["KT_hand"] == 0 and col["KT_src"] == 0
motion_lado = motion_dict()
motion_lado["hand"] = kig.HAND_SIDE_CODE["left"]          # 2 = esquerda
motion_lado["src_code"] = p.kt_src_code("triangulado_laptop")
col_lado = dict(zip(row_keys, p.build_motion_row(motion_lado, origin, orpy)))
assert col_lado["KT_hand"] == 2.0 and col_lado["KT_src"] == 1.0


# ---------------------------------------------------------------------------
# 2) Origem de ORIENTACAO: o IMU tem a origem subtraida (wrap +-180); os
#    versores/angulos da palma e os angulos articulares NAO mudam.
# ---------------------------------------------------------------------------
row2 = p.build_motion_row(motion_dict(), origin, np.array([10.0, 0.0, 20.0]))
col2 = dict(zip(row_keys, row2))
assert abs(col2["IMU_roll_deg"] - 2.0) < 1e-9, col2["IMU_roll_deg"]
assert abs(col2["IMU_yaw_deg"] - 27.0) < 1e-9, col2["IMU_yaw_deg"]
assert abs(col2["PALM_azim_deg"] - col["PALM_azim_deg"]) < 1e-12
assert col2["ARM_elbow_angle_deg"] == col["ARM_elbow_angle_deg"]
# wrap: 170 graus relativos a origem 170 -> 0 (nao -340)
motion_wrap = motion_dict()
motion_wrap["yaw"] = 170.0
col_wrap = dict(zip(row_keys, p.build_motion_row(
    motion_wrap, origin, np.array([0.0, 0.0, 170.0]))))
assert abs(col_wrap["IMU_yaw_deg"]) < 1e-9, col_wrap["IMU_yaw_deg"]


# ---------------------------------------------------------------------------
# 3) Sem braco/palma (Kinect perdeu a mao ou sem corpo rastreado):
#    NaN nas medidas + flags de validade em 0 (nunca 0.0 "fingindo" medida).
# ---------------------------------------------------------------------------
row3 = p.build_motion_row(motion_dict(arm=False, palm=False), origin, orpy)
col3 = dict(zip(row_keys, row3))
assert len(row3) == p.N_MOTION_COLS
for key in ("ARM_shoulder_x_m", "ARM_elbow_y_m", "ARM_wrist_z_m",
            "ARM_elbow_angle_deg", "ARM_shoulder_elev_deg",
            "ARM_shoulder_azim_deg", "ARM_upper_len_m", "ARM_fore_len_m",
            "PALM_nx", "PALM_nz", "PALM_diry",
            "PALM_elev_deg", "PALM_azim_deg", "PALM_pitch_deg",
            "PALM_yaw_deg"):
    assert np.isnan(col3[key]), f"{key} deveria ser NaN: {col3[key]}"
for key in ("ARM_valid", "ARM_ik", "ARM_clamped", "ARM_side", "PALM_valid"):
    assert col3[key] == 0.0, f"{key} deveria ser 0: {col3[key]}"
# KT_* do punho continuam validos (o braco pode falhar sem perder a mao)
assert abs(col3["KT_x_m"] - (0.31 - 0.60)) < 1e-9
assert col3["KTT_valid"] == 1.0


# ---------------------------------------------------------------------------
# 4) Sem origem ainda (antes do evento 501): valores ABSOLUTOS, sem subtracao.
# ---------------------------------------------------------------------------
row4 = p.build_motion_row(motion_dict(), None, None)
col4 = dict(zip(row_keys, row4))
assert abs(col4["KT_x_m"] - 0.31) < 1e-9
assert abs(col4["ARM_shoulder_x_m"] - 0.60) < 1e-9
assert abs(col4["ARM_wrist_z_m"] - 1.90) < 1e-9
assert abs(col4["IMU_roll_deg"] - 12.0) < 1e-9
assert col4["PALM_valid"] == 1


# ---------------------------------------------------------------------------
# 5) Largura total: Time(1) + EEG(32) + 42 movimento + Marker(1) = 76 colunas.
#    (Com o movimento em arquivo separado o CSV de EEG sai com 34 colunas;
#    aqui medimos o formato EMBUTIDO, que o muxer anuncia quando write_motion.)
# ---------------------------------------------------------------------------
assert n_eeg + p.N_MOTION_COLS + 1 == 75        # canais que o muxer anuncia
assert 1 + n_eeg + p.N_MOTION_COLS + 1 == 76    # + a coluna Time do CsvWriter
assert len(header) == 76, len(header)
print("ARM_CSV_OK: 5/5 blocos (42 colunas de movimento; formato embutido de "
      "76 colunas; padrao = 34 colunas + CSV de movimento separado)")
sys.exit(0)