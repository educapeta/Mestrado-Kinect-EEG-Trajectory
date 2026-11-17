"""Teste da GRAVACAO SIMULTANEA dos dois IMUs (sem hardware, sem EEG, sem GUI).

Motivo: as duas luvas (ESP32 + MPU6050, uma por mao) precisam entrar no MESMO
registro, cada uma no seu bloco de colunas (IMU_L_*/IMU_R_*), sem trocar de lado
e sem misturar os relogios. Aqui dois EMISSORES falsos (tools/emissores_imu_
falsos.py) fazem o papel das luvas e o caminho real de gravacao e' exercitado:

    ImuBank (receptor + fusao por lado) -> blocos_motion() -> build_motion_row()

Casos:
1. `ImuBank` recebe dos dois emissores e separa por porta (cada lado com a SUA
   assinatura: direita roll=+5/1,2 g; esquerda roll=-5/1,0 g);
2. `blocos_motion()` roda a fusao de cada lado e devolve os dois blocos validos;
3. `build_motion_row()` grava cada lado no bloco certo, com IMU_hand correto;
4. um lado sem emissor sai NaN/valid=0 (nunca herda o dado do outro lado);
5. a linha continua com N_MOTION_COLS colunas (o contrato nao muda).
"""
import os
import sys
import time

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tools"))

import emissores_imu_falsos as falsos     # noqa: E402
import eeg_motor_paradigm as p            # noqa: E402
from imu import ImuBank                   # noqa: E402

PORTA_D, PORTA_E = 4510, 4511
LIMITE = 3.0


def espera(banco, lados, limite=LIMITE):
    """Espera ate' todos os lados terem amostra (ou o limite)."""
    fim = time.monotonic() + limite
    while time.monotonic() < fim:
        if all(banco.amostra(lado) is not None for lado in lados):
            return True
        time.sleep(0.05)
    return False


# 1) recepcao dos dois lados (emissores falsos)
threads, parada, sockets = falsos.emissores(
    [(1, PORTA_D), (2, PORTA_E)], hz=50.0, ruido=0.0)
banco = ImuBank(f"{PORTA_D},{PORTA_E}")
banco.start()
try:
    assert espera(banco, [1, 2]), "nao recebeu dos dois emissores"
    direita, esquerda = banco.amostra(1), banco.amostra(2)
    assert abs(direita.roll_deg - 5.0) < 1e-6, direita.roll_deg
    assert abs(esquerda.roll_deg + 5.0) < 1e-6, esquerda.roll_deg
    assert abs(direita.accel_g[2] - 1.2) < 1e-6, direita.accel_g
    assert abs(esquerda.accel_g[2] - 1.0) < 1e-6, esquerda.accel_g
    print("1) OK: os dois lados chegaram e cada um com a SUA assinatura "
          f"(direita roll={direita.roll_deg:+.1f} e "
          f"az={direita.accel_g[2]:.2f} g; esquerda roll="
          f"{esquerda.roll_deg:+.1f} e az={esquerda.accel_g[2]:.2f} g)")

    # 2) blocos por lado (fusao propria de cada um)
    blocos = banco.blocos_motion(lado_ativo=1, posicao_camera=None)
    assert set(blocos) == {1, 2}, blocos
    for lado in (1, 2):
        assert blocos[lado]["valid"] == 1, (lado, blocos[lado])
        assert np.isfinite(blocos[lado]["accel_g"]).all()
        assert np.isfinite(blocos[lado]["rpy_deg"]).all()
    assert blocos[1]["rpy_deg"][0] != blocos[2]["rpy_deg"][0]
    print("2) OK: blocos_motion com os dois lados validos e independentes")

    # 3) a LINHA do CSV: cada lado no seu bloco
    estado = p.SharedState()
    estado.motion.update({"x": 0.1, "y": 0.0, "z": -0.05, "valid": 1,
                          "hand": 2, "src_code": 1, "onset": 0})
    motion = dict(estado.motion)
    motion["imu"] = blocos
    motion["hand"] = 2                      # mao ativa: esquerda
    row = p.build_motion_row(motion, None, None)
    assert len(row) == p.N_MOTION_COLS, (len(row), p.N_MOTION_COLS)
    col = dict(zip(p.MOTION_COLUMNS, row))
    assert col["IMU_hand"] == 2.0
    assert abs(col["IMU_R_roll_deg"] - 5.0) < 1e-6, col["IMU_R_roll_deg"]
    assert abs(col["IMU_L_roll_deg"] + 5.0) < 1e-6, col["IMU_L_roll_deg"]
    assert abs(col["IMU_R_acc_z_g"] - 1.2) < 1e-6, col["IMU_R_acc_z_g"]
    assert abs(col["IMU_L_acc_z_g"] - 1.0) < 1e-6, col["IMU_L_acc_z_g"]
    assert col["IMU_R_valid"] == 1.0 and col["IMU_L_valid"] == 1.0
    print(f"3) OK: linha de {len(row)} colunas com R(roll="
          f"{col['IMU_R_roll_deg']:+.1f}, az={col['IMU_R_acc_z_g']:.2f} g) e "
          f"L(roll={col['IMU_L_roll_deg']:+.1f}, az={col['IMU_L_acc_z_g']:.2f}"
          f" g), IMU_hand={col['IMU_hand']:.0f}")

    # 4) lado sem emissor: NaN + valid=0 (nao herda nada)
    #    OBS: portas DIFERENTES das de cima -- dois sockets UDP na mesma porta
    #    (mesmo com SO_REUSEADDR) dividem os datagramas entre si.
    PORTA_SO_DIR = 4512
    banco2 = ImuBank(f"{PORTA_SO_DIR},{PORTA_E + 2}")
    banco2.start()
    try:
        threads2, parada2, sockets2 = falsos.emissores([(1, PORTA_SO_DIR)],
                                                    hz=50.0, ruido=0.0)
        try:
            assert espera(banco2, [1]), "nao recebeu do lado direito"
            blocos2 = banco2.blocos_motion(lado_ativo=1)
            assert blocos2[1]["valid"] == 1, blocos2[1]
            assert blocos2[2]["valid"] == 0, blocos2[2]
            assert not np.isfinite(blocos2[2]["accel_g"]).any()
            motion2 = dict(estado.motion)
            motion2["imu"] = blocos2
            col2 = dict(zip(p.MOTION_COLUMNS,
                            p.build_motion_row(motion2, None, None)))
            assert col2["IMU_R_valid"] == 1.0 and col2["IMU_L_valid"] == 0.0
            assert np.isnan(col2["IMU_L_roll_deg"])
            print("4) OK: com um so' emissor, o outro lado sai NaN e valid=0 "
                  "(sem herdar o dado do lado ativo)")
        finally:
            parada2.set()
            for thread in threads2:
                thread.join(timeout=0.5)
            for socket_udp in sockets2:
                socket_udp.close()
    finally:
        banco2.stop()
finally:
    parada.set()
    for thread in threads:
        thread.join(timeout=0.5)
    for socket_udp in sockets:
        socket_udp.close()
    banco.stop()

# 5) contrato de colunas inalterado
assert p.N_MOTION_COLS == 59, p.N_MOTION_COLS
print("5) OK: contrato de colunas preservado (59 em MOTION_COLUMNS)")

print("IMU_GRAVACAO_DUAL_OK: 5/5 casos (dois emissores por porta, blocos por "
      "lado, linha do CSV com cada lado no seu bloco, lado ausente NaN/valid=0, "
      "contrato de colunas)")
sys.exit(0)
