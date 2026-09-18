"""Teste do ImuBank: DOIS IMUs (um por mao), cada um com receptor e fusao proprios.

Motivo (hardware de 18/09): havera dois MPU6050, um em cada mao, cada um com o
seu ESP32. Cada dispositivo tem o seu relogio, entao amostras de lados diferentes
nunca podem entrar no mesmo filtro; e o protocolo alterna as maos a cada trial,
logo as colunas IMU_* precisam dizer de QUAL mao sao.

Casos:
1. `parse_portas_imu`: uma porta (retrocompativel), duas portas e formato
   explicito 'lado:porta'; lado invalido levanta erro;
2. convencao do projeto: lado 1 = direita na porta 4210, lado 2 = esquerda na
   4211 (mesma convencao de ARM_SIDE_CODE);
3. com dois EMISSORES falsos (UDP, formato ASCII do ESP32): cada lado recebe a
   SUA amostra, com atitude e aceleracao corretas -- e' a atribuicao por porta;
4. as fusoes (zeragem/bias) e os filtros sao INDEPENDENTES por lado;
5. `resumo()`/`idade_s()` e parada limpa.
"""
import socket
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from imu import (IMU_PORTAS_PADRAO, ImuBank, parse_portas_imu,   # noqa: E402
                 rotation_matrix)

PORTA_D, PORTA_E = 4410, 4411


def envia(socket_udp, porta, timestamp_us, roll, pitch, yaw, accel_g,
          gyro_dps=(0.0, 0.0, 0.0)):
    """Datagrama no formato do ESP32: t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz."""
    parte = [f"{timestamp_us}", f"{roll}", f"{pitch}", f"{yaw}"]
    parte += [f"{valor}" for valor in accel_g]
    parte += [f"{valor}" for valor in gyro_dps]
    socket_udp.sendto(",".join(parte).encode("ascii"),
                      ("127.0.0.1", porta))


# 1) parsing das portas
assert parse_portas_imu("4210") == [(1, 4210)]
assert parse_portas_imu("4210,4211") == [(1, 4210), (2, 4211)]
assert parse_portas_imu("1:5000, 2:5001") == [(1, 5000), (2, 5001)]
for ruim in ("", "abc", "3:4210"):
    try:
        parse_portas_imu(ruim)
        raise AssertionError(f"aceitou porta invalida: {ruim!r}")
    except ValueError:
        pass
print("1) OK: parse das portas (uma, duas, 'lado:porta'; invalidas recusadas)")

# 2) convencao do projeto
assert IMU_PORTAS_PADRAO[1] == 4210 and IMU_PORTAS_PADRAO[2] == 4211
print("2) OK: lado 1 = direita (4210), lado 2 = esquerda (4211), igual a "
      "ARM_SIDE_CODE")

# 3) dois emissores falsos
banco = ImuBank(portas=f"{PORTA_D},{PORTA_E}")
assert banco.lados == [1, 2], banco.lados
banco.start()
remetente = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    agora_us = int(time.monotonic() * 1e6)
    # direita: 2 g no eixo vertical (=> 9,807 m/s^2 apos remover a gravidade)
    envia(remetente, PORTA_D, agora_us, 0.0, 0.0, 0.0, (0.0, 0.0, 2.0))
    # esquerda: PARADO porem inclinado 30 graus -- um sensor inclinado mede a
    # gravidade no SEU referencial: a_medido = R^T . z_hat (mesma convencao do
    # rotation_matrix do projeto), logo o resultado tem de ser ~0.
    roll_graus = 30.0
    accel_parado = rotation_matrix(roll_graus, 0.0, 0.0).T @ np.array([0.0,
                                                                       0.0,
                                                                       1.0])
    envia(remetente, PORTA_E, agora_us + 7, roll_graus, 0.0, 0.0,
          tuple(accel_parado))
    limite = time.monotonic() + 3.0
    while time.monotonic() < limite and (banco.amostra(1) is None
                                         or banco.amostra(2) is None):
        time.sleep(0.05)
    direita, esquerda = banco.amostra(1), banco.amostra(2)
    assert direita is not None and esquerda is not None, "faltou amostra"
    assert direita.timestamp_us == agora_us
    assert esquerda.timestamp_us == agora_us + 7
    assert abs(direita.roll_deg) < 1e-9 and abs(esquerda.roll_deg - 30.0) < 1e-9
    accel_d = banco.accel_no_referencial(1)
    accel_e = banco.accel_no_referencial(2)
    assert accel_d is not None and accel_e is not None
    assert abs(accel_d[2] - 9.80665) < 1e-6, accel_d
    assert np.abs(accel_e).max() < 1e-6, accel_e      # parado (so' gravidade)
    print(f"3) OK: atribuicao por porta (direita: az={accel_d[2]:+.3f} m/s^2, "
          f"roll=0; esquerda: |a|={np.abs(accel_e).max():.1e} m/s^2, roll=30)")

    # 4) fusoes independentes
    assert banco.fusoes[1] is not banco.fusoes[2]
    posicao_d, zero_d = banco.atualiza_fusao(1, np.array([0.10, 0.0, 0.0]))
    posicao_e, zero_e = banco.atualiza_fusao(2)
    assert posicao_d.shape == (3,) and posicao_e.shape == (3,)
    # o lado sem camera nenhuma fica na origem (nada de herdar o do outro lado)
    assert np.abs(posicao_e).max() < 0.05, posicao_e
    assert banco.contagem[1] >= 1 and banco.contagem[2] >= 1
    print(f"4) OK: fusoes independentes (direita pos="
          f"{np.round(posicao_d, 3)}, esquerda pos={np.round(posicao_e, 3)}, "
          f"zero_lock={zero_d}/{zero_e})")

    # 5) resumo, idade e parada
    resumo = banco.resumo()
    assert "direita" in resumo and "esquerda" in resumo, resumo
    assert banco.idade_s(1) is not None and banco.idade_s(2) is not None
    assert banco.lados == [1, 2]
    print(f"5) OK: resumo -> {resumo}")
finally:
    remetente.close()
    banco.stop()
    assert banco.receivers[1].running is False

print("IMU_BANK_OK: 5/5 casos (parse, convencao dos lados, atribuicao por porta "
      "com dois emissores, fusoes independentes, resumo/parada)")
sys.exit(0)