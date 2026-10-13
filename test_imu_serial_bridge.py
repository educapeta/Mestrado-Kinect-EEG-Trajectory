"""Testes do ponte serial -> UDP (tools/imu_serial_bridge.py) -- sem hardware.

Roda em qualquer maquina: nao abre porta serial real nem requer o ESP32.
Cobertura:

  1. LINE_RE aceita o formato do firmware (lado + 10 floats) e rejeita o que
     nao casa (cabecalho, firmware antigo sem lado, lixo, linha truncada);
  2. o payload UDP reenviado casa com o formato do ImuReceiver (imu.py):
     10 campos, int(t_us) no campo 0 -- ou seja, imu.py NAO muda;
  3. roteamento: 1 -> 4210 (direita), 2 -> 4211 (esquerda), igual a
     imu.IMU_PORTAS_PADRAO; lado desconhecido nao publica;
  4. ImuReceiver real do proprio imu.py aceita o que o bridge emite;
  5. descarte de linha invalida em ImuSerialReceiver._run (serial fake).
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from imu_serial_bridge import (  # noqa: E402
    LINE_RE, UDP_PORTS, ImuSerialReceiver,
)

_passou = 0
_falhas = []


def check(cond, nome):
    global _passou
    if cond:
        _passou += 1
        print(f"  [OK] {nome}")
    else:
        _falhas.append(nome)
        print(f"  [FALHA] {nome}")


def test_line_re():
    print("1) LINE_RE: formato do firmware esp32_duas_imus")
    boa_d = "1,1790000000,12.34,-5.67,90.12,0.123456,-0.654321,0.987654,1.5000,-2.2500,0.0000"
    boa_e = "2,1790000001,-1.00,2.00,3.00,0.0,0.0,1.0,0.0,0.0,0.0"
    m = LINE_RE.match(boa_d)
    check(m is not None, "linha lado 1 aceita")
    check(m and int(m.group(1)) == 1, "lado extraido = 1")
    check(m and abs(float(m.group(2)) - 1790000000) < 1e-6, "t_us extraido")
    m2 = LINE_RE.match(boa_e)
    check(m2 and int(m2.group(1)) == 2, "linha lado 2 aceita, lado = 2")
    check(LINE_RE.match("1,100,-12.5,0.0,0.0,-1.0,-2.0,-3.0,-4.0,-5.0,-6.0") is not None,
          "valores negativos aceitos")
    check(LINE_RE.match(
        "esp_timestamp_us,roll_deg,pitch_deg,yaw_deg,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"
    ) is None, "cabecalho do ESP32 rejeitado")
    check(LINE_RE.match("1790000000,12.34,-5.67,90.12,0.1,0.2,0.9,1.0,2.0,0.0") is None,
          "firmware antigo (sem campo lado) rejeitado")
    check(LINE_RE.match("erro de leitura I2C") is None, "texto solto rejeitado")
    check(LINE_RE.match("1,1790000000,12.34,-5.67") is None, "linha truncada rejeitada")
    check(LINE_RE.match("") is None, "linha vazia rejeitada")
    m3 = LINE_RE.match("9,1790000000,1,2,3,4,5,6,7,8,9")
    check(m3 is not None and int(m3.group(1)) not in UDP_PORTS,
          "lado fora de {1,2} passa no regex e e' filtrado por UDP_PORTS")


def _payload_de(valores):
    """Reproduz a montagem do payload em ImuSerialBridge._run (unitario)."""
    return (f"{valores[0]:.0f}," + ",".join(f"{v:.6f}" for v in valores[1:])
            + "\n").encode("ascii")


def test_payload_formato():
    print("2) payload reenviado casa com o ImuReceiver (imu.py)")
    valores = (1790000000.0, 12.34, -5.67, 90.12, 0.123456, -0.654321,
               0.987654, 1.5, -2.25, 0.0)
    payload = _payload_de(valores)
    campos = payload.decode().strip().split(",")
    check(len(campos) == 10, "payload tem 10 campos")
    check(campos[0] == "1790000000", "campo 0 e' inteiro textual (int() funciona)")
    import numpy as np
    t = int(campos[0])
    accel = np.array([float(campos[4]), float(campos[5]), float(campos[6])])
    gyro = np.array([float(campos[7]), float(campos[8]), float(campos[9])])
    check(t == 1790000000 and abs(float(campos[1]) - 12.34) < 1e-4, "parse t_us/roll")
    check(np.allclose(accel, [0.123456, -0.654321, 0.987654], atol=1e-6), "parse accel")
    check(np.allclose(gyro, [1.5, -2.25, 0.0], atol=1e-6), "parse gyro")


def test_roteamento():
    print("3) roteamento lado -> porta (contrato imu.IMU_PORTAS_PADRAO)")
    check(UDP_PORTS == {1: 4210, 2: 4211},
          "UDP_PORTS == {1: 4210, 2: 4211}")
    import imu as _imu
    esperado = dict(_imu.IMU_PORTAS_PADRAO)
    check(esperado == UDP_PORTS, f"bate com imu.IMU_PORTAS_PADRAO = {esperado}")
    check(3 not in UDP_PORTS and 0 not in UDP_PORTS,
          "lado 0/3 nao tem porta (nao publica)")


def test_fim_a_fim_receiver():
    print("4) ImuReceiver real aceita o que o bridge emite (sem ESP32)")
    import imu as _imu
    import numpy as np
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("0.0.0.0", 0))
    porta = recv_sock.getsockname()[1]
    recv_sock.close()
    valores = (42424242.0, 1.5, -2.5, 3.5, 0.1, 0.2, 0.9, 0.01, 0.02, 0.03)
    payload = _payload_de(valores)
    recv = _imu.ImuReceiver(porta)
    recv.start()
    time.sleep(0.1)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(payload, ("127.0.0.1", porta))
    time.sleep(0.3)
    amostra = recv.get_latest()
    recv.stop()
    s.close()
    check(amostra is not None, "ImuReceiver.get_latest() devolveu amostra")
    check(amostra is not None and amostra.timestamp_us == 42424242,
          "timestamp_us do ImuReceiver = o que o bridge emitiu")
    check(amostra is not None and abs(float(amostra.roll_deg) - 1.5) < 1e-4,
          "roll_deg coerente no ImuReceiver")
    check(amostra is not None and np.isfinite(amostra.accel_g).all(),
          "accel_g finito no ImuReceiver")



class _SerialFalsa:
    """Simula serial.Serial: devolve as linhas depois some (b'' = sem dado)."""

    def __init__(self, linhas):
        self._linhas = list(linhas)

    def readline(self):
        if self._linhas:
            return self._linhas.pop(0)
        time.sleep(0.05)
        return b""

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


def test_descarte_linhas_ruins():
    print("5) ImuSerialReceiver descarta linhas ruins sem morrer")
    r = ImuSerialReceiver.__new__(ImuSerialReceiver)   # sem abrir serial real
    r.port = "FAKE"
    r.baud = 115200
    r.running = True
    r.latest = None
    r.lock = threading.Lock()
    r.serial = _SerialFalsa([
        b"esp_timestamp_us,roll_deg,...\n",                    # cabecalho
        b"1,1790000000,1.0,2.0,3.0,0.1,0.2,0.9,0,0,0\n",       # boa (lado 1)
        b"lixo sem formato\n",                                 # ignora
        b"2,1790000001,na,2.0,3.0,0.1,0.2,0.9,0,0,0\n",        # float ruim
        b"9,1790000002,1.0,2.0,3.0,0.1,0.2,0.9,0,0,0\n",       # lado 3
        b"2,1790000003,-1.0,-2.0,-3.0,0.0,0.0,1.0,0.0,0.0,0.0\n",  # boa 2
    ])
    t = threading.Thread(target=r._run, daemon=True)
    t.start()
    time.sleep(1.0)
    r.running = False
    t.join(timeout=2.0)
    check(not t.is_alive(), "thread terminou (nao morreu com linhas ruins)")
    check(r.latest is not None, "ultima amostra valida guardada")
    check(r.latest is not None and r.latest[0] == 2, "ultima valida e do lado 2")
    check(r.latest is not None and r.latest[1][0] == 1790000003,
          "t_us da ultima valida")


if __name__ == "__main__":
    test_line_re()
    test_payload_formato()
    test_roteamento()
    test_fim_a_fim_receiver()
    test_descarte_linhas_ruins()
    print()
    if _falhas:
        print(f"FALHAS: {len(_falhas)} -> {_falhas}")
        sys.exit(1)
    print(f"teste_imu_serial_bridge: {_passou} assercoes OK")
    sys.exit(0)
