"""Ponte serial USB -> UDP para o ESP32 com 2 MPU6050 (firmware esp32_duas_imus).

O firmware de uma luva inteira publica via Wi-Fi/UDP (padrao). Quando o Wi-Fi
nao esta disponivel (bancada sem hotspot, SSH, CI, ou quando o ESP32 esta
conectado apenas via cabo USB), este modulo lê a MESMA linha serial que o
firmware imprime e encaminha para as portas UDP corretas (4210 / 4211).

Formato da linha serial (esp32_duas_imus):

    <lado>,<t_us>,<roll>,<pitch>,<yaw>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>

Onde ``<lado>`` e' 1 (direita, MPU @ 0x68 -> UDP 4210) ou 2 (esquerda, MPU @
0x69 -> UDP 4211). Os outros 10 campos sao identicos ao firmware original de
uma luva so.

Uso:

    python tools/imu_serial_bridge.py                    # COM4, padrao
    python tools/imu_serial_bridge.py --port COM5        # porta personalizada
    python tools/imu_serial_bridge.py --escutar 10       # mostra quem envia

Nao altera o ``imu.py``: ele continua recebendo via UDP (ImuReceiver), que e'
o que este modulo produz. O fluxo fica:

    ESP32 (2x MPU6050) -> Serial USB -> imu_serial_bridge.py -> UDP 4210/4211
                                                              -> ImuReceiver
"""
from __future__ import annotations

import math
import re
import socket
import threading
import time
from typing import Optional

import serial

# Porta serial padrao onde o ESP32 com 2 MPU6050 esta conectado.
DEFAULT_PORT = "COM4"
# Baud rate do firmware (Serial.begin no setup).
DEFAULT_BAUD = 115200
# Portas UDP -- casa com imu.py: IMU_PORTAS_PADRAO = {1: 4210, 2: 4211}.
UDP_PORTS = {1: 4210, 2: 4211}
# UDP recebe pacotes de qualquer interface (o ESP32 esta na mesma rede).
UDP_BIND = ("0.0.0.0", 0)
# Regex do formato serial: lado inteiro + 10 campos floats
# (lado,t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz = 11 campos no total).
_F = r"(-?\d+(?:\.\d+)?)"
LINE_RE = re.compile(
    r"^\s*(\d)\s*,\s*"
    + _F + r"\s*,\s*" + _F + r"\s*,\s*" + _F + r"\s*,\s*" + _F + r"\s*,\s*"
    + _F + r"\s*,\s*" + _F + r"\s*,\s*" + _F + r"\s*,\s*" + _F + r"\s*,\s*"
    + _F + r"\s*,\s*" + _F + r"\s*$"
)


class ImuSerialReceiver:
    """Recebe as amostras do ESP32 por Serial USB em uma thread propria.

    Guarda sempre a ULTIMA amostra valida (``get_latest``); linhas vazias,
    truncadas ou com valores nao finitos sao descartadas em silencio.
    """

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD):
        self.port = port
        self.baud = baud
        self.running = False
        self.latest: Optional[tuple[int, tuple[float, ...]]] = None
        self.lock = threading.Lock()
        self.serial: Optional[serial.Serial] = None

    def start(self) -> None:
        self.serial = serial.Serial(self.port, self.baud, timeout=0.2)
        self.serial.reset_input_buffer()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        assert self.serial is not None
        while self.running:
            line = self.serial.readline()
            if not line:
                continue
            try:
                text = line.decode("utf-8", errors="replace").strip()
            except UnicodeDecodeError:
                continue
            m = LINE_RE.match(text)
            if not m:
                continue
            lado = int(m.group(1))
            try:
                values = tuple(float(m.group(i)) for i in range(2, 12))
            except ValueError:
                continue
            if not all(math.isfinite(v) for v in values):
                continue
            if lado not in UDP_PORTS:
                continue
            with self.lock:
                self.latest = (lado, values)

    def get_latest(self) -> Optional[tuple[int, tuple[float, ...]]]:
        with self.lock:
            return self.latest

    def stop(self) -> None:
        self.running = False
        if self.serial is not None:
            self.serial.close()
        self.serial = None



# ==================== PUENTE SERIAL -> UDP ============================
class ImuSerialBridge:
    """Le a serial USB do ESP32 e publica UDP nas portas 4210/4211.

    Linha serial (firmware esp32_duas_imus):
        <lado>,<t_us>,<roll>,<pitch>,<yaw>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>
    lado 1 -> UDP 4210 (direita), lado 2 -> UDP 4211 (esquerda).
    O pacote UDP e' identico ao original (10 campos, sem lado), entao
    imu.py nao precisa de nenhuma alteracao.
    """

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD):
        self.receiver = ImuSerialReceiver(port, baud)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(UDP_BIND)
        self.running = False

    def start(self) -> None:
        self.receiver.start()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        assert self.sock is not None
        while self.running:
            sample = self.receiver.get_latest()
            if sample is None:
                time.sleep(0.01)
                continue
            lado, values = sample
            port = UDP_PORTS[lado]
            payload = (
                f"{values[0]:.0f},"
                + ",".join(f"{v:.6f}" for v in values[1:])
                + "\n"
            ).encode("ascii")
            self.sock.sendto(payload, ("127.0.0.1", port))

    def stop(self) -> None:
        self.running = False
        self.receiver.stop()
        if self.sock is not None:
            self.sock.close()
        self.sock = None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ponte serial USB -> UDP")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--escutar", type=int, default=0, metavar="S")
    args = parser.parse_args()
    bridge = ImuSerialBridge(args.port, args.baud)
    bridge.start()
    print("[bridge] ouvindo", args.port, "-> UDP 4210/4211. Ctrl+C para parar.")
    if args.escutar > 0:
        from collections import Counter
        seen = Counter()
        fim = time.monotonic() + args.escutar
        while time.monotonic() < fim:
            s = bridge.receiver.get_latest()
            if s is not None:
                seen[s[0]] += 1
            time.sleep(0.2)
        print("[bridge] remetentes:", dict(seen))
        bridge.stop()
        return
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[bridge] parando...")
        bridge.stop()


if __name__ == "__main__":
    main()
