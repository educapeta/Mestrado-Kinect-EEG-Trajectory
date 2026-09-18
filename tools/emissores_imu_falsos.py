"""Emissores FALSOS de luva (ESP32 + MPU6050) por UDP, para testar sem hardware.

Serve para exercitar toda a cadeia do IMU (recepcao, fusao por lado, colunas
IMU_L_*/IMU_R_* e filtro de Kalman) antes de ter as luvas na mao -- e tambem para
conferir, com as luvas, se o que chega e' o que se espera.

Cada lado ganha uma "assinatura" distinta e reconhecivel:
  - direita (lado 1): roll = +5 graus e aceleracao com 1,2 g no eixo vertical;
  - esquerda (lado 2): roll = -5 graus e aceleracao com 1,0 g (parada).

Uso:
    .venv\\Scripts\\python.exe tools\\emissores_imu_falsos.py --segundos 10
    .venv\\Scripts\\python.exe tools\\emissores_imu_falsos.py --hz 100 --portas 4210,4211
    .venv\\Scripts\\python.exe tools\\emissores_imu_falsos.py --porta 4210   # so' um
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time

import numpy as np

#: (lado, roll, pitch, yaw, accel_g)
ASSINATURA = {
    1: (5.0, -2.0, 1.0, (0.0, 0.0, 1.2)),
    2: (-5.0, 2.0, -1.0, (0.0, 0.0, 1.0)),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portas", default="4210,4211",
                        help="Uma porta por lado (padrao direita, esquerda)")
    parser.add_argument("--porta", type=int, default=None,
                        help="Manda os DOIS lados para a mesma porta (com ID)")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--segundos", type=float, default=10.0)
    parser.add_argument("--com-id", action="store_true",
                        help="Prefixa o lado no pacote (11 campos)")
    parser.add_argument("--sem-id", action="store_true",
                        help="No modo --porta, manda o formato de 10 campos "
                             "(para testar a separacao pelo REMETENTE)")
    parser.add_argument("--ruido", type=float, default=0.002,
                        help="Ruido (g) somado a aceleracao")
    return parser.parse_args()


def emissores(portas, hz=30.0, com_id=False, ruido=0.002, semente=0,
              parar=None):
    """Cria os emissores: devolve (threads, evento de parada, lista de sockets).

    `portas` aceita [(lado, porta)] ou uma porta unica (int) para o modo de ID.
    Cada lado usa o SEU proprio socket, como dois dispositivos de verdade (isso
    importa no modo de demultiplexacao por remetente: dois lados no mesmo socket
    teriam o MESMO endereco de origem).
    """
    if isinstance(portas, int):
        destinos = [(1, portas), (2, portas)]
    else:
        destinos = list(portas)
    parada = parar if parar is not None else threading.Event()
    rng = np.random.default_rng(semente)
    threads, sockets = [], []

    def trabalha(lado, porta, socket_udp):
        periodo = 1.0 / max(hz, 1e-6)
        proximo = time.monotonic()
        while not parada.is_set():
            roll, pitch, yaw, accel = ASSINATURA[lado]
            accel = np.asarray(accel, np.float64) + rng.normal(0, ruido, 3)
            campos = [f"{int(time.monotonic() * 1e6)}", f"{roll}", f"{pitch}",
                      f"{yaw}"]
            campos += [f"{valor:.6f}" for valor in accel]
            campos += ["0.0", "0.0", "0.0"]
            if com_id:
                campos = [str(lado)] + campos
            try:
                socket_udp.sendto(",".join(campos).encode("ascii"),
                                  ("127.0.0.1", int(porta)))
            except OSError:
                pass
            proximo += periodo
            atraso = proximo - time.monotonic()
            if atraso > 0:
                time.sleep(atraso)
            else:
                proximo = time.monotonic()

    for lado, porta in destinos:
        socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockets.append(socket_udp)
        thread = threading.Thread(target=trabalha, args=(lado, porta,
                                                         socket_udp),
                                  daemon=True, name=f"imu-falso-{lado}")
        thread.start()
        threads.append(thread)
    return threads, parada, sockets


def main():
    args = parse_args()
    if args.porta is not None:
        destinos = args.porta
        descricao = f"porta unica {args.porta} (com ID)"
    else:
        portas = [int(p) for p in str(args.portas).split(",") if p.strip()]
        destinos = list(zip([1, 2][:len(portas)], portas))
        descricao = "portas " + str(portas)
    threads, parada, sockets = emissores(destinos, hz=args.hz,
                                         com_id=(args.porta is not None
                                                 and not args.sem_id),
                                         ruido=args.ruido)
    print(f"Emissores falsos ativos em {descricao}: {args.hz:g} Hz por lado, "
          f"por {args.segundos:g} s (assinaturas: direita roll=+5 e 1,2 g; "
          "esquerda roll=-5 e 1,0 g)", flush=True)
    try:
        time.sleep(max(args.segundos, 0.1))
    except KeyboardInterrupt:
        pass
    finally:
        parada.set()
        for thread in threads:
            thread.join(timeout=0.5)
        for socket_udp in sockets:
            socket_udp.close()
    print("emissores encerrados")
    return 0


if __name__ == "__main__":
    sys.exit(main())