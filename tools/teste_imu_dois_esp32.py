"""Teste de bancada dos DOIS IMUs (luvas) -- sem EEG, sem Kinect, sem g.Pype.

Para que serve: conferir, antes da sessao, que os dois ESP32 estao enviando, a
que taxa, com que atraso, e (se pedido) GRAVAR as duas amostras simultaneamente
num CSV para inspecao offline.

Dois modos:

1. **Uma porta por lado** (recomendado, casa com `--imu-portas 4210,4211`):
   --portas 4210,4211
2. **Uma porta com demultiplexacao por remetente** (para firmware que manda os
   dois para a mesma porta): --porta 4210 --por-remetente
   Nesse modo o lado e' atribuido pela ORDEM em que os remetentes aparecem; para
   fixar, use --ips "192.168.0.101=1,192.168.0.102=2".

Formato aceito (ASCII, um datagrama por amostra):
    t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz          (10 campos, forma atual)
    lado,t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz     (11 campos, com ID)

Uso:
    .venv\\Scripts\\python.exe tools\\teste_imu_dois_esp32.py
    .venv\\Scripts\\python.exe tools\\teste_imu_dois_esp32.py --segundos 20 --gravar _imu.csv
    .venv\\Scripts\\python.exe tools\\teste_imu_dois_esp32.py --porta 4210 --por-remetente

Codigo de saida 0 = os dois lados receberam; 1 = faltou lado.
"""
from __future__ import annotations

import argparse
import os
import select
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRAVIDADE = 9.80665
UP_AXIS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--portas", default="4210,4211",
                        help="Uma porta por lado ('4210,4211' = direita, "
                             "esquerda)")
    parser.add_argument("--porta", type=int, default=4210,
                        help="Porta unica do modo --por-remetente")
    parser.add_argument("--por-remetente", action="store_true",
                        help="Modo porta unica: separa os lados pelo remetente")
    parser.add_argument("--ips", default="",
                        help="Mapa fixo 'ip=1,ip=2' (1 direita, 2 esquerda) no "
                             "modo --por-remetente")
    parser.add_argument("--segundos", type=float, default=15.0,
                        help="Duracao do teste (0 = ate' Ctrl+C)")
    parser.add_argument("--gravar", default=None,
                        help="CSV para gravar as amostras cruas dos dois lados")
    parser.add_argument("--sem-gravidade", action="store_true",
                        help="Nao remove a gravidade na aceleracao mostrada")
    return parser.parse_args()


def abre_socket(porta):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", int(porta)))
    sock.setblocking(False)
    return sock


def interpreta(payload):
    """(lado_do_pacote, t_us, roll, pitch, yaw, accel_g, gyro) ou None."""
    try:
        valores = np.fromstring(payload.decode("ascii").strip(), sep=",")
    except (UnicodeDecodeError, ValueError):
        return None
    lado = 0
    if valores.size == 11:
        lado = int(valores[0])
        valores = valores[1:]
    if valores.size < 4 or not np.isfinite(valores[:4]).all():
        return None
    accel = valores[4:7] if valores.size >= 7 else np.zeros(3)
    gyro = valores[7:10] if valores.size >= 10 else np.zeros(3)
    if not (np.isfinite(accel).all() and np.isfinite(gyro).all()):
        return None
    return (lado, int(valores[0]), float(valores[1]), float(valores[2]),
            float(valores[3]), np.asarray(accel, np.float64),
            np.asarray(gyro, np.float64))


class Lado:
    """Estatistica de um lado (um ESP32)."""

    def __init__(self, nome, porta):
        self.nome = nome
        self.porta = porta
        self.pacotes = 0
        self.marcas = []
        self.ultima = None
        self.remetentes = set()

    def registra(self, t_mono, dados, remetente):
        self.pacotes += 1
        self.marcas.append(t_mono)
        self.ultima = (t_mono,) + tuple(dados)
        self.remetentes.add(remetente)

    def taxa(self, janela=2.0):
        agora = time.monotonic()
        return len([m for m in self.marcas if agora - m <= janela]) / janela

    def idade(self):
        return None if self.ultima is None else time.monotonic() - self.ultima[0]

    def jitter_ms(self):
        if len(self.marcas) < 4:
            return None
        return float(np.std(np.diff(self.marcas[-200:]) * 1e3))

    def aceleracao(self, sem_gravidade=False):
        if self.ultima is None:
            return None
        # ultima = (t_mono, t_us, roll, pitch, yaw, accel, gyro)
        accel = np.asarray(self.ultima[5], np.float64) * GRAVIDADE
        if sem_gravidade:
            accel[UP_AXIS] -= GRAVIDADE
        return accel


def imprime(lados, sem_gravidade):
    print(f"\n{'lado':<10} {'porta':>6} {'Hz':>7} {'pacotes':>8} {'idade':>7} "
          f"{'jitter':>9} {'roll':>7} {'pitch':>7} {'yaw':>7} "
          f"{'acc(x,y,z) m/s^2':>26}")
    for lado in lados:
        if lado.pacotes == 0:
            print(f"{lado.nome:<10} {lado.porta:>6} {'--':>7} {0:>8} "
                  f"{'--':>7} {'--':>9} {'--':>7} {'--':>7} {'--':>7} "
                  f"{'sem dado':>26}")
            continue
        _, _, roll, pitch, yaw, _, _ = lado.ultima
        accel = lado.aceleracao(sem_gravidade)
        jitter = lado.jitter_ms()
        texto_jitter = "--" if jitter is None else f"{jitter:.1f}ms"
        print(f"{lado.nome:<10} {lado.porta:>6} {lado.taxa():>7.1f} "
              f"{lado.pacotes:>8} {lado.idade():>6.2f}s "
              f"{texto_jitter:>9} "
              f"{roll:>7.2f} {pitch:>7.2f} {yaw:>7.2f} "
              f"({accel[0]:+7.2f},{accel[1]:+7.2f},{accel[2]:+7.2f})")


def main():
    args = parse_args()
    nomes = ["direita", "esquerda"]
    if args.por_remetente:
        sockets = [abre_socket(args.porta)]
        lados = [Lado(nomes[i], args.porta) for i in range(2)]
    else:
        portas = [int(p) for p in str(args.portas).split(",") if p.strip()]
        sockets = [abre_socket(porta) for porta in portas]
        lados = [Lado(nomes[i] if i < len(nomes) else f"lado{i + 1}", porta)
                 for i, porta in enumerate(portas)]
    mapa_fixo = {}
    for item in str(args.ips).split(","):
        if "=" in item:
            ip, lado = item.split("=", 1)
            mapa_fixo[ip.strip()] = int(lado)
    por_remetente = {}
    arquivo = None
    if args.gravar:
        arquivo = open(args.gravar, "w", encoding="utf-8", newline="")
        arquivo.write("t_mono_s,t_us,lado,remetente,roll_deg,pitch_deg,"
                      "yaw_deg,acc_x_g,acc_y_g,acc_z_g,gyro_x,gyro_y,gyro_z\n")

    destino = ("ate' Ctrl+C" if args.segundos <= 0
               else f"por {args.segundos:g} s")
    print(f"Ouvindo {destino}: "
          + (f"porta unica {args.porta} (separando por remetente)"
             if args.por_remetente else f"portas {args.portas}"), flush=True)
    inicio = time.monotonic()
    proximo = inicio + 1.0
    try:
        while True:
            if args.segundos > 0 and time.monotonic() - inicio >= args.segundos:
                break
            prontos, _, _ = select.select(sockets, [], [], 0.2)
            for sock in prontos:
                try:
                    payload, remetente = sock.recvfrom(512)
                except OSError:
                    continue
                dados = interpreta(payload)
                if dados is None:
                    continue
                lado_pacote, t_us, roll, pitch, yaw, accel, gyro = dados
                agora = time.monotonic()
                chave = f"{remetente[0]}:{remetente[1]}"
                if args.por_remetente:
                    if lado_pacote in (1, 2):
                        indice = lado_pacote - 1
                    elif remetente[0] in mapa_fixo:
                        indice = mapa_fixo[remetente[0]] - 1
                    else:
                        if chave not in por_remetente:
                            livres = [i for i in range(len(lados))
                                      if i not in set(por_remetente.values())]
                            por_remetente[chave] = (livres[0] if livres
                                                    else len(lados) - 1)
                            print(f"[aviso] remetente novo {chave} assumindo o "
                                  f"lado '{lados[por_remetente[chave]].nome}'; "
                                  "fixe com --ips se estiver trocado.",
                                  flush=True)
                        indice = por_remetente[chave]
                else:
                    indice = sockets.index(sock)
                lados[indice].registra(agora, (t_us, roll, pitch, yaw, accel,
                                               gyro), chave)
                if arquivo is not None:
                    arquivo.write(
                        f"{agora - inicio:.6f},{t_us},{indice + 1},{chave},"
                        + ",".join(f"{v:.6g}" for v in (roll, pitch, yaw)) + ","
                        + ",".join(f"{v:.6g}" for v in accel) + ","
                        + ",".join(f"{v:.6g}" for v in gyro) + "\n")
            if time.monotonic() >= proximo:
                proximo += 1.0
                imprime(lados, args.sem_gravidade)
    except KeyboardInterrupt:
        print("\ninterrompido pelo usuario")
    finally:
        for sock in sockets:
            sock.close()
        if arquivo is not None:
            arquivo.close()
            print(f"amostras cruas gravadas em {args.gravar}")

    imprime(lados, args.sem_gravidade)
    print()
    for lado in lados:
        if lado.remetentes:
            print(f"  {lado.nome}: remetentes {sorted(lado.remetentes)}")
    faltando = [lado.nome for lado in lados if lado.pacotes == 0]
    if faltando:
        print(f"RESULTADO: FALTOU {', '.join(faltando)}. Confira: (1) as duas "
              "luvas ligadas e no mesmo Wi-Fi; (2) a porta de destino do "
              "firmware (uma por mao, ou --por-remetente); (3) o firewall do "
              "Windows liberando UDP.")
        return 1
    print("RESULTADO: os dois lados receberam amostras.")
    return 0


if __name__ == "__main__":
    sys.exit(main())