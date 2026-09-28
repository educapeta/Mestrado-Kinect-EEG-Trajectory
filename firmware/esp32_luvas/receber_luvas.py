"""Recebe os dados das duas luvas (ESP32 + MPU6050) via UDP e mostra no terminal.

Portas (definidas em platformio.ini / src/main.cpp):
    4210 -> luva DIREITA    (env luva_direita)
    4211 -> luva ESQUERDA   (env luva_esquerda)

Formato do pacote enviado pelo ESP32:
    timestamp_us,roll_deg,pitch_deg,yaw_deg,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps

Uso:
    python receber_luvas.py        (Ctrl+C para sair)
"""

import socket
import sys
import time

PORTA_ESQUERDA = 4211
PORTA_DIREITA = 4210

ROTULOS = {
    PORTA_ESQUERDA: "ESQUERDA",
    PORTA_DIREITA: "DIREITA ",
}

INTERVALO_IMPRESSAO_S = 0.5


def abrir_socket(porta):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    s.bind(("", porta))
    s.settimeout(0.2)
    return s


def converter(pacote):
    """Converte o CSV do ESP32 em lista de floats. Retorna None se invalido."""
    texto = pacote.decode("ascii", "replace").strip()
    partes = texto.split(",")
    if len(partes) != 10:
        return None
    try:
        return [float(p) for p in partes]
    except ValueError:
        return None


def formatar_linha(porta, valores, origem, quantidade):
    rotulo = ROTULOS[porta]
    if valores is None:
        return "  {r} :{p}   -- sem dados --".format(r=rotulo, p=porta)
    ip = origem[0] if origem else "-"
    return (
        "  {r} :{p}   ip={ip:<15s} pacotes={n:<7d} "
        "roll={roll:8.2f} pitch={pitch:8.2f} yaw={yaw:8.2f} "
        "ax={ax:7.3f} ay={ay:7.3f} az={az:7.3f} "
        "gx={gx:8.2f} gy={gy:8.2f} gz={gz:8.2f}"
    ).format(
        r=rotulo,
        p=porta,
        ip=ip,
        n=quantidade,
        roll=valores[1],
        pitch=valores[2],
        yaw=valores[3],
        ax=valores[4],
        ay=valores[5],
        az=valores[6],
        gx=valores[7],
        gy=valores[8],
        gz=valores[9],
    )


def main():
    sockets = {}
    for porta in (PORTA_ESQUERDA, PORTA_DIREITA):
        try:
            sockets[porta] = abrir_socket(porta)
        except OSError as exc:
            print("ERRO: nao consegui abrir a porta UDP {}: {}".format(porta, exc))
            print("Feche outros programas que estejam usando essas portas e tente de novo.")
            return 1

    ultimo = {PORTA_ESQUERDA: None, PORTA_DIREITA: None}
    origem = {PORTA_ESQUERDA: None, PORTA_DIREITA: None}
    contagem = {PORTA_ESQUERDA: 0, PORTA_DIREITA: 0}

    print("Escutando as luvas via UDP:")
    print("  ESQUERDA -> porta {}".format(PORTA_ESQUERDA))
    print("  DIREITA  -> porta {}".format(PORTA_DIREITA))
    print("Ctrl+C para sair.\n")

    proximo_print = 0.0
    try:
        while True:
            for porta, s in sockets.items():
                try:
                    pacote, endereco = s.recvfrom(4096)
                except socket.timeout:
                    continue
                valores = converter(pacote)
                if valores is None:
                    continue
                ultimo[porta] = valores
                origem[porta] = endereco
                contagem[porta] += 1

            agora = time.time()
            if agora >= proximo_print:
                proximo_print = agora + INTERVALO_IMPRESSAO_S
                print(formatar_linha(PORTA_ESQUERDA, ultimo[PORTA_ESQUERDA],
                                     origem[PORTA_ESQUERDA], contagem[PORTA_ESQUERDA]))
                print(formatar_linha(PORTA_DIREITA, ultimo[PORTA_DIREITA],
                                     origem[PORTA_DIREITA], contagem[PORTA_DIREITA]))
                print()
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        for s in sockets.values():
            s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
