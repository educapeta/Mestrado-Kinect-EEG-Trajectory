"""Mede a DERIVA dos IMUs parados (bias do giroscopio, yaw/roll/pitch).

POR QUE EXISTE (medido em 21/10/2026, item 99): a calibracao de bias do firmware
roda UMA vez no boot e assume a luva PARADA. Se religar com a luva em movimento,
o bias fica com a media do movimento e o yaw deriva sem parar (medido: -12,5 dps
na direita = **434 graus/min**; a esquerda ~6 graus/min mesmo com bias menor).
Roll/pitch sao corrigidos pela gravidade (ficam perfeitos); so' o yaw integra o
bias residual.

Este utilitario:
  1) captura N minutos de UDP 4210/4211 (luvas PARADAS!) para um CSV, e
  2) analisa por lado: taxa/gaps, deriva de roll/pitch/yaw (graus e graus/min),
     media/std do giroscopio parado (= bias residual), acelerometro (norma ~1 g),
     saltos/descontinuidades e o yaw previsto pela projecao do gyro na vertical.

Uso:
    .venv\\Scripts\\python.exe tools\\deriva_estatica_imu.py
    .venv\\Scripts\\python.exe tools\\deriva_estatica_imu.py --segundos 300
    .venv\\Scripts\\python.exe tools\\deriva_estatica_imu.py --analisar deriva.csv

Leitura do resultado:
  - gyro medio ~0 e yaw ~0 graus/min  => bias calibrado e IMU apto para sessao;
  - yaw deriva > ~1 grau/min         => rezerar (tools\\reinicia_luvas_serial.py
                                        com a luva parada) ou, no limite,
                                        re-zero em runtime (firmware, item 100);
  - roll/pitch andando              => problema serio (filtro/MPU/acelerometro).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np


def capturar(caminho, portas, segundos):
    """Grava `segundos` de UDP (t_mono, porta, roll..gz) em `caminho`."""
    import socket

    print(f"  gravando {segundos:g} s de {portas} em {caminho} ... "
          "MANTENHA AS LUVAS PARADAS", flush=True)
    sockets = {}
    for porta in portas:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", porta))
        s.settimeout(0.05)
        sockets[porta] = s
    total = 0
    t0 = time.monotonic()
    try:
        with open(caminho, "w", encoding="utf-8") as saida:
            saida.write("t_mono,porta,roll,pitch,yaw,ax,ay,az,gx,gy,gz\n")
            while time.monotonic() - t0 < segundos:
                for porta, s in sockets.items():
                    try:
                        dado, _ = s.recvfrom(2048)
                    except socket.timeout:
                        continue
                    campos = dado.decode("utf-8", "replace").strip().split(",")
                    if len(campos) >= 10:
                        saida.write("%.3f,%d,%s\n" % (
                            time.monotonic(), porta,
                            ",".join(campos[1:10])))
                        total += 1
                saida.flush()
    finally:
        for s in sockets.values():
            s.close()
    print(f"  captura terminada: {total} amostras", flush=True)
    return total > 0


def _carregar(caminho):
    dados = {}
    with open(caminho, encoding="utf-8") as fh:
        for linha in csv.DictReader(fh):
            try:
                valores = [float(linha[c]) for c in
                           ("t_mono", "roll", "pitch", "yaw", "ax", "ay",
                            "az", "gx", "gy", "gz")]
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(valores).all():
                dados.setdefault(str(linha.get("porta")), []).append(valores)
    return dados


def _rpy_para_R(usar_imu=True):
    """Matriz R(rpy) do projeto (ZYX), com fallback local se imu nao importar."""
    if usar_imu:
        try:
            import imu  # noqa: F401
            return imu.rotation_matrix
        except Exception:                    # noqa: BLE001 - ambiente sem projeto
            pass

    def rotation_matrix(roll, pitch, yaw):
        r, p, y = np.radians([roll, pitch, yaw])
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    return rotation_matrix


def analisar(caminho):
    """Imprime o relatorio de deriva e devolve True se algum lado e' problemático."""
    dados = _carregar(caminho)
    rot = _rpy_para_R()
    problematico = False
    for porta, arr in sorted(dados.items()):
        arr = np.array(arr)
        if len(arr) < 100:
            print(f"\nporta {porta}: {len(arr)} amostras -- insuficiente")
            continue
        t = arr[:, 0] - arr[0, 0]
        dt = t[-1]
        rate = len(arr) / dt
        gaps = np.diff(t)
        roll, pitch = arr[:, 1], arr[:, 2]
        yaw = np.degrees(np.unwrap(np.radians(arr[:, 3])))
        acc = arr[:, 4:7]
        gyro = arr[:, 7:10]

        def janela(vetor, a, b):
            m = (t >= a) & (t <= b)
            return float(np.median(vetor[m])) if m.any() else float("nan")

        def deriva(vetor):
            return janela(vetor, dt - 20, dt) - janela(vetor, 0, 20)

        print(f"\nporta {porta}: {len(arr)} amostras em {dt:.0f} s "
              f"({rate:.1f} Hz) | maior gap {gaps.max():.2f} s")
        for nome, vetor in (("roll", roll), ("pitch", pitch), ("yaw", yaw)):
            dr = deriva(vetor)
            coef = float(np.polyfit(t, vetor, 1)[0])
            print(f"  {nome:5s}: deriva {dr:+7.2f} deg em {dt/60:.1f} min "
                  f"({coef*60:+.2f} deg/min) | std {np.std(vetor):5.2f}")
            if nome == "yaw" and abs(coef * 60) > 1.0:
                problematico = True
            if nome != "yaw" and abs(dr) > 2.0:
                problematico = True
        for i, eixo in enumerate("xyz"):
            g = gyro[:, i]
            print(f"  gyro_{eixo}: media {np.mean(g):+7.3f} dps (std "
                  f"{np.std(g):.3f}, min {np.min(g):+6.2f} max {np.max(g):+6.2f})")
        mag = np.linalg.norm(acc, axis=1)
        print(f"  accel: norma {np.mean(mag):.4f} g (std {np.std(mag):.4f}) | "
              f"std eixos {np.round(np.std(acc, axis=0), 4)}")
        vertical = np.array([0.0, 0.0, 1.0])
        previsto = []
        amostrar = np.linspace(0, len(arr) - 1, min(len(arr), 600)).astype(int)
        for i in amostrar:
            R = rot(roll[i], pitch[i], yaw[i] - yaw[0])
            previsto.append(float((R @ gyro[i]) @ vertical))
        previsto = np.array(previsto)
        coef_yaw = float(np.polyfit(t, yaw, 1)[0])
        print(f"  yaw medido {coef_yaw*60:+.2f} deg/min | yaw previsto pelo gyro "
              f"{np.mean(previsto)*60:+.2f} deg/min")
        saltos = np.abs(np.diff(yaw))
        print(f"  saltos de yaw >5 deg: {(saltos > 5).sum()} | max {saltos.max():.1f}")

    print()
    if problematico:
        print("VEREDITO: DERIVA ALTA -- rezero com a luva parada "
              "(tools\\reinicia_luvas_serial.py) e meca de novo.")
    else:
        print("VEREDITO: comportamento BOM para sessao (yaw <= 1 deg/min e "
              "roll/pitch estaveis).")
    return not problematico


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segundos", type=float, default=600.0,
                        help="duracao da captura (padrao 600 s = 10 min)")
    parser.add_argument("--portas", default="4210,4211",
                        help="portas UDP dos IMUs (padrao 4210,4211)")
    parser.add_argument("--csv", default="deriva_estatica.csv",
                        help="arquivo de saida/entrada (padrao deriva_estatica.csv)")
    parser.add_argument("--analisar", metavar="CSV", default=None,
                        help="apenas analisa um CSV ja' gravado")
    args = parser.parse_args(argv)
    portas = [int(p) for p in str(args.portas).split(",") if p.strip()]
    if args.analisar:
        ok = analisar(args.analisar)
        return 0 if ok else 1
    if not capturar(args.csv, portas, args.segundos):
        return 2
    ok = analisar(args.csv)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
