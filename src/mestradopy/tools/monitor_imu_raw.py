#!/usr/bin/env python
"""Monitor continuo dos valores crus do IMU na serial do ESP32.

Le a serial USB (formato do firmware esp32_duas_imus):
    lado,t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz

Uso:
    .venv\\Scripts\\python.exe tools\\monitor_imu_raw.py
    .venv\\Scripts\\python.exe tools\\monitor_imu_raw.py --porta COM4 --formatado
    .venv\\Scripts\\python.exe tools\\monitor_imu_raw.py --log imu_raw.log

NAO reseta a placa (DTR/RTS em nivel inativo). Ctrl+C encerra.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Monitor continuo dos valores crus do IMU (serial USB)")
    p.add_argument("--porta", default="COM4")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--formatado", action="store_true",
                   help="mostra campos com nomes em vez da linha crua")
    p.add_argument("--so-dados", action="store_true",
                   help="mostra apenas linhas de dados (ignora boot/Wi-Fi)")
    p.add_argument("--log", type=str, default=None,
                   help="arquivo para gravar as linhas cruas")
    p.add_argument("--segundos", type=float, default=0.0,
                   help="0 = infinito; >0 encerra depois de N s")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import serial
    except ImportError:
        print("pyserial nao instalado; use .venv\\Scripts\\python.exe", file=sys.stderr)
        return 1

    try:
        porta = serial.Serial(
            args.porta, args.baud, timeout=0.3,
            dsrdtr=False, rtscts=False,
        )
        # NAO mexer em DTR/RTS: nesta placa isso segura o ESP32 em
        # boot/reset e a serial entrega so' lixo de boot (medido 09/11/2026).
        porta.reset_input_buffer()
    except Exception as exc:  # noqa: BLE001 - hardware
        print(f"[ERRO] nao abri {args.porta}: {type(exc).__name__}: {exc}")
        print("      Feche a bridge/fusion_live/protocolo que esteja usando a porta.")
        return 2

    log_file = None
    if args.log:
        log_file = open(Path(args.log), "a", encoding="utf-8")

    print(f"[monitor] {args.porta} @ {args.baud} | Ctrl+C para parar", flush=True)
    print("[formato] lado,t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz", flush=True)
    t0 = time.monotonic()
    linhas = 0
    try:
        while True:
            if args.segundos > 0 and time.monotonic() - t0 >= args.segundos:
                break
            try:
                raw = porta.readline()
            except Exception as exc:  # noqa: BLE001
                print(f"[ERRO] leitura: {type(exc).__name__}: {exc}")
                break
            if not raw:
                continue
            texto = raw.decode("utf-8", errors="replace").strip()
            if not texto:
                continue
            if log_file is not None:
                log_file.write(texto + "\n")
                log_file.flush()
            campos = texto.split(",")
            if len(campos) == 11 and campos[0] in ("1", "2"):
                if args.formatado:
                    print(
                        f"lado={campos[0]} t_us={campos[1]} "
                        f"rpy=({campos[2]},{campos[3]},{campos[4]}) "
                        f"acc=({campos[5]},{campos[6]},{campos[7]}) "
                        f"gyr=({campos[8]},{campos[9]},{campos[10]})",
                        flush=True)
                else:
                    print(texto, flush=True)
                linhas += 1
            elif not args.so_dados:
                print(f"[serial] {texto}", flush=True)
    except KeyboardInterrupt:
        print("\n[monitor] interrompido", flush=True)
    finally:
        if log_file is not None:
            log_file.close()
        porta.close()
    print(f"[monitor] fim: {linhas} linhas de dados em "
          f"{time.monotonic() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
