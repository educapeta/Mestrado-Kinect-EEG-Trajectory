#!/usr/bin/env python
"""Analisa se a normal prevista pelo IMU gira no MESMO sentido da verdade
durante o roll do punho.

Aceita:
- palm_diag_*.csv (tecla D): colunas roll_deg, nx/ny/nz, prev_x/prev_y/prev_z;
- palma_vetor_log_*.csv novo (tecla G): mesmas colunas prev_* + normal_estavel_*.

Para cada intervalo com |delta roll| >= limiar, remove a componente radial das
duas variacoes e mede o dot entre as componentes tangenciais. Dot negativo =
sentido oposto naquele intervalo. O veredito e' por lado.

Uso:
    python tools/analisa_roll_palma.py [csv]
    (sem csv pega o palm_diag mais recente)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _col(df, *nomes):
    for n in nomes:
        if n in df.columns:
            return n
    return None


def _norm(df, prefixo):
    cols = [f"{prefixo}{e}" for e in ("x", "y", "z")]
    if not all(c in df.columns for c in cols):
        return None
    v = df[cols].to_numpy(float)
    n = np.linalg.norm(v, axis=1)
    ok = np.isfinite(v).all(axis=1) & (n > 1e-9)
    v[ok] /= n[ok, None]
    return v, ok


def analisa(path: Path, limiar: float = 2.0):
    df = pd.read_csv(path)
    col_roll = _col(df, "roll_deg", "imu_roll_deg")
    col_lado = _col(df, "lado", "hand", "hand_side")
    if col_roll is None or col_lado is None:
        print(f"{path}: sem coluna roll_deg/lado")
        return False
    verdade = _norm(df, "n")
    if verdade is None:
        verdade = _norm(df, "normal_estavel_")
    previsto = _norm(df, "prev_")
    if verdade is None or previsto is None:
        print(f"{path}: sem normais nx/ny/nz e prev_x/y/z")
        return False
    v, okv = verdade
    p, okp = previsto
    print(f"\n=== {path.name} ===")
    for lado in ("right", "left"):
        sub = df[col_lado].astype(str).str.lower() == lado
        idx = np.where(sub.to_numpy())[0]
        dots = []
        for a, b in zip(idx[:-1], idx[1:]):
            if not (okv[a] and okv[b] and okp[a] and okp[b]):
                continue
            dr = float(df[col_roll].iloc[b] - df[col_roll].iloc[a])
            if abs(dr) < limiar:
                continue
            dv = v[b] - v[a]
            dp = p[b] - p[a]
            # remove componente radial (a direcao da normal nao "gira" nela)
            dv = dv - np.dot(dv, v[a]) * v[a]
            dp = dp - np.dot(dp, p[a]) * p[a]
            nv = np.linalg.norm(dv)
            npp = np.linalg.norm(dp)
            if nv < 1e-6 or npp < 1e-6:
                continue
            dots.append(float(np.dot(dv, dp) / (nv * npp)))
        if not dots:
            print(f"  {lado:5s}: sem intervalos de roll suficientes")
            continue
        dots = np.asarray(dots)
        opostos = int((dots < 0).sum())
        conc = int((dots >= 0).sum())
        med = float(np.median(dots))
        conclusao = ("SENTIDO OPOSTO predominante" if med < -0.25
                     else "sem inversao consistente")
        print(f"  {lado:5s}: n={len(dots)} concordantes={conc} "
              f"opostos={opostos} dot_med={med:+.2f} -> {conclusao}")
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Analisa sentido do roll no log")
    p.add_argument("csv", nargs="?")
    p.add_argument("--limiar", type=float, default=2.0,
                   help="delta minimo de roll em graus (padrao 2)")
    args = p.parse_args(argv)
    if args.csv:
        caminho = Path(args.csv)
    else:
        candidatos = sorted(glob.glob("palm_diag_*.csv"),
                            key=lambda f: Path(f).stat().st_mtime,
                            reverse=True)
        if not candidatos:
            print("Nenhum palm_diag_*.csv no diretorio.")
            return 1
        caminho = Path(candidatos[0])
    return 0 if analisa(caminho, args.limiar) else 2


if __name__ == "__main__":
    raise SystemExit(main())
