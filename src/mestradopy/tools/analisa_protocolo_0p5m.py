#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analisa o protocolo de deslocamento de 0,5 m (par aux0/aux1 vs Kinect).

Lê o CSV de trajetória gerado por
`compara_pair_kinect_online.py --protocolo-0p5m`.

Fases esperadas:
  A_parado   0-5 s
  ida_0p5m   5-15 s
  B_parado   15-20 s
  volta_0p5m 20-30 s
  A_retorno  30-35 s

Métrica principal: deslocamento 3D reconstruído A->B comparado à régua de
0,5 m, para o par e para o Kinect; quem chega mais perto da verdade externa.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _v(row, prefix):
    try:
        return np.array([float(row[f"{prefix}_{a}"]) for a in ("x", "y", "z")])
    except (KeyError, ValueError):
        return np.array([np.nan, np.nan, np.nan])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", type=Path)
    ap.add_argument("--lado", default="right")
    ap.add_argument("--distancia-m", type=float, default=0.5)
    ap.add_argument("--saida", type=Path, default=None)
    args = ap.parse_args()
    out = args.saida or args.traj.with_name(args.traj.stem + "_analise")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.traj, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise SystemExit("CSV de trajetoria vazio")

    lados = sorted({r.get("lado") for r in rows if r.get("lado")})
    lado = args.lado
    if lado not in lados:
        # A lateralidade do MediaPipe pode sair invertida (frame espelhado).
        # Nao invente: use o unico lado que realmente foi gravado.
        preferido = [x for x in lados if x in ("right", "left")]
        if not preferido:
            raise SystemExit(f"lados no CSV: {lados}")
        lado = preferido[0]
        print(f"[aviso] lado={args.lado!r} sem linhas; usando lado={lado!r} "
              f"(lados disponiveis: {lados})")
    rows = [r for r in rows if r.get("lado") == lado]
    print(f"lado={lado} frames={len(rows)}")

    def por_fase(fase):
        return [r for r in rows if r["fase"] == fase]

    def posicao(fase):
        rr = por_fase(fase)
        P = np.array([_v(r, "par") for r in rr])
        K = np.array([_v(r, "kin") for r in rr])
        Pm = np.nanmedian(P, axis=0) if np.isfinite(P).any() else np.full(3, np.nan)
        Km = np.nanmedian(K, axis=0) if np.isfinite(K).any() else np.full(3, np.nan)
        return Pm, Km, len(rr), int(np.isfinite(P).all(axis=1).sum()), int(np.isfinite(K).all(axis=1).sum())

    A = posicao("A_parado")
    B = posicao("B_parado")
    A2 = posicao("A_retorno")

    def disp(P, Q):
        return float(np.linalg.norm(Q - P)) if np.isfinite(P).all() and np.isfinite(Q).all() else float("nan")

    def trajeto(fase):
        rr = por_fase(fase)
        P = np.array([_v(r, "par") for r in rr])
        K = np.array([_v(r, "kin") for r in rr])
        def comp(X):
            X = X[np.isfinite(X).all(axis=1)]
            if len(X) < 2:
                return float("nan"), 0
            # ignora saltos improváveis (> 0,25 m entre frames)
            d = np.linalg.norm(np.diff(X, axis=0), axis=1)
            d = d[d < 0.25]
            return float(np.nansum(d)), int(len(d))
        return comp(P), comp(K)

    res = {
        "lado": args.lado,
        "distancia_referencia_m": args.distancia_m,
        "frames": len(rows),
        "A_parado": {"par": A[0].tolist(), "kin": A[1].tolist(),
                     "n": A[2], "par_validos": A[3], "kin_validos": A[4]},
        "B_parado": {"par": B[0].tolist(), "kin": B[1].tolist(),
                     "n": B[2], "par_validos": B[3], "kin_validos": B[4]},
        "A_retorno": {"par": A2[0].tolist(), "kin": A2[1].tolist(),
                      "n": A2[2], "par_validos": A2[3], "kin_validos": A2[4]},
        "deslocamento_A_B_m": {
            "par": disp(A[0], B[0]),
            "kinect": disp(A[1], B[1]),
            "erro_par_vs_referencia_m": abs(disp(A[0], B[0]) - args.distancia_m),
            "erro_kinect_vs_referencia_m": abs(disp(A[1], B[1]) - args.distancia_m),
        },
        "drift_retorno_A_m": {
            "par": disp(A[0], A2[0]),
            "kinect": disp(A[1], A2[1]),
        },
        "trajeto_ida": None,
        "trajeto_volta": None,
    }
    for nome, fase in (("trajeto_ida", "ida_0p5m"), ("trajeto_volta", "volta_0p5m")):
        tp, tk = trajeto(fase)
        res[nome] = {"par": tp[0], "par_n": tp[1],
                     "kinect": tk[0], "kinect_n": tk[1]}
    # erro frame-a-frame par x Kinect
    P = np.array([_v(r, "par") for r in rows])
    K = np.array([_v(r, "kin") for r in rows])
    ok = np.isfinite(P).all(axis=1) & np.isfinite(K).all(axis=1)
    err = np.linalg.norm(P[ok] - K[ok], axis=1) * 1000.0
    res["erro_par_kinect_mm"] = {
        "n": int(ok.sum()),
        "mediana": float(np.median(err)) if len(err) else float("nan"),
        "p90": float(np.percentile(err, 90)) if len(err) else float("nan"),
    }
    (out / "resultado.json").write_text(json.dumps(res, indent=2,
                                                   ensure_ascii=False), encoding="utf-8")
    linhas = ["# Protocolo 0,5 m — par × Kinect", "",
              f"- lado: {args.lado}",
              f"- frames: {len(rows)}",
              f"- referência: {args.distancia_m:.3f} m", "",
              "| métrica | par aux | Kinect |", "|---|---:|---:|",
              f"| deslocamento A→B (m) | {res['deslocamento_A_B_m']['par']:.4f} | {res['deslocamento_A_B_m']['kinect']:.4f} |",
              f"| erro vs régua (m) | {res['deslocamento_A_B_m']['erro_par_vs_referencia_m']:.4f} | {res['deslocamento_A_B_m']['erro_kinect_vs_referencia_m']:.4f} |",
              f"| drift retorno A (m) | {res['drift_retorno_A_m']['par']:.4f} | {res['drift_retorno_A_m']['kinect']:.4f} |",
              f"| trajeto ida (m) | {res['trajeto_ida']['par']:.4f} | {res['trajeto_ida']['kinect']:.4f} |",
              f"| trajeto volta (m) | {res['trajeto_volta']['par']:.4f} | {res['trajeto_volta']['kinect']:.4f} |",
              "", f"Erro par×Kinect mediano: {res['erro_par_kinect_mm']['mediana']:.1f} mm "
              f"(p90 {res['erro_par_kinect_mm']['p90']:.1f} mm; n={res['erro_par_kinect_mm']['n']})"]
    (out / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))
    print("saida:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
