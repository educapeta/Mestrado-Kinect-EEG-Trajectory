#!/usr/bin/env python
"""Refit de T/f a partir de um CSV do FusionCsvLogger.

Usa as colunas vis_n_* (normal da visao no frame do Kinect, Y-up) e
imu_roll/pitch/yaw para resolver `n_vis = T @ R_imu @ f` com
`solve_palm_alignment`, o mesmo modelo do Y. Serve para refinar T/f usando
logs do multi-view sem nova captura.

Uso:
    python tools/refit_tf_log.py --log hoje_mv_yfix_movimento.csv
    python tools/refit_tf_log.py --log ... --salvar

Sem --salvar apenas mostra o residuo; com --salvar grava o JSON oficial com
backup .json.bak (so' se o residuo estiver dentro do limite).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imu import (  # noqa: E402
    PALM_ALIGNMENT_MAX_RESIDUAL_DEG,
    PALM_ALIGNMENT_MIN_SAMPLES,
    rotation_matrix,
    solve_palm_alignment,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refit T/f a partir de CSV de fusao")
    p.add_argument("--log", action="append", required=True,
                   help="CSV do FusionCsvLogger (pode repetir)")
    p.add_argument("--lado", choices=["right", "left", "ambos"], default="ambos")
    p.add_argument("--sem-edge", action="store_true",
                   help="descarta frames edge_on=1")
    p.add_argument("--sem-held", action="store_true",
                   help="descarta frames held=1")
    p.add_argument("--min-amostras", type=int,
                   default=PALM_ALIGNMENT_MIN_SAMPLES)
    p.add_argument("--salvar", action="store_true")
    p.add_argument("--dir-saida", default=".")
    return p


def carregar_amostras(caminho: Path, lado: str, sem_edge: bool,
                      sem_held: bool):
    df = pd.read_csv(caminho)
    df = df[df["hand"] == lado]
    if "valid" in df:
        df = df[df["valid"].astype(str).isin(["1", "True", "true"])]
    if sem_edge and "vis_edge_on" in df:
        df = df[~df["vis_edge_on"].astype(str).isin(["1", "True", "true"])]
    if sem_held and "vis_held" in df:
        df = df[~df["vis_held"].astype(str).isin(["1", "True", "true"])]
    rotacoes = []
    normais = []
    for _, row in df.iterrows():
        try:
            rpy = (float(row["imu_roll_deg"]), float(row["imu_pitch_deg"]),
                   float(row["imu_yaw_deg"]))
            n = np.array([float(row["vis_n_x"]), float(row["vis_n_y"]),
                          float(row["vis_n_z"])], dtype=np.float64)
        except (TypeError, ValueError, KeyError):
            continue
        if not (np.isfinite(rpy).all() and np.isfinite(n).all()):
            continue
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        rotacoes.append(rotation_matrix(*rpy))
        normais.append(n / nn)
    return rotacoes, normais


def salvar_alinhamento(diretorio: Path, lado: str, rotacao: np.ndarray,
                      corpo: np.ndarray, residuo: float) -> Path:
    caminho = diretorio / f"imu_palm_alignment_{lado}.json"
    if caminho.exists():
        shutil.copy2(caminho, caminho.with_suffix(".json.bak"))
    dados = {
        "rotation": np.asarray(rotacao, np.float64).tolist(),
        "body": np.asarray(corpo, np.float64).tolist(),
        "residual_deg": float(residuo),
        "yaw_boot_delta_deg": 0.0,
        "source": "refit_multiview_log",
    }
    caminho.write_text(json.dumps(dados, indent=2), encoding="utf-8")
    return caminho


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dir_saida = Path(args.dir_saida)
    lados = ["right", "left"] if args.lado == "ambos" else [args.lado]
    for lado in lados:
        rots = []
        nrms = []
        for nome in args.log:
            caminho = Path(nome)
            if not caminho.exists():
                print(f"[AVISO] nao existe: {caminho}")
                continue
            r, n = carregar_amostras(caminho, lado, args.sem_edge,
                                     args.sem_held)
            rots.extend(r)
            nrms.extend(n)
            print(f"  {caminho.name}: {len(r)} amostras uteis")
        if len(rots) < args.min_amostras:
            print(f"{lado}: insuficiente ({len(rots)} < {args.min_amostras})")
            continue
        resolved = solve_palm_alignment(rots, nrms)
        if resolved is None:
            print(f"{lado}: refit falhou")
            continue
        t_lado, f_lado, residuo = resolved
        print(f"{lado}: n={len(rots)} residuo_refit={residuo:.2f} deg")
        if residuo > PALM_ALIGNMENT_MAX_RESIDUAL_DEG:
            print(f"  residuo > {PALM_ALIGNMENT_MAX_RESIDUAL_DEG:g}; NAO salvar")
            continue
        if args.salvar:
            caminho = salvar_alinhamento(dir_saida, lado, t_lado, f_lado,
                                         residuo)
            print(f"  gravado {caminho} (backup .json.bak)")
        else:
            print("  dry-run: use --salvar para gravar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
