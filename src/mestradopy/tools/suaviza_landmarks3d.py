#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Suaviza landmarks 3D por track (Savitzky-Golay) e mede o jitter.

Motivo (revisÃ£o ATHENA/DLC): o ganho real do `triangulate_optim`/ATHENA Ã©
suavizaÃ§Ã£o temporal; nÃ£o Ã© necessÃ¡rio adotar a stack inteira. Este tool aplica
SavGol por track/junta, preserva a mÃ¡scara `valid` e gera um NPZ compatÃ­vel com
`handmvnet_export_2views.py`.

Uso:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\suaviza_landmarks3d.py ^
        --session F:\\handmvnet_videos\\teste_sync3 ^
        --npz F:\\handmvnet_videos\\teste_sync3\\processed_track1b\\landmarks_3d.npz ^
        --out outputs\\visao\\landmarks3d_suavizado_teste_sync3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None


def _jitter(xyz: np.ndarray, valid: np.ndarray, track: np.ndarray) -> Dict[str, float]:
    vals = []
    for tid in sorted(set(int(t) for t in track if int(t) >= 0)):
        idx = np.where(track == tid)[0]
        if len(idx) < 3:
            continue
        x = xyz[idx]
        v = valid[idx]
        for j in range(x.shape[1]):
            ok = v[:, j] & np.isfinite(x[:, j]).all(axis=1)
            if ok.sum() < 3:
                continue
            q = x[ok, j]
            # segunda diferenca (aceleracao discreta), em mm
            d2 = np.linalg.norm(np.diff(q, n=2, axis=0), axis=1) * 1000.0
            vals.extend(d2.tolist())
    if not vals:
        return {"n": 0, "mediana_mm": float("nan"), "p90_mm": float("nan")}
    arr = np.asarray(vals, dtype=np.float64)
    return {"n": int(arr.size), "mediana_mm": round(float(np.median(arr)), 4),
            "p90_mm": round(float(np.percentile(arr, 90)), 4)}


def suaviza(npz: Path, out: Path, *, window: int = 5, polyorder: int = 2,
            min_points: int = 5, only_tracks: bool = True) -> Dict[str, Any]:
    d = np.load(npz, allow_pickle=True)
    xyz = np.asarray(d["joints_xyz"], np.float64).copy()
    valid = np.asarray(d["valid"], dtype=bool).copy()
    track = (np.asarray(d["track_id"], np.int64).copy()
             if "track_id" in d.files else np.zeros(len(xyz), np.int64))
    tempo = (np.asarray(d["time_sync_s"], np.float64).copy()
             if "time_sync_s" in d.files else np.arange(len(xyz), dtype=np.float64))
    antes = _jitter(xyz, valid, track)
    out_xyz = xyz.copy()
    n_tracks = 0
    for tid in sorted(set(int(t) for t in track)):
        if tid < 0 and only_tracks:
            continue
        idx = np.where(track == tid)[0]
        if len(idx) < 3:
            continue
        idx = idx[np.argsort(tempo[idx])]
        n_tracks += 1
        for j in range(xyz.shape[1]):
            ok = valid[idx, j] & np.isfinite(xyz[idx, j]).all(axis=1)
            if int(ok.sum()) < max(3, min_points):
                continue
            pos = np.where(ok)[0]
            q = xyz[idx[pos], j]
            grid = np.arange(len(idx))
            # curva cheia: interpola lacunas para o filtro, mas so' devolve
            # aos frames com landmark valido.
            full = np.column_stack([
                np.interp(grid, pos.astype(float), q[:, dim])
                for dim in range(3)])
            w = int(min(window, len(full) if len(full) % 2 else len(full) - 1))
            if w < 3:
                continue
            if w % 2 == 0:
                w -= 1
            po = min(int(polyorder), w - 1)
            if savgol_filter is None:
                sm = full.copy()
                sm[1:-1] = (full[:-2] + full[1:-1] + full[2:]) / 3.0
            else:
                sm = savgol_filter(full, window_length=w, polyorder=po, axis=0,
                                   mode="interp")
            out_xyz[idx[pos], j] = sm[pos]
    depois = _jitter(out_xyz, valid, track)
    out.mkdir(parents=True, exist_ok=True)
    npz_out = out / "landmarks_3d_suavizado.npz"
    arrays = {k: d[k] for k in d.files if k not in ("joints_xyz", "meta_json")}
    arrays["joints_xyz"] = out_xyz.astype(np.float32)
    arrays["joints_xyz_antes"] = xyz.astype(np.float32)
    meta = {
        "source": str(npz),
        "window": int(window), "polyorder": int(polyorder),
        "only_tracks": bool(only_tracks), "n_tracks_suavizados": n_tracks,
        "jitter_antes": antes, "jitter_depois": depois,
        "reducao_mediana_pct": (round(100.0 * (1.0 - depois["mediana_mm"]
                                              / antes["mediana_mm"]), 2)
                                if antes.get("mediana_mm") and antes["mediana_mm"] > 0
                                and np.isfinite(depois.get("mediana_mm", np.nan))
                                else None),
        "note": ("SavGol por track/junta; valid original preservado; "
                 "joints_xyz_antes incluido para auditoria."),
    }
    np.savez_compressed(npz_out, **arrays)
    (out / "smoothing_report.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta, str(npz_out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Suaviza landmarks 3D por track")
    p.add_argument("--npz", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--polyorder", type=int, default=2)
    p.add_argument("--min-points", type=int, default=5)
    p.add_argument("--incluir-sem-track", action="store_true",
                   help="tambem suaviza registros com track_id=-1 (nao recomendado)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    meta, path = suaviza(args.npz, args.out, window=args.window,
                         polyorder=args.polyorder, min_points=args.min_points,
                         only_tracks=not args.incluir_sem_track)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("salvo:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

