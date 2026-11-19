#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QC offline dos landmarks 3D processados (par aux0↔aux1).

Lê `landmarks_3d.npz` (ou `landmarks_3d_suavizado.npz`) e gera:
  - qc_report.json/md: tracks, cobertura, reprojecao, jitter, rigidez;
  - bones.csv: comprimento de cada osso por track (mediana, std, CV bruto e
    robusto);
  - frames.csv: metricas por frame/track (cobertura, reproj, jitter local).

Uso:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\qc_landmarks3d.py ^
        --npz F:\\handmvnet_videos\\teste_sync3\\processed_track1b\\landmarks_3d.npz ^
        --out outputs\\visao\\qc_teste_sync3
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys
_SRC = Path(__file__).resolve().parents[1]
for _p in (_SRC, _SRC / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from kinect_imu_groundtruth import HAND_CONNECTIONS  # noqa: E402

BONES = tuple(HAND_CONNECTIONS)


def _cv(vals):
    a = np.asarray(vals, np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 3:
        return None
    med = float(np.median(a)); mad = float(np.median(np.abs(a - med)))
    sigma = 1.4826 * mad
    keep = np.abs(a - med) <= 3.0 * sigma if sigma > 0 else np.ones(a.size, bool)
    b = a[keep]
    if b.size < 3:
        b = a
    return {"n": int(a.size), "n_limpos": int(b.size),
            "n_outliers": int(a.size - b.size),
            "mediana_mm": round(med * 1000.0, 3),
            "std_mm": round(float(np.std(a)) * 1000.0, 3),
            "cv_pct": round(100.0 * float(np.std(a)) / max(med, 1e-9), 3),
            "cv_robusto_pct": round(100.0 * float(np.std(b)) / max(med, 1e-9), 3),
            "mad_mm": round(mad * 1000.0, 3)}


def _jitter_xyz(xyz, valid):
    vals = []
    for j in range(21):
        q = xyz[:, j]
        ok = valid[:, j] & np.isfinite(q).all(axis=1)
        if ok.sum() < 3:
            continue
        # segunda diferenca nos frames validos consecutivos
        d2 = np.linalg.norm(np.diff(q, n=2, axis=0), axis=1) * 1000.0
        vals.extend(d2.tolist())
    if not vals:
        return {"n": 0, "mediana_mm": float("nan"), "p90_mm": float("nan")}
    a = np.asarray(vals, np.float64)
    return {"n": int(a.size), "mediana_mm": round(float(np.median(a)), 4),
            "p90_mm": round(float(np.percentile(a, 90)), 4)}


def analisa(npz_path: Path, out: Path) -> Dict[str, Any]:
    d = np.load(npz_path, allow_pickle=True)
    xyz = np.asarray(d["joints_xyz"], np.float64)
    valid = np.asarray(d["valid"], dtype=bool)
    track = (np.asarray(d["track_id"], np.int64)
             if "track_id" in d.files else np.zeros(len(xyz), np.int64))
    hand = (np.asarray(d["hand"]).astype(str)
            if "hand" in d.files else np.array(["unknown"] * len(xyz)))
    tempo = (np.asarray(d["time_sync_s"], np.float64)
             if "time_sync_s" in d.files else np.arange(len(xyz), np.float64))
    reproj0 = np.asarray(d["reproj_aux0"], np.float64)
    reproj1 = np.asarray(d["reproj_aux1"], np.float64)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(int(t) for t in track if int(t) >= 0))
    tracks_info = []
    bones_rows = []
    for tid in ids:
        idx = np.where(track == tid)[0]
        idx = idx[np.argsort(tempo[idx])]
        x = xyz[idx]; v = valid[idx]; t = tempo[idx]
        n = len(idx)
        counts = {str(k): int(v[:, k].sum()) for k in range(21)}
        dur = float(t[-1] - t[0]) if n else 0.0
        tracks_info.append({
            "track_id": int(tid), "n": int(n), "t0": float(t[0]),
            "t1": float(t[-1]), "dur_s": round(dur, 3),
            "hands": {h: int((hand[idx] == h).sum())
                      for h in sorted(set(hand[idx].tolist()))},
            "valid_mean": round(float(v.mean()), 3),
            "valid_min_joint": min(counts.values()) if counts else 0,
            "valid_max_joint": max(counts.values()) if counts else 0,
            "reproj_aux0_med_px": round(float(np.nanmedian(reproj0[idx])), 3),
            "reproj_aux1_med_px": round(float(np.nanmedian(reproj1[idx])), 3),
            "jitter_2a_mm": _jitter_xyz(x, v),
        })
        for k, (a, b) in enumerate(BONES):
            ok = v[:, a] & v[:, b] & np.isfinite(x[:, a]).all(axis=1) \
                & np.isfinite(x[:, b]).all(axis=1)
            ln = np.linalg.norm(x[ok, a] - x[ok, b], axis=1) if ok.any() else np.array([])
            cv = _cv(ln)
            if cv is not None:
                bones_rows.append({"track_id": tid, "osso": f"{a}-{b}", **cv})
    # frames
    frames_rows = []
    for i in range(len(xyz)):
        v = valid[i]
        nv = int(v.sum())
        med0 = float(np.nanmedian(reproj0[i])) if all(
            np.isfinite(reproj0[i])) else float("nan")
        med1 = float(np.nanmedian(reproj1[i])) if all(
            np.isfinite(reproj1[i])) else float("nan")
        frames_rows.append({
            "i": i, "t": float(tempo[i]), "track_id": int(track[i]),
            "hand": str(hand[i]), "n_valid": nv,
            "reproj_aux0_med_px": round(med0, 3) if np.isfinite(med0) else "",
            "reproj_aux1_med_px": round(med1, 3) if np.isfinite(med1) else "",
        })
    # resumo de ossos por lado
    lados = {}
    for row in bones_rows:
        # recupera lado majoritario do track
        tid = row["track_id"]; q = track == tid
        if not q.any():
            continue
        lado = max(set(hand[q].tolist()), key=hand[q].tolist().count)
        lados.setdefault(lado, []).append(row)
    resumo_lados = {}
    for lado, rows in lados.items():
        cv = [r["cv_pct"] for r in rows]
        cvr = [r["cv_robusto_pct"] for r in rows]
        resumo_lados[lado] = {
            "n_ossos": len(rows),
            "cv_mediana_pct": round(float(np.median(cv)), 3),
            "cv_p90_pct": round(float(np.percentile(cv, 90)), 3),
            "cv_robusto_mediana_pct": round(float(np.median(cvr)), 3),
        }
    report = {
        "source": str(npz_path),
        "n_records": len(xyz),
        "n_tracks": len(ids),
        "tracks": tracks_info,
        "sides": resumo_lados,
        "global": {
            "valid_mean": round(float(valid.mean()), 4),
            "reproj_aux0_med_px": round(float(np.nanmedian(reproj0)), 3),
            "reproj_aux1_med_px": round(float(np.nanmedian(reproj1)), 3),
            "jitter_2a_mm": _jitter_xyz(xyz, valid),
        },
    }
    (out / "qc_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "bones.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bones_rows[0].keys()) if bones_rows
                           else ["track_id", "osso"])
        w.writeheader()
        for row in bones_rows:
            w.writerow(row)
    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frames_rows[0].keys()))
        w.writeheader(); w.writerows(frames_rows)
    linhas = ["# QC landmarks 3D", "", f"- Fonte: `{npz_path}`",
              f"- Registros: {len(xyz)}",
              f"- Tracks: {len(ids)}", ""]
    for side, r in resumo_lados.items():
        linhas.append(f"- {side}: CV mediana {r['cv_mediana_pct']}% "
                      f"(robusto {r['cv_robusto_mediana_pct']}%), "
                      f"{r['n_ossos']} ossos")
    for tr in tracks_info:
        linhas.append(f"- track {tr['track_id']}: n={tr['n']}, "
                      f"dur={tr['dur_s']}s, valid_mean={tr['valid_mean']}, "
                      f"jitter={tr['jitter_2a_mm']['mediana_mm']}mm")
    (out / "qc_report.md").write_text("\n".join(linhas) + "\n",
                                      encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rep = analisa(args.npz, args.out)
    print(f"QC: {rep['n_records']} registros, {rep['n_tracks']} tracks")
    for side, r in rep["sides"].items():
        print(f"  {side}: CV mediana {r['cv_mediana_pct']}% "
              f"(robusto {r['cv_robusto_mediana_pct']}%)")
    print("salvo:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
