#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analisa o CSV cru de `aquisicao_ruido_eeg.py`.

Mede, por condição:
  - PSD mediana (todos os canais);
  - potência em 50 Hz e 60 Hz e harmônicos (100/120/180/240);
  - banda larga 20-100 Hz;
  - correlação entre canais (modo comum);
  - desvio-padrão por canal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfiltfilt

FS = 500.0


def _le_csv(caminho: Path):
    linhas = []
    with open(caminho, encoding="utf-8", errors="replace") as f:
        for r in f:
            if not r or r.startswith("#") or r.lower().startswith("time"):
                continue
            partes = [p.strip() for p in r.replace(";", ",").split(",")]
            try:
                linhas.append([float(p) for p in partes])
            except ValueError:
                continue
    arr = np.asarray(linhas, np.float64)
    time = arr[:, 0]
    marker = arr[:, -1].astype(int)
    eeg = arr[:, 1:-1]
    return time, eeg, marker


def _psd(x):
    f, p = welch(x, fs=FS, nperseg=min(4096, x.shape[0]), axis=0)
    return f, p


def _rms_banda(x, lo, hi):
    sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="bandpass", output="sos")
    y = sosfiltfilt(sos, x, axis=0)
    return float(np.median(np.std(y, axis=0)))


def _pot(p, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(p[m].mean()) if np.any(m) else 0.0


def analisa(caminho: Path, out: Path, meta: dict | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    time, eeg, marker = _le_csv(caminho)
    if eeg.size == 0:
        raise RuntimeError("CSV sem dados numéricos")
    codes = [c for c in np.unique(marker) if c > 0]
    nomes = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    if meta and meta.get("eventos"):
        mapa = {int(e["code"]): e["condicao"] for e in meta["eventos"]}
    else:
        mapa = {v: k for k, v in nomes.items()}
    resultados = {}
    psds = {}
    alvo = float(meta.get("duracao_alvo_s", 60.0)) if meta else 60.0
    for code in codes:
        idx = np.where(marker == code)[0]
        if not len(idx):
            continue
        ini = int(idx[0])
        prox = [int(np.where(marker == c2)[0][0]) for c2 in codes
                if c2 != code and np.any(np.where(marker == c2)[0] > ini)]
        fim = min(prox) if prox else len(time)
        if fim - ini < int(2 * FS):
            continue
        nome = mapa.get(int(code), f"C{code}")
        # ignora 1 s de estabilizacao e limita a janela alvo de aquisicao
        fim = min(fim, ini + int((1.0 + alvo) * FS))
        seg = eeg[ini + int(FS):fim]
        seg = seg - np.mean(seg, axis=0)
        f, p = _psd(seg)
        pmed = np.median(p, axis=1)
        total = _pot(pmed, f, 1, 100)
        r = {
            "n_amostras": int(seg.shape[0]),
            "duracao_s": round(float(seg.shape[0] / FS), 2),
            "std_mediano_uV": round(float(np.median(np.std(seg, axis=0))), 3),
            "pot_1_100": round(total, 4),
            "rms_50_uV": round(_rms_banda(seg, 49.5, 50.5), 3),
            "rms_60_uV": round(_rms_banda(seg, 59.5, 60.5), 3),
            "ratio_50": round(_pot(pmed, f, 49.5, 50.5) / max(total, 1e-20), 5),
            "ratio_60": round(_pot(pmed, f, 59.5, 60.5) / max(total, 1e-20), 5),
            "ratio_100": round(_pot(pmed, f, 99.5, 100.5) / max(total, 1e-20), 5),
            "ratio_120": round(_pot(pmed, f, 119.5, 120.5) / max(total, 1e-20), 5),
            "ratio_180": round(_pot(pmed, f, 179.5, 180.5) / max(total, 1e-20), 5),
            "ratio_20_100": round(_pot(pmed, f, 20, 100) / max(total, 1e-20), 5),
            "psd_40_55": round(_pot(pmed, f, 40, 55), 3),
            "psd_65_80": round(_pot(pmed, f, 65, 80), 3),
            "psd_80_95": round(_pot(pmed, f, 80, 95), 3),
            "corr_media_canais": round(float(np.median([
                np.corrcoef(seg[:, i], seg[:, j])[0, 1]
                for i in range(min(8, seg.shape[1]))
                for j in range(i + 1, min(8, seg.shape[1]))
            ])), 3) if seg.shape[1] > 1 else float("nan"),
        }
        resultados[nome] = r
        psds[nome] = (f, pmed)
        print(f"{nome}: {r}")
    if psds:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for nome, (f, p) in psds.items():
            ax.semilogy(f, p, label=nome, lw=1.2)
        for freq in (50, 60, 100, 120, 180):
            ax.axvline(freq, color="gray", alpha=.15, lw=.8)
        ax.set_xlim(1, 120); ax.set_xlabel("Hz"); ax.set_ylabel("PSD mediana")
        ax.set_title("PSD por condição — EEG cru")
        ax.grid(alpha=.2, which="both"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "psd_condicoes.png", dpi=160)
        plt.close(fig)
    (out / "resultado.json").write_text(
        json.dumps({"csv": str(caminho), "condicoes": resultados,
                    "mapa": mapa}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    linhas = ["# Análise de ruído EEG cru", "",
              "| cond | dur | std | RMS60 | RMS50 | 50/60 | 60Hz | 100 | 120 | 180 | 40-55 | 65-80 | 20-100 | corr |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for nome, r in resultados.items():
        linhas.append(f"| {nome} | {r['duracao_s']}s | {r['std_mediano_uV']} | "
                      f"{r['rms_60_uV']} | {r['rms_50_uV']} | "
                      f"{r['ratio_50']}/{r['ratio_60']} | "
                      f"{r['ratio_60']} | {r['ratio_100']} | "
                      f"{r['ratio_120']} | {r['ratio_180']} | {r['psd_40_55']} | "
                      f"{r['psd_65_80']} | {r['ratio_20_100']} | "
                      f"{r['corr_media_canais']} |")
    (out / "relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return resultados


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    args = ap.parse_args()
    meta = json.loads(args.meta.read_text(encoding="utf-8")) if args.meta else None
    out = args.saida or (args.csv.parent / "analise_ruido")
    analisa(args.csv, out, meta)
    print("saida:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
