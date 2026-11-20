#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Varredura de offset temporal entre o marcador 780 e a MI real.

Para cada shift delta (segundos), recorta a MI como [780+delta, 780+delta+4s]
e roda CSP+LDA com 5-fold agrupado por trial. Se o melhor delta for diferente
de zero de forma reprodutÃ­vel, hÃ¡ evidÃªncia de offset marcadorâ†”MI.

IMPORTANTE: varredura Ã© exploratÃ³ria (mÃºltiplas comparaÃ§Ãµes); o pico nÃ£o Ã©
prova definitiva.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parents[1]
for _p in (_SRC, _SRC / "eeg", _SRC / "tools", _SRC / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eeg.mi_eegnet.data import _load_own_csv, filt_resample, CH_OWN_IDX  # noqa: E402
from analisa_janela_mi import _pipe, avalia_cv  # noqa: E402

FS_ORIG = 500.0
FS = 128
WIN = 2.0
STEP = 0.5
TRIAL = 4.0
SHIFT_MIN = -1.0
SHIFT_MAX = 1.0
SHIFT_STEP = 0.1


def carrega_sessao(pasta: Path):
    csv_path = sorted(Path(pasta).glob("eeg_mi3c*.csv"))[-1]
    eeg, markers, _ = _load_own_csv(csv_path)
    eeg = eeg[:, CH_OWN_IDX]
    eeg = filt_resample(eeg.T, FS_ORIG, FS, 8.0, 30.0).T
    # posiÃ§Ãµes dos marcadores na grade de 128 Hz
    mi = (np.where(markers == 780)[0] * FS / FS_ORIG).round().astype(int)
    # rÃ³tulo por Ãºltimo cue antes de cada 780
    labels = []
    for a in np.where(markers == 780)[0]:
        prev = markers[:a]
        ult = -1; lab = -1
        for code, cls in ((769, 0), (770, 1)):
            pos = np.where(prev == code)[0]
            if len(pos) and pos[-1] > ult:
                ult = pos[-1]; lab = cls
        labels.append(lab)
    return eeg, mi, np.array(labels, int)


def monta(eeg, mi, labels, delta, janela=WIN):
    nwin = int(janela * FS); step = int(STEP * FS); ntr = int(janela * FS)
    X, y, g = [], [], []
    gid = 0
    for a, lab in zip(mi, labels):
        if lab not in (0, 1):
            continue
        start = int(a + round(delta * FS))
        end = start + ntr
        if start < 0 or end > eeg.shape[0]:
            continue
        seg = eeg[start:end].T
        for w in range(0, seg.shape[1] - nwin + 1, step):
            X.append(seg[:, w:w + nwin]); y.append(lab); g.append(gid)
        gid += 1
    return np.stack(X), np.array(y), np.array(g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alvo", type=Path, nargs="+",
                    default=[Path("outputs/eeg/piloto_me_mi_s01"),
                             Path("outputs/eeg/piloto_me_mi_s02")])
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/offset_marcador"))
    ap.add_argument("--janela", type=float, default=WIN,
                    help="duracao da janela de MI (s); 2 evita invadir o ITI")
    ap.add_argument("--shift-min", type=float, default=SHIFT_MIN)
    ap.add_argument("--shift-max", type=float, default=SHIFT_MAX)
    ap.add_argument("--shift-step", type=float, default=SHIFT_STEP)
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    sessoes = [carrega_sessao(p) for p in args.alvo]
    shifts = np.round(np.arange(args.shift_min, args.shift_max + 1e-9,
                               args.shift_step), 3)
    resultados = {}
    for delta in shifts:
        Xs, ys, gs = [], [], []; off = 0
        for eeg, mi, labels in sessoes:
            X, y, g = monta(eeg, mi, labels, float(delta), args.janela)
            Xs.append(X); ys.append(y); gs.append(g + off)
            off += int(g.max()) + 1
        X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
        r = avalia_cv(X, y, g)
        resultados[str(float(delta))] = r
        print(f"delta={delta:+.2f}s BA={r['ba']:.3f}", flush=True)
    (args.saida / "resultado.json").write_text(
        json.dumps({"shifts_s": [float(x) for x in shifts],
                    "resultados": resultados,
                    "observacao": ("varredura exploratoria; MI recortada em "
                                   "780+delta por 4 s; 5-fold por trial.")},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    deltas = [float(k) for k in resultados]
    bas = [resultados[k]["ba"] for k in resultados]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(deltas, bas, "o-", color="tab:blue")
    ax.axvline(0, color="gray", ls="--", label="marcador 780")
    ax.set_xlabel("offset aplicado ao marcador (s)")
    ax.set_ylabel("BA (5-fold agrupado)")
    ax.set_title("Varredura de offset marcador Ã— MI")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(args.saida / "offset_scan.png", dpi=160)
    plt.close(fig)
    melhor = max(resultados, key=lambda k: resultados[k]["ba"])
    linhas = ["# Varredura de offset do marcador 780", "",
              f"- melhor offset: **{float(melhor):+.2f} s** "
              f"(BA={resultados[melhor]['ba']:.3f})",
              f"- BA em 0 s: {resultados.get('0.0', {}).get('ba', float('nan')):.3f}",
              "", "| offset (s) | BA | n trials |", "|---:|---:|---:|"]
    for k in sorted(resultados, key=lambda z: float(z)):
        linhas.append(f"| {float(k):+.2f} | {resultados[k]['ba']:.3f} | "
                      f"{resultados[k]['n_trials']} |")
    (args.saida / "relatorio.md").write_text("\n".join(linhas) + "\n",
                                             encoding="utf-8")
    print("\nmelhor:", melhor, resultados[melhor])
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

