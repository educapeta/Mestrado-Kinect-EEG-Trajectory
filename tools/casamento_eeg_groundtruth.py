#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Casamento groundtruth x EEG: funcoes puras + verificacao com os dados reais.

Tres eixos de casamento, os TRES existindo nas gravacoes de hoje:

  1) RELOGIO: o `*_eventos.json` traz, para cada evento, `monotonic_s`
     (time.monotonic) e `sample_index` (indice 0-based da linha do EEG).
     Duas ancoras bastam para mapear qualquer timestamp de camera/IMU
     (`t_mono_s` do *_movimento.csv, timestamps do overlay_deviation_log)
     em sample_index. MAIS ROBUSTO do que fs fixa: medido em 16/10/2026, a
     ancora da sessao deu 486,26 Hz enquanto a coluna `Time` do CSV tem os
     500,0 Hz nominais exatos -- a ancora inclui o jitter de escrita do
     pipeline e NAO deve ser usada para estimar FS (só para POSICAO).

  2) MARCADOR: o CSV de EEG tem coluna `Marker` com os codigos dos eventos
     (899/898 = link perdido/recuperado). O casamento e' codigo -> amostra.

  3) MOVIMENTO: *_movimento.csv a 30 Hz com `t_mono_s` por linha -> cada
     linha vira (sample_index, coluna_de_valor).

Uso (diagnostico de uma sessao):
    .venv\\Scripts\\python.exe tools\\casamento_eeg_groundtruth.py --prefixo gravacao_MEMI_20261016_153213
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------- funcoes puras

def mono_para_sample_index(ts_mono, ancoras):
    """Interpola timestamps monotonicos em sample_index pelas ancoras.

    ancoras: lista de pares (t_mono, sample_index), ORDENADA por t_mono.
    Fora do intervalo ancorado e' clampado na primeira/ultima ancora (o
    movimento comeca um pouco antes e termina um pouco depois do primeiro/
    ultimo evento -- medido, 4% das linhas ficam fora).
    """
    a = np.asarray(ancoras, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2:
        raise ValueError("sao necessarias 2+ ancoras (t_mono, sample_index)")
    t = np.asarray(ts_mono, dtype=float)
    return np.interp(t, a[:, 0], a[:, 1])


def ancora_dos_eventos(eventos):
    """Ancoras (monotonic_s, sample_index) dos eventos que trazem os dois."""
    pts = []
    for e in eventos or []:
        if not isinstance(e, dict):
            continue
        t = e.get("monotonic_s", e.get("t_mono_s"))
        si = e.get("sample_index")
        if t is not None and si is not None:
            pts.append((float(t), int(si)))
    return sorted(pts)


def marcadores_do_csv(marker, codigos=None):
    """Primeira amostra de cada codigo da coluna Marker (0 = sem marcador)."""
    out = {}
    for codigo in np.unique(marker):
        if codigo == 0:
            continue
        if codigos is not None and int(codigo) not in codigos:
            continue
        m = np.flatnonzero(marker == codigo)
        out[int(codigo)] = (int(m[0]), int(m.size))
    return out


# ------------------------------------------------------------------ diagnostico

def carrega_sessao(prefixo, pasta="gravacoes"):
    eeg = sorted(glob.glob(os.path.join(pasta, prefixo + "_*.csv")))
    eeg = [p for p in eeg if not p.endswith("_movimento.csv")]
    mov = os.path.join(pasta, prefixo + "_movimento.csv")
    # o JSON pode nascer alinhado ao CSV real (prefixo_<stamp>_eventos.json)
    # ou, em sessoes interrompidas, com o nome pedido (prefixo_eventos.json)
    ev = sorted(glob.glob(os.path.join(pasta, prefixo + "_*_eventos.json"))
                + glob.glob(os.path.join(pasta, prefixo + "_eventos.json")))
    return (eeg[-1] if eeg else None,
            mov if os.path.exists(mov) else None,
            ev[-1] if ev else None)


def le_csv_eeg(caminho):
    """Le Time + Marker do CSV de EEG (pula o cabecalho do g.tec '# ...')."""
    cab, n_skip = [], 0
    with open(caminho, "r", encoding="utf-8", errors="replace") as f:
        for linha in f:
            n_skip += 1
            if not linha.startswith("#"):
                cab = [c.strip() for c in linha.rstrip("\r\n").split(",")]
                break
    if not cab:
        return None, None, ["(arquivo vazio)"]
    col_time = cab.index("Time") if "Time" in cab else 0
    col_marker = cab.index("Marker") if "Marker" in cab else None
    usados = [col_time] + ([col_marker] if col_marker is not None else [])
    dados = np.loadtxt(caminho, delimiter=",", skiprows=n_skip,
                       usecols=usados, ndmin=2)
    return (dados[:, 0],
            dados[:, 1] if col_marker is not None else None, cab)


def diagnostico(prefixo, pasta="gravacoes", fs_nominal=500.0, stream=None):
    w = stream.write if stream is not None else (lambda s: None)
    eeg, mov, ev = carrega_sessao(prefixo, pasta)
    w(f"== CASAMENTO groundtruth x EEG -- {prefixo} ==\n")
    w(f"eeg:       {eeg or '(nao achou)'}\n")
    w(f"movimento: {mov or '(nao achou)'}\n")
    w(f"eventos:   {ev or '(nao achou)'}\n")
    if eeg is None:
        w("SEM CSV de EEG: nada a casar.\n")
        return 2

    time_s, marker, cab = le_csv_eeg(eeg)
    w("\n-- CSV de EEG --\n")
    w(f"amostras:  {time_s.size}   Time: {time_s[0]:.4f} .. {time_s[-1]:.4f} s"
      f"  (duracao {time_s[-1] - time_s[0]:.1f} s)\n")
    dt = np.diff(time_s)
    fs_csv = 1.0 / np.median(dt)
    w(f"passo:     med {np.median(dt) * 1e3:.4f} ms -> {fs_csv:.1f} Hz"
      f"  (min {dt.min() * 1e3:.4f}  max {dt.max() * 1e3:.4f})\n")
    if marker is not None:
        vals = np.unique(marker[marker != 0])
        w(f"Marker:    {int((marker != 0).sum())} amostras marcadas,"
          f" {vals.size} codigos distintos\n")

    eventos, meta = [], {}
    if ev:
        with open(ev, "r", encoding="utf-8") as f:
            bruto = json.load(f)
        if isinstance(bruto, dict):
            eventos, meta = bruto.get("eventos", []), bruto.get("meta", {})
        else:
            eventos = bruto
        w(f"\n-- Eventos ({len(eventos)}) --\n")
        if meta:
            w("meta: " + "  ".join(
                f"{k}={meta[k]}" for k in
                ("participante", "sessao", "fs_eeg_hz", "inicio_monotonic_s")
                if k in meta) + "\n")

    ancoras = ancora_dos_eventos(eventos)
    if len(ancoras) >= 2:
        (t0, si0), (t1, si1) = ancoras[0], ancoras[-1]
        w(f"\n-- ANCORAS mono <-> sample_index ({len(ancoras)} eventos) --\n")
        w(f"primeira:  t_mono {t0:.3f} s -> sample_index {si0}\n")
        w(f"ultima:    t_mono {t1:.3f} s -> sample_index {si1}\n")
        fs_ancora = (si1 - si0) / (t1 - t0)
        w(f"FS da ancora: {fs_ancora:.3f} Hz x {fs_csv:.1f} Hz da coluna Time"
          f"  ({(fs_ancora - fs_csv) / fs_csv * 1e6:.0f} ppm)\n")
        # A ancora so' e' confiavel se abrange a gravacao de verdade: ancoras
        # coladas (ex.: 899 link perdido -> 898 recuperado, com o link morto
        # no meio) nao refletem a taxa -- a janela ancora quase nada.
        if (si1 - si0) < 0.5 * time_s.size:
            w("AVISO: as ancoras cobrem so' "
              f"{100.0 * (si1 - si0) / time_s.size:.0f}% das amostras do CSV\n"
              "      (link perdido no meio, ou sessao interrompida cedo).\n"
              "      O mapeamento mono->sample_index e' valido SO' na faixa\n"
              "      entre as ancoras; fora dela e' clamp/extrapolacao.\n")
        else:
            w("NOTA: a ancora serve para POSICAO (mono -> sample_index); quem\n"
              "      dita o FS e' a coluna Time (a ancora mede o jitter de\n"
              "      escrita do pipeline, nao a taxa do ADC).\n")
    else:
        w("\n-- ANCORAS: menos de 2 eventos com monotonic_s + sample_index --\n")
        t0 = si0 = t1 = si1 = None
        fs_ancora = float("nan")

    if marker is not None and eventos:
        codigos = {}
        for e in eventos:
            if isinstance(e, dict) and e.get("code") is not None:
                codigos.setdefault(int(e["code"]), e)
        achados = []
        for codigo in sorted(codigos):
            m = np.flatnonzero(marker == codigo)
            if m.size:
                e = codigos[codigo]
                si = e.get("sample_index")
                achados.append((codigo, int(m[0]), int(m.size), si))
        w(f"\n-- Casamento Marker(CSV) x codigos(JSON):"
          f" {len(achados)}/{len(codigos)} codigos encontrados --\n")
        for codigo, linha, n, si in achados[:6]:
            delta = f"delta {linha - si:+d}" if si is not None else "?"
            w(f"  codigo {codigo:>4}: 1a amostra Marker {linha}"
              f"  ({n} amostras)  |  sample_index JSON {si}  |  {delta}\n")
        perdidos = [c for c in sorted(codigos) if not (marker == c).any()]
        if perdidos:
            w(f"  codigos do JSON AUSENTES no Marker: {perdidos[:12]}"
              f"{' ...' if len(perdidos) > 12 else ''}\n")
    elif marker is None:
        w("\n-- Marker: CSV SEM coluna Marker -- casamento por marcador"
          " INDISPONIVEL.\n")

    if mov and t0 is not None:
        with open(mov, "r", encoding="utf-8", errors="replace") as f:
            cab_mov = [c.strip() for c in f.readline().rstrip("\r\n").split(",")]
        if "t_mono_s" in cab_mov:
            idx = cab_mov.index("t_mono_s")
            tsm = np.atleast_1d(np.loadtxt(mov, delimiter=",", skiprows=1,
                                           usecols=idx))
            dentro = (tsm >= t0) & (tsm <= t1)
            w(f"\n-- Movimento/cameras ({tsm.size} linhas) --\n")
            w(f"linhas com t_mono_s DENTRO da ancora: {int(dentro.sum())}"
              f" ({100.0 * dentro.sum() / max(1, tsm.size):.1f}%)\n")
            if dentro.any():
                si = mono_para_sample_index(tsm[dentro], ancoras)
                w(f"sample_index equivalente: {si.min():.0f} .. {si.max():.0f}\n")
                w("CASAMENTO: SIM -- cada linha de movimento vira"
                  " (sample_index, valor).\n")
            else:
                w("CASAMENTO: NAO -- nenhuma linha de movimento cai na ancora.\n")
        else:
            w(f"\n-- Movimento: SEM coluna `t_mono_s` (colunas:"
              f" {cab_mov[:8]}) --\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefixo", default="gravacao_MEMI_20261016_153213",
                    help="prefixo do arquivo de EEG (sem o sufixo _<stamp>.csv)")
    ap.add_argument("--pasta", default="gravacoes")
    ap.add_argument("--fs", type=float, default=500.0,
                    help="taxa nominal do EEG em Hz (padrao %(default)s)")
    args = ap.parse_args(argv)
    return diagnostico(args.prefixo, args.pasta, args.fs, stream=sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
