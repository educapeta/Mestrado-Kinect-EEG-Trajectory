#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Carregamento dos dados: BCIC-IV 2b local e dataset proprio."""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
import mne
from scipy import signal

mne.set_log_level("ERROR")

CH_2B = ["EEG:C3", "EEG:Cz", "EEG:C4"]
CH_OWN_IDX = [14, 15, 16]  # 15=C3, 16=C4, 17=Cz na aquisicao
CLASSES = ["esquerda", "direita"]


def filt_resample(eeg: np.ndarray, fs_in: float, fs_out: float,
                  l_freq=8.0, h_freq=32.0) -> np.ndarray:
    """eeg (channels, samples) -> filtrado e reamostrado."""
    x = mne.filter.filter_data(np.asarray(eeg, float), fs_in,
                               l_freq, h_freq, verbose=False)
    if abs(fs_out - fs_in) > 1e-6:
        up = int(round(fs_out))
        down = int(round(fs_in))
        x = signal.resample_poly(x, up, down, axis=1)
    return x.astype(np.float32)


def carrega_bci2b(pasta: Path, fs=128, l_freq=8.0, h_freq=32.0,
                  window_sec=2.0, cue_offset=1.0):
    """Sessoes T do 2b; janela 2 s comecando cue_offset apos o cue."""
    X, y, subj = [], [], []
    for f in sorted(Path(pasta).glob("*T.gdf")):
        m = re.search(r"B(\d\d)", f.name)
        if not m:
            continue
        s = int(m.group(1))
        raw = mne.io.read_raw_gdf(str(f), preload=True, verbose=False)
        raw.filter(l_freq, h_freq, verbose=False)
        raw.resample(fs, verbose=False)
        picks = [c for c in CH_2B if c in raw.ch_names]
        if len(picks) != 3:
            raise RuntimeError(f"{f.name}: canais {raw.ch_names}")
        raw.pick(picks)
        data = raw.get_data()
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        code_to_label = {}
        for code, nome in [(769, 0), (770, 1)]:
            if str(code) in event_id:
                code_to_label[int(event_id[str(code)])] = nome
        ev = events[np.isin(events[:, 2], list(code_to_label))]
        n = int(window_sec * fs)
        for sample, _, code in ev:
            start = int(sample + cue_offset * fs)
            end = start + n
            if start < 0 or end > data.shape[1]:
                continue
            X.append(data[:, start:end])
            y.append(code_to_label[int(code)])
            subj.append(s)
    return np.stack(X).astype(np.float32), np.array(y), np.array(subj)


def _load_own_csv(csv_path: Path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.reader(f):
            if r and not r[0].startswith("#"):
                rows.append(r)
    header = [h.strip() for h in rows[0]]
    arr = np.array([[float(v) for v in r[1:-1]] for r in rows[1:]],
                   dtype=np.float64)
    markers = np.array([int(float(r[-1])) for r in rows[1:]], dtype=int)
    return arr, markers, header


def carrega_own(pasta: Path, fs=128, l_freq=8.0, h_freq=32.0,
                window_sec=2.0, step_sec=0.5, trial_sec=4.0):
    """4 s de MI -> janelas deslizantes de 2 s, somente left/right."""
    csv_path = sorted(Path(pasta).glob("eeg_mi3c*.csv"))[-1]
    json_path = sorted(Path(pasta).glob("eventos*.json"))[-1]
    meta = json.load(open(json_path, encoding="utf-8"))
    eeg, markers, _ = _load_own_csv(csv_path)
    fs_orig = float(meta.get("fs", 500.0))
    eeg = eeg[:, CH_OWN_IDX]
    eeg = filt_resample(eeg.T, fs_orig, fs, l_freq, h_freq).T

    mapa = {"esquerda": 0, "direita": 1, "repouso": 2}
    starts = np.where(markers == 780)[0]
    ends = np.where(markers == 781)[0]
    X, y, groups, trial_y = [], [], [], []
    n_win = int(window_sec * fs)
    step = int(step_sec * fs)
    n_trial = int(trial_sec * fs)
    for a, b in zip(starts, ends):
        s = int(round(a * fs / fs_orig))
        e = int(round(b * fs / fs_orig))
        seg = eeg[s:e].T
        if seg.shape[1] < n_trial:
            seg = np.pad(seg, ((0, 0), (0, n_trial - seg.shape[1])))
        seg = seg[:, :n_trial]
        # ultimo cue (incluindo repouso) antes do inicio do MI
        prev = markers[:a]
        ultimo = -1; cls = 2
        for code, nome in [(769, "esquerda"), (770, "direita"),
                           (784, "repouso")]:
            pos = np.where(prev == code)[0]
            if len(pos) and pos[-1] > ultimo:
                ultimo = pos[-1]; cls = mapa[nome]
        if cls not in (0, 1):
            continue
        tid = len(trial_y)
        trial_y.append(cls)
        for w in range(0, seg.shape[1] - n_win + 1, step):
            X.append(seg[:, w:w + n_win])
            y.append(cls)
            groups.append(tid)
    return (np.stack(X).astype(np.float32), np.array(y),
            np.array(groups), np.array(trial_y))


def carrega_config(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
