#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decoder de feedback online para o protocolo L/R.

Fluxo:
  EEGTap (gpype) -> queue -> OnlineFeedbackDecoder
  - buffer de 2 s estritamente dentro da MI;
  - reamostra 500 -> 128 Hz, banda 8-30 Hz;
  - CSP+LDA calibrado no bloco 1; score 2*P(direita)-1;
  - adapta apenas ao fim de blocos (nunca durante o trial);
  - modelo salvo/carregado via joblib.

Este modulo NAO abre hardware; e' integrado ao `validacao_eeg_3classes.py`.
"""
from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mne.decoding import CSP

try:  # gpype so' existe no ambiente de aquisicao
    from gpype.backend.core.io_node import IONode
    from gpype.common.constants import Constants
    PORT_IN = Constants.Defaults.PORT_IN
    PORT_OUT = Constants.Defaults.PORT_OUT
except Exception:  # pragma: no cover
    IONode = object
    PORT_IN = "in"
    PORT_OUT = "out"


FS = 500.0
FS_MODEL = 128.0
WINDOW_S = 2.0
STEP_S = 0.25
BAND = (8.0, 30.0)
CH_OWN = (14, 15, 16)  # C3, C4, Cz (0-based)
_LABELS = {"esquerda": 0, "direita": 1}


def _finite_np(z):
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _pipeline(n_csp=2):
    return Pipeline([
        ("csp", CSP(n_components=n_csp, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(_finite_np)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def _preprocess(window_500: np.ndarray) -> np.ndarray:
    """(N,3) em 500 Hz -> (3,256) filtrado/reamostrado."""
    x = np.asarray(window_500, np.float64).T          # (3,N)
    down = int(round(FS / FS_MODEL))
    # anti-alias e reamostragem
    x = resample_poly(x, up=int(FS_MODEL), down=int(FS), axis=-1)
    sos = butter(4, [BAND[0] / (FS_MODEL / 2), BAND[1] / (FS_MODEL / 2)],
                 btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    n = int(WINDOW_S * FS_MODEL)
    if x.shape[-1] < n:
        x = np.pad(x, ((0, 0), (0, n - x.shape[-1])))
    return x[:, -n:].astype(np.float64)


class EEGTap(IONode):
    """Sink pass-through: copia cada bloco de EEG para uma queue."""

    def __init__(self, queue_: queue.Queue, **kwargs):
        super().__init__(**kwargs)
        self._queue = queue_

    def step(self, data):
        block = data.get(PORT_IN)
        if block is not None and len(block):
            try:
                self._queue.put_nowait(np.asarray(block, dtype=np.float64).copy())
            except queue.Full:
                pass
        return {PORT_OUT: block}


class OnlineFeedbackDecoder:
    """Decodificador CSP+LDA com janelas dentro da MI."""

    def __init__(self, eeg_queue: queue.Queue, *, mode: str = "calibration",
                 model_path: Optional[Path] = None,
                 salvar_modelo: Optional[Path] = None,
                 congelado: bool = False,
                 verbose: bool = True):
        self.q = eeg_queue
        self.mode = str(mode)                  # calibration | adaptive | frozen
        self.model_path = Path(model_path) if model_path else None
        self.salvar_modelo = Path(salvar_modelo) if salvar_modelo else None
        self.congelado = bool(congelado)
        self.verbose = bool(verbose)
        self.pipeline = None
        self.limiar = 0.0
        self.buffer = np.zeros((0, len(CH_OWN)), np.float64)
        self._last_eval_n = 0
        self._label: Optional[int] = None
        self._collecting = False
        self._trial_windows: List[np.ndarray] = []
        self._X: List[np.ndarray] = []
        self._y: List[int] = []
        self.last_score: Optional[float] = None
        self.last_p_right: Optional[float] = None
        self.n_updates = 0
        self.history: List[Dict] = []
        if self.model_path and self.model_path.is_file():
            self.load(self.model_path)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    @property
    def n_calibracao(self) -> int:
        return len(self._y)

    def drain(self) -> None:
        while True:
            try:
                block = self.q.get_nowait()
            except queue.Empty:
                break
            if block is None or len(block) == 0:
                continue
            b = np.asarray(block, np.float64)
            if b.ndim == 1:
                b = b.reshape(-1, 1)
            if b.shape[1] <= max(CH_OWN):
                continue
            self.buffer = np.concatenate([self.buffer, b[:, CH_OWN]], axis=0)
            if len(self.buffer) > int(30 * FS):
                self.buffer = self.buffer[-int(30 * FS):]

    def begin_trial(self, label: str, *, collect: bool = True) -> None:
        """Chamado no inicio da MI: zera o buffer e prepara coleta."""
        self.drain()
        self.buffer = np.zeros((0, len(CH_OWN)), np.float64)
        self._last_eval_n = 0
        self._label = _LABELS.get(str(label).lower())
        self._collecting = bool(collect and self._label is not None)
        self._trial_windows = []
        self.last_score = None
        self.last_p_right = None

    def update(self) -> Optional[float]:
        """Drena EEG e, se houver 2 s, calcula o score. Devolve score ou None."""
        self.drain()
        n = len(self.buffer)
        win_n = int(WINDOW_S * FS)
        step_n = int(STEP_S * FS)
        if n < win_n or (n - self._last_eval_n) < step_n:
            return self.last_score
        seg = self.buffer[-win_n:]
        try:
            x = _preprocess(seg)                  # (3,256)
        except Exception:
            return self.last_score
        self._last_eval_n = n
        self.n_updates += 1
        if self._collecting:
            self._trial_windows.append(x.copy())
        if self.ready:
            prob = self.pipeline.predict_proba(x.reshape(1, 3, -1))[0]
            self.last_p_right = float(prob[1])
            self.last_score = float(2.0 * prob[1] - 1.0) - float(self.limiar)
            self.history.append({
                "t": round(float(time.time()), 3),
                "label": self._label, "score": round(self.last_score, 4),
                "p_right": round(self.last_p_right, 4)})
        return self.last_score

    def end_trial(self) -> List[np.ndarray]:
        """Finaliza a coleta do trial e devolve as janelas."""
        wins = list(self._trial_windows)
        if self._label is not None and wins:
            self._X.extend(wins)
            self._y.extend([int(self._label)] * len(wins))
            if self.verbose:
                print(f"  [online] trial label={self._label} "
                      f"janelas={len(wins)}")
        self._trial_windows = []
        return wins

    def fit(self) -> bool:
        if len(self._X) < 10 or len(set(self._y)) < 2:
            if self.verbose:
                print(f"[online] fit adiado: n={len(self._X)} "
                      f"classes={set(self._y)}")
            return False
        X = np.stack(self._X)
        y = np.asarray(self._y, int)
        self.pipeline = _pipeline()
        self.pipeline.fit(X, y)
        # limiar que equilibra as classes na calibracao (mediana L/R)
        try:
            proba = self.pipeline.predict_proba(X)[:, 1]
            scores = 2.0 * proba - 1.0
            esq = scores[y == 0]; dir_ = scores[y == 1]
            if len(esq) and len(dir_):
                self.limiar = float((float(np.median(esq)) +
                                     float(np.median(dir_))) / 2.0)
        except Exception:
            self.limiar = 0.0
        if self.verbose:
            print(f"[online] modelo CSP+LDA ajustado com {len(y)} janelas")
        return True

    def finish_block(self) -> bool:
        """Chamado no fim de cada bloco; adapta apenas se nao congelado."""
        if self.congelado or self.mode == "frozen":
            return self.ready
        if not self.ready:
            return self.fit()
        if self.mode == "adaptive":
            return self.fit()
        return self.ready

    def save(self) -> Optional[Path]:
        if not self.ready or self.salvar_modelo is None:
            return None
        import joblib
        self.salvar_modelo.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "limiar": float(self.limiar),
                     "n_calibracao": len(self._y),
                     "online": True}, self.salvar_modelo)
        if self.verbose:
            print(f"[online] modelo salvo em {self.salvar_modelo}")
        return self.salvar_modelo

    def load(self, caminho: Path) -> bool:
        import joblib
        try:
            d = joblib.load(caminho)
            self.pipeline = d["pipeline"]
            self.limiar = float(d.get("limiar", 0.0))
            if self.verbose:
                print(f"[online] modelo carregado de {caminho}")
            return True
        except Exception as exc:               # noqa: BLE001
            print(f"[online] falha ao carregar {caminho}: {exc}")
            self.pipeline = None
            return False

    def save_log(self, caminho: Path) -> Optional[Path]:
        """Grava historico de scores online (para auditoria do feedback)."""
        if not self.history:
            return None
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps({"n_updates": len(self.history),
                                        "scores": self.history},
                                       indent=2, ensure_ascii=False),
                           encoding="utf-8")
        return caminho

    def score_texto(self) -> str:
        """Texto curto para a UI (barra proporcional)."""
        if not self.ready:
            return "CALIBRANDO...\n(mova a mao imaginada conforme o cue)"
        s = 0.0 if self.last_score is None else float(self.last_score)
        # 21 posicoes; centro = 0 (score ja subtraiu o limiar)
        pos = int(round((s + 1.0) * 10))
        barra = [" "] * 21
        barra[max(0, min(20, pos))] = "|"
        lado = "DIREITA" if s >= 0 else "ESQUERDA"
        return (f"{'ESQUERDA':<10}{''.join(barra)}{'DIREITA':>10}\n"
                f"score={s:+.2f}  p(dir)={self.last_p_right if self.last_p_right is not None else 0:.2f}\n"
                f"decodificador: {lado}")
