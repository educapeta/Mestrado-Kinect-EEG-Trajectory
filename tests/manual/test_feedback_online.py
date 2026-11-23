#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Teste offline do decoder de feedback online (sem hardware)."""
from __future__ import annotations

import queue
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
for p in (RAIZ / "src" / "mestradopy", RAIZ / "src" / "mestradopy" / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import feedback_online as fo  # noqa: E402


def bloco_sintetico(label: str, n: int = 1200, ch: int = 32, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.0, (n, ch)).astype(np.float64)
    t = np.arange(n) / fo.FS
    # padrao espacial: direita -> C3; esquerda -> C4 (12 Hz)
    c3, c4, cz = fo.CH_OWN
    if label == "direita":
        x[:, c3] += 4.0 * np.sin(2 * np.pi * 12 * t)
    elif label == "esquerda":
        x[:, c4] += 4.0 * np.sin(2 * np.pi * 12 * t)
    else:
        x[:, cz] += 2.0 * np.sin(2 * np.pi * 8 * t)
    return x


def main() -> int:
    q = queue.Queue()
    dec = fo.OnlineFeedbackDecoder(q, mode="calibration", verbose=False)
    for i in range(40):
        label = "direita" if i % 2 else "esquerda"
        dec.begin_trial(label)
        q.put(bloco_sintetico(label, n=1100, seed=i))
        dec.update()
        dec.end_trial()
    assert not dec.ready
    dec.finish_block()
    assert dec.ready, "modelo deveria estar pronto apos o bloco de calibracao"
    assert dec.score_texto()
    # aquisicao de um trial de teste
    dec.begin_trial("direita")
    q.put(bloco_sintetico("direita", n=1100, seed=999))
    s = dec.update()
    assert s is None or -1.0 <= s <= 1.0
    assert dec.n_calibracao == 40 * 3 or dec.n_calibracao > 0
    # save/load
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.joblib"
        dec.salvar_modelo = p
        assert dec.save() is not None
        dec2 = fo.OnlineFeedbackDecoder(queue.Queue(), mode="frozen", model_path=p,
                                        verbose=False)
        assert dec2.ready
        assert dec2.score_texto()
    print("FEEDBACK_ONLINE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
