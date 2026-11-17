#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baselines no mesmo split do fine-tuning: CSP+LDA e EEGNet do zero."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .data import carrega_config, carrega_own
from .finetune import _split_group_ids
from .model import build_model
from .pretrain import escolhe_device
from .train_utils import (ChannelScaler, group_folds, predict_windows,
                          set_seed, soft_vote, split_trials, train_model,
                          trial_metrics)


def _finite(Z):
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def csp_baseline(Xw, yw, gw, y_trial, train_trial, cfg):
    folds = group_folds(y_trial[train_trial], train_trial,
                        cfg["finetune"]["folds"], cfg["seed"])
    yt, yp = [], []
    for k, (tr_pos, va_pos) in enumerate(folds):
        tr_ids = train_trial[tr_pos]; va_ids = train_trial[va_pos]
        tr_ids, in_val = _split_group_ids(
            y_trial, tr_ids, max(1, int(round(len(tr_ids) * 0.2))),
            cfg["seed"] + k)
        mtr = np.isin(gw, tr_ids); mva = np.isin(gw, va_ids)
        sc = ChannelScaler().fit(Xw[mtr])
        clf = Pipeline([
            ("csp", CSP(n_components=2, reg="ledoit_wolf", log=True)),
            ("finite", FunctionTransformer(_finite)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr",
                                               shrinkage="auto")),
        ])
        clf.fit(sc.transform(Xw[mtr]), yw[mtr])
        p = clf.predict_proba(sc.transform(Xw[mva]))
        _, _, pred = soft_vote(p, gw[mva])
        y_true = np.array([yw[mva][gw[mva] == g][0]
                           for g in np.unique(gw[mva])])
        m = trial_metrics(y_true, pred)
        yt.extend(y_true.tolist()); yp.extend(pred.tolist())
    return trial_metrics(yt, yp)


def scratch_baseline(Xw, yw, gw, y_trial, train_trial, cfg, device):
    folds = group_folds(y_trial[train_trial], train_trial,
                        cfg["finetune"]["folds"], cfg["seed"])
    yt, yp = [], []
    for k, (tr_pos, va_pos) in enumerate(folds):
        tr_ids = train_trial[tr_pos]; va_ids = train_trial[va_pos]
        tr_ids, in_val = _split_group_ids(
            y_trial, tr_ids, max(1, int(round(len(tr_ids) * 0.2))),
            cfg["seed"] + k)
        mtr = np.isin(gw, tr_ids); mva = np.isin(gw, in_val)
        mfold = np.isin(gw, va_ids)
        model = build_model(cfg)
        res = train_model(
            Xw[mtr], yw[mtr], Xw[mva], yw[mva], model,
            epochs=cfg["finetune"]["epochs"],
            patience=cfg["finetune"]["patience"],
            lr=cfg["finetune"]["strategy_b_lr"],
            weight_decay=cfg["finetune"]["strategy_b_weight_decay"],
            batch_size=cfg["finetune"]["batch_size"], augment=True,
            seed=cfg["seed"] + k, device=device)
        probs = predict_windows(res.model, res.scaler, Xw[mfold], device)
        _, _, pred = soft_vote(probs, gw[mfold])
        y_true = np.array([yw[mfold][gw[mfold] == g][0]
                           for g in np.unique(gw[mfold])])
        yt.extend(y_true.tolist()); yp.extend(pred.tolist())
    return trial_metrics(yt, yp)


def final_test_csp(Xw, yw, gw, y_trial, train_trial, test_trial, cfg):
    mtr = np.isin(gw, train_trial); mte = np.isin(gw, test_trial)
    sc = ChannelScaler().fit(Xw[mtr])
    clf = Pipeline([
        ("csp", CSP(n_components=2, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(_finite)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr",
                                           shrinkage="auto")),
    ])
    clf.fit(sc.transform(Xw[mtr]), yw[mtr])
    p = clf.predict_proba(sc.transform(Xw[mte]))
    _, _, pred = soft_vote(p, gw[mte])
    yt = np.array([yw[mte][gw[mte] == g][0] for g in np.unique(gw[mte])])
    return trial_metrics(yt, pred)


def final_test_scratch(Xw, yw, gw, y_trial, train_trial, test_trial,
                       cfg, device):
    tr_ids, val_ids = _split_group_ids(
        y_trial, train_trial, max(1, int(round(len(train_trial) * 0.2))),
        cfg["seed"])
    mtr = np.isin(gw, tr_ids); mva = np.isin(gw, val_ids)
    mte = np.isin(gw, test_trial)
    model = build_model(cfg)
    res = train_model(
        Xw[mtr], yw[mtr], Xw[mva], yw[mva], model,
        epochs=cfg["finetune"]["epochs"],
        patience=cfg["finetune"]["patience"],
        lr=cfg["finetune"]["strategy_b_lr"],
        weight_decay=cfg["finetune"]["strategy_b_weight_decay"],
        batch_size=cfg["finetune"]["batch_size"], augment=True,
        seed=cfg["seed"], device=device)
    probs = predict_windows(res.model, res.scaler, Xw[mte], device)
    _, _, pred = soft_vote(probs, gw[mte])
    yt = np.array([yw[mte][gw[mte] == g][0] for g in np.unique(gw[mte])])
    return trial_metrics(yt, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("mi_eegnet/configs/default.yaml"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = carrega_config(args.config)
    if args.epochs:
        cfg["finetune"]["epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = escolhe_device(args.device)
    out = Path(args.saida or cfg["paths"]["out"]) / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    ds = cfg["preprocessing"]
    Xw, yw, gw, y_trial = carrega_own(
        Path(cfg["paths"]["own"]), fs=ds["fs"], l_freq=ds["l_freq"],
        h_freq=ds["h_freq"], window_sec=ds["window_sec"],
        step_sec=ds["step_sec"])
    tr, te = split_trials(y_trial, cfg["finetune"]["n_test_trials"],
                          cfg["seed"])
    res = {
        "csp_lda_cv": csp_baseline(Xw, yw, gw, y_trial, tr, cfg),
        "eegnet_scratch_cv": scratch_baseline(Xw, yw, gw, y_trial, tr, cfg,
                                               device),
        "csp_lda_final_test": final_test_csp(
            Xw, yw, gw, y_trial, tr, te, cfg),
        "eegnet_scratch_final_test": final_test_scratch(
            Xw, yw, gw, y_trial, tr, te, cfg, device),
    }
    print("CSP CV:", res["csp_lda_cv"])
    print("scratch CV:", res["eegnet_scratch_cv"])
    print("CSP final:", res["csp_lda_final_test"])
    print("scratch final:", res["eegnet_scratch_final_test"])
    # relatorio do fine-tuned, se existir
    fin = Path(cfg["paths"]["out"]) / "finetune" / "results.json"
    if fin.exists():
        res["pretrained_finetuned"] = json.load(
            open(fin, encoding="utf-8")).get("final_test")
    (out / "baselines.json").write_text(json.dumps(res, indent=2),
                                         encoding="utf-8")
    print("Salvo em", out)


if __name__ == "__main__":
    main()
