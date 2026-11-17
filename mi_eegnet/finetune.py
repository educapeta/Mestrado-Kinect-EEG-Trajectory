#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tuning no dataset proprio com GroupKFold por trial e teste final."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from .data import carrega_own, carrega_config
from .model import build_model, count_parameters
from .pretrain import escolhe_device
from .train_utils import (ChannelScaler, bootstrap_ci_trial, group_folds,
                          predict_windows, set_seed, soft_vote,
                          split_trials, train_model, trial_metrics)


def carrega_pretrained(path: Path, cfg: dict, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    return model, ck


def _split_group_ids(y, ids, n_val, seed=42):
    """Split de IDs de trial, aproximadamente balanceado por classe."""
    rng = np.random.default_rng(seed)
    ids = np.array(ids)
    lab = np.array([y[int(t)] for t in ids])
    val = []
    for c in (0, 1):
        cand = ids[lab == c]
        rng.shuffle(cand)
        n = max(1, n_val // 2)
        val.extend(cand[:n].tolist())
    val = np.array(sorted(set(val)))
    tr = ids[~np.isin(ids, val)]
    return tr, val


def evaluate_groups(model, scaler, X, y, groups, device):
    probs = predict_windows(model, scaler, X, device)
    _, _, pred = soft_vote(probs, groups)
    y_trial = np.array([y[groups == g][0] for g in np.unique(groups)])
    return trial_metrics(y_trial, pred)


def run_strategy(Xw, yw, gw, y_trial, train_trial, test_trial,
                 strategy, pretrained_state, cfg, device):
    c = cfg["finetune"]
    folds = group_folds(y_trial[train_trial], train_trial, c["folds"],
                        cfg["seed"])
    rows = []
    for k, (tr_pos, va_pos) in enumerate(folds):
        tr_ids = train_trial[tr_pos]
        va_ids = train_trial[va_pos]
        tr_ids, in_val_ids = _split_group_ids(
            y_trial, tr_ids, max(1, int(round(len(tr_ids) *
                                              c["inner_val_fraction"]))),
            cfg["seed"] + k)
        mtr = np.isin(gw, tr_ids)
        mva = np.isin(gw, in_val_ids)
        mfold = np.isin(gw, va_ids)
        model = build_model(cfg)
        if strategy.startswith("pretrained"):
            model.load_state_dict(pretrained_state)
        freeze = strategy.endswith("A")
        lr = c["strategy_a_lr"] if freeze else c["strategy_b_lr"]
        wd = c["strategy_a_weight_decay"] if freeze else c["strategy_b_weight_decay"]
        res = train_model(
            Xw[mtr], yw[mtr], Xw[mva], yw[mva], model,
            epochs=c["epochs"], patience=c["patience"], lr=lr,
            weight_decay=wd, batch_size=c["batch_size"],
            augment=True, freeze_backbone=freeze, seed=cfg["seed"] + k,
            device=device, verbose=False)
        m = evaluate_groups(res.model, res.scaler, Xw[mfold], yw[mfold],
                            gw[mfold], device)
        rows.append({"fold": k + 1, "val_ba": m["balanced_accuracy"],
                     "val_acc": m["accuracy"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("mi_eegnet/configs/default.yaml"))
    ap.add_argument("--pretrained", type=Path,
                    default=Path("_mi_eegnet_out/pretrain/eegnet_pretrained.pt"))
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = carrega_config(args.config)
    if args.epochs:
        cfg["finetune"]["epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = escolhe_device(args.device)
    out = Path(args.saida or cfg["paths"]["out"]) / "finetune"
    out.mkdir(parents=True, exist_ok=True)

    ds = cfg["preprocessing"]
    Xw, yw, gw, y_trial = carrega_own(
        Path(cfg["paths"]["own"]), fs=ds["fs"], l_freq=ds["l_freq"],
        h_freq=ds["h_freq"], window_sec=ds["window_sec"],
        step_sec=ds["step_sec"])
    print("own:", Xw.shape, "trials:", len(y_trial), flush=True)
    train_trial, test_trial = split_trials(
        y_trial, cfg["finetune"]["n_test_trials"], cfg["seed"])
    print("train trials:", len(train_trial), "test trials:",
          len(test_trial), flush=True)
    assert not set(train_trial) & set(test_trial)

    pretrained, ck = carrega_pretrained(args.pretrained, cfg, device)
    print("parametros do modelo:", count_parameters(pretrained), flush=True)

    results = {}
    for strategy in ("pretrained_A", "pretrained_B"):
        rows = run_strategy(Xw, yw, gw, y_trial, train_trial, test_trial,
                            strategy, ck["state_dict"], cfg, device)
        ba = float(np.mean([r["val_ba"] for r in rows]))
        results[strategy] = {"folds": rows, "mean_val_ba": ba}
        print(f"[{strategy}] mean val BA={ba:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(out / f"{strategy}_folds.csv", index=False)

    best_name = max(results, key=lambda k: results[k]["mean_val_ba"])
    print("melhor estrategia:", best_name, flush=True)

    # retreina a melhor estrategia nos 20 trials de treino, com split interno
    c = cfg["finetune"]
    tr_ids, in_val_ids = _split_group_ids(
        y_trial, train_trial, max(1, int(round(len(train_trial) *
                                              c["inner_val_fraction"]))),
        cfg["seed"])
    mtr = np.isin(gw, tr_ids); mva = np.isin(gw, in_val_ids)
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    freeze = best_name.endswith("A")
    lr = c["strategy_a_lr"] if freeze else c["strategy_b_lr"]
    wd = c["strategy_a_weight_decay"] if freeze else c["strategy_b_weight_decay"]
    res = train_model(
        Xw[mtr], yw[mtr], Xw[mva], yw[mva], model,
        epochs=c["epochs"], patience=c["patience"], lr=lr,
        weight_decay=wd, batch_size=c["batch_size"], augment=True,
        freeze_backbone=freeze, seed=cfg["seed"], device=device)
    torch.save({"state_dict": res.model.state_dict(),
                "model_config": cfg["model"],
                "strategy": best_name,
                "scaler_mean": res.scaler.mean,
                "scaler_std": res.scaler.std},
               out / "eegnet_finetuned.pt")

    # teste final: uma unica vez nos 20 trials reservados
    mtest = np.isin(gw, test_trial)
    probs = predict_windows(res.model, res.scaler, Xw[mtest], device)
    _, pm, pred = soft_vote(probs, gw[mtest])
    y_test = np.array([yw[mtest][gw[mtest] == g][0]
                       for g in np.unique(gw[mtest])])
    final = trial_metrics(y_test, pred)
    ci = bootstrap_ci_trial(y_test, pred, cfg["evaluation"]["bootstrap"],
                            cfg["seed"])
    print(f"TESTE FINAL: {final} CI BA={ci}", flush=True)
    results["final_test"] = {**final, "bootstrap_ci": ci,
                             "strategy": best_name,
                             "params": count_parameters(res.model)}
    (out / "results.json").write_text(json.dumps(results, indent=2),
                                      encoding="utf-8")
    print("Salvo em", out)


if __name__ == "__main__":
    main()
