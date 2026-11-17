#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEGNet 2b pre-treinado + fine-tune no novo piloto (ME/MI, 2 sessoes).

Para cada fase (ME/MI) e direcao cross-session (s1->s2 e s2->s1):
- carrega sessoes;
- separa 80/20 dos trials da sessao de treino para early stopping;
- testa estrategias A (feature extraction) e B (full fine-tune);
- avalia na outra sessao com soft voting por trial.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from mi_eegnet.data import carrega_config, carrega_own
from mi_eegnet.finetune import _split_group_ids
from mi_eegnet.model import build_model, count_parameters
from mi_eegnet.pretrain import escolhe_device
from mi_eegnet.train_utils import (bootstrap_ci_trial, predict_windows,
                                   set_seed, soft_vote, trial_metrics,
                                   train_model)


def load_phase(path, cfg, fase):
    ds = cfg["preprocessing"]
    return carrega_own(path, fs=ds["fs"], l_freq=ds["l_freq"],
                       h_freq=ds["h_freq"], window_sec=ds["window_sec"],
                       step_sec=ds["step_sec"], fase=fase)


def trial_arrays(probs, yw, gw):
    uniq = np.unique(gw)
    ytrue = np.array([yw[gw == g][0] for g in uniq])
    _, _, pred = soft_vote(probs, gw)
    return ytrue, pred


def train_eval(Xtr, ytr, gtr, Xte, yte, gte, cfg, state, strategy,
               device, seed):
    ytrial = np.array([ytr[gtr == g][0] for g in np.unique(gtr)])
    ids = np.unique(gtr)
    nval = max(2, int(round(len(ids) * 0.2)))
    tr_ids, val_ids = _split_group_ids(ytrial, ids, nval, seed)
    mtr = np.isin(gtr, tr_ids); mval = np.isin(gtr, val_ids)
    model = build_model(cfg)
    model.load_state_dict(state)
    c = cfg["finetune"]
    freeze = strategy.endswith("A")
    lr = c["strategy_a_lr"] if freeze else c["strategy_b_lr"]
    wd = c["strategy_a_weight_decay"] if freeze else c["strategy_b_weight_decay"]
    res = train_model(
        Xtr[mtr], ytr[mtr], Xtr[mval], ytr[mval], model,
        epochs=c["epochs"], patience=c["patience"], lr=lr,
        weight_decay=wd, batch_size=c["batch_size"], augment=True,
        freeze_backbone=freeze, seed=seed, device=device)
    probs = predict_windows(res.model, res.scaler, Xte, device)
    ytrue, pred = trial_arrays(probs, yte, gte)
    m = trial_metrics(ytrue, pred)
    m["bootstrap_ci_bal_acc"] = bootstrap_ci_trial(
        ytrue, pred, cfg["evaluation"]["bootstrap"], seed)
    m["n_test_trials"] = int(len(ytrue))
    m["n_train_trials"] = int(len(tr_ids))
    m["n_internal_val"] = int(len(val_ids))
    return m, res.best_val_ba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("mi_eegnet/configs/default.yaml"))
    ap.add_argument("--pretrained", type=Path,
                    default=Path("_mi_eegnet_out/pretrain/eegnet_pretrained.pt"))
    ap.add_argument("--s1", type=Path, default=Path("_piloto_me_mi_s01"))
    ap.add_argument("--s2", type=Path, default=Path("_piloto_me_mi_s02"))
    ap.add_argument("--saida", type=Path, default=Path("_eegnet2b_novo_piloto"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = carrega_config(args.config)
    if args.epochs:
        cfg["finetune"]["epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = escolhe_device(args.device)
    out = args.saida
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    print("device:", device, "params:", count_parameters(build_model(cfg)),
          flush=True)

    data = {}
    for sname, path in (("s1", args.s1), ("s2", args.s2)):
        data[sname] = {}
        for fase in ("ME", "MI"):
            X, y, g, yt = load_phase(path, cfg, fase)
            data[sname][fase] = (X, y, g, yt)
            print(sname, fase, X.shape, "trials", len(yt), flush=True)

    res = {}
    for fase in ("ME", "MI"):
        res[fase] = {}
        for direction, a, b in [("s1_to_s2", "s1", "s2"),
                                ("s2_to_s1", "s2", "s1")]:
            Xtr, ytr, gtr, _ = data[a][fase]
            Xte, yte, gte, _ = data[b][fase]
            res[fase][direction] = {}
            for strategy in ("pretrained_A", "pretrained_B"):
                key = f"{direction}_{strategy}"
                print("\n", fase, key, flush=True)
                m, vba = train_eval(Xtr, ytr, gtr, Xte, yte, gte, cfg,
                                    ck["state_dict"], strategy, device,
                                    cfg["seed"])
                m["internal_val_ba"] = float(vba)
                res[fase][key] = m
                print(f"  BA={m['balanced_accuracy']:.3f} "
                      f"CI={m['bootstrap_ci_bal_acc']} "
                      f"cm={m['confusion']}", flush=True)

    (out / "resultados.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
    lines = ["# EEGNet 2b pre-treinado + fine-tune (novo piloto)", ""]
    for fase, fs in res.items():
        lines += [f"## {fase}", ""]
        for key, m in fs.items():
            if not isinstance(m, dict) or "balanced_accuracy" not in m:
                continue
            lines += [f"### {key}", "",
                      f"- BA: {m['balanced_accuracy']:.3f} "
                      f"CI {m['bootstrap_ci_bal_acc']}",
                      f"- acc: {m['accuracy']:.3f}",
                      f"- sens/espec: {m['sensitivity']:.3f}/{m['specificity']:.3f}",
                      f"- confusão: {m['confusion']}",
                      f"- val interna: {m['internal_val_ba']:.3f}", ""]
    (out / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")
    print("Salvo em", out)


if __name__ == "__main__":
    main()
