#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compara o pipeline atual com RSF (Riemannian spatial filtering).

Avaliacao:
- CSP+LDA: baseline x RSF+CSP+LDA;
- EEGNet 2b pre-treinado + fine-tune: baseline x RSF+EEGNet;
- fases ME (controle) e MI (alvo);
- cross-session nas duas direcoes (60 trials de teste).

O RSF e ajustado SOMENTE em trials da sessao de treino. Para o CSP/EEGNet,
as janelas de 2 s sao transformadas pelo W aprendido; nenhuma informacao da
sessao de teste entra no fit do RSF.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mne.decoding import CSP

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "mestradopy"))
sys.path.insert(0, str(_ROOT / "src" / "mestradopy" / "eeg"))

from mestradopy.eeg.rsf_filter import RSFFilter
from mi_eegnet.data import carrega_config, carrega_own
from mi_eegnet.finetune import _split_group_ids
from mi_eegnet.model import build_model, count_parameters
from mi_eegnet.pretrain import escolhe_device
from mi_eegnet.train_utils import (ChannelScaler, bootstrap_ci_trial,
                                   predict_windows, set_seed, soft_vote,
                                   trial_metrics, train_model)


def _finite(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def load_phase(path, cfg, phase):
    d = cfg["preprocessing"]
    kw = dict(fs=d["fs"], l_freq=d["l_freq"], h_freq=d["h_freq"],
              fase=phase)
    X4, y4, g4, trial4 = carrega_own(path, window_sec=4.0, step_sec=4.0, **kw)
    X2, y2, g2, trial2 = carrega_own(path, window_sec=d["window_sec"],
                                     step_sec=d["step_sec"], **kw)
    return {"X4": X4, "y4": y4, "trial4": trial4,
            "X2": X2, "y2": y2, "g2": g2, "trial2": trial2}


def fit_rsf(d, dim=3, method="default", seed=42):
    rsf = RSFFilter(dim=dim, method=method, random_state=seed)
    rsf.fit(d["X4"], d["y4"])
    return rsf


def eval_csp(Xtr, ytr, gtr, Xte, yte, gte, seed=42):
    sc = ChannelScaler().fit(Xtr)
    clf = Pipeline([
        ("csp", CSP(n_components=2, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(_finite)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr",
                                           shrinkage="auto")),
    ])
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))
    yt = np.array([yte[gte == g][0] for g in np.unique(gte)])
    _, _, pred = soft_vote(p, gte)
    m = trial_metrics(yt, pred)
    m["n"] = len(yt)
    return m


def eval_eegnet(Xtr, ytr, gtr, Xte, yte, gte, state, strategy, cfg,
                device, seed=42):
    ytrial = np.array([ytr[gtr == g][0] for g in np.unique(gtr)])
    ids = np.unique(gtr)
    nval = max(2, int(round(len(ids) * 0.2)))
    tr_ids, val_ids = _split_group_ids(ytrial, ids, nval, seed)
    mtr = np.isin(gtr, tr_ids); mval = np.isin(gtr, val_ids)
    model = build_model(cfg)
    model.load_state_dict(state)
    c = cfg["finetune"]
    freeze = strategy.endswith("A")
    res = train_model(
        Xtr[mtr], ytr[mtr], Xtr[mval], ytr[mval], model,
        epochs=c["epochs"], patience=c["patience"],
        lr=c["strategy_a_lr"] if freeze else c["strategy_b_lr"],
        weight_decay=(c["strategy_a_weight_decay"] if freeze
                      else c["strategy_b_weight_decay"]),
        batch_size=c["batch_size"], augment=True, freeze_backbone=freeze,
        seed=seed, device=device)
    probs = predict_windows(res.model, res.scaler, Xte, device)
    yt = np.array([yte[gte == g][0] for g in np.unique(gte)])
    _, _, pred = soft_vote(probs, gte)
    m = trial_metrics(yt, pred)
    m["n"] = len(yt)
    m["internal_val_ba"] = float(res.best_val_ba)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=Path,
                    default=Path("outputs/eeg/piloto_me_mi_s01"))
    ap.add_argument("--s2", type=Path,
                    default=Path("outputs/eeg/piloto_me_mi_s02"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("outputs/eeg/mi_eegnet_out/pretrain/"
                                 "eegnet_pretrained.pt"))
    ap.add_argument("--config", type=Path,
                    default=Path("src/mestradopy/eeg/mi_eegnet/configs/"
                                 "default.yaml"))
    ap.add_argument("--saida", type=Path,
                    default=Path("outputs/eeg/rsf_novo_piloto"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--rsf-dim-csp", type=int, default=2)
    ap.add_argument("--rsf-dim-eegnet", type=int, default=3)
    args = ap.parse_args()
    cfg = carrega_config(args.config)
    if args.epochs:
        cfg["finetune"]["epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = escolhe_device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False)
    args.saida.mkdir(parents=True, exist_ok=True)

    print("device:", device, "params:",
          count_parameters(build_model(cfg)), flush=True)
    data = {}
    for sname, path in (("s1", args.s1), ("s2", args.s2)):
        data[sname] = {}
        for phase in ("ME", "MI"):
            data[sname][phase] = load_phase(path, cfg, phase)
            print(sname, phase, "4s", data[sname][phase]["X4"].shape,
                  "2s", data[sname][phase]["X2"].shape, flush=True)

    res = {}
    for phase in ("ME", "MI"):
        res[phase] = {}
        for direction, tr, te in (("s1_to_s2", "s1", "s2"),
                                  ("s2_to_s1", "s2", "s1")):
            dtr, dte = data[tr][phase], data[te][phase]
            print("\n", phase, direction, flush=True)
            rsf_csp = fit_rsf(dtr, dim=args.rsf_dim_csp, seed=cfg["seed"])
            Xtr_csp = rsf_csp.transform(dtr["X2"])
            Xte_csp = rsf_csp.transform(dte["X2"])
            rsf_eeg = fit_rsf(dtr, dim=args.rsf_dim_eegnet, seed=cfg["seed"])
            Xtr_eeg = rsf_eeg.transform(dtr["X2"])
            Xte_eeg = rsf_eeg.transform(dte["X2"])
            out = {"rsf_distance_csp": rsf_csp.distance_,
                   "rsf_dim_csp": args.rsf_dim_csp,
                   "rsf_distance_eegnet": rsf_eeg.distance_,
                   "rsf_dim_eegnet": args.rsf_dim_eegnet,
                   "rsf_method": "default"}
            out["csp_baseline"] = eval_csp(
                dtr["X2"], dtr["y2"], dtr["g2"],
                dte["X2"], dte["y2"], dte["g2"], cfg["seed"])
            out["csp_rsf"] = eval_csp(
                Xtr_csp, dtr["y2"], dtr["g2"],
                Xte_csp, dte["y2"], dte["g2"], cfg["seed"])
            print("  CSP base:", out["csp_baseline"]["balanced_accuracy"],
                  "RSF:", out["csp_rsf"]["balanced_accuracy"], flush=True)
            for strategy in ("pretrained_A", "pretrained_B"):
                out[f"eegnet_{strategy}_baseline"] = eval_eegnet(
                    dtr["X2"], dtr["y2"], dtr["g2"],
                    dte["X2"], dte["y2"], dte["g2"],
                    ck["state_dict"], strategy, cfg, device, cfg["seed"])
                out[f"eegnet_{strategy}_rsf"] = eval_eegnet(
                    Xtr_eeg, dtr["y2"], dtr["g2"],
                    Xte_eeg, dte["y2"], dte["g2"],
                    ck["state_dict"], strategy, cfg, device, cfg["seed"])
                print("  EEG", strategy, "base:",
                      out[f"eegnet_{strategy}_baseline"]["balanced_accuracy"],
                      "RSF:", out[f"eegnet_{strategy}_rsf"]["balanced_accuracy"],
                      flush=True)
            res[phase][direction] = out

    (args.saida / "resultados.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    lines = ["# RSF no novo piloto (ME/MI)", "",
             "RSF fit apenas na sessao de treino; teste na outra sessao.",
             "Janelas de 2 s; RSF ajustado em janelas de 4 s.", ""]
    for phase, dirs in res.items():
        lines += [f"## {phase}", ""]
        for direction, out in dirs.items():
            lines += [f"### {direction}", "",
                      f"- RSF distância (CSP, dim {out['rsf_dim_csp']}): "
                      f"{out['rsf_distance_csp']:.4f}; "
                      f"(EEGNet, dim {out['rsf_dim_eegnet']}): "
                      f"{out['rsf_distance_eegnet']:.4f}", ""]
            for k in ("csp_baseline", "csp_rsf",
                      "eegnet_pretrained_A_baseline",
                      "eegnet_pretrained_A_rsf",
                      "eegnet_pretrained_B_baseline",
                      "eegnet_pretrained_B_rsf"):
                m = out.get(k)
                if not m:
                    continue
                lines += [f"- **{k}**: BA={m['balanced_accuracy']:.3f}, "
                          f"acc={m['accuracy']:.3f}, "
                          f"conf={m['confusion']}", ""]
    (args.saida / "relatorio.md").write_text("\n".join(lines),
                                              encoding="utf-8")
    print("Salvo em", args.saida)


if __name__ == "__main__":
    main()
