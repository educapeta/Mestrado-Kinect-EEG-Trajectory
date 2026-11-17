#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilitarios de normalizacao, treino e avaliacao por trial."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold


class ChannelScaler:
    """StandardScaler por canal, ajustado apenas no conjunto de treino."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        # X: (n, channels, times)
        self.mean = X.mean(axis=(0, 2), keepdims=True)
        self.std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def soft_vote(probs, groups):
    """Media de probabilidades das janelas; retorna predicoes por trial."""
    uniq = np.unique(groups)
    pm = np.stack([probs[groups == g].mean(axis=0) for g in uniq])
    return uniq, pm, pm.argmax(axis=1)


def trial_metrics(y_trial, pred, proba=None):
    y = np.asarray(y_trial); p = np.asarray(pred)
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "accuracy": float(np.mean(y == p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "confusion": cm.tolist(),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
    }
    if proba is not None and len(np.unique(y)) == 2:
        out["auc_roc"] = float(roc_auc_score(y, np.asarray(proba)[:, 1]))
    return out


def bootstrap_ci_trial(y_trial, pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_trial); p = np.asarray(pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vals.append(balanced_accuracy_score(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _augment(X, noise_sigma=0.0, channel_dropout=0.0, rng=None):
    """Data augmentation simples no treino."""
    rng = rng or np.random.default_rng()
    X = X.copy()
    if noise_sigma > 0:
        X = X + rng.normal(0, noise_sigma * X.std(), X.shape)
    if channel_dropout > 0:
        drop = rng.random((len(X), X.shape[1])) < channel_dropout
        X[drop] = 0.0
    return X.astype(np.float32)


@dataclass
class TrainResult:
    model: torch.nn.Module
    scaler: ChannelScaler
    best_val_ba: float
    history: list


def build_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay)


def train_model(Xtr, ytr, Xval, yval, model, scaler=None,
                epochs=100, patience=20, lr=1e-3, weight_decay=1e-4,
                batch_size=8, augment=True, freeze_backbone=False,
                noise_sigma=0.0, channel_dropout=0.0, seed=42,
                device="cpu", verbose=False):
    set_seed(seed)
    scaler = scaler or ChannelScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xval_s = scaler.transform(Xval).astype(np.float32)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

    model = model.to(device)
    opt = build_optimizer(model, lr, weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    hist = []
    best = -1.0
    best_state = None
    best_epoch = 0
    bad = 0

    for ep in range(epochs):
        model.train()
        Xa = _augment(Xtr_s, noise_sigma, channel_dropout, rng) if augment else Xtr_s
        perm = np.random.permutation(len(Xa))
        for i in range(0, len(Xa), batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(Xa[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).long().to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xval_s).to(device)).softmax(1).cpu().numpy()
        ba = balanced_accuracy_score(yval, pv.argmax(1))
        hist.append({"epoch": ep + 1, "val_ba": float(ba)})
        if ba > best + 1e-5:
            best = ba
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep + 1
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and (ep == 0 or (ep + 1) % 20 == 0):
            print(f"  ep {ep+1:03d}: val_BA={ba:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model, scaler, best, hist)


def predict_windows(model, scaler, X, device="cpu"):
    model.eval()
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(Xs).to(device))
        return logits.softmax(1).cpu().numpy()


def group_folds(y, groups, n_splits=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups=groups))


def split_trials(y_trial, n_test, seed=42):
    """Split estratificado no nivel do trial."""
    rng = np.random.default_rng(seed)
    idx0 = np.where(y_trial == 0)[0]
    idx1 = np.where(y_trial == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    n0 = n_test // 2; n1 = n_test - n0
    test = np.concatenate([idx0[:n0], idx1[:n1]])
    train = np.concatenate([idx0[n0:], idx1[n1:]])
    return np.sort(train), np.sort(test)
