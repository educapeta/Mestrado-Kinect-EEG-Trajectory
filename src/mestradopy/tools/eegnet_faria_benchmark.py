#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replicação do EEGNet do Faria (2023), Tabela 4.1.

Fonte do modelo: https://github.com/gbrlfaria/bci-data-augmentation
  project/models/eegnet.py
  project/experiments/runner.py
  project/__main__.py

Configurações da Tabela 4.1:
  BCIC-IV 2a: 22ch/4cls (1.716), 3ch/4cls (1.412), 22ch/2cls (1.458)
  BCIC-IV 2b: 3ch/2cls (1.154)

Pré-processamento do repositório:
  - filtro 4-40 Hz
  - 128 Hz
  - padronização exponencial por eletrodo
  - janela de 2 s: 0,5 a 2,5 s após o cue
  - 4-fold estratificado blockwise; validação interna 3-fold
"""
from __future__ import annotations

import argparse
import warnings
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", message=".*Channel names are not unique.*")
warnings.filterwarnings("ignore", message=".*duplicates for.*")
FS = 128.0
N_TIMES = 256
CHANCE = {2: 0.5, 4: 0.25}


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """Cópia fiel de project/models/eegnet.py (Faria 2023)."""

    def __init__(self, n_channels, n_times, n_classes, kernel_length=32,
                 F1=8, D=2, F2=16, pool1_stride=4, pool2_stride=8,
                 dropout_rate=0.5, norm_rate=0.25):
        super().__init__()
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length), bias=False, padding="same")
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.conv_spatial = Conv2dWithConstraint(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False,
            padding="valid", max_norm=1.0)
        self.bnorm_spatial = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=0.001)
        self.pool1 = nn.AvgPool2d((1, 4), stride=(1, pool1_stride))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv_depth1 = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), bias=False, groups=F1 * D, padding="same")
        self.conv_point1 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False,
                                     padding=(0, 0))
        self.bnorm3 = nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.pool2 = nn.AvgPool2d((1, 8), stride=(1, pool2_stride))
        self.dropout2 = nn.Dropout(dropout_rate)
        n_feat = F2 * ((((n_times - 4) // pool1_stride + 1) - 8)
                       // pool2_stride + 1)
        self.fc = LinearWithConstraint(n_feat, n_classes, max_norm=norm_rate)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        x = self.conv_spatial(x)
        x = self.bnorm_spatial(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv_depth1(x)
        x = self.conv_point1(x)
        x = self.bnorm3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


def conta_parametros(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ema_standardize(data, factor_new=0.001):
    """Padronização exponencial por canal, como no braindecode."""
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        s = pd.Series(data[i])
        mean = s.ewm(alpha=factor_new).mean()
        std = s.ewm(alpha=factor_new).std()
        out[i] = (data[i] - mean.to_numpy()) / (std.to_numpy() + 1e-9)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _eventos_cue(raw):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    code_to_label = {}
    for code, label in [(769, 0), (770, 1), (771, 2), (772, 3)]:
        chave = str(code)
        if chave in event_id:
            code_to_label[int(event_id[chave])] = label
    if not code_to_label:
        return None, None
    ev = events[np.isin(events[:, 2], list(code_to_label))]
    return ev, code_to_label


def carrega_2a(pasta: Path, n_channels: int, n_classes: int,
               sujeitos=None):
    xs, ys, subs, runs = [], [], [], []
    arquivos = sorted(pasta.glob("*T.gdf"))
    for f in arquivos:
        m = re.search(r"A(\d\d)T", f.name)
        if not m:
            continue
        s = int(m.group(1))
        if sujeitos is not None and s not in sujeitos:
            continue
        raw = mne.io.read_raw_gdf(str(f), preload=True, verbose=False)
        raw.filter(4.0, 40.0, verbose=False)
        raw.resample(FS, verbose=False)
        if n_channels == 3:
            picks = ["EEG-C3", "EEG-C4", "EEG-Cz"]
        else:
            picks = raw.ch_names[:22]
        raw.pick(picks)
        dados = raw.get_data()
        dados = ema_standardize(dados)
        ev, code_to_label = _eventos_cue(raw)
        if ev is None:
            print(f"[aviso] sem cues em {f.name}")
            continue
        for k, (sample, _, code) in enumerate(ev):
            label = code_to_label[int(code)]
            if n_classes == 2 and label not in (0, 1):
                continue
            start = int(sample + 0.5 * FS)
            end = start + N_TIMES
            if start < 0 or end > dados.shape[1]:
                continue
            xs.append(dados[:, start:end].astype(np.float32))
            ys.append(0 if label == 0 and n_classes == 2 else
                      (1 if label == 1 and n_classes == 2 else label))
            subs.append(s)
            runs.append(k // 48)  # 288 trials / 6 runs = 48
        print(f"[2a] {f.name}: {len(xs)} acumuladas")
    return (np.stack(xs), np.array(ys, dtype=np.int64),
            np.array(subs), np.array(runs))


def carrega_2b(pasta: Path):
    """BCIC-IV 2b: 3 canais, 2 classes, apenas sessões T rotuladas."""
    xs, ys, subs, runs = [], [], [], []
    arquivos = sorted(pasta.glob("*T.gdf"))
    if not arquivos:
        raise SystemExit(f"Dataset 2b nao encontrado em {pasta}")
    for f in arquivos:
        m = re.search(r"B(\d\d)", f.name)
        if not m:
            continue
        subj = int(m.group(1))
        raw = mne.io.read_raw_gdf(str(f), preload=True, verbose=False)
        raw.filter(4.0, 40.0, verbose=False)
        raw.resample(FS, verbose=False)
        picks = [c for c in ["EEG:C3", "EEG:C4", "EEG:Cz"]
                 if c in raw.ch_names]
        if len(picks) != 3:
            raise RuntimeError(f"{f.name}: canais encontrados {raw.ch_names}")
        raw.pick(picks)
        dados = ema_standardize(raw.get_data())
        ev, code_to_label = _eventos_cue(raw)
        if ev is None:
            print(f"[aviso] sem cues rotulados em {f.name}")
            continue
        for k, (sample, _, code) in enumerate(ev):
            label = code_to_label[int(code)]
            if label not in (0, 1):
                continue
            start = int(sample + 0.5 * FS)
            end = start + N_TIMES
            if start < 0 or end > dados.shape[1]:
                continue
            xs.append(dados[:, start:end].astype(np.float32))
            ys.append(label)
            subs.append(subj)
            runs.append(k // 120)  # 120 trials por sessão T
        print(f"[2b] {f.name}: {len(xs)} acumuladas")
    return (np.stack(xs), np.array(ys, dtype=np.int64),
            np.array(subs), np.array(runs))

def treina_fold(Xtr, ytr, Xval, yval, n_classes, epochs, seed, device,
                batch_size=32, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(device)
    model = EEGNet(Xtr.shape[1], Xtr.shape[2], n_classes,
                   kernel_length=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    # Tensores ficam uma vez na XPU; evita o overhead de DataLoader para
    # um modelo minúsculo com poucos trials.
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    Xv = torch.from_numpy(Xval).to(device)
    yv = torch.from_numpy(yval).to(device)
    n = Xt.shape[0]
    melhor = 1e9; melhor_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if verbose and (ep == 0 or (ep + 1) % 50 == 0 or ep == epochs - 1):
            print(f"      epoca {ep + 1:03d}/{epochs}: val_loss={val_loss:.4f}",
                  flush=True)
        if val_loss < melhor:
            melhor = val_loss
            melhor_state = {k: v.clone() for k, v in model.state_dict().items()}
    if melhor_state:
        model.load_state_dict(melhor_state)
    return model


def prediz(model, X, device):
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).argmax(1).cpu().numpy()


def run_config(X, y, subj, n_classes, folds, epochs, nome, device,
               verbose=False):
    accs = []
    for s in np.unique(subj):
        m = subj == s
        Xs, ys = X[m], y[m]
        # 4-fold estratificado; validação = 1 dos 3 folds de treino
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(Xs, ys)):
            tr_idx = np.array(tr_idx); te_idx = np.array(te_idx)
            skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            sub_tr, sub_val = next(skf2.split(Xs[tr_idx], ys[tr_idx]))
            tr = tr_idx[sub_tr]; val = tr_idx[sub_val]
            print(f"  sujeito {int(s):02d} fold {fold + 1}/{folds}: treinando...",
                  flush=True)
            model = treina_fold(Xs[tr], ys[tr], Xs[val], ys[val],
                                n_classes, epochs, seed=42 + fold,
                                device=device, verbose=verbose)
            pred = prediz(model, Xs[te_idx], device)
            fold_acc = float(np.mean(pred == ys[te_idx]))
            accs.append(fold_acc)
            print(f"  sujeito {int(s):02d} fold {fold + 1}/{folds}: "
                  f"test_acc={fold_acc:.3f}", flush=True)
        print(f"  sujeito {int(s):02d}: {len(accs)} folds acumulados")
    return np.mean(accs), np.std(accs), len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["2a", "2b"], default="2a")
    ap.add_argument("--n-canais", type=int, choices=[3, 22], default=22)
    ap.add_argument("--n-classes", type=int, choices=[2, 4], default=4)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--sujeitos", default="")
    ap.add_argument("--saida", type=Path, default=Path("outputs/eeg/eegnet_faria"))
    ap.add_argument("--device", default="auto",
                    help="auto, cuda ou cpu")
    ap.add_argument("--pasta-2a", type=Path, default=None)
    ap.add_argument("--pasta-2b", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="Mostra learning curve a cada 50 epochs")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[device] {device} | cuda={torch.cuda.is_available()} | "
          f"xpu={hasattr(torch, 'xpu') and torch.xpu.is_available()}")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA nao disponivel neste ambiente; use --device cpu")
    if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("XPU nao disponivel; instale torch XPU ou use --device cpu")

    # valida a contagem de parâmetros da Tabela 4.1
    esperado = {(22, 4): 1716, (3, 4): 1412, (22, 2): 1458, (3, 2): 1154}
    for (nch, ncl), alvo in esperado.items():
        p = conta_parametros(EEGNet(nch, N_TIMES, ncl, kernel_length=32))
        print(f"[params] {nch}ch/{ncl}cls = {p} (Tabela 4.1: {alvo})")
        assert p == alvo, f"contagem {p} != {alvo}"

    if args.dataset == "2b":
        pasta_2b = args.pasta_2b or Path("data/raw/Datasets/BCICIV_2b_gdf")
        if not pasta_2b.exists():
            pasta_2b = Path("BCICIV_2b_gdf")
        X, y, subj, _ = carrega_2b(pasta_2b)
        args.n_canais, args.n_classes = 3, 2
    else:
        sujeitos = None
        if args.sujeitos:
            sujeitos = {int(x) for x in args.sujeitos.split(",")}
        pasta_2a = args.pasta_2a or Path("data/raw/Datasets/BCICIV_2a_gdf")
        if not pasta_2a.exists():
            pasta_2a = Path("BCICIV_2a_gdf")
        X, y, subj, _ = carrega_2a(pasta_2a,
                                    args.n_canais, args.n_classes, sujeitos)
    print("Dados:", X.shape, "classes:", dict(zip(*np.unique(y, return_counts=True))))
    acc, sd, n = run_config(X, y, subj, args.n_classes, args.folds,
                            args.epochs,
                            f"{args.dataset}_{args.n_canais}c_{args.n_classes}c",
                            device, verbose=args.verbose)
    chance = CHANCE[args.n_classes]
    print(f"[{args.dataset} {args.n_canais}ch {args.n_classes}cls] "
          f"acc={acc:.3f} ± {sd:.3f} (n={n}, chance={chance})")
    (args.saida / f"resultado_{args.dataset}_{args.n_canais}ch_{args.n_classes}cls.md").write_text(
        f"# EEGNet Faria — {args.dataset}, {args.n_canais} canais, "
        f"{args.n_classes} classes\n\n"
        f"- acurácia: {acc:.4f} ± {sd:.4f}\n"
        f"- folds/sujeitos: {n}\n"
        f"- chance: {chance}\n", encoding="utf-8")
    print("Salvo em", args.saida)


if __name__ == "__main__":
    main()
