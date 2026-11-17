#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEGNet configuravel.

Base: blocos EEGNet-8,2 (Lawhern et al., 2018).
Variante: classificador com uma camada densa oculta configuravel.

Com C3/Cz/C4, 256 amostras, 2 classes e classifier_hidden=36, a variante tem
5.870 parametros treinaveis (~5.900). Nao e o EEGNet original de 1.410
parametros; e uma variante explicitamente configuravel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EEGNetConfigurable(nn.Module):
    def __init__(self, n_channels=3, n_times=256, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5,
                 norm_rate=0.25, classifier_hidden=36):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.classifier_hidden = classifier_hidden

        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length), bias=False, padding="same")
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.conv_spatial = Conv2dWithConstraint(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False,
            padding="valid", max_norm=1.0)
        self.bnorm_spatial = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=0.001)
        self.pool1 = nn.AvgPool2d((1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(dropout)

        self.conv_depth1 = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), bias=False, groups=F1 * D,
            padding="same")
        self.conv_point1 = nn.Conv2d(
            F1 * D, F2, (1, 1), bias=False, padding=(0, 0))
        self.bnorm3 = nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.pool2 = nn.AvgPool2d((1, 8), stride=(1, 8))
        self.dropout2 = nn.Dropout(dropout)

        n_feat = F2 * ((((n_times - 4) // 4 + 1) - 8) // 8 + 1)
        self.n_feat = n_feat
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(n_feat, classifier_hidden,
                                 max_norm=norm_rate),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_classes),
        )

    def forward(self, x):
        # entradas aceitas: (B, C, T) ou (B, 1, C, T)
        if x.ndim == 3:
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
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: dict, n_channels=None, n_times=None,
                n_classes=None):
    m = config["model"]
    return EEGNetConfigurable(
        n_channels=n_channels or m["n_channels"],
        n_times=n_times or m["n_times"],
        n_classes=n_classes or m["n_classes"],
        F1=m["F1"], D=m["D"], F2=m["F2"],
        kernel_length=m["kernel_length"], dropout=m["dropout"],
        norm_rate=m["norm_rate"],
        classifier_hidden=m["classifier_hidden"],
    )
