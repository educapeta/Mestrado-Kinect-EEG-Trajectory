#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Treinamento baseline para validação EEG 3 classes (MI direita/esquerda/repouso).
Implementa CSP + LDA usando apenas numpy/scipy (sem sklearn).

Carrega CSV/JSON da aquisição, pré-processa, extrai features CSP+LDA,
validação cruzada por trial/bloco, relatório markdown.

Uso:
    python tools/treino_baseline_3classes.py --csv dados_validacao/eeg_mi3c_S01_s01_*.csv \\
        --json dados_validacao/eventos_S01_s01_*.json --saida ./relatorio_validacao
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, linalg, stats

# Suprime warnings de scipy
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# CONSTANTES E CONFIGURAÇÃO
# =============================================================================

# Códigos de marcador (devem bater com validacao_eeg_3classes.py)
CODE_CUE_LEFT = 769
CODE_CUE_RIGHT = 770
CODE_CUE_REST = 784
CODE_ME_START = 778
CODE_ME_END = 779
CODE_MI_START = 780
CODE_MI_END = 781
CODE_BLOCK_START = 790
CODE_BLOCK_END = 791

CLASS_NAMES = ["esquerda", "direita", "repouso"]
CLASS_CODES = {
    "esquerda": CODE_CUE_LEFT,
    "direita": CODE_CUE_RIGHT,
    "repouso": CODE_CUE_REST,
}
CODE_TO_CLASS = {v: k for k, v in CLASS_CODES.items()}

FS = 500.0  # Hz


# =============================================================================
# PRÉ-PROCESSAMENTO
# =============================================================================

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Projeta filtro Butterworth passa-banda."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandstop(lowcut: float, highcut: float, fs: float, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Projeta filtro Butterworth rejeita-banda (notch)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandstop')
    return b, a


def apply_filter(data: np.ndarray, b: np.ndarray, a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Aplica filtro zero-phase (filtfilt) ao longo do eixo especificado."""
    return signal.filtfilt(b, a, data, axis=axis)


def preprocess_eeg(
    eeg_data: np.ndarray,
    fs: float = FS,
    bp_low: float = 0.5,
    bp_high: float = 40.0,
    notch_freq: float = 60.0,
    notch_width: float = 2.0,
) -> np.ndarray:
    """
    Pré-processamento padrão: bandpass 0.5-40 Hz + notch 50/60 Hz.
    
    Args:
        eeg_data: Shape (n_samples, n_channels)
        fs: Taxa de amostragem
        bp_low: Highpass cutoff
        bp_high: Lowpass cutoff
        notch_freq: Frequência do notch (50 ou 60 Hz)
        notch_width: Largura do notch (Hz)
        
    Returns:
        Dados filtrados, shape (n_samples, n_channels)
    """
    # Robustez para dados reais do g.Nautilus: counts grandes e eventuais
    # amostras invalidas. Sem isso o filtfilt propaga NaN por todo o canal.
    data = np.asarray(eeg_data, np.float64).copy()
    for ch in range(data.shape[1]):
        x = data[:, ch]
        bons = np.isfinite(x)
        if bons.sum() < 3:
            data[:, ch] = 0.0
            continue
        if not bons.all():
            idx = np.arange(len(x))
            x[~bons] = np.interp(idx[~bons], idx[bons], x[bons])
            data[:, ch] = x
        if np.std(x) < 1e-12:
            data[:, ch] = 0.0

    # Bandpass
    b_bp, a_bp = butter_bandpass(bp_low, bp_high, fs, order=4)
    data_filt = apply_filter(data, b_bp, a_bp, axis=0)
    
    # Notch
    b_ns, a_ns = butter_bandstop(notch_freq - notch_width/2, notch_freq + notch_width/2, fs, order=4)
    data_filt = apply_filter(data_filt, b_ns, a_ns, axis=0)
    
    return data_filt


def detect_bad_epochs(
    epochs: np.ndarray,
    threshold_std: float = 5.0,
    threshold_range: float = 200.0,  # µV
) -> np.ndarray:
    """
    Detecta épocas com artefatos baseado em amplitude excessiva.
    
    Args:
        epochs: Shape (n_epochs, n_samples, n_channels)
        threshold_std: Limiar em desvios padrão da mediana
        threshold_range: Limiar de amplitude pico-a-pico (µV)
        
    Returns:
        Boolean array (n_epochs,) - True = época boa
    """
    n_epochs = epochs.shape[0]
    good = np.ones(n_epochs, dtype=bool)
    
    for i in range(n_epochs):
        epoch = epochs[i]  # (n_samples, n_channels)
        
        # Critério 1: amplitude pico-a-pico
        ptp = np.ptp(epoch, axis=0)  # por canal
        if np.any(ptp > threshold_range):
            good[i] = False
            continue
            
        # Critério 2: desvio padrão excessivo
        std_ch = np.std(epoch, axis=0)
        median_std = np.median(std_ch)
        if np.any(std_ch > threshold_std * median_std):
            good[i] = False
            continue
    
    return good


# =============================================================================
# EXTRAÇÃO DE ÉPOCAS
# =============================================================================

def extract_epochs_from_markers(
    eeg_data: np.ndarray,
    markers: np.ndarray,
    marker_times: np.ndarray,
    fs: float,
    mi_start_code: int = CODE_MI_START,
    mi_end_code: int = CODE_MI_END,
    cue_codes: Optional[Dict[int, str]] = None,
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> Tuple[List[np.ndarray], List[str], List[int], List[int]]:
    """
    Extrai épocas de MI baseadas nos marcadores.
    
    Args:
        eeg_data: (n_samples, n_channels)
        markers: Array de códigos de marcador por amostra (mesmo tamanho que eeg_data)
        marker_times: Timestamps de cada amostra
        fs: Taxa de amostragem
        mi_start_code: Código do início MI
        mi_end_code: Código do fim MI
        cue_codes: Mapeamento código -> classe para cues
        tmin: Início relativo ao mi_start (s)
        tmax: Fim relativo ao mi_start (s)
        
    Returns:
        (epochs_list, labels, trial_indices, block_indices)
    """
    if cue_codes is None:
        cue_codes = CODE_TO_CLASS
    
    # Encontra índices dos marcadores MI_START
    mi_start_idx = np.where(markers == mi_start_code)[0]
    mi_end_idx = np.where(markers == mi_end_code)[0]
    
    # Para cada MI_START, encontra o cue anterior mais próximo
    epochs = []
    labels = []
    trial_indices = []
    block_indices = []
    
    # Detecta blocos
    block_start_idx = np.where(markers == CODE_BLOCK_START)[0]
    block_end_idx = np.where(markers == CODE_BLOCK_END)[0]
    
    current_block = 0
    trial_in_block = 0
    last_block_start = -1
    
    for i, start_idx in enumerate(mi_start_idx):
        if i >= len(mi_end_idx):
            break
        end_idx = mi_end_idx[i]
        
        # Verifica se mudou de bloco
        if len(block_start_idx) > 0 and start_idx > block_start_idx[0]:
            if current_block < len(block_start_idx) - 1 and start_idx > block_start_idx[current_block + 1]:
                current_block += 1
                trial_in_block = 0
        
        # Encontra cue neste trial (entre block_start e mi_start)
        trial_cue_idx = -1
        trial_class = "desconhecido"
        
        # Busca cue no intervalo [last_block_start, start_idx]
        search_start = block_start_idx[current_block] if current_block < len(block_start_idx) else 0
        cue_candidates = []
        for code, cls in cue_codes.items():
            cue_idx = np.where((markers[search_start:start_idx] == code))[0]
            if len(cue_idx) > 0:
                cue_candidates.append((cue_idx[-1] + search_start, code, cls))
        
        if cue_candidates:
            # Pega o mais próximo do mi_start
            trial_cue_idx, cue_code, trial_class = max(cue_candidates, key=lambda x: x[0])
        else:
            # Fallback: usa a classe do mi_start se houver mapeamento direto
            trial_class = "repouso"
        
        # Extrai época
        start_sample = start_idx + int(tmin * fs)
        end_sample = start_idx + int(tmax * fs)
        
        if end_sample <= eeg_data.shape[0] and start_sample >= 0:
            epoch = eeg_data[start_sample:end_sample, :].copy()
            epochs.append(epoch)
            labels.append(trial_class)
            trial_indices.append(trial_in_block)
            block_indices.append(current_block)
            trial_in_block += 1
    
    return epochs, labels, trial_indices, block_indices


def bandpower_features(epochs: np.ndarray, fs: float = FS,
                      band: Tuple[float, float] = (8.0, 13.0)) -> np.ndarray:
    """Log-bandpower por canal (baseline robusto para validacao de hardware).

    Diferente do CSP, nao estima filtros espaciais no treino; por isso e'
    menos propenso a overfit com poucos trials. Usa a banda mu/alpha, a mais
    classica para MI.
    """
    from scipy import signal as _signal
    n_epochs, n_samples, n_channels = epochs.shape
    freqs, psd = _signal.welch(epochs, fs=fs, nperseg=min(256, n_samples),
                              axis=1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    bp = psd[:, mask, :].mean(axis=1)             # (n_epochs, n_channels)
    return np.log(bp + 1e-12)


# =============================================================================
# CSP (COMMON SPATIAL PATTERNS)
# =============================================================================

def compute_covariance(epoch: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Calcula matriz de covariância normalizada de uma época.
    
    Args:
        epoch: (n_samples, n_channels)
        reg: Regularização (traço médio * reg)
        
    Returns:
        Covariância (n_channels, n_channels)
    """
    # Centraliza
    epoch = epoch - np.mean(epoch, axis=0, keepdims=True)
    # Covariância
    cov = epoch.T @ epoch / (epoch.shape[0] - 1)
    # Regularização
    cov = cov + reg * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return cov


def csp_train(
    epochs: np.ndarray,
    labels: np.ndarray,
    n_components: int = 4,
    class_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Treina filtros CSP para múltiplas classes (One-vs-Rest ou pares).
    
    Args:
        epochs: (n_epochs, n_samples, n_channels)
        labels: Array de strings com nomes das classes
        n_components: Número de componentes CSP por classe/par
        class_pairs: Pares de classes para CSP binário. Se None, usa One-vs-Rest.
        
    Returns:
        Dict com filtros, autovalores, classes, etc.
    """
    classes = np.unique(labels)
    n_classes = len(classes)
    n_channels = epochs.shape[2]
    
    if class_pairs is None:
        # One-vs-Rest para cada classe
        class_pairs = [(cls, "rest") for cls in classes]
    
    filters_dict = {}
    eigenvalues_dict = {}
    
    for cls1, cls2 in class_pairs:
        if cls2 == "rest":
            # One-vs-Rest: cls1 vs todas as outras
            idx1 = np.where(labels == cls1)[0]
            idx2 = np.where(labels != cls1)[0]
        else:
            # Par binário
            idx1 = np.where(labels == cls1)[0]
            idx2 = np.where(labels == cls2)[0]
        
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        
        # Covariâncias médias por classe
        cov1 = np.mean([compute_covariance(epochs[i]) for i in idx1], axis=0)
        cov2 = np.mean([compute_covariance(epochs[i]) for i in idx2], axis=0)
        
        # Covariância composta
        cov_composite = cov1 + cov2
        
        # Decomposição da composta: C = U * lambda * U^T
        eigvals, eigvecs = linalg.eigh(cov_composite)
        # Ordena
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        # Whitening matrix: P = lambda^-1/2 * U^T
        diag_sqrt_inv = np.diag(1.0 / np.sqrt(eigvals + 1e-12))
        P = diag_sqrt_inv @ eigvecs.T
        
        # Transforma covariâncias
        S1 = P @ cov1 @ P.T
        S2 = P @ cov2 @ P.T
        
        # Generalized eigenvalue problem: S1 * w = lambda * S2 * w
        # Resolve via S1 * w = lambda * (S1+S2) * w já que S1+S2 = I após whitening
        eigvals_csp, eigvecs_csp = linalg.eigh(S1, S1 + S2)
        
        # Ordena por distância de 0.5 (mais discriminativo)
        order_csp = np.argsort(np.abs(eigvals_csp - 0.5))[::-1]
        eigvals_csp = eigvals_csp[order_csp]
        eigvecs_csp = eigvecs_csp[:, order_csp]
        
        # Filtros CSP no espaço original: W = P^T * eigvecs_csp
        W = P.T @ eigvecs_csp
        
        # Seleciona componentes (primeiros e últimos)
        n_comp = min(n_components // 2, W.shape[1] // 2)
        if n_comp > 0:
            selected = np.concatenate([np.arange(n_comp), np.arange(-n_comp, 0)])
            W = W[:, selected]
            eigvals_csp = eigvals_csp[selected]
        
        key = f"{cls1}_vs_{cls2}"
        filters_dict[key] = W
        eigenvalues_dict[key] = eigvals_csp
    
    return {
        "filters": filters_dict,
        "eigenvalues": eigenvalues_dict,
        "classes": classes,
        "class_pairs": class_pairs,
        "n_components": n_components,
        "n_channels": n_channels,
    }


def csp_transform(epochs: np.ndarray, csp_model: Dict[str, Any]) -> np.ndarray:
    """
    Aplica filtros CSP às épocas e extrai features (log-variância).
    
    Args:
        epochs: (n_epochs, n_samples, n_channels)
        csp_model: Output de csp_train
        
    Returns:
        Features: (n_epochs, n_features)
    """
    n_epochs = epochs.shape[0]
    filters_dict = csp_model["filters"]
    
    all_features = []
    
    for epoch_idx in range(n_epochs):
        epoch = epochs[epoch_idx]  # (n_samples, n_channels)
        epoch_features = []
        
        for pair_name, W in filters_dict.items():
            # Projeta: Z = epoch @ W  -> (n_samples, n_components)
            Z = epoch @ W
            # Log-variância de cada componente
            var = np.var(Z, axis=0)
            logvar = np.log(var + 1e-12)
            epoch_features.append(logvar)
        
        all_features.append(np.concatenate(epoch_features))
    
    return np.array(all_features)  # (n_epochs, n_features)


# =============================================================================
# LDA (LINEAR DISCRIMINANT ANALYSIS)
# =============================================================================

def lda_train(X: np.ndarray, y: np.ndarray, reg: float = 1e-4) -> Dict[str, Any]:
    """
    Treina classificador LDA multiclasse.
    
    Args:
        X: (n_samples, n_features)
        y: Labels (strings ou ints)
        reg: Regularização da covariância
        
    Returns:
        Modelo LDA com pesos, bias, classes, etc.
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]
    
    # Média global
    mean_global = np.mean(X, axis=0)
    
    # Matrizes de scatter
    Sw = np.zeros((n_features, n_features))  # Within-class scatter
    Sb = np.zeros((n_features, n_features))  # Between-class scatter
    
    class_means = {}
    class_priors = {}
    
    for cls in classes:
        X_cls = X[y == cls]
        n_cls = X_cls.shape[0]
        mean_cls = np.mean(X_cls, axis=0)
        class_means[cls] = mean_cls
        class_priors[cls] = n_cls / len(y)
        
        # Within-class scatter
        centered = X_cls - mean_cls
        Sw += centered.T @ centered
        
        # Between-class scatter
        diff = mean_cls - mean_global
        Sb += class_priors[cls] * np.outer(diff, diff)
    
    # Regularização
    Sw = Sw + reg * np.trace(Sw) / n_features * np.eye(n_features)
    
    # Resolve autovalores generalizados: Sb * w = lambda * Sw * w
    eigvals, eigvecs = linalg.eigh(Sb, Sw)
    
    # Ordena por autovalor decrescente
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    # Projeção LDA (máx n_classes - 1 dimensões)
    n_lda = min(n_classes - 1, n_features)
    W_lda = eigvecs[:, :n_lda]
    
    # Projeta médias das classes no espaço LDA
    projected_means = {}
    for cls in classes:
        projected_means[cls] = W_lda.T @ (class_means[cls] - mean_global)
    
    return {
        "W_lda": W_lda,
        "mean_global": mean_global,
        "class_means": class_means,
        "projected_means": projected_means,
        "class_priors": class_priors,
        "classes": classes,
        "eigenvalues": eigvals[:n_lda],
        "n_features": n_features,
        "n_lda": n_lda,
    }


def lda_predict(X: np.ndarray, lda_model: Dict[str, Any]) -> np.ndarray:
    """
    Prediz classes usando modelo LDA.
    
    Args:
        X: (n_samples, n_features)
        lda_model: Output de lda_train
        
    Returns:
        Predições (n_samples,)
    """
    W_lda = lda_model["W_lda"]
    mean_global = lda_model["mean_global"]
    projected_means = lda_model["projected_means"]
    class_priors = lda_model["class_priors"]
    classes = lda_model["classes"]
    
    # Projeta dados
    X_centered = X - mean_global
    X_proj = X_centered @ W_lda  # (n_samples, n_lda)
    
    # Calcula discriminante para cada classe
    n_samples = X.shape[0]
    scores = np.zeros((n_samples, len(classes)))
    
    for i, cls in enumerate(classes):
        mean_proj = projected_means[cls]  # (n_lda,)
        prior = class_priors[cls]
        # Distância de Mahalanobis simplificada (espaço LDA tem covariância ~ I)
        diff = X_proj - mean_proj
        # Score = -0.5 * ||diff||^2 + log(prior)
        scores[:, i] = -0.5 * np.sum(diff**2, axis=1) + np.log(prior + 1e-12)
    
    # Classe com maior score
    pred_indices = np.argmax(scores, axis=1)
    return classes[pred_indices]


def lda_predict_proba(X: np.ndarray, lda_model: Dict[str, Any]) -> np.ndarray:
    """Retorna probabilidades (softmax dos scores)."""
    W_lda = lda_model["W_lda"]
    mean_global = lda_model["mean_global"]
    projected_means = lda_model["projected_means"]
    class_priors = lda_model["class_priors"]
    classes = lda_model["classes"]
    
    X_centered = X - mean_global
    X_proj = X_centered @ W_lda
    
    n_samples = X.shape[0]
    scores = np.zeros((n_samples, len(classes)))
    
    for i, cls in enumerate(classes):
        mean_proj = projected_means[cls]
        prior = class_priors[cls]
        diff = X_proj - mean_proj
        scores[:, i] = -0.5 * np.sum(diff**2, axis=1) + np.log(prior + 1e-12)
    
    # Softmax
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    return probas


# =============================================================================
# VALIDAÇÃO CRUZADA
# =============================================================================

def stratified_kfold_indices(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Gera índices de validação cruzada estratificada por grupo (bloco/trial).
    Nunca mistura épocas do mesmo trial/bloco entre treino e teste.
    
    Args:
        y: Labels
        groups: IDs de grupo (ex: block_index * 1000 + trial_index)
        n_splits: Número de folds
        shuffle: Embaralhar
        random_state: Seed
        
    Returns:
        Lista de (train_idx, test_idx)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    # Labels por grupo (maiority vote)
    group_labels = []
    for g in unique_groups:
        mask = groups == g
        vals, cnts = np.unique(y[mask], return_counts=True)
        group_labels.append(vals[int(np.argmax(cnts))])
    group_labels = np.array(group_labels)
    
    # Stratified split nos grupos
    from collections import defaultdict
    class_to_groups = defaultdict(list)
    for i, label in enumerate(group_labels):
        class_to_groups[label].append(i)
    
    folds = [[] for _ in range(n_splits)]
    
    for cls, grp_indices in class_to_groups.items():
        if shuffle:
            np.random.shuffle(grp_indices)
        for i, grp_idx in enumerate(grp_indices):
            folds[i % n_splits].append(grp_idx)
    
    # Converte índices de grupos para índices de amostras
    splits = []
    for fold_idx in range(n_splits):
        test_groups = [unique_groups[i] for i in folds[fold_idx]]
        train_groups = [g for g in unique_groups if g not in test_groups]
        
        test_idx = np.where(np.isin(groups, test_groups))[0]
        train_idx = np.where(np.isin(groups, train_groups))[0]
        
        splits.append((train_idx, test_idx))
    
    return splits


def cross_validate(
    epochs: np.ndarray,
    labels: np.ndarray,
    trial_indices: np.ndarray,
    block_indices: np.ndarray,
    csp_n_components: int = 6,
    n_splits: int = 5,
    random_state: int = 42,
    feature_mode: str = "bandpower",
) -> Dict[str, Any]:
    """
    Validação cruzada completa: CSP + LDA por fold.
    
    Args:
        epochs: (n_epochs, n_samples, n_channels)
        labels: Classes
        trial_indices: Índice do trial dentro do bloco
        block_indices: Índice do bloco
        csp_n_components: Componentes CSP
        n_splits: Folds
        random_state: Seed
        
    Returns:
        Dict com métricas, predições, modelos por fold
    """
    # Grupo = trial unico (bloco + trial): nao mistura epocas do mesmo trial
    # entre treino e teste e evita folds vazios quando ha' poucos blocos.
    groups = (np.asarray(block_indices).astype(int) * 1000
              + np.asarray(trial_indices).astype(int))
    
    splits = stratified_kfold_indices(labels, groups, n_splits, random_state=random_state)
    
    all_predictions = []
    all_true = []
    all_probas = []
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {fold + 1}/{n_splits}: treino={len(train_idx)}, teste={len(test_idx)}")
        
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"  Fold {fold + 1}/{n_splits}: VAZIO (ignorado)")
            continue
        # Dados do fold
        X_train_epochs = epochs[train_idx]
        y_train = labels[train_idx]
        X_test_epochs = epochs[test_idx]
        y_test = labels[test_idx]
        
        if feature_mode == "csp":
            csp_model = csp_train(X_train_epochs, y_train,
                                  n_components=csp_n_components)
            X_train_feat = csp_transform(X_train_epochs, csp_model)
            X_test_feat = csp_transform(X_test_epochs, csp_model)
        elif feature_mode == "bandpower":
            X_train_feat = bandpower_features(X_train_epochs, FS)
            X_test_feat = bandpower_features(X_test_epochs, FS)
        else:
            raise ValueError(f"feature_mode invalido: {feature_mode}")
        
        # LDA no treino
        lda_model = lda_train(X_train_feat, y_train)
        
        # Predição
        y_pred = lda_predict(X_test_feat, lda_model)
        y_proba = lda_predict_proba(X_test_feat, lda_model)
        
        # Métricas do fold
        acc = np.mean(y_pred == y_test)
        bal_acc = balanced_accuracy(y_test, y_pred)
        kappa = cohen_kappa(y_test, y_pred)
        
        fold_metrics.append({
            "fold": fold,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "kappa": kappa,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })
        
        all_predictions.extend(y_pred)
        all_true.extend(y_test)
        all_probas.append(y_proba)
    
    # Métricas globais
    all_predictions = np.array(all_predictions)
    all_true = np.array(all_true)
    all_probas = np.vstack(all_probas) if all_probas else np.array([])
    
    global_acc = np.mean(all_predictions == all_true)
    global_bal_acc = balanced_accuracy(all_true, all_predictions)
    global_kappa = cohen_kappa(all_true, all_predictions)
    cm = confusion_matrix(all_true, all_predictions, CLASS_NAMES)
    
    return {
        "splits": splits,
        "fold_metrics": fold_metrics,
        "predictions": all_predictions,
        "true_labels": all_true,
        "probabilities": all_probas,
        "global_accuracy": global_acc,
        "global_balanced_accuracy": global_bal_acc,
        "global_kappa": global_kappa,
        "confusion_matrix": cm,
        "classes": CLASS_NAMES,
    }


# =============================================================================
# MÉTRICAS
# =============================================================================

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy (média do recall por classe)."""
    classes = np.unique(y_true)
    recalls = []
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            recalls.append(np.mean(y_pred[mask] == cls))
    return np.mean(recalls) if recalls else 0.0


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente Kappa de Cohen."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    
    # Matriz de confusão
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[cls_to_idx[t], cls_to_idx[p]] += 1
    
    # Acordo observado
    po = np.trace(cm) / n
    
    # Acordo esperado
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)
    
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """Matriz de confusão com labels."""
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    
    for t, p in zip(y_true, y_pred):
        if t in name_to_idx and p in name_to_idx:
            cm[name_to_idx[t], name_to_idx[p]] += 1
    
    # Normalizada por linha (recall)
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    
    return {
        "matrix": cm.tolist(),
        "matrix_normalized": cm_norm.tolist(),
        "classes": class_names,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Intervalo de confiança bootstrap para uma métrica."""
    np.random.seed(random_state)
    n = len(y_true)
    values = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        values.append(metric_func(y_true[idx], y_pred[idx]))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(values, 100 * alpha)
    upper = np.percentile(values, 100 * (1 - alpha))
    return lower, upper


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================

def load_eeg_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega CSV de EEG com coluna Marker e timestamp.
    
    Retorna:
        eeg_data: (n_samples, n_channels)
        markers: (n_samples,) - códigos de marcador
        timestamps: (n_samples,) - timestamps
    """
    print(f"[INFO] Carregando CSV: {csv_path}")
    
    # Lê header para detectar nomes de colunas
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        # Pula linhas de comentário
        while first_line.startswith('#'):
            first_line = f.readline().strip()
    
    # Detecta delimitador e se a primeira linha e' header
    if ',' in first_line:
        delimiter = ','
    else:
        delimiter = None  # auto
    def _num(x):
        try:
            float(x)
            return True
        except ValueError:
            return False
    header = any(not _num(tok.strip()) for tok in first_line.split(delimiter)
                 if tok.strip())
    skip = 1 if header else 0
    
    # Carrega dados (header nao pode entrar no genfromtxt)
    data = np.genfromtxt(csv_path, delimiter=delimiter, comments='#',
                         skip_header=skip)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"CSV com colunas insuficientes: {data.shape}")
    
    # Primeira coluna = Time, ultima = Marker, meio = canais EEG
    timestamps = data[:, 0]
    markers = np.nan_to_num(data[:, -1], nan=0).astype(int)
    eeg_data = data[:, 1:-1]  # Canais EEG
    
    print(f"  Formato: {eeg_data.shape[0]} amostras, {eeg_data.shape[1]} canais")
    print(f"  Marcadores únicos: {np.unique(markers[markers != 0])}")
    
    return eeg_data, markers, timestamps


def ler_nomes_canais(csv_path: Path) -> list:
    """Lê os nomes de canais do header do CSV de EEG."""
    with open(csv_path, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            if "," in linha and not linha.split(",")[0].replace(".", "", 1).isdigit():
                nomes = [x.strip() for x in linha.split(",")]
                return nomes[1:-1]
            return []
    return []


def load_events_json(json_path: Path) -> Dict[str, Any]:
    """Carrega JSON de eventos/metadados."""
    print(f"[INFO] Carregando JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# RELATÓRIO MARKDOWN
# =============================================================================

def generate_report(
    results: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: Path,
    subject_id: str,
    session_id: str,
) -> Path:
    """Gera relatório markdown com resultados."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"relatorio_validacao_{subject_id}_s{session_id}_{timestamp}.md"
    
    cm = results["confusion_matrix"]
    classes = cm["classes"]
    matrix = cm["matrix"]
    matrix_norm = cm["matrix_normalized"]
    
    # Bootstrap CI
    bal_acc_ci = bootstrap_ci(
        results["true_labels"], results["predictions"], balanced_accuracy
    )
    kappa_ci = bootstrap_ci(
        results["true_labels"], results["predictions"], cohen_kappa
    )
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Relatório de Validação EEG 3 Classes\n\n")
        f.write(f"**Sujeito:** {subject_id}  \n")
        f.write(f"**Sessão:** {session_id}  \n")
        f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Protocolo:** MI 3 classes (direita/esquerda/repouso)  \n")
        f.write(f"**Fonte:** {metadata.get('source', 'desconhecido')}  \n\n")
        
        f.write(f"## Configuração da Aquisição\n\n")
        f.write(f"- **Blocos:** {metadata.get('n_blocks', '?')}\n")
        f.write(f"- **Trials por classe:** {metadata.get('trials_per_class', '?')}\n")
        f.write(f"- **Total trials:** {metadata.get('total_trials', '?')}\n")
        f.write(f"- **Duração MI:** {metadata.get('mi_sec', '?')} s\n")
        f.write(f"- **Duração ME:** {metadata.get('me_sec', '?')} s\n")
        f.write(f"- **Fase analisada:** {metadata.get('fase', '?')}\n")
        f.write(f"- **Duração Cue:** {metadata.get('cue_sec', '?')} s\n")
        f.write(f"- **ITI:** {metadata.get('iti_range', '?')} s\n")
        f.write(f"- **Fs:** {metadata.get('fs', FS)} Hz\n")
        f.write(f"- **Canais:** {metadata.get('channels', '?')}\n\n")
        
        f.write(f"## Pré-processamento\n\n")
        f.write(f"- Bandpass: 0.5-40 Hz (Butterworth ordem 4, zero-phase)\n")
        f.write(f"- Notch: 58-62 Hz (Butterworth ordem 4, zero-phase)\n")
        f.write(f"- Rejeição de artefatos: amplitude pico-a-pico > 200 µV ou std > 5× mediana\n\n")
        
        f.write(f"## Features e Classificador\n\n")
        f.write(f"- **CSP:** {results.get('csp_n_components', 6)} componentes (pares one-vs-rest)\n")
        f.write(f"- **LDA:** Multiclasse com regularização (shrinkage)\n")
        f.write(f"- **Validação cruzada:** {len(results['fold_metrics'])} folds estratificados por bloco\n\n")
        
        f.write(f"## Resultados Globais\n\n")
        f.write(f"| Métrica | Valor |\n")
        f.write(f"|---------|-------|\n")
        f.write(f"| Acurácia global | {results['global_accuracy']:.4f} |\n")
        f.write(f"| **Balanced Accuracy** | **{results['global_balanced_accuracy']:.4f}** |\n")
        f.write(f"| Kappa de Cohen | {results['global_kappa']:.4f} |\n")
        f.write(f"| IC 95% Bal. Acc. | [{bal_acc_ci[0]:.4f}, {bal_acc_ci[1]:.4f}] |\n")
        f.write(f"| IC 95% Kappa | [{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}] |\n\n")
        
        f.write(f"## Matriz de Confusão\n\n")
        f.write(f"### Contagens absolutas\n\n")
        f.write(f"| | " + " | ".join(classes) + " |\n")
        f.write(f"|{'---|' * (len(classes) + 1)}\n")
        for i, cls_true in enumerate(classes):
            row = [str(matrix[i][j]) for j in range(len(classes))]
            f.write(f"| {cls_true} | " + " | ".join(row) + " |\n")
        
        f.write(f"\n### Normalizada por linha (Recall por classe)\n\n")
        f.write(f"| | " + " | ".join(classes) + " |\n")
        f.write(f"|{'---|' * (len(classes) + 1)}\n")
        for i, cls_true in enumerate(classes):
            row = [f"{matrix_norm[i][j]:.2f}" for j in range(len(classes))]
            f.write(f"| {cls_true} | " + " | ".join(row) + " |\n")
        
        f.write(f"\n## Métricas por Fold\n\n")
        f.write(f"| Fold | Acurácia | Bal. Acc. | Kappa |\n")
        f.write(f"|------|----------|-----------|-------|\n")
        for fm in results["fold_metrics"]:
            f.write(f"| {fm['fold']+1} | {fm['accuracy']:.4f} | {fm['balanced_accuracy']:.4f} | {fm['kappa']:.4f} |\n")
        
        f.write(f"\n## Métricas por Classe\n\n")
        f.write(f"| Classe | Precision | Recall | F1-Score |\n")
        f.write(f"|--------|-----------|--------|----------|\n")
        for i, cls in enumerate(classes):
            tp = matrix[i][i]
            fp = sum(matrix[j][i] for j in range(len(classes)) if j != i)
            fn = sum(matrix[i][j] for j in range(len(classes)) if j != i)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f.write(f"| {cls} | {precision:.4f} | {recall:.4f} | {f1:.4f} |\n")
        
        f.write(f"\n## Critério de Aceite (Gate C)\n\n")
        f.write(f"- **Balanced Accuracy ≥ 0.60**: {'✅ PASSA' if results['global_balanced_accuracy'] >= 0.60 else '❌ FALHA'}\n")
        f.write(f"- **IC 95% exclui 0.50**: {'✅ PASSA' if bal_acc_ci[0] > 0.50 else '❌ FALHA'}\n")
        f.write(f"- Kappa > 0: {'✅ PASSA' if results['global_kappa'] > 0 else '❌ FALHA'}\n\n")
        
        f.write(f"## Observações\n\n")
        f.write(f"- Validação cruzada por bloco (não mistura dados do mesmo bloco entre treino/teste)\n")
        f.write(f"- CSP one-vs-rest para cada classe (esquerda vs resto, direita vs resto, repouso vs resto)\n")
        f.write(f"- LDA com regularização automática baseada no traço da covariância\n")
        f.write(f"- Intervalos de confiança via bootstrap (1000 reamostragens)\n")
        if metadata.get('source') == 'generator':
            f.write(f"- **Dados sintéticos** (fonte generator) - usado para teste do pipeline\n")
    
    print(f"[OK] Relatório salvo em: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Treinamento baseline CSP+LDA para validação EEG 3 classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--csv", type=Path, required=True, help="Arquivo CSV de EEG")
    parser.add_argument("--json", type=Path, required=True, help="Arquivo JSON de eventos")
    parser.add_argument("--saida", type=Path, default=Path("./relatorio_validacao"), help="Pasta de saída")
    parser.add_argument("--fase", choices=["MI", "ME"], default="MI",
                        help="Fase a analisar: MI (780/781) ou ME (778/779)")
    parser.add_argument("--mi-tmin", type=float, default=0.5, help="Início da época relativo ao start da fase (s)")
    parser.add_argument("--mi-tmax", type=float, default=4.0, help="Fim da época relativo ao MI start (s)")
    parser.add_argument("--bp-low", type=float, default=0.5, help="Highpass (Hz)")
    parser.add_argument("--bp-high", type=float, default=40.0, help="Lowpass (Hz)")
    parser.add_argument("--notch", type=float, default=60.0, help="Notch frequency (Hz)")
    parser.add_argument("--canais", type=str, default="todos",
                        help="Canais a usar, ex.: C3,Cz,C4 (default: todos)")
    parser.add_argument("--features", choices=["bandpower", "csp"],
                        default="bandpower",
                        help="Features: bandpower (robusto) ou csp (experimental)")
    parser.add_argument("--csp-components", type=int, default=6, help="Componentes CSP por par")
    parser.add_argument("--cv-folds", type=int, default=5, help="Folds de validação cruzada")
    parser.add_argument("--seed", type=int, default=42, help="Seed aleatória")
    
    args = parser.parse_args()
    
    # Cria pasta de saída
    output_dir = Path(args.saida)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TREINAMENTO BASELINE CSP+LDA - VALIDAÇÃO EEG 3 CLASSES")
    print("=" * 60)
    
    # Carrega dados
    try:
        eeg_data, markers, timestamps = load_eeg_csv(args.csv)
        metadata = load_events_json(args.json)
        metadata['fase'] = args.fase
        if args.canais and args.canais.strip().lower() not in ("todos", "all"):
            # O CsvWriter do gpype nomeia as colunas Ch01..Ch32; os nomes
            # anatomicos estao no JSON de metadados. Usar o JSON quando o
            # header for generico.
            nomes = ler_nomes_canais(args.csv)
            if (not nomes or all(n.lower().startswith("ch") for n in nomes)):
                nomes = list(metadata.get("channel_names") or [])
            desejados = [c.strip() for c in args.canais.split(",") if c.strip()]
            indices = []
            nao_encontrados = []
            for token in desejados:
                if token in nomes:
                    indices.append(nomes.index(token))
                elif token.isdigit():
                    # índice 1-based do canal físico (como no g.recorder)
                    idx = int(token) - 1
                    if 0 <= idx < len(nomes):
                        indices.append(idx)
                    else:
                        nao_encontrados.append(token)
                else:
                    nao_encontrados.append(token)
            if nao_encontrados:
                raise SystemExit(f"Canais nao encontrados no CSV/JSON: "
                                 f"{nao_encontrados}; disponiveis: {nomes}")
            eeg_data = eeg_data[:, indices]
            print(f"[INFO] Canais selecionados: {desejados} -> indices "
                  f"{[i+1 for i in indices]} -> shape {eeg_data.shape}")
    except Exception as e:
        print(f"[ERRO] Falha ao carregar dados: {e}")
        return 1
    
    subject_id = metadata.get('subject_id', 'S01')
    session_id = metadata.get('session_id', '01')
    
    # Pré-processamento
    print(f"[INFO] Pré-processando EEG (bandpass {args.bp_low}-{args.bp_high} Hz, notch {args.notch} Hz)...")
    eeg_filt = preprocess_eeg(
        eeg_data,
        fs=metadata.get('fs', FS),
        bp_low=args.bp_low,
        bp_high=args.bp_high,
        notch_freq=args.notch,
    )
    
    # Extração de épocas
    print("[INFO] Extraindo épocas de MI...")
    inicio_fase = CODE_ME_START if args.fase == "ME" else CODE_MI_START
    fim_fase = CODE_ME_END if args.fase == "ME" else CODE_MI_END
    print(f"[INFO] Fase analisada: {args.fase} "
          f"(marcadores {inicio_fase}/{fim_fase})")
    epochs_list, labels, trial_idx, block_idx = extract_epochs_from_markers(
        eeg_filt,
        markers,
        timestamps,
        fs=metadata.get('fs', FS),
        mi_start_code=inicio_fase,
        mi_end_code=fim_fase,
        tmin=args.mi_tmin,
        tmax=args.mi_tmax,
        cue_codes=CODE_TO_CLASS,
    )
    
    if len(epochs_list) == 0:
        print("[ERRO] Nenhuma época extraída. Verifique marcadores.")
        return 1
    
    epochs = np.array(epochs_list)  # (n_epochs, n_samples, n_channels)
    labels = np.array(labels)
    trial_idx = np.array(trial_idx)
    block_idx = np.array(block_idx)
    
    print(f"  Épocas extraídas: {len(epochs)}")
    print(f"  Classes: {np.unique(labels, return_counts=True)}")
    print(f"  Blocos: {np.unique(block_idx)}")
    
    # Rejeição de artefatos
    print("[INFO] Detectando épocas com artefatos...")
    good_mask = detect_bad_epochs(epochs)
    n_bad = np.sum(~good_mask)
    if n_bad > 0:
        print(f"  Removendo {n_bad} épocas com artefatos")
        epochs = epochs[good_mask]
        labels = labels[good_mask]
        trial_idx = trial_idx[good_mask]
        block_idx = block_idx[good_mask]
    
    print(f"  Épocas finais: {len(epochs)}")
    
    # Validação cruzada
    print(f"[INFO] Iniciando validação cruzada ({args.cv_folds} folds, por bloco)...")
    results = cross_validate(
        epochs,
        labels,
        trial_idx,
        block_idx,
        csp_n_components=args.csp_components,
        n_splits=args.cv_folds,
        feature_mode=args.features,
        random_state=args.seed,
    )
    results['csp_n_components'] = args.csp_components
    
    # Exibe resultados
    print(f"\n{'='*60}")
    print(f"RESULTADOS FINAIS")
    print(f"{'='*60}")
    print(f"Balanced Accuracy: {results['global_balanced_accuracy']:.4f}")
    print(f"Kappa: {results['global_kappa']:.4f}")
    print(f"Acurácia: {results['global_accuracy']:.4f}")
    
    # Gera relatório
    report_path = generate_report(results, metadata, output_dir, subject_id, session_id)
    
    # Critério de aceite
    if results['global_balanced_accuracy'] >= 0.60:
        print(f"\n✅ CRITÉRIO DE ACEITE ATENDIDO: Balanced Accuracy ≥ 0.60")
        return 0
    else:
        print(f"\n❌ CRITÉRIO DE ACEITE NÃO ATENDIDO: Balanced Accuracy < 0.60")
        return 1


if __name__ == "__main__":
    sys.exit(main())
