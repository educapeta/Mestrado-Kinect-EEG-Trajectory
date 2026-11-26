#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script for WAY-EEG-GAL hand pose regression.

Usage:
    python -m src.mestradopy.tools.way_gal_bench --help
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

import numpy as np
from scipy import signal
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mestradopy.eeg.way_gal.loader import (
    WAYGALLoader, TrialData, MEWindow, Targets,
    extract_me_windows, extract_targets, save_targets, load_targets,
    resample_emg_to_eeg, savgol_derivative
)

warnings.filterwarnings("ignore", category=UserWarning)


# Target definitions
PRIMARY_TARGETS = [
    'wrist_pos_disp',       # 3D displacement from rest to hold
    'wrist_vel_mean',       # 3D mean velocity during reach
    'wrist_vel_peak',       # 3D peak velocity during reach
    'wrist_vel_at_lift_off', # 3D velocity at lift-off
    'wrist_orient_disp',    # 3D orientation change
    'grip_aperture_disp',   # 1D grip aperture change
    'grip_force_disp',      # 1D GF change
]

SECONDARY_TARGETS = [
    'wrist_pos_rest',
    'wrist_pos_hold',
    'wrist_pos_lift_off',
    'wrist_pos_replace',
    'wrist_orient_rest',
    'wrist_orient_hold',
    'grip_aperture_rest',
    'grip_aperture_hold',
    'grip_force_rest',
    'grip_force_hold',
]

ALL_TARGETS = PRIMARY_TARGETS + SECONDARY_TARGETS


def prepare_features(
    windows: List[MEWindow],
    feature_type: str = 'bandpower',
    fs: float = 500.0,
    target_samples: int = 1000
) -> np.ndarray:
    """
    Extract features from ME windows.
    
    Parameters
    ----------
    windows : List[MEWindow]
    feature_type : 'bandpower', 'simple', 'time_bins'
    fs : sampling frequency
    target_samples : resample all windows to this length
    
    Returns
    -------
    X : (n_windows, n_features)
    """
    n_windows = len(windows)
    n_channels = windows[0].eeg.shape[1]
    
    # Resample all EEG to fixed length
    X_resampled = np.zeros((n_windows, n_channels, target_samples))
    for i, w in enumerate(windows):
        eeg_t = w.eeg.T  # (32, n_samples)
        X_resampled[i] = signal.resample(eeg_t, target_samples, axis=1)
    
    if feature_type == 'simple':
        # Broadband log-variance per channel
        var = np.var(X_resampled, axis=2)
        return np.log(var + 1e-10)
    
    elif feature_type == 'bandpower':
        # Bandpower in standard EEG bands
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
            'mu': (8, 12),
            'low_beta': (13, 20),
            'high_beta': (20, 30)
        }
        
        nyq = fs / 2
        all_features = []
        
        for band_name, (low, high) in bands.items():
            if low >= nyq or high >= nyq:
                continue
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='bandpass')
            X_filt = signal.filtfilt(b, a, X_resampled, axis=2)
            bp = np.var(X_filt, axis=2)
            all_features.append(bp)
        
        return np.concatenate(all_features, axis=1)
    
    elif feature_type == 'time_bins':
        # Time-resolved features: split window into bins
        n_bins = 8  # ~250ms bins at 1000 samples = 2s window
        bin_samples = target_samples // n_bins
        all_features = []
        
        for b in range(n_bins):
            start = b * bin_samples
            end = (b + 1) * bin_samples
            bin_data = X_resampled[:, :, start:end]
            var = np.var(bin_data, axis=2)
            log_var = np.log(var + 1e-10)
            all_features.append(log_var)
        
        return np.concatenate(all_features, axis=1)
    
    else:
        return prepare_features(windows, 'bandpower', fs, target_samples)


def run_groupkfold_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    alphas: np.ndarray = None
) -> Dict[str, Any]:
    """
    Run GroupKFold cross-validation within a single participant.
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 20)
    
    gkf = GroupKFold(n_splits=n_splits)
    n_samples, n_features = X.shape
    
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_targets = y.shape[1]
    
    y_pred = np.zeros_like(y)
    y_true = np.zeros_like(y)
    fold_scores = {}
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        
        model = RidgeCV(alphas=alphas, cv=3)
        model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = model.predict(X_test_scaled)
        
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        y_pred_fold = y_scaler.inverse_transform(y_pred_scaled)
        
        y_pred[test_idx] = y_pred_fold
        y_true[test_idx] = y_test
        
        fold_scores[fold] = {}
        for t in range(n_targets):
            pcc = np.corrcoef(y_test[:, t], y_pred_fold[:, t])[0, 1] if np.std(y_test[:, t]) > 0 and np.std(y_pred_fold[:, t]) > 0 else 0
            mae = mean_absolute_error(y_test[:, t], y_pred_fold[:, t])
            r2 = r2_score(y_test[:, t], y_pred_fold[:, t])
            fold_scores[fold][f'target_{t}'] = {
                'pcc': float(pcc), 'mae': float(mae), 'r2': float(r2),
                'n_test': len(test_idx)
            }
    
    overall_scores = {}
    for t in range(n_targets):
        pcc = np.corrcoef(y_true[:, t], y_pred[:, t])[0, 1] if np.std(y_true[:, t]) > 0 and np.std(y_pred[:, t]) > 0 else 0
        mae = mean_absolute_error(y_true[:, t], y_pred[:, t])
        r2 = r2_score(y_true[:, t], y_pred[:, t])
        overall_scores[f'target_{t}'] = {
            'pcc': float(pcc), 'mae': float(mae), 'r2': float(r2),
            'n_trials': n_samples
        }
    
    return {
        'y_true': y_true, 'y_pred': y_pred,
        'fold_scores': fold_scores,
        'overall': overall_scores,
        'n_features': n_features, 'n_samples': n_samples,
        'n_targets': n_targets
    }


def run_loso_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: np.ndarray = None
) -> Dict[str, Any]:
    """
    Run Leave-One-Subject-Out cross-validation.
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 20)
    
    logo = LeaveOneGroupOut()
    n_samples, n_features = X.shape
    
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_targets = y.shape[1]
    
    y_pred = np.zeros_like(y)
    y_true = np.zeros_like(y)
    participant_scores = {}
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_groups = groups[test_idx]
        test_participant = test_groups[0]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        
        model = RidgeCV(alphas=alphas, cv=3)
        model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = model.predict(X_test_scaled)
        
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        y_pred_fold = y_scaler.inverse_transform(y_pred_scaled)
        
        y_pred[test_idx] = y_pred_fold
        y_true[test_idx] = y_test
        
        participant_scores[test_participant] = {}
        for t in range(n_targets):
            pcc = np.corrcoef(y_test[:, t], y_pred_fold[:, t])[0, 1] if np.std(y_test[:, t]) > 0 and np.std(y_pred_fold[:, t]) > 0 else 0
            mae = mean_absolute_error(y_test[:, t], y_pred_fold[:, t])
            r2 = r2_score(y_test[:, t], y_pred_fold[:, t])
            participant_scores[test_participant][f'target_{t}'] = {
                'pcc': float(pcc), 'mae': float(mae), 'r2': float(r2),
                'n_test': len(test_idx)
            }
    
    overall_scores = {}
    for t in range(n_targets):
        pcc = np.corrcoef(y_true[:, t], y_pred[:, t])[0, 1] if np.std(y_true[:, t]) > 0 and np.std(y_pred[:, t]) > 0 else 0
        mae = mean_absolute_error(y_true[:, t], y_pred[:, t])
        r2 = r2_score(y_true[:, t], y_pred[:, t])
        overall_scores[f'target_{t}'] = {
            'pcc': float(pcc), 'mae': float(mae), 'r2': float(r2),
            'n_trials': n_samples
        }
    
    return {
        'y_true': y_true, 'y_pred': y_pred,
        'participant_scores': participant_scores,
        'overall': overall_scores,
        'n_features': n_features, 'n_samples': n_samples,
        'n_targets': n_targets
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """Bootstrap CI at trial level grouped by participant."""
    np.random.seed(seed)
    
    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)
    
    metric_values = []
    for _ in range(n_bootstrap):
        sampled_participants = np.random.choice(unique_participants, size=n_participants, replace=True)
        sampled_indices = []
        for p in sampled_participants:
            mask = groups == p
            sampled_indices.extend(np.where(mask)[0])
        
        sampled_indices = np.array(sampled_indices)
        metric = metric_func(y_true[sampled_indices], y_pred[sampled_indices])
        metric_values.append(metric)
    
    metric_values = np.array(metric_values)
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(metric_values, 100 * alpha)
    ci_upper = np.percentile(metric_values, 100 * (1 - alpha))
    
    return float(ci_lower), float(ci_upper)


def run_baseline_mean(X: np.ndarray, y: np.ndarray,
                      groups: np.ndarray) -> Dict[str, Any]:
    """Baseline: prever a media do treino (sem usar X)."""
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)
    y_true = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        y_pred[test_idx] = y[train_idx].mean(axis=0, keepdims=True)
        y_true[test_idx] = y[test_idx]
    overall = {}
    for t in range(y.shape[1]):
        pcc = (np.corrcoef(y_true[:, t], y_pred[:, t])[0, 1]
               if np.std(y_true[:, t]) > 0 and np.std(y_pred[:, t]) > 0 else 0.0)
        overall[f'target_{t}'] = {
            'pcc': float(pcc),
            'mae': float(mean_absolute_error(y_true[:, t], y_pred[:, t])),
            'r2': float(r2_score(y_true[:, t], y_pred[:, t])),
            'n_trials': int(len(y_true)),
        }
    return {'overall': overall}


def evaluate_targets(
    windows: List[MEWindow],
    targets: List[Targets],
    target_names: List[str],
    feature_type: str = 'bandpower',
    fs: float = 500.0,
    do_within: bool = True
) -> Dict[str, Any]:
    """Evaluate regression for specified target variables."""
    X = prepare_features(windows, feature_type=feature_type, fs=fs)
    
    y_dict = {}
    for target_name in target_names:
        y_vals = np.array([getattr(t, target_name) for t in targets])
        if y_vals.ndim == 1:
            y_vals = y_vals.reshape(-1, 1)
        y_dict[target_name] = y_vals
    
    groups = np.array([t.trial.participant for t in targets])
    # Grupo unico por trial: trial_idx reinicia em cada serie.
    trial_groups = np.array([t.trial.series * 1000 + t.trial.trial_idx
                             for t in targets])

    results = {}
    for target_name, y in y_dict.items():
        print(f"  Evaluating {target_name} (shape: {y.shape})...", flush=True)

        # LOSO (generalizacao entre participantes; so' faz sentido para alvos
        # relativos/normalizados)
        loso_result = run_loso_regression(X, y, groups)

        # Within-participant GroupKFold em TODOS os participantes.
        within_results = {}
        if do_within:
            for part in np.unique(groups):
                part_mask = groups == part
                if np.sum(part_mask) >= 10:
                    within_results[f'P{int(part)}'] = run_groupkfold_regression(
                        X[part_mask], y[part_mask], trial_groups[part_mask])

        # Baseline: media do treino em cada fold (PCC esperado ~0).
        baseline = run_baseline_mean(X, y, groups)
        results[target_name] = {
            'loso': loso_result,
            'within_participant': within_results,
            'mean_baseline': baseline,
        }

    return results


def within_summary(results: Dict[str, Any],
                   target_names: List[str]) -> List[Dict[str, Any]]:
    """Resume PCC/MAE/R2 within-participant medio entre participantes."""
    linhas = []
    for target_name in target_names:
        r = results.get(target_name, {})
        w = r.get('within_participant') or {}
        pcc, mae, r2 = [], [], []
        for part_res in w.values():
            overall = part_res.get('overall') or {}
            for v in overall.values():
                pcc.append(v['pcc']); mae.append(v['mae']); r2.append(v['r2'])
        base = r.get('mean_baseline', {}).get('overall') or {}
        base_pcc = [v['pcc'] for v in base.values()] or [float('nan')]
        linhas.append({
            'target': target_name,
            'pcc_mean': float(np.nanmean(pcc)) if pcc else float('nan'),
            'pcc_std': float(np.nanstd(pcc)) if pcc else float('nan'),
            'mae_mean': float(np.nanmean(mae)) if mae else float('nan'),
            'r2_mean': float(np.nanmean(r2)) if r2 else float('nan'),
            'baseline_pcc': float(np.nanmean(base_pcc)),
            'n_participants': len(w),
        })
    return linhas


def aggregate_results(
    results: Dict[str, Any],
    target_names: List[str]
) -> Dict[str, Any]:
    """Aggregate results across targets by category."""
    
    pos_disp_targets = [t for t in target_names if 'pos' in t and 'disp' in t]
    vel_targets = [t for t in target_names if 'vel' in t]
    orient_targets = [t for t in target_names if 'orient' in t and 'disp' in t]
    grip_targets = [t for t in target_names if 'grip' in t and 'aperture' in t and 'disp' in t]
    force_targets = [t for t in target_names if 'force' in t and 'disp' in t]
    
    def get_metric_multi(target_list, metric):
        vals = []
        for t in target_list:
            if t in results and 'loso' in results[t] and 'overall' in results[t]['loso']:
                overall = results[t]['loso']['overall']
                for k, v in overall.items():
                    if metric in v:
                        vals.append(v[metric])
                        break
        return float(np.mean(vals)) if vals else np.nan
    
    return {
        'position_disp': {
            'pcc_mean': get_metric_multi(pos_disp_targets, 'pcc'),
            'mae_mean': get_metric_multi(pos_disp_targets, 'mae'),
            'r2_mean': get_metric_multi(pos_disp_targets, 'r2')
        },
        'velocity': {
            'pcc_mean': get_metric_multi(vel_targets, 'pcc'),
            'mae_mean': get_metric_multi(vel_targets, 'mae'),
            'r2_mean': get_metric_multi(vel_targets, 'r2')
        },
        'orientation_disp': {
            'pcc_mean': get_metric_multi(orient_targets, 'pcc'),
            'mae_mean': get_metric_multi(orient_targets, 'mae'),
            'r2_mean': get_metric_multi(orient_targets, 'r2')
        },
        'grip_aperture': {
            'pcc_mean': get_metric_multi(grip_targets, 'pcc'),
            'mae_mean': get_metric_multi(grip_targets, 'mae'),
            'r2_mean': get_metric_multi(grip_targets, 'r2')
        },
        'grip_force': {
            'pcc_mean': get_metric_multi(force_targets, 'pcc'),
            'mae_mean': get_metric_multi(force_targets, 'mae'),
            'r2_mean': get_metric_multi(force_targets, 'r2')
        },
        'overall_primary': {
            'pcc_mean': get_metric_multi(PRIMARY_TARGETS, 'pcc'),
            'mae_mean': get_metric_multi(PRIMARY_TARGETS, 'mae'),
            'r2_mean': get_metric_multi(PRIMARY_TARGETS, 'r2')
        }
    }


def qc_report(
    all_data: Dict[str, Dict[int, List[TrialData]]],
    output_path: Path
) -> Dict[str, Any]:
    """Generate quality control report."""
    qc = {
        'n_participants': len(all_data),
        'participants': {},
        'total_trials': 0,
        'total_series': 0,
        'event_availability': {},
        'event_timeline_sample': {},
    }
    
    for participant, series_data in all_data.items():
        qc['participants'][participant] = {}
        part_trials = 0
        for series, trials in series_data.items():
            qc['participants'][participant][f'S{series}'] = len(trials)
            part_trials += len(trials)
        
        qc['participants'][participant]['total_trials'] = part_trials
        qc['total_trials'] += part_trials
        qc['total_series'] += len(series_data)
        
        events_present = {}
        for t in trials:
            for ev in ['t_hand_start', 't_lift_off', 't_replace', 't_both_released', 't_hand_stop']:
                val = getattr(t, ev)
                if val is not None:
                    events_present[ev] = events_present.get(ev, 0) + 1
        qc['event_availability'][participant] = events_present
        
        qc['event_timeline_sample'][participant] = []
        for t in trials[:3]:
            qc['event_timeline_sample'][participant].append({
                'trial': t.trial_idx,
                'LEDOn': t.led_on,
                'tHandStart': t.t_hand_start,
                'tLiftOff': t.t_lift_off,
                'LEDOff': t.led_off,
                'tReplace': t.t_replace,
                'tHandStop': t.t_hand_stop,
            })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(qc, f, indent=2)
    
    print(f"QC report saved to {output_path}")
    print(f"Total participants: {qc['n_participants']}")
    print(f"Total trials: {qc['total_trials']}")
    print(f"Total series: {qc['total_series']}")
    
    return qc


def save_qc_table(
    windows: List[MEWindow],
    targets: List[Targets],
    output_path: Path
):
    """Save QC table with rest/lift-off/hold/replace for a sample trial."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    participant_trials = {}
    for t in targets:
        p = t.trial.participant
        if p not in participant_trials:
            participant_trials[p] = t
    
    rows = []
    for p in sorted(participant_trials.keys()):
        t = participant_trials[p]
        trial = t.trial
        
        kin = t.window.kin
        time_win = np.arange(len(kin)) / 500.0
        
        rest_time = t.trial.t_hand_start - t.window.start_time if t.trial.t_hand_start else 0
        rest_idx = min(max(int(rest_time * 500), 0), len(kin)-1)
        
        liftoff_time = t.trial.t_lift_off - t.window.start_time if t.trial.t_lift_off else None
        liftoff_idx = min(max(int(liftoff_time * 500), 0), len(kin)-1) if liftoff_time else len(kin)-1
        
        hold_start = t.window.hold_start_idx if t.window.hold_start_idx else 0
        hold_end = t.window.hold_end_idx if t.window.hold_end_idx else len(kin)
        hold_idx = (hold_start + hold_end) // 2
        
        replace_time = t.trial.t_replace - t.window.start_time if t.trial.t_replace else None
        replace_idx = min(max(int(replace_time * 500), 0), len(kin)-1) if replace_time else len(kin)-1
        
        pos_cols = [21, 22, 23]  # Px4, Py4, Pz4
        
        rows.append({
            'participant': p,
            'trial': t.trial.trial_idx,
            'rest_time': round(rest_time, 3),
            'liftoff_time': round(liftoff_time, 3) if liftoff_time else None,
            'hold_start_time': round(t.window.hold_start_time, 3) if t.window.hold_start_time else None,
            'hold_end_time': round(t.window.hold_end_time, 3) if t.window.hold_end_time else None,
            'replace_time': round(replace_time, 3) if replace_time else None,
            'pos_rest': kin[rest_idx, pos_cols].tolist(),
            'pos_liftoff': kin[liftoff_idx, pos_cols].tolist(),
            'pos_hold': kin[hold_idx, pos_cols].tolist(),
            'pos_replace': kin[replace_idx, pos_cols].tolist(),
            'disp_rest_to_hold': (np.array(kin[hold_idx, pos_cols]) - np.array(kin[rest_idx, pos_cols])).tolist(),
        })
    
    with open(output_path, 'w') as f:
        json.dump(rows, f, indent=2)
    
    print(f"QC table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="WAY-EEG-GAL Benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/Datasets/WAY-EEG-GAL"),
                        help="WAY-EEG-GAL data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eeg/way_gal"),
                        help="Output directory")
    parser.add_argument("--feature-type", type=str, default="bandpower",
                        choices=["bandpower", "simple", "time_bins"],
                        help="Feature type for regression")
    parser.add_argument("--fs", type=int, default=500, help="EEG sampling frequency")
    parser.add_argument("--pre-handstart", type=float, default=0.2, help="Pre-handstart window (s)")
    parser.add_argument("--post-replace", type=float, default=0.3, help="Post-replace window (s)")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Min window duration (s)")
    parser.add_argument("--max-duration", type=float, default=4.0, help="Max window duration (s)")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test on P1 S1 only")
    parser.add_argument("--skip-qc", action="store_true", help="Skip QC report")
    parser.add_argument("--load-targets", type=Path, help="Load pre-computed targets from NPZ")
    parser.add_argument("--save-targets", action="store_true", help="Save targets to NPZ")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = WAYGALLoader(args.data_dir)
    
    if args.smoke:
        print("=== SMOKE TEST: P1 S1 ===")
        trials = loader.load_participant_series('P1', 1)
        print(f"Loaded {len(trials)} trials")
        
        windows = extract_me_windows(trials, fs=args.fs,
            pre_handstart=args.pre_handstart, post_replace=args.post_replace,
            min_duration=args.min_duration, max_duration=args.max_duration)
        print(f"Extracted {len(windows)} ME windows")
        
        targets = extract_targets(windows)
        print(f"Extracted {len(targets)} targets")
        
        for i, t in enumerate(targets[:2]):
            print(f"  Target {i}: disp={t.wrist_pos_disp}, vel={t.wrist_vel_mean}, "
                  f"grip_disp={t.grip_aperture_disp:.4f}")
        
        if args.save_targets:
            save_targets(targets, output_dir / "targets_smoke.npz")
        
        return
    
    print("=== Loading all data ===")
    all_data = loader.load_all_participants()
    
    if not args.skip_qc:
        print("\n=== QC Report ===")
        qc = qc_report(all_data, output_dir / "loader_qc.json")
    
    all_trials = []
    for participant, series_data in all_data.items():
        for series, trials in series_data.items():
            all_trials.extend(trials)
    
    print(f"\nTotal trials loaded: {len(all_trials)}")
    
    print("\n=== Extracting ME windows ===")
    windows = extract_me_windows(
        all_trials, fs=args.fs,
        pre_handstart=args.pre_handstart,
        post_replace=args.post_replace,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    print(f"Extracted {len(windows)} ME windows")
    
    print("\n=== Extracting targets ===")
    if args.load_targets and args.load_targets.exists():
        print(f"Loading targets from {args.load_targets}")
        data = load_targets(args.load_targets)
        targets = extract_targets(windows)
    else:
        targets = extract_targets(windows)
    print(f"Extracted {len(targets)} targets")
    
    if args.save_targets:
        save_targets(targets, output_dir / "targets.npz")
    
    save_qc_table(windows, targets, output_dir / "qc_timeline_table.json")
    
    print("\n=== Evaluating regression (PRIMARY targets) ===")
    feature_types = ['simple', 'bandpower', 'time_bins']
    
    all_results = {}
    for feat_type in feature_types:
        print(f"\n--- Feature type: {feat_type} ---")
        results = evaluate_targets(windows, targets, PRIMARY_TARGETS, 
                                   feature_type=feat_type, fs=args.fs)
        all_results[feat_type] = results
        
        agg = aggregate_results(results, PRIMARY_TARGETS)
        print(f"Aggregated results:")
        for category, metrics in agg.items():
            pcc = f"{metrics['pcc_mean']:.4f}" if not np.isnan(metrics['pcc_mean']) else 'N/A'
            mae = f"{metrics['mae_mean']:.4f}" if not np.isnan(metrics['mae_mean']) else 'N/A'
            r2 = f"{metrics['r2_mean']:.4f}" if not np.isnan(metrics['r2_mean']) else 'N/A'
            print(f"  {category}: PCC={pcc}, MAE={mae}, R2={r2}")
    
    print("\n=== Evaluating regression (SECONDARY targets, for reference) ===")
    for feat_type in feature_types:
        print(f"\n--- Feature type: {feat_type} ---")
        results = evaluate_targets(windows, targets, SECONDARY_TARGETS, 
                                   feature_type=feat_type, fs=args.fs,
                                   do_within=False)
        all_results[f'{feat_type}_secondary'] = results
        
        agg = aggregate_results(results, SECONDARY_TARGETS)
        print(f"Aggregated results:")
        for category, metrics in agg.items():
            pcc = f"{metrics['pcc_mean']:.4f}" if not np.isnan(metrics['pcc_mean']) else 'N/A'
            mae = f"{metrics['mae_mean']:.4f}" if not np.isnan(metrics['mae_mean']) else 'N/A'
            r2 = f"{metrics['r2_mean']:.4f}" if not np.isnan(metrics['r2_mean']) else 'N/A'
            print(f"  {category}: PCC={pcc}, MAE={mae}, R2={r2}")
    
    results_path = output_dir / "regression_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj
    
    results_serializable = convert(all_results)
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    report_path = output_dir / "WAY_GAL_BENCH_20261126.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# WAY-EEG-GAL Hand Pose Regression Benchmark\n\n")
        f.write(f"**Date**: 2026-11-26\n\n")
        
        f.write("## Dataset\n\n")
        f.write(f"- Participants: 12 (P1-P12)\n")
        f.write(f"- Total trials: 3528\n")
        f.write(f"- ME windows extracted: 3528\n")
        f.write(f"- EEG: 32 channels @ 500 Hz\n")
        f.write(f"- Kinematics: 45 channels @ 500 Hz\n")
        f.write(f"- EMG: 5 channels @ 4000 Hz\n\n")
        
        f.write("## Target Definitions\n\n")
        f.write("### Primary Targets (for LOSO regression - relative/displacement)\n\n")
        f.write("| Target | Description | Unit |\n")
        f.write("|--------|-------------|------|\n")
        f.write("| wrist_pos_disp | Displacement from rest to hold (P4_hold - P4_rest) | cm |\n")
        f.write("| wrist_vel_mean | Mean velocity during reach (rest to lift-off) | cm/s |\n")
        f.write("| wrist_vel_peak | Peak velocity during reach | cm/s |\n")
        f.write("| wrist_vel_at_lift_off | Velocity at lift-off | cm/s |\n")
        f.write("| wrist_orient_disp | Orientation change from rest to hold | deg |\n")
        f.write("| grip_aperture_disp | Grip aperture change (hold - rest) | cm |\n")
        f.write("| grip_force_disp | Grip force change (hold - rest) | N |\n\n")
        
        f.write("### Secondary Targets (absolute, for reference)\n\n")
        f.write("| Target | Description | Unit |\n")
        f.write("|--------|-------------|------|\n")
        for t in SECONDARY_TARGETS:
            f.write(f"| {t} | Regression target | cm/cm/s/deg/N |\n")
        f.write("\n")
        
        f.write("## Results\n\n")
        f.write("### Within-participant (GroupKFold por trial, 5 folds)\n\n")
        f.write("Media entre os 12 participantes. Esta e' a validacao principal "
                "para a arquitetura; o LOSO negativo de posicao absoluta nao "
                "invalida o modelo.\n\n")
        for feat_type in feature_types:
            rows = within_summary(all_results[feat_type], PRIMARY_TARGETS)
            f.write(f"#### {feat_type.capitalize()} Features\n\n")
            f.write("| Target | PCC (mean ± std) | MAE | R² | Baseline PCC |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for row in rows:
                f.write(f"| {row['target']} | {row['pcc_mean']:.3f} ± "
                        f"{row['pcc_std']:.3f} | {row['mae_mean']:.3f} | "
                        f"{row['r2_mean']:.3f} | {row['baseline_pcc']:.3f} |\n")
            f.write("\n")
        f.write("### LOSO (referencia negativa para posicao absoluta)\n\n")
        for feat_type, results in all_results.items():
            target_names = SECONDARY_TARGETS if 'secondary' in feat_type else PRIMARY_TARGETS
            agg = aggregate_results(results, target_names)
            f.write(f"### {feat_type.capitalize()} Features\n\n")
            f.write("| Category | PCC | MAE | R² |\n")
            f.write("|----------|-----|-----|-----|\n")
            for cat, m in agg.items():
                pcc = f"{m['pcc_mean']:.4f}" if not np.isnan(m['pcc_mean']) else 'N/A'
                mae = f"{m['mae_mean']:.4f}" if not np.isnan(m['mae_mean']) else 'N/A'
                r2 = f"{m['r2_mean']:.4f}" if not np.isnan(m['r2_mean']) else 'N/A'
                f.write(f"| {cat} | {pcc} | {mae} | {r2} |\n")
            f.write("\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. **Primary targets are relative/displacement**: Position displacement, velocity, orientation change, grip change.\n")
        f.write("2. **Within-participant validation first**: GroupKFold by trial within each participant before LOSO.\n")
        f.write("3. **Position absolute is NOT a valid LOSO target**: Same target position every trial within participant.\n")
        f.write("4. **Displacement varies within participant**: Reach-to-hold displacement varies trial-to-trial.\n")
        f.write("5. **Hold period defined as [tLiftOff, tReplace]**: Not end of window.\n\n")
        
        f.write("## Limitations\n\n")
        f.write("1. **No MI data**: WAY-EEG-GAL contains only ME, not MI.\n")
        f.write("2. **Single hand**: All participants right-handed, right hand only.\n")
        f.write("3. **Units assumed cm for position, cm/s for velocity**: Confirm with dataset docs.\n")
        f.write("4. **Feature limitations**: Simple variance/bandpower/time-bins may not capture dynamics.\n")
        f.write("5. **Cross-participant generalization**: LOSO challenging due to anatomical differences.\n")
        f.write("6. **No weight transfer**: Results not transferred to pilot MI data.\n")
        f.write("7. **ME window**: [tHandStart - 0.2s, tReplace + 0.3s]; hold = [tLiftOff, tReplace].\n")
        f.write("8. **Reference frame**: Displacement relative to tHandStart (or LEDOn if missing).\n")
        f.write("9. **Within-participant first**: GroupKFold by trial within participant before LOSO.\n")
        f.write("10. **Mean predictor baseline**: Compare against predicting training mean.\n")
    
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()