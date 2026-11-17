"""
Metrics computation from fusion CSV logs.

Provides summarize_fusion_log() to analyze a CSV log and produce
summary statistics per hand and overall.
"""

from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# Expected columns (from fusion/logging.py CSV_HEADERS)
EXPECTED_COLUMNS = {
    't_mono_s': float,
    'hand': str,
    'source': str,
    'confidence': float,
    'valid': str,  # '1'/'0'
    'latency_ms': float,
    'q_w': float, 'q_x': float, 'q_y': float, 'q_z': float,
    'n_x': float, 'n_y': float, 'n_z': float,
    'dir_x': float, 'dir_y': float, 'dir_z': float,
    'vis_n_x': float, 'vis_n_y': float, 'vis_n_z': float,
    'err_fused_vis_deg': float,
    'err_cal_vis_deg': float,
    'imu_age_ms': float,
    'imu_roll_deg': float, 'imu_pitch_deg': float, 'imu_yaw_deg': float,
    'imu_gx_dps': float, 'imu_gy_dps': float, 'imu_gz_dps': float,
}


def _safe_float(s: str) -> Optional[float]:
    """Convert string to float, return None if empty or invalid."""
    if not s or s.strip() == '':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_str(s: str) -> str:
    """Return stripped string."""
    return s.strip() if s else ''


def _percentile(data: List[float], p: float) -> Optional[float]:
    """Compute percentile (0-100) of data, or None if empty."""
    if not data:
        return None
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[f] * (c - k)
    d1 = data_sorted[c] * (k - f)
    return d0 + d1


def _rmse(data: List[float]) -> Optional[float]:
    """Compute RMSE of data (treating as errors)."""
    if not data:
        return None
    return math.sqrt(sum(x * x for x in data) / len(data))


def _mean(data: List[float]) -> Optional[float]:
    """Compute mean of data."""
    if not data:
        return None
    return sum(data) / len(data)


def _std(data: List[float]) -> Optional[float]:
    """Compute standard deviation of data."""
    if not data:
        return None
    m = _mean(data)
    if m is None:
        return None
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read CSV file and return list of row dicts with typed values."""
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
        for row in reader:
            typed = {}
            for col, caster in EXPECTED_COLUMNS.items():
                val = row.get(col, '')
                if caster is float:
                    typed[col] = _safe_float(val)
                else:
                    typed[col] = _safe_str(val)
            rows.append(typed)
    return rows


def _filter_hand(rows: List[Dict[str, Any]], hand: str) -> List[Dict[str, Any]]:
    """Filter rows for a specific hand."""
    return [r for r in rows if r.get('hand') == hand]


def _filter_valid(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter rows where valid == '1'."""
    return [r for r in rows if r.get('valid') == '1']


def _filter_source(rows: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Filter rows by source."""
    return [r for r in rows if r.get('source') == source]


def _has_column(rows: List[Dict[str, Any]], col: str) -> bool:
    """Check if any row has non-None value for column."""
    return any(r.get(col) is not None for r in rows)


def _collect(rows: List[Dict[str, Any]], col: str) -> List[float]:
    """Collect non-None values for a column."""
    return [r[col] for r in rows if r.get(col) is not None]


def summarize_fusion_log(path: str | Path) -> Dict[str, Any]:
    """
    Analyze a fusion CSV log and return summary metrics.

    Args:
        path: Path to CSV log file

    Returns:
        Dictionary with summary statistics per hand and overall.
        If a column is missing/empty, reports n=0 for that metric.
    """
    path = Path(path)
    if not path.exists():
        return {'error': f'File not found: {path}'}

    rows = _read_csv(path)
    if not rows:
        return {'error': 'Empty log file', 'path': str(path)}

    # Overall stats
    total_rows = len(rows)
    hands_present = sorted(set(r.get('hand', '') for r in rows if r.get('hand')))

    result = {
        'path': str(path),
        'total_rows': total_rows,
        'hands': hands_present,
        'per_hand': {},
        'overall': {},
    }

    # Per-hand analysis
    for hand in hands_present:
        hand_rows = _filter_hand(rows, hand)
        valid_rows = _filter_valid(hand_rows)
        n_total = len(hand_rows)
        n_valid = len(valid_rows)

        # Source counts
        source_counts = {}
        for r in hand_rows:
            src = r.get('source', 'none')
            source_counts[src] = source_counts.get(src, 0) + 1

        # Fused availability
        fused_rows = _filter_source(valid_rows, 'fused')
        fused_count = len(fused_rows)
        fused_pct = (fused_count / n_valid * 100) if n_valid > 0 else 0.0

        imu_only_rows = _filter_source(valid_rows, 'imu_only')
        vision_only_rows = _filter_source(valid_rows, 'vision_only:kinect_depth')
        vision_only_kinect_rows = _filter_source(valid_rows, 'vision_only:kinect_depth')
        none_rows = _filter_source(hand_rows, 'none')

        # Errors (fused vs vision)
        err_fused_vis = _collect(fused_rows, 'err_fused_vis_deg')
        err_cal_vis = _collect(fused_rows, 'err_cal_vis_deg')

        # Latency
        latency = _collect(valid_rows, 'latency_ms')

        # IMU age
        imu_age = _collect(valid_rows, 'imu_age_ms')

        # Confidence
        confidence = _collect(valid_rows, 'confidence')

        hand_stats = {
            'hand': hand,
            'total_rows': n_total,
            'valid_rows': n_valid,
            'availability_pct': (n_valid / n_total * 100) if n_total > 0 else 0.0,
            'source_counts': source_counts,
            'fused_count': fused_count,
            'fused_pct': fused_pct,
            'imu_only_count': len(imu_only_rows),
            'vision_only_count': len(vision_only_rows),
            'none_count': len(none_rows),
        }

        # Error metrics (only if column has data)
        if err_fused_vis:
            hand_stats['err_fused_vis'] = {
                'n': len(err_fused_vis),
                'mean_deg': _mean(err_fused_vis),
                'std_deg': _std(err_fused_vis),
                'rmse_deg': _rmse(err_fused_vis),
                'p50_deg': _percentile(err_fused_vis, 50),
                'p95_deg': _percentile(err_fused_vis, 95),
                'max_deg': max(err_fused_vis),
            }
        else:
            hand_stats['err_fused_vis'] = {'n': 0}

        if err_cal_vis:
            hand_stats['err_cal_vis'] = {
                'n': len(err_cal_vis),
                'mean_deg': _mean(err_cal_vis),
                'std_deg': _std(err_cal_vis),
                'rmse_deg': _rmse(err_cal_vis),
                'p50_deg': _percentile(err_cal_vis, 50),
                'p95_deg': _percentile(err_cal_vis, 95),
                'max_deg': max(err_cal_vis),
            }
        else:
            hand_stats['err_cal_vis'] = {'n': 0}

        # Latency metrics
        if latency:
            hand_stats['latency_ms'] = {
                'n': len(latency),
                'mean_ms': _mean(latency),
                'std_ms': _std(latency),
                'p50_ms': _percentile(latency, 50),
                'p95_ms': _percentile(latency, 95),
                'max_ms': max(latency),
            }
        else:
            hand_stats['latency_ms'] = {'n': 0}

        # IMU age metrics
        if imu_age:
            hand_stats['imu_age_ms'] = {
                'n': len(imu_age),
                'mean_ms': _mean(imu_age),
                'std_ms': _std(imu_age),
                'p50_ms': _percentile(imu_age, 50),
                'p95_ms': _percentile(imu_age, 95),
                'max_ms': max(imu_age),
            }
        else:
            hand_stats['imu_age_ms'] = {'n': 0}

        # Confidence metrics
        if confidence:
            hand_stats['confidence'] = {
                'n': len(confidence),
                'mean': _mean(confidence),
                'std': _std(confidence),
                'p50': _percentile(confidence, 50),
                'p95': _percentile(confidence, 95),
                'min': min(confidence),
                'max': max(confidence),
            }
        else:
            hand_stats['confidence'] = {'n': 0}

        result['per_hand'][hand] = hand_stats

    # Overall aggregates
    all_valid = _filter_valid(rows)
    all_fused = _filter_source(all_valid, 'fused')

    all_err_fused_vis = _collect(all_fused, 'err_fused_vis_deg')
    all_err_cal_vis = _collect(all_fused, 'err_cal_vis_deg')
    all_latency = _collect(all_valid, 'latency_ms')
    all_imu_age = _collect(all_valid, 'imu_age_ms')

    overall = {
        'total_rows': total_rows,
        'valid_rows': len(all_valid),
        'fused_rows': len(all_fused),
        'fused_pct': (len(all_fused) / len(all_valid) * 100) if all_valid else 0.0,
    }

    if all_err_fused_vis:
        overall['err_fused_vis'] = {
            'n': len(all_err_fused_vis),
            'mean_deg': _mean(all_err_fused_vis),
            'rmse_deg': _rmse(all_err_fused_vis),
            'p95_deg': _percentile(all_err_fused_vis, 95),
        }
    else:
        overall['err_fused_vis'] = {'n': 0}

    if all_err_cal_vis:
        overall['err_cal_vis'] = {
            'n': len(all_err_cal_vis),
            'mean_deg': _mean(all_err_cal_vis),
            'rmse_deg': _rmse(all_err_cal_vis),
            'p95_deg': _percentile(all_err_cal_vis, 95),
        }
    else:
        overall['err_cal_vis'] = {'n': 0}

    if all_latency:
        overall['latency_ms'] = {
            'n': len(all_latency),
            'mean_ms': _mean(all_latency),
            'p95_ms': _percentile(all_latency, 95),
        }
    else:
        overall['latency_ms'] = {'n': 0}

    if all_imu_age:
        overall['imu_age_ms'] = {
            'n': len(all_imu_age),
            'mean_ms': _mean(all_imu_age),
            'p95_ms': _percentile(all_imu_age, 95),
        }
    else:
        overall['imu_age_ms'] = {'n': 0}

    result['overall'] = overall
    return result


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    if 'error' in summary:
        print(f"[ERRO] {summary['error']}")
        return

    print("\n" + "=" * 60)
    print("RESUMO DO LOG DE FUSÃO")
    print("=" * 60)
    print(f"Arquivo:    {summary['path']}")
    print(f"Linhas:     {summary['total_rows']}")
    print(f"Mãos:       {', '.join(summary['hands']) if summary['hands'] else 'nenhuma'}")

    for hand in summary['hands']:
        h = summary['per_hand'][hand]
        print(f"\n--- {hand.upper()} ---")
        print(f"  Total/Validas: {h['total_rows']}/{h['valid_rows']} ({h['availability_pct']:.1f}%)")
        print(f"  Fontes: {h['source_counts']}")
        print(f"  Fused: {h['fused_count']} ({h['fused_pct']:.1f}% das válidas)")

        ev = h['err_fused_vis']
        if ev['n'] > 0:
            print(f"  Erro fused vs visão: mean={ev['mean_deg']:.2f}° rmse={ev['rmse_deg']:.2f}° p95={ev['p95_deg']:.2f}° max={ev['max_deg']:.2f}° (n={ev['n']})")
        else:
            print(f"  Erro fused vs visão: n=0")

        ec = h['err_cal_vis']
        if ec['n'] > 0:
            print(f"  Erro cal vs visão:   mean={ec['mean_deg']:.2f}° rmse={ec['rmse_deg']:.2f}° p95={ec['p95_deg']:.2f}° max={ec['max_deg']:.2f}° (n={ec['n']})")
        else:
            print(f"  Erro cal vs visão:   n=0")

        lat = h['latency_ms']
        if lat['n'] > 0:
            print(f"  Latência: mean={lat['mean_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms max={lat['max_ms']:.1f}ms (n={lat['n']})")
        else:
            print(f"  Latência: n=0")

        imu = h['imu_age_ms']
        if imu['n'] > 0:
            print(f"  Idade IMU: mean={imu['mean_ms']:.1f}ms p95={imu['p95_ms']:.1f}ms max={imu['max_ms']:.1f}ms (n={imu['n']})")
        else:
            print(f"  Idade IMU: n=0")

    ov = summary['overall']
    print(f"\n--- GERAL ---")
    print(f"  Total/Validas/Fused: {ov['total_rows']}/{ov['valid_rows']}/{ov['fused_rows']} ({ov['fused_pct']:.1f}%)")

    ev = ov['err_fused_vis']
    if ev['n'] > 0:
        print(f"  Erro fused vs visão: mean={ev['mean_deg']:.2f}° rmse={ev['rmse_deg']:.2f}° p95={ev['p95_deg']:.2f}° (n={ev['n']})")

    lat = ov['latency_ms']
    if lat['n'] > 0:
        print(f"  Latência: mean={lat['mean_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms (n={lat['n']})")

    imu = ov['imu_age_ms']
    if imu['n'] > 0:
        print(f"  Idade IMU: mean={imu['mean_ms']:.1f}ms p95={imu['p95_ms']:.1f}ms (n={imu['n']})")

    print("=" * 60)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m fusion.metrics <csv_log_path>")
        sys.exit(1)
    summary = summarize_fusion_log(sys.argv[1])
    print_summary(summary)