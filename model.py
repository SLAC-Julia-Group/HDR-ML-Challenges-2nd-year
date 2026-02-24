#!/usr/bin/env python3
"""
Coastal Flooding Prediction Model
iHARP ML Challenge 2 -- Submission

Architecture:
    ATLAS foundation model (3.3M params, pretrained on particle track reconstruction)
    transferred to coastal flooding prediction via cross-domain feature engineering.

    Encoder:  [2048, 1024, 512] -> latent_dim=128  (pretrained + fine-tuned)
    Head:     Dense(15)  --  15-bin softmax, one bin per flood day count (0–14)
    y_prob:   1 − P(bin=0) = P(≥1 flood day)

Feature engineering:
    7-day historical sea level (168h) -> 30 temporal segments x 6 features
    6 features match ATLAS track features: energy_centroid, std, time_centroid,
    tail_asymmetry, dt (normalized longitude), kurtosis_shifted

Usage (called by ingestion program):
    python model.py \\
        --train_hourly /path/to/train_hourly.csv \\
        --test_hourly  /path/to/test_hourly.csv \\
        --test_index   /path/to/test_index.csv \\
        --predictions_out /path/to/predictions.csv
"""

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

N_SEGMENTS = 30
MAX_TRACKS = 30
HISTORY_HOURS = 168     # 7 days x 24 hours
N_FEATURES = 6

FEATURE_NAMES = [
    'energy_centroid',  # eta analog: normalized energy-weighted time position [-1, 1]
    'std',              # pt analog:  sea level variability
    'time_centroid',    # phi analog: time-weighted centroid [-pi, pi]
    'tail_asymmetry',   # d0 analog:  distribution tail asymmetry
    'dt',               # z0 analog:  normalized station longitude [-1, 1]
    'kurtosis_shifted', # chi2 analog: excess kurtosis + 3
]

# Preprocessing parameters (from flooding_data_prep.yaml)
PREPROCESSING_CONFIG = {
    'energy_centroid': {'method': 'scale',  'divisor': 0.35},
    'std':             {'method': 'log',    'add': 0.001, 'scale': 0.5, 'divisor': 2.0},
    'time_centroid':   {'method': 'scale',  'divisor': 3.14159},
    'tail_asymmetry':  {'method': 'scale',  'divisor': 1.0},
    'dt':              {'method': 'none'},
    'kurtosis_shifted':{'method': 'scale',  'divisor': 2.0},
}


# ============================================================================
# FEATURE EXTRACTION
# Inlined from flooding_challenge/src/flooding_transform.py
# ============================================================================

def extract_segments(data: np.ndarray, n_segments: int = 30) -> np.ndarray:
    """
    Split (N, 168) time series into (N, n_segments, segment_length) segments.

    For n_segments=30, segment_length = 168 // 30 = 5 hours.
    """
    N = data.shape[0]
    n_hours = data.shape[1]
    segment_length = n_hours // n_segments

    segments = np.zeros((N, n_segments, segment_length), dtype=np.float32)
    for seg_idx in range(n_segments):
        start = seg_idx * segment_length
        end = start + segment_length
        if end <= n_hours:
            segments[:, seg_idx, :] = data[:, start:end]

    return segments


def compute_features(segments: np.ndarray, lon_norm: float) -> np.ndarray:
    """
    Compute 6 features per segment for all N samples.

    Args:
        segments:  (N, n_segments, segment_length)
        lon_norm:  Normalized longitude for this station (scalar, in [-1, 1])

    Returns:
        features:  (N, n_segments, 6)
    """
    N, n_segments, segment_length = segments.shape
    indices = np.arange(segment_length, dtype=np.float32)

    mean = np.mean(segments, axis=-1, keepdims=True)
    std_vals = np.std(segments, axis=-1, keepdims=True) + 1e-10
    normalized = (segments - mean) / std_vals
    abs_signal = np.abs(segments)

    features = np.zeros((N, n_segments, N_FEATURES), dtype=np.float32)

    # Feature 0: energy_centroid [-1, 1]
    ec = np.sum(indices * abs_signal, axis=-1) / (np.sum(abs_signal, axis=-1) + 1e-10)
    features[:, :, 0] = 2.0 * (ec / max(segment_length - 1, 1)) - 1.0

    # Feature 1: std (sea level variability)
    features[:, :, 1] = std_vals.squeeze(-1)

    # Feature 2: time_centroid [-pi, pi]
    tc = np.sum(indices * abs_signal, axis=-1) / (np.sum(abs_signal, axis=-1) + 1e-10)
    seg_indices = np.arange(n_segments, dtype=np.float32)
    gpos = (seg_indices[np.newaxis, :] + tc / segment_length) / n_segments
    features[:, :, 2] = 2.0 * np.pi * (gpos - 0.5)

    # Feature 3: tail_asymmetry (p95-median) - (median-p5)
    p95 = np.percentile(segments, 95, axis=-1)
    p5 = np.percentile(segments, 5, axis=-1)
    median = np.percentile(segments, 50, axis=-1)
    features[:, :, 3] = (p95 - median) - (median - p5)

    # Feature 4: dt = normalized station longitude (static per station)
    features[:, :, 4] = float(lon_norm)

    # Feature 5: kurtosis_shifted = excess_kurtosis + 3
    excess_kurtosis = np.mean(normalized ** 4, axis=-1) - 3.0
    features[:, :, 5] = excess_kurtosis + 3.0

    return features


def sort_by_strength(features: np.ndarray) -> np.ndarray:
    """Sort segments by std (feature index 1) in descending order."""
    strength = features[:, :, 1]  # (N, n_segments)
    sort_indices = np.argsort(-strength, axis=1)
    sorted_features = np.zeros_like(features)
    for feat_idx in range(features.shape[2]):
        sorted_features[:, :, feat_idx] = np.take_along_axis(
            features[:, :, feat_idx], sort_indices, axis=1
        )
    return sorted_features


def apply_preprocessing(tracks: np.ndarray) -> np.ndarray:
    """
    Apply per-feature preprocessing to (N, MAX_TRACKS, 6) tracks array.
    Parameters hard-coded from flooding_data_prep.yaml.
    """
    preprocessed = tracks.copy()
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        cfg = PREPROCESSING_CONFIG.get(feat_name, {'method': 'none'})
        method = cfg.get('method', 'none')
        kwargs = {k: float(v) for k, v in cfg.items() if k != 'method'}

        data = preprocessed[..., feat_idx]
        if method == 'none':
            pass
        elif method == 'log':
            add = kwargs.get('add', 1.0)
            scale = kwargs.get('scale', 1.0)
            divisor = kwargs.get('divisor', 1.0)
            preprocessed[..., feat_idx] = np.log(add + data / scale) / divisor
        elif method == 'scale':
            preprocessed[..., feat_idx] = data / kwargs.get('divisor', 1.0)
        elif method == 'arcsinh':
            preprocessed[..., feat_idx] = np.arcsinh(data) / kwargs.get('divisor', 1.0)
        elif method == 'tanh':
            preprocessed[..., feat_idx] = np.tanh(data / kwargs.get('scale', 50.0))
    return preprocessed


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(pkl_path: str):
    """
    Reconstruct functional Keras model from saved weights + architecture dict.
    No custom Keras class definitions needed.

    Returns (model, arch) where arch contains 'task', 'decision_threshold_days', etc.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    arch    = data['architecture']
    weights = data['weights']
    task    = arch.get('task', 'classification')

    tracks_in = tf.keras.Input(
        shape=(arch['max_tracks'], arch['n_features']), name='tracks'
    )
    x = tf.keras.layers.Flatten()(tracks_in)

    for i, units in enumerate(arch['encoder_layers']):
        x = tf.keras.layers.Dense(units, name=f'enc_dense_{i}')(x)
        x = tf.keras.layers.Activation(arch['activation'], name=f'enc_act_{i}')(x)
        d = arch['dropout'][i] if i < len(arch['dropout']) else 0.0
        if d > 0:
            x = tf.keras.layers.Dropout(d, name=f'enc_drop_{i}')(x)

    x = tf.keras.layers.Dense(arch['latent_dim'], name='latent')(x)

    # --- head hidden layers (stored in arch['head'] by convert_to_pkl.py) ---
    head = arch.get('head', {})
    for i, units in enumerate(head.get('hidden_layers') or []):
        x = tf.keras.layers.Dense(units, name=f'head_dense_{i}')(x)
        x = tf.keras.layers.Activation(head.get('activation', 'relu'),
                                        name=f'head_act_{i}')(x)

    # --- output layer ---
    if task == 'classification':
        out_units = arch['num_classes']
    else:
        out_units = arch.get('output_dim', 1)
    output = tf.keras.layers.Dense(out_units, name='output_head')(x)

    if arch.get('output_activation'):
        output = tf.keras.layers.Activation(
            arch['output_activation'], name='output_activation'
        )(output)

    model = tf.keras.Model(inputs=tracks_in, outputs=output)
    model.set_weights(weights)
    return model, arch


# ============================================================================
# FEATURE MATRIX CONSTRUCTION
# ============================================================================

def process_station_windows(
    station_name: str,
    station_windows: pd.DataFrame,
    hourly_df: pd.DataFrame,
    lon_norm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and transform all historical windows for a single station.

    Args:
        station_name:     Station identifier (for logging)
        station_windows:  Subset of test_index for this station
        hourly_df:        Hourly sea level data for this station (sorted by time)
        lon_norm:         Normalized longitude [-1, 1] for this station

    Returns:
        tracks:    (N, N_SEGMENTS, N_FEATURES) float32 -- sorted, not yet preprocessed
        valid_mask:(N,) bool -- True for windows with enough valid data
    """
    times = hourly_df['time'].values           # numpy datetime64
    sea_levels = hourly_df['sea_level'].values.astype(np.float64)  # keep float64 for interp

    N = len(station_windows)
    window_matrix = np.zeros((N, HISTORY_HOURS), dtype=np.float32)
    valid_mask    = np.ones(N, dtype=bool)

    for j, (_, row) in enumerate(station_windows.iterrows()):
        hist_start = np.datetime64(row['hist_start'])
        # hist_end is the LAST day of history (inclusive), so we go +1 day
        hist_end_excl = np.datetime64(row['hist_end']) + np.timedelta64(1, 'D')

        # Binary search for the hourly range
        start_idx = int(np.searchsorted(times, hist_start, side='left'))
        end_idx   = int(np.searchsorted(times, hist_end_excl, side='left'))

        n_avail = end_idx - start_idx
        n_copy  = min(n_avail, HISTORY_HOURS)
        if n_copy < 24:
            valid_mask[j] = False
            continue

        chunk = sea_levels[start_idx: start_idx + n_copy].copy()

        # Interpolate NaN values
        nan_idx = np.where(np.isnan(chunk))[0]
        if len(nan_idx) > 0:
            valid_idx = np.where(~np.isnan(chunk))[0]
            if len(valid_idx) >= 2:
                chunk[nan_idx] = np.interp(nan_idx, valid_idx, chunk[valid_idx])
            elif len(valid_idx) == 1:
                chunk[:] = chunk[valid_idx[0]]
            else:
                valid_mask[j] = False
                continue

        window_matrix[j, :n_copy] = chunk.astype(np.float32)
        # Edge-pad if shorter than 168 hours
        if n_copy < HISTORY_HOURS:
            window_matrix[j, n_copy:] = float(chunk[-1])

    # Vectorized feature extraction
    segments      = extract_segments(window_matrix, N_SEGMENTS)   # (N, 30, 5)
    features      = compute_features(segments, lon_norm)           # (N, 30, 6)
    sorted_feats  = sort_by_strength(features)                     # (N, 30, 6)

    return sorted_feats, valid_mask


def build_feature_matrix(
    test_hourly: pd.DataFrame,
    train_hourly: pd.DataFrame,
    test_index:  pd.DataFrame,
) -> np.ndarray:
    """
    Build (N, MAX_TRACKS, N_FEATURES) feature matrix for all test windows.

    Uses train_hourly to compute longitude normalization across all 12 stations.
    Returns preprocessed tracks array.
    """
    # Compute lon normalization from ALL stations (train + test) for consistency
    all_lons = pd.concat([train_hourly, test_hourly]).groupby('station_name')['longitude'].first()
    lon_min = float(all_lons.min())
    lon_max = float(all_lons.max())
    lon_range = max(lon_max - lon_min, 1e-10)

    # Parse timestamps once
    test_hourly = test_hourly.copy()
    test_hourly['time'] = pd.to_datetime(test_hourly['time']).values.astype('datetime64[ns]')
    test_hourly['time'] = test_hourly['time'].values.astype('datetime64[h]')  # hourly precision

    # Precompute per-station sorted hourly series
    station_hourly = {}
    station_lons   = {}
    for station, group in test_hourly.groupby('station_name'):
        group = group.sort_values('time').reset_index(drop=True)
        station_hourly[station] = group
        station_lons[station] = float(group['longitude'].iloc[0])

    N = len(test_index)
    all_tracks = np.zeros((N, MAX_TRACKS, N_FEATURES), dtype=np.float32)

    test_index = test_index.reset_index(drop=True)

    for station in test_index['station_name'].unique():
        station_mask     = (test_index['station_name'] == station).values
        station_pos      = np.where(station_mask)[0]   # row positions in test_index
        station_windows  = test_index.iloc[station_pos]

        if station not in station_hourly:
            print(f"  WARNING: no hourly data for station '{station}', skipping.",
                  flush=True)
            continue

        lon = station_lons[station]
        lon_norm = (lon - lon_min) / lon_range * 2.0 - 1.0

        print(f"  {station}: {len(station_windows):,} windows...", end='', flush=True)
        tracks, valid = process_station_windows(
            station, station_windows, station_hourly[station], lon_norm
        )
        all_tracks[station_pos, :N_SEGMENTS, :] = tracks
        n_invalid = int((~valid).sum())
        suffix = f" ({n_invalid} skipped due to missing data)" if n_invalid else ""
        print(f" done.{suffix}", flush=True)

    # Apply per-feature preprocessing
    all_tracks = apply_preprocessing(all_tracks)
    return all_tracks


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(
    model: tf.keras.Model,
    tracks: np.ndarray,
    task: str = 'classification',
    regression_mode: str = 'unbinned',
    decision_threshold_days: float = 1.0,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Run batched inference.

    Returns y_prob = P(flood) in [0, 1]:

      Classification:        y_prob = softmax[:, 1]
      Regression (unbinned): y_prob = clip(count / (2 * decision_threshold_days), 0, 1)
                             → y_prob = 0.5 iff count = decision_threshold_days
      Regression (binned):   y_prob = 1 - softmax(logits)[:, 0]
                             = P(count ≥ 1 day)  [P(flood)]
                             → threshold 0.5 means "more likely flood than not"
    """
    N = tracks.shape[0]
    y_prob = np.zeros(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        batch = tf.constant(tracks[start:end], dtype=tf.float32)
        out   = model(batch, training=False).numpy()
        if task == 'classification':
            y_prob[start:end] = out[:, 1]
        elif regression_mode == 'binned':
            probs = np.exp(out - out.max(axis=1, keepdims=True))  # numerically stable softmax
            probs /= probs.sum(axis=1, keepdims=True)
            y_prob[start:end] = 1.0 - probs[:, 0]   # P(≥1 flood day)
        else:  # unbinned
            y_prob[start:end] = np.clip(
                out[:, 0] / (2.0 * decision_threshold_days), 0.0, 1.0
            )

    # Fill any NaN predictions with 0.5 (neutral / uncertain)
    nan_mask = ~np.isfinite(y_prob)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()} NaN predictions filled with 0.5", flush=True)
        y_prob[nan_mask] = 0.5
    return y_prob


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='iHARP ML Challenge 2 -- Coastal Flooding Prediction'
    )
    parser.add_argument('--train_hourly',    required=True,
                        help='Path to train_hourly.csv')
    parser.add_argument('--test_hourly',     required=True,
                        help='Path to test_hourly.csv')
    parser.add_argument('--test_index',      required=True,
                        help='Path to test_index.csv')
    parser.add_argument('--predictions_out', required=True,
                        help='Output path for predictions.csv')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Inference batch size (default: 2048)')
    args = parser.parse_args()

    # Locate model.pkl relative to this script
    script_dir = Path(__file__).parent
    pkl_path   = script_dir / 'model.pkl'
    if not pkl_path.exists():
        pkl_path = Path('model.pkl')
    if not pkl_path.exists():
        print(f"ERROR: model.pkl not found (tried {script_dir / 'model.pkl'})",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 60, flush=True)
    print("iHARP ML Challenge 2 -- Coastal Flooding Prediction", flush=True)
    print("=" * 60, flush=True)

    # 1. Load CSVs
    print("\n[1/4] Loading data...", flush=True)
    train_hourly = pd.read_csv(args.train_hourly)
    test_hourly  = pd.read_csv(args.test_hourly)
    test_index   = pd.read_csv(args.test_index)
    print(f"  Train hourly rows: {len(train_hourly):,}", flush=True)
    print(f"  Test  hourly rows: {len(test_hourly):,}",  flush=True)
    print(f"  Test  windows:     {len(test_index):,}",   flush=True)
    print(f"  Test  stations:    {sorted(test_index['station_name'].unique())}",
          flush=True)

    # 2. Load model
    print("\n[2/4] Loading model...", flush=True)
    model, arch = load_model(str(pkl_path))
    task = arch['task']
    regression_mode = arch.get('regression_mode', 'unbinned')
    decision_threshold_days = arch.get('decision_threshold_days', 1.0)
    print(f"  Loaded: {pkl_path}", flush=True)
    print(f"  Task:                   {task}", flush=True)
    if task == 'regression':
        if regression_mode == 'binned':
            print(f"  Mode: binned regression  "
                  f"(y_prob = 1−P(bin=0) = P(≥1 occurrence), threshold 0.5)", flush=True)
        else:
            print(f"  Decision threshold:     {decision_threshold_days} day(s)  "
                  f"(y_prob=0.5 ↔ count≥{decision_threshold_days})", flush=True)
    print(f"  Parameters:             {model.count_params():,}", flush=True)

    # 3. Build feature matrix
    print("\n[3/4] Extracting features...", flush=True)
    tracks = build_feature_matrix(test_hourly, train_hourly, test_index)
    print(f"  Feature matrix shape: {tracks.shape}", flush=True)

    # 4. Run inference
    print("\n[4/4] Running inference...", flush=True)
    y_prob = run_inference(model, tracks, task=task,
                           regression_mode=regression_mode,
                           decision_threshold_days=decision_threshold_days,
                           batch_size=args.batch_size)
    print(f"  y_prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]", flush=True)
    print(f"  Predicted flood (>0.5): {(y_prob > 0.5).sum():,} / {len(y_prob):,}",
          flush=True)

    # 5. Save predictions
    out_df = pd.DataFrame({
        'id':     test_index['id'].values,
        'y_prob': y_prob.astype(np.float32),
    })
    out_df.to_csv(args.predictions_out, index=False)
    print(f"\nWrote {len(out_df):,} predictions to {args.predictions_out}", flush=True)


if __name__ == '__main__':
    main()
