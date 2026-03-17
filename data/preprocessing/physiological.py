#!/usr/bin/env python3
"""
data/preprocessing/physiological.py
--------------------------------------
ECG / EDA / Respiration feature extraction (Algorithms 5-6).

Processes CSV files of synchronized physiological signals and produces
(T_video, 3, 250) numpy arrays: 3 signal channels × 250-sample windows,
aligned to the video frame rate.

Usage
-----
    python data/preprocessing/physiological.py \
        --input  data/icope/physio \
        --output data/icope/features/physio \
        --video_fps 30 \
        --physio_sr 250
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import entropy as scipy_entropy


N_SIGNALS   = 3     # ECG, EDA, Respiration
WINDOW_LEN  = 250   # samples per window (1 second at 250 Hz)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      type=str, required=True)
    p.add_argument("--output",     type=str, required=True)
    p.add_argument("--video_fps",  type=int, default=30)
    p.add_argument("--physio_sr",  type=int, default=250)
    p.add_argument("--overlap",    type=float, default=0.5,
                   help="Window overlap fraction")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Signal filters
# ---------------------------------------------------------------------------

def bandpass(signal: np.ndarray, lo: float, hi: float, fs: float,
             order: int = 4) -> np.ndarray:
    nyq  = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def lowpass(signal: np.ndarray, cutoff: float, fs: float,
            order: int = 4) -> np.ndarray:
    nyq  = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Per-channel preprocessing
# ---------------------------------------------------------------------------

def preprocess_ecg(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass 0.5–40 Hz, z-score normalize."""
    filtered = bandpass(ecg, 0.5, 40.0, fs)
    mu, std  = filtered.mean(), filtered.std() + 1e-8
    return (filtered - mu) / std


def preprocess_eda(eda: np.ndarray, fs: float) -> np.ndarray:
    """Lowpass 1.0 Hz, polynomial detrend, z-score."""
    filtered = lowpass(eda, 1.0, fs)
    # Polynomial detrend (degree 3)
    t = np.arange(len(filtered))
    coef = np.polyfit(t, filtered, 3)
    trend = np.polyval(coef, t)
    filtered -= trend
    mu, std = filtered.mean(), filtered.std() + 1e-8
    return (filtered - mu) / std


def preprocess_resp(resp: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass 0.1–2.0 Hz, normalize amplitude to [0, 1]."""
    filtered = bandpass(resp, 0.1, 2.0, fs)
    lo, hi   = filtered.min(), filtered.max()
    return (filtered - lo) / (hi - lo + 1e-8)


PREPROCESS_FNS = [preprocess_ecg, preprocess_eda, preprocess_resp]
SIGNAL_COLS    = ["ecg", "eda", "resp"]   # expected CSV column names


# ---------------------------------------------------------------------------
# Sliding window segmentation
# ---------------------------------------------------------------------------

def sliding_windows(signal: np.ndarray, window: int, overlap: float
                    ) -> np.ndarray:
    """
    signal  : (L,) 1-D signal
    Returns : (N_windows, window) array of segments
    """
    step = max(1, int(window * (1.0 - overlap)))
    L    = len(signal)
    windows = []
    for start in range(0, L - window + 1, step):
        windows.append(signal[start : start + window])
    if not windows:
        windows.append(np.zeros(window, dtype=np.float32))
    return np.stack(windows)   # (N, window)


def process_file(csv_path: Path, physio_sr: int, video_fps: int,
                 overlap: float = 0.5) -> np.ndarray:
    """
    Returns (T_video, 3, 250) array of windowed physiological signals.
    """
    df = pd.read_csv(csv_path)

    # Accept flexible column names
    col_map = {}
    for target in SIGNAL_COLS:
        matches = [c for c in df.columns if target.lower() in c.lower()]
        col_map[target] = matches[0] if matches else None

    signals = []
    for target, fn in zip(SIGNAL_COLS, PREPROCESS_FNS):
        col = col_map[target]
        if col and col in df.columns:
            raw = df[col].values.astype(np.float64)
            preprocessed = fn(raw, physio_sr)
        else:
            # Missing signal: zeros
            preprocessed = np.zeros(max(len(df), 1), dtype=np.float64)
        signals.append(preprocessed)

    # Align lengths
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]

    # Sliding windows → (N_windows, 3, 250)
    n_windows = None
    windowed  = []
    for sig in signals:
        w = sliding_windows(sig, WINDOW_LEN, overlap)   # (N, 250)
        n_windows = w.shape[0]
        windowed.append(w)

    # Stack: (N_windows, 3, 250)
    stacked = np.stack(windowed, axis=1).astype(np.float32)  # (N, 3, 250)

    # Resample to video frame rate
    T_video = max(1, int(n_windows * video_fps / (physio_sr / WINDOW_LEN)))
    indices = np.round(np.linspace(0, n_windows - 1, T_video)).astype(int)
    return stacked[indices]   # (T_video, 3, 250)


def main():
    args   = parse_args()
    in_dir = Path(args.input)
    out_dir= Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(in_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files.")

    for csv_path in csv_files:
        out_path = out_dir / f"{csv_path.stem}.npy"
        if out_path.exists():
            print(f"  [skip] {csv_path.stem}")
            continue
        try:
            arr = process_file(csv_path, args.physio_sr,
                               args.video_fps, args.overlap)
            np.save(str(out_path), arr)
            print(f"  [ok]   {csv_path.stem}  shape={arr.shape}")
        except Exception as e:
            print(f"  [err]  {csv_path.stem}: {e}")
            np.save(str(out_path),
                    np.zeros((1, N_SIGNALS, WINDOW_LEN), dtype=np.float32))

    print("Physiological signal processing complete.")


if __name__ == "__main__":
    main()
