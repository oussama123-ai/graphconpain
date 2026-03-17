#!/usr/bin/env python3
"""
data/preprocessing/facial_au.py
---------------------------------
Facial Action Unit (AU) extraction via OpenFace 2.2.0 (Algorithm 1).

Extracts 17 pain-relevant AUs per video frame, applies temporal smoothing
(3-frame moving average), and saves (T, 17) numpy arrays.

Usage
-----
    python data/preprocessing/facial_au.py \
        --input  data/icope/videos \
        --output data/icope/features/facial
"""

from __future__ import annotations
import argparse
import subprocess
import re
import csv
from pathlib import Path

import numpy as np


# 17 pain-relevant AUs (Table in Section 3.3.1)
PAIN_AUS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU15_r", "AU17_r", "AU18_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU43_r",
]
N_AUS = len(PAIN_AUS)  # 17


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--openface_bin", type=str, default="FeatureExtraction",
                   help="Path to OpenFace FeatureExtraction binary")
    p.add_argument("--fps",    type=int, default=30)
    return p.parse_args()


def run_openface(video_path: Path, out_dir: Path, openface_bin: str):
    """Call OpenFace FeatureExtraction and return path to CSV output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        openface_bin,
        "-f", str(video_path),
        "-out_dir", str(out_dir),
        "-aus",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"OpenFace failed: {result.stderr}")
    # OpenFace writes <stem>/<stem>.csv
    csv_path = out_dir / video_path.stem / f"{video_path.stem}.csv"
    return csv_path


def read_openface_csv(csv_path: Path) -> np.ndarray:
    """Parse OpenFace CSV and return (T, 17) AU intensity array, normalized [0,1]."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows   = list(reader)

    T   = len(rows)
    aus = np.zeros((T, N_AUS), dtype=np.float32)

    for t, row in enumerate(rows):
        for j, au in enumerate(PAIN_AUS):
            col = au if au in row else au.replace("_r", "_c")
            try:
                aus[t, j] = float(row.get(col, 0.0))
            except ValueError:
                aus[t, j] = 0.0

    # Normalize [0, 5] → [0, 1] per AU
    max_vals = aus.max(axis=0, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    aus /= max_vals

    return aus


def temporal_smooth(aus: np.ndarray, window: int = 3) -> np.ndarray:
    """3-frame moving average (Algorithm 1, line 27)."""
    T, C = aus.shape
    smoothed = aus.copy()
    half = window // 2
    for t in range(half, T - half):
        smoothed[t] = aus[t - half : t + half + 1].mean(axis=0)
    return smoothed


def interpolate_missing(aus: np.ndarray) -> np.ndarray:
    """Linear interpolation for zero frames (Algorithm 1, line 34)."""
    T, C = aus.shape
    for c in range(C):
        signal = aus[:, c]
        zero_mask = signal == 0.0
        if zero_mask.all():
            continue
        indices = np.arange(T)
        valid   = ~zero_mask
        aus[:, c] = np.interp(indices, indices[valid], signal[valid])
    return aus


def main():
    args   = parse_args()
    in_dir = Path(args.input)
    out_dir= Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(list(in_dir.glob("*.mp4")) + list(in_dir.glob("*.avi")))
    print(f"Found {len(video_paths)} videos in {in_dir}")

    for vid_path in video_paths:
        out_path = out_dir / f"{vid_path.stem}.npy"
        if out_path.exists():
            print(f"  [skip] {vid_path.stem}")
            continue
        try:
            tmp_dir  = out_dir / "_openface_tmp"
            csv_path = run_openface(vid_path, tmp_dir, args.openface_bin)
            aus      = read_openface_csv(csv_path)
            aus      = interpolate_missing(aus)
            aus      = temporal_smooth(aus)
            np.save(str(out_path), aus)
            print(f"  [ok]   {vid_path.stem}  shape={aus.shape}")
        except Exception as e:
            print(f"  [err]  {vid_path.stem}: {e}")
            # Save zero array as placeholder
            np.save(str(out_path), np.zeros((1, N_AUS), dtype=np.float32))

    print("Facial AU extraction complete.")


if __name__ == "__main__":
    main()
