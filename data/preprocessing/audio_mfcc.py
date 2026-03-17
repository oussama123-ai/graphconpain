#!/usr/bin/env python3
"""
data/preprocessing/audio_mfcc.py
----------------------------------
Cry acoustic feature extraction (Algorithm 4).

Extracts per-video-frame feature vectors of dimension 65:
  20 MFCC + 20 Δ-MFCC + 20 ΔΔ-MFCC + 5 additional acoustics
  (F0, spectral centroid, ZCR, rolloff, RMS)

Usage
-----
    python data/preprocessing/audio_mfcc.py \
        --input  data/icope/audio \
        --output data/icope/features/audio \
        --video_fps 30
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import librosa


N_MFCC      = 20
N_EXTRA     = 5      # F0, SpectCentroid, ZCR, Rolloff, RMS
OUTPUT_DIM  = N_MFCC * 3 + N_EXTRA   # 65


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",     type=str, required=True)
    p.add_argument("--output",    type=str, required=True)
    p.add_argument("--video_fps", type=int, default=30)
    p.add_argument("--sr",        type=int, default=16000)
    return p.parse_args()


def extract_features(audio_path: Path, sr: int = 16000,
                     video_fps: int = 30) -> np.ndarray:
    """
    Returns (T, 65) feature array aligned to video frame rate.
    """
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    # Pre-emphasis (α = 0.97)
    y_pe = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # STFT parameters (25ms window, 10ms hop)
    n_fft    = int(0.025 * sr)   # 400
    hop_len  = int(0.010 * sr)   # 160

    # MFCC (20 coeffs, skip C0)
    mfcc = librosa.feature.mfcc(
        y=y_pe, sr=sr, n_mfcc=N_MFCC + 1,
        n_fft=n_fft, hop_length=hop_len
    )[1:]                          # (20, T_audio)

    # Delta and delta-delta
    d_mfcc  = librosa.feature.delta(mfcc)
    dd_mfcc = librosa.feature.delta(mfcc, order=2)

    # Additional features
    f0, _, _ = librosa.pyin(y_pe, fmin=50, fmax=600, sr=sr,
                             hop_length=hop_len)
    f0 = np.nan_to_num(f0, nan=0.0)

    spec_centroid = librosa.feature.spectral_centroid(
        y=y_pe, sr=sr, n_fft=n_fft, hop_length=hop_len)[0]
    zcr           = librosa.feature.zero_crossing_rate(
        y_pe, frame_length=n_fft, hop_length=hop_len)[0]
    rolloff       = librosa.feature.spectral_rolloff(
        y=y_pe, sr=sr, n_fft=n_fft, hop_length=hop_len)[0]
    rms           = librosa.feature.rms(
        y=y_pe, frame_length=n_fft, hop_length=hop_len)[0]

    # Stack → (65, T_audio)
    feats = np.vstack([mfcc, d_mfcc, dd_mfcc,
                       f0[None], spec_centroid[None],
                       zcr[None], rolloff[None], rms[None]])   # (65, T_audio)

    # Resample to video frame rate
    T_audio = feats.shape[1]
    T_video = max(1, int(T_audio * video_fps / (sr / hop_len)))

    feats_video = np.zeros((OUTPUT_DIM, T_video), dtype=np.float32)
    for i in range(OUTPUT_DIM):
        feats_video[i] = np.interp(
            np.linspace(0, T_audio - 1, T_video),
            np.arange(T_audio),
            feats[i],
        )

    return feats_video.T   # (T_video, 65)


def main():
    args   = parse_args()
    in_dir = Path(args.input)
    out_dir= Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(list(in_dir.glob("*.wav")) +
                         list(in_dir.glob("*.mp3")) +
                         list(in_dir.glob("*.flac")))
    print(f"Found {len(audio_files)} audio files.")

    for af in audio_files:
        out_path = out_dir / f"{af.stem}.npy"
        if out_path.exists():
            print(f"  [skip] {af.stem}")
            continue
        try:
            feats = extract_features(af, sr=args.sr, video_fps=args.video_fps)
            np.save(str(out_path), feats)
            print(f"  [ok]   {af.stem}  shape={feats.shape}")
        except Exception as e:
            print(f"  [err]  {af.stem}: {e}")
            np.save(str(out_path), np.zeros((1, OUTPUT_DIM), dtype=np.float32))

    print("Audio MFCC extraction complete.")


if __name__ == "__main__":
    main()
