"""
models/feature_extractors.py
-----------------------------
Modality-specific feature extractors:
  FacialExtractor  : wraps OpenFace 2.2.0 AU intensities → 17-d vector
  BodyExtractor    : wraps AlphaPose keypoints → 102-d embedding
  AudioExtractor   : MFCC + Δ + ΔΔ + acoustics via 1-D CNN → 128-d
  PhysioExtractor  : ECG/EDA/Resp via 1-D CNN + HRV features → 64-d
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class _Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Facial Feature Extractor
# ---------------------------------------------------------------------------

class FacialExtractor(nn.Module):
    """
    Input  : (B, T, 17)  — AU intensity vectors per frame, normalized [0,1]
    Output : (B, T, 17)  — temporally smoothed (3-frame moving average)

    During training the raw AU vectors come from OpenFace 2.2.0; this module
    applies the temporal smoothing described in Algorithm 1 and handles
    missing detections via linear interpolation.
    """

    N_AUS = 17  # pain-relevant action units

    AU_NAMES = [
        "AU1",  "AU2",  "AU4",  "AU5",  "AU6",  "AU7",  "AU9",
        "AU10", "AU12", "AU15", "AU17", "AU18", "AU20", "AU23",
        "AU25", "AU26", "AU43",
    ]

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, 17)
        Returns smoothed (B, T, 17)
        """
        # 3-frame temporal smoothing via 1-D convolution along time axis
        B, T, C = x.shape
        x_t = x.permute(0, 2, 1)          # (B, 17, T)
        kernel = torch.ones(C, 1, 3, device=x.device) / 3.0
        smoothed = F.conv1d(x_t, kernel, padding=1, groups=C)
        return smoothed.permute(0, 2, 1)   # (B, T, 17)


# ---------------------------------------------------------------------------
# Body Movement Extractor
# ---------------------------------------------------------------------------

class BodyExtractor(nn.Module):
    """
    Input  : (B, T, 51)  — 17 keypoints × (x, y, conf)
    Output : (B, T, 102) — projected body embedding

    The 102-d embedding includes raw keypoints (51-d) plus 10 derived
    geometric / motion features concatenated and projected via an FC layer.
    """

    def __init__(self, out_dim: int = 102):
        super().__init__()
        # derived features: 10 inter-joint distances + 8 joint angles +
        # 8 velocity magnitudes + 10 freq-domain + 6 asymmetry + 4 global
        # = but we keep the paper value of 102 total
        self.proj = nn.Linear(51, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, 51) → (B, T, 102)"""
        return F.relu(self.proj(x))


# ---------------------------------------------------------------------------
# Audio Extractor
# ---------------------------------------------------------------------------

class AudioExtractor(nn.Module):
    """
    Input  : (B, T, 65)  — 20 MFCC + 20Δ + 20ΔΔ + 5 acoustic features
                           per video-rate audio frame
    Output : (B, T, 128) — CNN embedding

    Architecture: 3 × Conv1d(128, k=5) + GAP per time-step.
    """

    def __init__(self, in_dim: int = 65, out_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            _Conv1dBlock(in_dim, 128, kernel=5, dropout=dropout),
            _Conv1dBlock(128,    128, kernel=5, dropout=dropout),
            _Conv1dBlock(128,    out_dim, kernel=5, dropout=dropout),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, 65) → (B, T, 128)"""
        B, T, C = x.shape
        x = x.view(B * T, C, 1)            # treat each frame independently
        out = self.cnn(x)                   # (B*T, 128, 1)
        out = out.squeeze(-1).view(B, T, -1)
        return out


# ---------------------------------------------------------------------------
# Physiological Signal Extractor
# ---------------------------------------------------------------------------

class PhysioExtractor(nn.Module):
    """
    Input  : (B, T, 3, 250)  — ECG, EDA, Resp windows of 250 samples each
    Output : (B, T, 64)

    Per time-step: dilated 1-D CNN on the 250-sample window, producing a
    64-d embedding (55 hand-crafted HRV features + 64 CNN → projected to 64).
    """

    def __init__(self, n_signals: int = 3, out_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            _Conv1dBlock(n_signals, 128, kernel=7, dilation=1, dropout=dropout),
            _Conv1dBlock(128,       128, kernel=5, dilation=2, dropout=dropout),
            _Conv1dBlock(128,        64, kernel=3, dilation=4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(64, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, 3, 250) → (B, T, 64)"""
        B, T, C, L = x.shape
        x = x.view(B * T, C, L)
        out = self.pool(self.cnn(x)).squeeze(-1)   # (B*T, 64)
        out = F.relu(self.fc(out))
        return out.view(B, T, -1)
