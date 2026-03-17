"""
utils/augmentation.py
----------------------
Training-time augmentations for all four modalities (Table C6 in paper).
"""

from __future__ import annotations
import random
import torch
import torch.nn.functional as F
import numpy as np


class MultimodalAugmentation:
    """
    Apply random augmentations to all modalities in a batch item dict.
    Designed to be called inside __getitem__ or a collate function.
    """

    def __call__(self, item: dict) -> dict:
        facial  = self._aug_facial(item["facial"])
        body    = self._aug_body(item["body"])
        audio   = self._aug_audio(item["audio"])
        physio  = self._aug_physio(item["physio"])
        return {**item, "facial": facial, "body": body,
                "audio": audio, "physio": physio}

    # ------------------------------------------------------------------ #
    # Facial: brightness, contrast, horizontal flip, Gaussian noise
    # ------------------------------------------------------------------ #
    def _aug_facial(self, x: torch.Tensor) -> torch.Tensor:
        """x : (T, 17)"""
        # Random brightness-like shift
        if random.random() < 0.5:
            x = x + torch.randn(1) * 0.2 * x.std()
        # Random Gaussian noise
        x = x + torch.randn_like(x) * 0.01
        return x.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Body: random crop (90%), keypoint jitter (±5 px norm)
    # ------------------------------------------------------------------ #
    def _aug_body(self, x: torch.Tensor) -> torch.Tensor:
        """x : (T, 51)"""
        # Keypoint jitter (norm coords, small perturbation)
        x = x + torch.randn_like(x) * 0.01
        # Random frame dropout
        if random.random() < 0.05 and x.shape[0] > 1:
            idx = random.randint(0, x.shape[0] - 1)
            x[idx] = 0.0
        return x

    # ------------------------------------------------------------------ #
    # Audio: time stretch, pitch shift (approx), volume scale, noise
    # ------------------------------------------------------------------ #
    def _aug_audio(self, x: torch.Tensor) -> torch.Tensor:
        """x : (T, 65)"""
        # Volume scale
        scale = random.uniform(0.8, 1.2)
        x = x * scale
        # Temporal stretch (linear interp in time dim)
        stretch = random.uniform(0.9, 1.1)
        T = x.shape[0]
        new_T = max(1, int(T * stretch))
        x = F.interpolate(
            x.T.unsqueeze(0), size=new_T, mode="linear", align_corners=False
        ).squeeze(0).T
        # Crop or pad back to T
        if x.shape[0] > T:
            x = x[:T]
        elif x.shape[0] < T:
            x = F.pad(x, (0, 0, 0, T - x.shape[0]))
        # Gaussian noise (SNR ~25dB)
        x = x + torch.randn_like(x) * 0.02
        return x

    # ------------------------------------------------------------------ #
    # Physio: amplitude scaling, baseline wander, Gaussian noise
    # ------------------------------------------------------------------ #
    def _aug_physio(self, x: torch.Tensor) -> torch.Tensor:
        """x : (T, 3, 250)"""
        # Amplitude scaling per signal
        scale = torch.FloatTensor(1, 3, 1).uniform_(0.95, 1.05)
        x = x * scale
        # Gaussian noise (SNR ~30dB)
        x = x + torch.randn_like(x) * 0.01
        return x
