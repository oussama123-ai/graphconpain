"""
utils/data_loader.py
---------------------
PyTorch Dataset classes for GraphConPain:
  NeonatalPainDataset   : labeled iCOPE + NPAD data
  UnlabeledMultimodalDataset : unlabeled recordings for SSL pretraining
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Labeled dataset
# ---------------------------------------------------------------------------

class NeonatalPainDataset(Dataset):
    """
    Labeled neonatal pain dataset.

    Expected layout
    ---------------
    root/
      metadata.json      ← list of episode records
      features/
        facial/  *.npy   ← (T, 17) AU intensities
        body/    *.npy   ← (T, 51) keypoints
        audio/   *.npy   ← (T, 65) MFCC features
        physio/  *.npy   ← (T, 3, 250) ECG/EDA/Resp windows

    metadata.json schema (per episode)
    ------------------------------------
    {
        "id":         "icope_001",
        "split":      "train",            // "train" | "val" | "test"
        "y_cont":     6.5,                // continuous pain score [0,10]
        "y_class":    2,                  // 0=none,1=mild,2=moderate,3=severe
        "y_silent":   0,                  // 1 if cry-absent pain episode
        "infant_id":  "I001",
        "ga_weeks":   35,
        "sex":        "M",
        "skin_tone":  3,                  // Fitzpatrick 1-6
        "procedure":  "heel_stick"
    }
    """

    def __init__(
        self,
        root:  str,
        split: Optional[str] = None,   # None = all splits
        transform=None,
    ):
        self.root      = Path(root)
        self.transform = transform

        meta_path = self.root / "metadata.json"
        with open(meta_path) as f:
            records = json.load(f)

        self.records = [r for r in records
                        if split is None or r.get("split") == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        rid = rec["id"]

        def _load(subdir, suffix=".npy"):
            path = self.root / "features" / subdir / f"{rid}{suffix}"
            return np.load(str(path)).astype(np.float32)

        facial  = torch.from_numpy(_load("facial"))   # (T, 17)
        body    = torch.from_numpy(_load("body"))     # (T, 51)
        audio   = torch.from_numpy(_load("audio"))    # (T, 65)
        physio  = torch.from_numpy(_load("physio"))   # (T, 3, 250)

        item = {
            "facial":   facial,
            "body":     body,
            "audio":    audio,
            "physio":   physio,
            "y_cont":   torch.tensor(rec["y_cont"],   dtype=torch.float32),
            "y_class":  torch.tensor(rec["y_class"],  dtype=torch.long),
            "y_silent": torch.tensor(rec["y_silent"], dtype=torch.float32),
            "id":       rid,
            "infant_id":rec.get("infant_id", ""),
            "ga_weeks": rec.get("ga_weeks", -1),
            "sex":      rec.get("sex", ""),
            "skin_tone":rec.get("skin_tone", -1),
        }

        if self.transform:
            item = self.transform(item)

        return item


# ---------------------------------------------------------------------------
# Unlabeled dataset for SSL pretraining
# ---------------------------------------------------------------------------

class UnlabeledMultimodalDataset(Dataset):
    """
    Unlabeled multimodal recordings for contrastive pretraining.
    Returns stacked node feature tensor (N=4 modality nodes).
    """

    def __init__(self, root: str):
        self.root = Path(root)
        meta_path = self.root / "unlabeled_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.records = json.load(f)
        else:
            # Fallback: use all training records without labels
            all_meta = self.root / "metadata.json"
            if all_meta.exists():
                with open(all_meta) as f:
                    self.records = json.load(f)
            else:
                self.records = []

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        rid = rec["id"]

        def _load(subdir):
            p = self.root / "features" / subdir / f"{rid}.npy"
            return np.load(str(p)).astype(np.float32) if p.exists() else np.zeros((1, 1), dtype=np.float32)

        facial  = torch.from_numpy(_load("facial"))   # (T, 17)
        body    = torch.from_numpy(_load("body"))     # (T, 51)
        audio   = torch.from_numpy(_load("audio"))    # (T, 65)
        physio  = torch.from_numpy(_load("physio"))   # (T, 3, 250)

        # Flatten each modality to its mean over time → node features
        # (N=4, D_max) — padded to common dim for simple batching
        f_mean = facial.mean(0)   # (17,)
        b_mean = body.mean(0)     # (51,)
        a_mean = audio.mean(0)    # (65,)
        p_mean = physio.mean(0).flatten()  # (750,) → trimmed below

        # Use first 64 dims of each for prototype node features
        def _pad(v, d=64):
            if v.shape[0] >= d:
                return v[:d]
            return torch.cat([v, torch.zeros(d - v.shape[0])])

        H = torch.stack([_pad(f_mean), _pad(b_mean),
                         _pad(a_mean), _pad(p_mean[:64])])  # (4, 64)

        return {"node_features": H, "id": rid}
