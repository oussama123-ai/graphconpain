#!/usr/bin/env python3
"""
scripts/inference.py
---------------------
Single-episode inference with JSON output (Section 3.1 + Figure 16).

Usage
-----
    python scripts/inference.py \
        --checkpoint checkpoints/finetuned_full.pth \
        --video  data/test/episode_001.mp4 \
        --audio  data/test/episode_001.wav \
        --physio data/test/episode_001_ecg.csv \
        --output predictions/episode_001.json
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import GraphConPain
from data.preprocessing.audio_mfcc   import extract_features as extract_audio
from data.preprocessing.physiological import process_file as process_physio

PAIN_LEVELS = {
    0: "None (0-2)",
    1: "Mild (3-4)",
    2: "Moderate (5-7)",
    3: "Severe (8-10)",
}
MODALITIES = ["facial", "body", "audio", "physiological"]


def parse_args():
    p = argparse.ArgumentParser(description="GraphConPain Single-Episode Inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--video",      type=str, default=None)
    p.add_argument("--audio",      type=str, default=None)
    p.add_argument("--physio",     type=str, default=None)
    p.add_argument("--facial_npy", type=str, default=None,
                   help="Pre-extracted AU .npy (skip OpenFace)")
    p.add_argument("--body_npy",   type=str, default=None,
                   help="Pre-extracted body .npy (skip AlphaPose)")
    p.add_argument("--output",     type=str, default="prediction.json")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fps",        type=int, default=30)
    return p.parse_args()


def load_feature(path: str | None, shape: tuple, dtype=np.float32) -> np.ndarray:
    """Load .npy or return zeros."""
    if path and Path(path).exists():
        return np.load(path).astype(dtype)
    return np.zeros(shape, dtype=dtype)


def prepare_inputs(args, device) -> dict[str, torch.Tensor]:
    """Load / extract all modality features and return batched tensors."""

    # --- Facial ---
    if args.facial_npy:
        facial = load_feature(args.facial_npy, (90, 17))
    elif args.video:
        # In production: call OpenFace here
        print("  [warn] OpenFace not called; using zeros for facial AU.")
        facial = np.zeros((90, 17), dtype=np.float32)
    else:
        facial = np.zeros((90, 17), dtype=np.float32)

    # --- Body ---
    if args.body_npy:
        body = load_feature(args.body_npy, (90, 102))
    else:
        body = np.zeros((90, 102), dtype=np.float32)

    # --- Audio ---
    if args.audio and Path(args.audio).exists():
        audio = extract_audio(Path(args.audio), sr=16000, video_fps=args.fps)
    else:
        audio = np.zeros((90, 65), dtype=np.float32)

    # --- Physio ---
    if args.physio and Path(args.physio).exists():
        physio = process_physio(Path(args.physio), physio_sr=250, video_fps=args.fps)
    else:
        physio = np.zeros((90, 3, 250), dtype=np.float32)

    # Align temporal dimension
    T = min(facial.shape[0], body.shape[0], audio.shape[0], physio.shape[0])
    T = max(T, 1)
    facial  = facial[:T];  body   = body[:T]
    audio   = audio[:T];   physio = physio[:T]

    return {
        "facial":  torch.from_numpy(facial).unsqueeze(0).to(device),   # (1,T,17)
        "body":    torch.from_numpy(body).unsqueeze(0).to(device),     # (1,T,102)
        "audio":   torch.from_numpy(audio).unsqueeze(0).to(device),    # (1,T,65)
        "physio":  torch.from_numpy(physio).unsqueeze(0).to(device),   # (1,T,3,250)
    }


@torch.no_grad()
def run_inference(model: GraphConPain, inputs: dict) -> dict:
    model.eval()
    t0    = time.perf_counter()
    preds = model(**inputs)
    latency_ms = (time.perf_counter() - t0) * 1000

    cont_score    = float(preds["continuous"].item())
    class_probs   = preds["class_logits"].exp().squeeze().cpu().numpy()
    pain_level    = int(class_probs.argmax())
    silent_prob   = float(preds["silent_logit"].sigmoid().item())
    confidence    = float(class_probs.max())

    # Per-modality attention weights (from last GAT layer, averaged)
    attn_list = preds["attentions"]
    if attn_list:
        last_attn = attn_list[-1].cpu().numpy()   # (B*T, H, 4, 4)
        mod_importance = last_attn.mean(axis=(0, 1)).mean(axis=0)  # (4,)
        mod_importance /= mod_importance.sum() + 1e-8
    else:
        mod_importance = np.ones(4) / 4

    return {
        "continuous_score":          round(cont_score, 2),
        "pain_level":                PAIN_LEVELS[pain_level],
        "pain_level_index":          pain_level,
        "class_probabilities":       {
            PAIN_LEVELS[i]: round(float(p), 4)
            for i, p in enumerate(class_probs)
        },
        "silent_pain_probability":   round(silent_prob, 4),
        "confidence":                round(confidence, 4),
        "attention_weights": {
            mod: round(float(w), 4)
            for mod, w in zip(MODALITIES, mod_importance)
        },
        "latency_ms":                round(latency_ms, 1),
    }


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # Load model
    model = GraphConPain()
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model = model.to(device).eval()
    print(f"Model loaded from {args.checkpoint}")

    # Prepare inputs
    inputs  = prepare_inputs(args, device)
    result  = run_inference(model, inputs)

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n=== GraphConPain Prediction ===")
    print(f"  Pain score (0-10): {result['continuous_score']}")
    print(f"  Pain level:        {result['pain_level']}")
    print(f"  Silent pain prob:  {result['silent_pain_probability']}")
    print(f"  Confidence:        {result['confidence']}")
    print(f"  Latency:           {result['latency_ms']} ms")
    print(f"\n  Attention weights:")
    for mod, w in result["attention_weights"].items():
        print(f"    {mod:<15}: {w:.4f}")
    print(f"\nOutput saved → {out_path}")


if __name__ == "__main__":
    main()
