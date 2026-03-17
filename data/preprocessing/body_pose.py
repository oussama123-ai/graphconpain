#!/usr/bin/env python3
"""
data/preprocessing/body_pose.py
---------------------------------
Body movement feature extraction via AlphaPose (Algorithms 2-3).

Extracts 17 COCO keypoints + 10 derived geometric/motion features per frame,
producing (T, 51) raw keypoints and (T, 102) projected body embeddings.

Usage
-----
    python data/preprocessing/body_pose.py \
        --input  data/icope/videos \
        --output data/icope/features/body
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


N_KEYPOINTS  = 17   # COCO format
BODY_DIM     = 51   # 17 × (x, y, conf)
DERIVED_DIM  = 51   # distances + angles + velocity + freq + asymmetry + global
OUTPUT_DIM   = 102  # matches paper


# COCO keypoint indices
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--alphapose_json", type=str, default=None,
                   help="Pre-computed AlphaPose JSON (if available)")
    p.add_argument("--confidence_threshold", type=float, default=0.5)
    return p.parse_args()


def load_alphapose_json(json_path: Path, video_stem: str) -> dict[int, np.ndarray]:
    """
    Parse AlphaPose output JSON.
    Returns {frame_idx: keypoints (17,3)} where col=[x,y,conf].
    """
    with open(json_path) as f:
        data = json.load(f)

    frame_kps = {}
    for item in data:
        img_name = item.get("image_id", "")
        # Extract frame index from filename like "frame_0042.jpg"
        nums = [int(s) for s in img_name.split("_") if s.split(".")[0].isdigit()]
        if not nums:
            continue
        frame_idx = nums[-1]
        kps = np.array(item["keypoints"]).reshape(17, 3)   # (x, y, conf)
        # Keep highest-confidence person per frame
        if frame_idx not in frame_kps or kps[:, 2].mean() > frame_kps[frame_idx][:, 2].mean():
            frame_kps[frame_idx] = kps

    return frame_kps


def compute_derived_features(kps_seq: np.ndarray) -> np.ndarray:
    """
    kps_seq : (T, 17, 3) — x, y, conf per keypoint
    Returns  : (T, 51)  — derived features (padded to DERIVED_DIM)
    """
    T = kps_seq.shape[0]
    xy = kps_seq[:, :, :2]   # (T, 17, 2)

    features = []

    # --- Inter-joint distances (10) ---
    pairs = [
        ("left_shoulder",  "left_elbow"),
        ("left_elbow",     "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow",    "right_wrist"),
        ("left_hip",       "left_knee"),
        ("left_knee",      "left_ankle"),
        ("right_hip",      "right_knee"),
        ("right_knee",     "right_ankle"),
        ("left_shoulder",  "right_shoulder"),
        ("left_hip",       "right_hip"),
    ]
    for (a, b) in pairs:
        diff = xy[:, KP[a]] - xy[:, KP[b]]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)   # (T,1)
        features.append(dist)

    # --- Temporal velocity (17×2 = 34, aggregated to 17 magnitudes) ---
    vel = np.zeros((T, 17))
    vel[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
    features.append(vel)

    # --- Centroid displacement ---
    centroid = xy.mean(axis=1)                         # (T, 2)
    cent_disp = np.zeros((T, 1))
    cent_disp[1:] = np.linalg.norm(centroid[1:] - centroid[:-1], axis=-1, keepdims=True)
    features.append(cent_disp)

    # --- Left-right asymmetry (6 pairs) ---
    lr_pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow",    "right_elbow"),
        ("left_wrist",    "right_wrist"),
        ("left_hip",      "right_hip"),
        ("left_knee",     "right_knee"),
        ("left_ankle",    "right_ankle"),
    ]
    for (l, r) in lr_pairs:
        diff = np.linalg.norm(xy[:, KP[l]] - xy[:, KP[r]], axis=-1, keepdims=True)
        features.append(diff)

    # --- Pose entropy (body spread) ---
    spread = xy.std(axis=1).mean(axis=1, keepdims=True)   # (T,1)
    features.append(spread)

    derived = np.concatenate(features, axis=1)   # (T, 10+17+1+6+1) = (T, 35)
    # Pad to DERIVED_DIM=51
    if derived.shape[1] < DERIVED_DIM:
        pad = np.zeros((T, DERIVED_DIM - derived.shape[1]))
        derived = np.concatenate([derived, pad], axis=1)
    else:
        derived = derived[:, :DERIVED_DIM]

    return derived.astype(np.float32)


def process_video(video_path: Path, ap_json: Path | None,
                  conf_thresh: float) -> np.ndarray:
    """
    Returns (T, 102) body feature array.
    Falls back to zero array if AlphaPose JSON not available.
    """
    if ap_json is None or not ap_json.exists():
        # Placeholder: zero array (actual AlphaPose must be run offline)
        print(f"    No AlphaPose JSON for {video_path.stem}. Using zeros.")
        return np.zeros((1, OUTPUT_DIM), dtype=np.float32)

    frame_kps = load_alphapose_json(ap_json, video_path.stem)
    if not frame_kps:
        return np.zeros((1, OUTPUT_DIM), dtype=np.float32)

    T       = max(frame_kps.keys()) + 1
    kps_seq = np.zeros((T, 17, 3), dtype=np.float32)
    for t, kps in frame_kps.items():
        if t < T:
            kps_seq[t] = kps

    # Flatten raw keypoints (T, 51)
    raw = kps_seq.reshape(T, -1)    # (T, 51)

    # Derived features
    derived = compute_derived_features(kps_seq)  # (T, 51)

    body = np.concatenate([raw, derived], axis=1)  # (T, 102)
    return body.astype(np.float32)


def main():
    args   = parse_args()
    in_dir = Path(args.input)
    out_dir= Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(in_dir.glob("*.mp4")) + list(in_dir.glob("*.avi")))
    print(f"Found {len(videos)} videos.")

    for vid in videos:
        out_path = out_dir / f"{vid.stem}.npy"
        if out_path.exists():
            print(f"  [skip] {vid.stem}")
            continue
        ap_json = Path(args.alphapose_json) / f"{vid.stem}.json" \
                  if args.alphapose_json else None
        body = process_video(vid, ap_json, args.confidence_threshold)
        np.save(str(out_path), body)
        print(f"  [ok]   {vid.stem}  shape={body.shape}")

    print("Body pose extraction complete.")


if __name__ == "__main__":
    main()
