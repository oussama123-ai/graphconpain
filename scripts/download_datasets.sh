#!/bin/bash
# scripts/download_datasets.sh
# Downloads iCOPE and NPAD datasets (requires authentication for NPAD).

set -e

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "=== GraphConPain Dataset Download ==="
echo ""
echo "iCOPE (Infant Classification of Pain Expressions)"
echo "  → Public dataset: https://sites.google.com/view/cope-dataset"
echo "  → Please download manually and place under: $DATA_DIR/icope/"
echo ""
echo "NPAD (Neonatal Pain, Agitation and Sedation)"
echo "  → Requires consortium authentication"
echo "  → Request access: https://www.physionet.org/content/npad/"
echo "  → After approval, run:"
echo "      wget -r -N -c -np --user YOUR_USERNAME --ask-password \\"
echo "        https://physionet.org/files/npad/1.0.0/ -P $DATA_DIR/npad/"
echo ""
echo "After downloading, run preprocessing:"
echo "  python data/preprocessing/facial_au.py  --input $DATA_DIR/icope/videos --output $DATA_DIR/icope/features/facial"
echo "  python data/preprocessing/body_pose.py  --input $DATA_DIR/icope/videos --output $DATA_DIR/icope/features/body"
echo "  python data/preprocessing/audio_mfcc.py --input $DATA_DIR/icope/audio  --output $DATA_DIR/icope/features/audio"
echo "  python data/preprocessing/physiological.py --input $DATA_DIR/icope/physio --output $DATA_DIR/icope/features/physio"
