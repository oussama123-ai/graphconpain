# GraphConPain: Graph Neural Network Framework for Neonatal Pain Assessment

[![Paper](https://img.shields.io/badge/Paper-Springer_MVA-blue)](https://link.springer.com/journal/138)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **EvoGraphConPain: An Adaptive Multimodal Graph Intelligence Framework for Real-Time Neonatal Pain Detection**

Official PyTorch implementation of **GraphConPain** — a novel deep learning framework for objective, continuous neonatal pain assessment combining graph neural networks, contrastive self-supervised learning, and multi-task optimization.

**Authors**: Oussama El Othmani¹², Riadh Ouersighni¹²
¹Computer Science Department, Military Academy of Fondouk Jedid, Tunisia
²Military Research Center, Tunisia

---

## Key Features

- **Graph-Based Multimodal Fusion**: Dynamic Graph Attention Networks (GAT) model inter-modality relationships between facial expressions, body movements, cry acoustics, and physiological signals
- **State-of-the-Art Performance**: 88.5% accuracy, 92% silent pain recall (+27–34% over clinical scales)
- **Silent Pain Detection**: Dedicated module for cry-absent episodes (critical clinical gap)
- **Self-Supervised Pretraining**: Contrastive learning on unlabeled data for data efficiency
- **Clinical Interpretability**: Attention visualization and SHAP-based feature importance
- **Real-Time Inference**: <200ms latency, NVIDIA Jetson compatible
- **Fairness Validated**: Demographic parity ratio 0.97 across skin tones, ages, sex

---

## Performance Highlights

| Metric | GraphConPain | Best Baseline | Improvement |
|--------|-------------|---------------|-------------|
| Overall Accuracy | **88.5%** | 84.1% | +4.4% |
| Silent Pain Recall | **92.0%** | 87.2% | +4.8% |
| F1-Score | **0.86** | 0.81 | +6.2% |
| Continuous MSE | **0.32** | 0.37 | −13.5% |
| vs. NIPS (Clinical) | **88.5%** | 72.3% | +16.2% |
| vs. PIPP-R (Clinical) | **88.5%** | 75.1% | +13.4% |

---

## Installation

```bash
git clone https://github.com/oussama123-ai/graphconpain.git
cd graphconpain
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## Quick Start

```bash
# 1. Preprocess data
python data/preprocessing/facial_au.py --input data/icope/videos --output data/icope/features/facial
python data/preprocessing/body_pose.py --input data/icope/videos --output data/icope/features/body
python data/preprocessing/audio_mfcc.py --input data/icope/audio  --output data/icope/features/audio
python data/preprocessing/physiological.py --input data/icope/physio --output data/icope/features/physio

# 2. Contrastive pretraining (optional but recommended, ~18h on RTX 4090)
python training/pretrain.py --config config/pretrain.yaml --data_dir data/ --output_dir checkpoints/

# 3. Supervised fine-tuning (~23h on RTX 4090)
python training/finetune.py --config config/finetune.yaml \
    --pretrained_weights checkpoints/pretrained_ssl.pth \
    --data_dir data/ --output_dir checkpoints/

# 4. Evaluate
python evaluation/cross_validation.py --config config/finetune.yaml --data_dir data/ --folds 5

# 5. Single episode inference
python scripts/inference.py \
    --checkpoint checkpoints/finetuned_full.pth \
    --video data/test/episode_001.mp4 \
    --audio data/test/episode_001.wav \
    --physio data/test/episode_001_ecg.csv \
    --output predictions/episode_001.json
```

---

## Repository Structure

```
graphconpain/
├── config/
│   ├── default.yaml
│   ├── pretrain.yaml
│   └── finetune.yaml
├── data/
│   └── preprocessing/
│       ├── facial_au.py
│       ├── body_pose.py
│       ├── audio_mfcc.py
│       └── physiological.py
├── models/
│   ├── graph_attention.py      # GAT implementation
│   ├── feature_extractors.py   # Multimodal encoders
│   ├── temporal_model.py       # Bidirectional GRU
│   ├── contrastive.py          # SSL pretraining
│   └── multitask_head.py       # Prediction heads
├── training/
│   ├── pretrain.py
│   ├── finetune.py
│   └── losses.py
├── evaluation/
│   ├── metrics.py
│   ├── fairness.py
│   ├── explainability.py
│   └── cross_validation.py
├── utils/
│   ├── augmentation.py
│   ├── data_loader.py
│   └── visualization.py
├── scripts/
│   ├── inference.py
│   └── download_datasets.sh
├── notebooks/
│   ├── demo.ipynb
│   ├── visualization.ipynb
│   └── fairness_analysis.ipynb
├── tests/
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_training.py
├── checkpoints/               # Pre-trained weights (download separately)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Datasets

| Dataset | Infants | Duration | Modalities | Access |
|---------|---------|----------|-----------|--------|
| iCOPE | 26 | 15h | RGB video, audio, thermal | Public |
| NPAD | 108 | 30h | Video, audio, ECG, respiration | Request |
| Body Movement Ext. | — | 50h | 17-keypoint annotations @5fps | On request |

---

## Citation

```bibtex
@article{elothmani2025graphconpain,
  title   = {EvoGraphConPain: An Adaptive Multimodal Graph Intelligence Framework
             for Real-Time Neonatal Pain Detection},
  author  = {El Othmani, Oussama and Ouersighni, Riadh},
  journal = {Machine Vision and Applications},
  year    = {2025},
  publisher = {Springer}
}
```

---

## Ethics

All datasets used under IRB approval. Body movement annotation extension approved by IRB of the Military Research Center, Tunisia. Anonymized data available upon reasonable request.

Contact: `oussama.elothmani@ept.u-carthage.tn`

---

## License

MIT License — see [LICENSE](LICENSE).
