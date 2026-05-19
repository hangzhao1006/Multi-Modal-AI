# PG-MoE: Phase-Gated Mixture-of-Experts for Multimodal Human Activity Recognition

**MAS.S60 / 6.S985 — Multimodal AI — Harvard GSD / MIT**

Hang Zhao & Xiaoyang Wu

## Overview

PG-MoE is a multimodal fusion framework for Human Activity Recognition (HAR) that uses **physics-driven gating** to dynamically decide how much to trust each modality (RGB video vs IMU sensor) at each timestep during an action.

Unlike prior approaches that use fixed fusion weights or black-box attention, our Phase Arbitrator extracts biomechanical features (acceleration magnitude, jerk, energy rate) from raw IMU signals to produce interpretable per-token gating weights.

## Results

| Method | Type | Accuracy |
|--------|------|----------|
| IMU-only (Deep 1D-CNN) | Single | 82.3% |
| Vision-only (Qwen2.5-VL, frozen) | Single | 58.6% |
| Vision-only (ResNet3D, fine-tuned) | Single | 89.3% |
| Early Fusion (concat) | Fusion | 90.2% |
| Late Fusion (avg logits) | Fusion | 92.3% |
| Cross-Attn (no gating) | Fusion | 89.8% |
| PG-MoE (fine-tuned) | Ours | 91.4% |
| Oracle (upper bound) | — | 98.4% |

**Key Findings:**
- All fusion methods outperform single-modality best (89.3%)
- Late Fusion wins on small datasets (431 train samples) due to zero learnable parameters
- Oracle 98.4% confirms strong modality complementarity — room for improvement with larger datasets
- VLM (Qwen ViT) encodes semantics, not motion — task-specific pretraining matters for HAR

## Architecture

```
Vision Expert (ResNet3D, Kinetics pretrained) → (B, T_v, 256)
IMU Expert (Deep 1D-CNN with ResBlocks)       → (B, 12, 256)
                    ↓
        Phase Arbitrator (2,881 params)
        Raw IMU → |a|, d²|a|/dt², d/dt(|a|²)
        → PhaseEncoder → MLP → σ → α(t) ∈ [0,1]
                    ↓
        Per-token Gated Fusion
        z(t) = α(t)·vision(t) + (1-α(t))·imu(t)
                    ↓
        Classifier → 27 action classes
```

## Dataset

**UTD-MHAD** (University of Texas at Dallas - Multimodal Human Action Dataset)
- 861 samples, 27 action classes, 8 subjects
- We use **RGB + IMU only** (practical smartphone sensors)
- Train: subjects {1,3,5,7} | Test: subjects {2,4,6,8}
- Prior SOTA uses Skeleton+Depth (98%+), but requires Kinect hardware

## Code Structure

```
├── 01_qwen_token_extraction.py     # Step 1-3: Qwen2.5-VL vision token extraction
├── 02_qwen_token_visualization.py  # Step 4: Token heatmaps, patch analysis
├── 03_qwen_vision_classification.py # Step 5-6: Qwen vision-only classification
├── 04_resnet3d_vision_expert.py    # Step 6d: ResNet3D training (89.3%)
├── 05_resnet3d_visualization.py    # Grad-CAM, temporal dynamics, t-SNE
├── 06_imu_expert_verification.py   # IMU model loading and verification
├── 07_pgmoe_full.py                # Complete PG-MoE pipeline + ablations
├── 08_fusion_experiments.py        # Additional fusion method experiments
├── 09_presentation_figures.py      # Generate presentation/report figures
└── README.md
```

## Running

All scripts are designed for **Google Colab with GPU** (T4 or A100).

```bash
# Step 1: Extract Qwen vision tokens (requires A100, ~42GB VRAM)
python 01_qwen_token_extraction.py

# Step 2: Train ResNet3D Vision Expert
python 04_resnet3d_vision_expert.py

# Step 3: Run complete PG-MoE pipeline
python 07_pgmoe_full.py

# Step 4: Generate visualizations
python 05_resnet3d_visualization.py
python 09_presentation_figures.py
```

## Per-Class Modality Complementarity

| Action | IMU | Vision | Stronger |
|--------|-----|--------|----------|
| Tennis serve | 50% | 94% | Vision +44% |
| Throw | 25% | 56% | Vision +31% |
| Boxing | 69% | 100% | Vision +31% |
| Wave | 100% | 75% | IMU +25% |
| Draw circle | 100% | 75% | IMU +25% |
| Clap | 100% | 100% | Both |
| Walk | 100% | 100% | Both |

Oracle fusion (either modality correct): **98.4%**

## Requirements

```
torch >= 2.0
torchvision
transformers (for Qwen2.5-VL)
scipy
scikit-learn
matplotlib
numpy
opencv-python
Pillow
```

## Acknowledgments

- Course: MAS.S60/6.S985 Multimodal AI (MIT/Harvard)
- Dataset: UTD-MHAD (Chen et al., 2015)
- Vision backbone: ResNet3D pretrained on Kinetics-400
- VLM: Qwen2.5-VL-3B (Alibaba)
