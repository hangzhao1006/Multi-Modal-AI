# PG-MoE: Phase-Gated Mixture-of-Experts for Multimodal HAR

> **Hang Zhao** (Harvard GSD) & **Xiaoyang Wu** (MIT)  
> MAS.S60 / 6.S985 — Multimodal AI — Spring 2026

---

## Overview

PG-MoE fuses RGB video and IMU sensor data for Human Activity Recognition, using **physics-driven gating** to dynamically decide how much to trust each modality at each timestep. Unlike black-box attention mechanisms, our Phase Arbitrator extracts biomechanical features (acceleration magnitude, jerk, energy rate) from raw IMU to produce interpretable gating weights.

## Architecture

```
┌──────────────────┐        ┌──────────────────┐
│  Vision Expert   │        │   IMU Expert      │
│  ResNet3D        │        │  Deep 1D-CNN      │
│  (Kinetics-400)  │        │  (4× stride-2)    │
│  89.3% standalone│        │  82.3% standalone  │
└────────┬─────────┘        └────────┬──────────┘
         │                           │
    (B, T_v, 256)              (B, 12, 256)
         │                           │
         └─────────┬─────────────────┘
                   │
       ┌───────────▼────────────┐
       │   Phase Arbitrator     │
       │                        │
       │   Raw IMU (B, 6, 192)  │
       │      ↓                 │
       │   |a|, d²|a|/dt²,     │
       │   d/dt(|a|²)          │
       │      ↓                 │
       │   PhaseEncoder (3→32)  │
       │      ↓                 │
       │   MLP → σ → α(t)      │
       │   Only 2,881 params    │
       └───────────┬────────────┘
                   │
     z(t) = α(t)·vision(t) + (1-α(t))·imu(t)
                   │
            ┌──────▼──────┐
            │  Classifier  │
            │  → 27 classes│
            └─────────────┘
```

## Results

| Method | Type | Accuracy |
|--------|------|----------|
| IMU-only (Deep 1D-CNN) | Single | 82.3% |
| Vision-only (Qwen2.5-VL, frozen) | Single | 58.6% |
| Vision-only (ResNet3D, fine-tuned) | Single | **89.3%** |
| Early Fusion (concat) | Fusion | 90.2% |
| **Late Fusion (avg logits)** | Fusion | **92.3%** |
| Cross-Attention (no gating) | Fusion | 89.8% |
| PG-MoE (fine-tuned) | Ours | 91.4% |
| Oracle upper bound | — | 98.4% |

### Per-Class Modality Complementarity

| Action | IMU | Vision | Stronger |
|--------|-----|--------|----------|
| Tennis serve | 50% | 94% | Vision +44% |
| Throw | 25% | 56% | Vision +31% |
| Boxing | 69% | 100% | Vision +31% |
| Wave | 100% | 75% | IMU +25% |
| Draw circle | 100% | 75% | IMU +25% |
| Clap | 100% | 100% | Both |
| Walk | 100% | 100% | Both |

**Oracle (if perfect gating): 98.4%** — strong complementarity exists.

## Key Findings

1. **Multimodal fusion works** — all methods beat single-modality best (89.3%)
2. **Simple wins on small data** — Late Fusion (0 params) > learned fusion on 431 samples
3. **VLM ≠ action model** — Qwen2.5-VL: 58.6% (semantics) vs ResNet3D: 89.3% (motion)
4. **Oracle 98.4%** — huge untapped complementarity, needs larger datasets
5. **Physics-driven gating** — interpretable α(t) driven by real biomechanical quantities

## Repository Structure

```
pgmoe/
├── README.md
│
├── models/                          # Modular model definitions
│   ├── imu_expert.py                #   Deep 1D-CNN (Xiaoyang)
│   ├── vision_expert.py             #   ResNet3D / Qwen ViT (Hang)
│   ├── phase_arbitrator.py          #   Physics-driven gating (Xiaoyang)
│   ├── cross_attention.py           #   ViLBERT Co-TRM style (Hang)
│   ├── pgmoe.py                     #   Full PG-MoE + PGMoELate
│   └── train_loops.py               #   Reusable training functions
│
├── notebooks/
│   ├── hang/                        # Hang's experiments
│   │   ├── 01_qwen_token_extraction.py      # Qwen ViT token extraction
│   │   ├── 02_qwen_token_visualization.py   # Token heatmaps, patch anatomy
│   │   ├── 03_qwen_vision_classification.py # Qwen classification (58.6%)
│   │   ├── 04_resnet3d_vision_expert.py     # ResNet3D training (89.3%)
│   │   ├── 05_resnet3d_visualization.py     # Grad-CAM, t-SNE, temporal
│   │   ├── 08_fusion_experiments.py         # Ablation: 9 fusion methods
│   │   ├── 09_presentation_figures.py       # Figure generation
│   │   └── vision_analysis_figures.ipynb    # All vision analysis
│   │
│   ├── xiaoyang/                    # Xiaoyang's experiments
│   │   ├── train_imu.ipynb                  # IMU Expert training (82.3%)
│   │   ├── build_phase_arbitrator.ipynb     # Phase Arbitrator design
│   │   ├── save_imu_results.ipynb           # Results archiving
│   │   └── xiaoyang_slides_figures.ipynb    # IMU visualization figures
│   │
│   └── joint/                       # Joint training experiments
│       ├── train_pgmoe.ipynb                # PG-MoE with Qwen vision
│       ├── train_pgmoe_resnet3d.ipynb       # PG-MoE with ResNet3D
│       └── pgmoe_analysis_figures.ipynb     # Analysis & comparison figs
│
├── train_imu.py                     # Standalone IMU training script
├── train.py                         # PG-MoE training entry point
├── evaluate.py                      # Evaluation & visualization
│
├── figures/                         # Generated figures for report
├── report/                          # NeurIPS-format final report
└── presentation/                    # 4-minute presentation slides
    └── PG-MoE_Presentation.pptx
```

## Team Contributions

| Component | Hang | Xiaoyang |
|-----------|------|----------|
| Vision Expert (Qwen ViT) | ✅ | |
| Vision Expert (ResNet3D) | ✅ | |
| Vision visualizations | ✅ | |
| IMU Expert | | ✅ |
| Phase Arbitrator | | ✅ |
| Cross-Modal Attention | ✅ | |
| PG-MoE assembly | ✅ | ✅ |
| Fusion experiments | ✅ | ✅ |
| IMU visualizations | | ✅ |
| Report & presentation | ✅ | ✅ |

## Journey: How Ideas Evolved

| Stage | Plan | Reality |
|-------|------|---------|
| Proposal | MMTSA-style segment gating | Too coarse for per-timestep dynamics |
| Midterm | Qwen ViT as vision backbone | VLMs encode semantics not motion (58.6%) |
| Final | Physics-driven per-token gating | α stuck near 0.5 on small data; Late Fusion wins on accuracy but PG-MoE provides interpretability |

**Honest takeaway:** On 431 training samples, simple averaging beats learned gating. But oracle analysis (98.4%) proves strong complementarity exists — the right gating mechanism trained on sufficient data should dramatically outperform averaging.

## Setup

```bash
pip install torch torchvision transformers scipy scikit-learn matplotlib opencv-python Pillow
```

### Data
Download [UTD-MHAD](http://www.utdallas.edu/~kehtar/UTD-MHAD.html) and place RGB + Inertial folders under `data/utd_mhad/`.

### Run
```bash
# 1. Train IMU Expert
python train_imu.py --data_root data/utd_mhad --save_dir checkpoints/

# 2. Train Vision Expert (in Colab with GPU)
python notebooks/hang/04_resnet3d_vision_expert.py

# 3. Train PG-MoE (use notebooks/joint/train_pgmoe_resnet3d.ipynb in Colab)
```

## Dataset

**UTD-MHAD** — 861 samples, 27 action classes, 8 subjects  
We use **RGB + IMU only** (smartphone sensors), not Skeleton/Depth (Kinect hardware).  
Train: subjects {1,3,5,7} | Test: subjects {2,4,6,8}

## Acknowledgments

- Course: MAS.S60/6.S985 Multimodal AI (MIT/Harvard)
- Dataset: UTD-MHAD (Chen et al., 2015)
- ResNet3D: Kinetics-400 pretrained (torchvision)
- Qwen2.5-VL: Alibaba
