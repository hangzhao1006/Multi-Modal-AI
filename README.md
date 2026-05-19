# 🧠 Multi-Modal AI — MAS.S60 / 6.S985

> **Hang Zhao** · Harvard GSD  
> Spring 2026 · MIT / Harvard Multimodal AI Course

A living portfolio of weekly explorations, homework assignments, and the final research project in multimodal AI.

---

## 📂 Repository Structure

```
Multi-Modal-AI/
│
├── README.md                          ← You are here
│
├── assignments/
│   ├── hw1_multimodal_basics/
│   │   ├── README.md
│   │   └── ...
│   ├── hw2_fusion_methods/
│   │   ├── README.md
│   │   └── ...
│   ├── hw3_lora_qwen/
│   │   ├── README.md
│   │   └── ...
│   ├── hw4_grpo_vlm/
│   │   ├── README.md
│   │   └── ...
│   └── hw5_.../
│       ├── README.md
│       └── ...
│
├── reading_assignments/
│   └── ...
│
└── final_project/
    ├── README.md                      ← Project overview & results
    ├── code/
    │   ├── 01_qwen_token_extraction.py
    │   ├── 02_qwen_token_visualization.py
    │   ├── 03_qwen_vision_classification.py
    │   ├── 04_resnet3d_vision_expert.py
    │   ├── 05_resnet3d_visualization.py
    │   ├── 06_imu_expert_verification.py
    │   ├── 07_pgmoe_full.py
    │   ├── 08_fusion_experiments.py
    │   └── 09_presentation_figures.py
    ├── figures/
    │   ├── architecture_diagram.png
    │   ├── gradcam.png
    │   ├── tsne.png
    │   ├── per_class_table.png
    │   ├── token_activation.png
    │   ├── temporal_dynamics.png
    │   └── patch_anatomy.png
    ├── report/
    │   └── PG_MoE_Final_Report.pdf
    └── presentation/
        └── PG_MoE_Presentation.pptx
```

---

## 🎯 Final Project: PG-MoE

**Phase-Gated Mixture-of-Experts for Multimodal Human Activity Recognition**

*with Xiaoyang Wu*

### The Idea

Different modalities (video, IMU sensor) have different strengths at different phases of an action. During the preparation phase of a throw, vision clearly captures posture. During the acceleration phase, IMU captures the acceleration peak while vision gets blurry. We use **physics-driven gating** to dynamically route between modalities.

### Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Vision Expert  │     │    IMU Expert     │
│  ResNet3D       │     │  Deep 1D-CNN     │
│  89.3% alone    │     │  82.3% alone     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
    (B, T_v, 256)          (B, 12, 256)
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Phase Arbitrator    │
         │   Raw IMU → Physics   │
         │   |a|, d²|a|/dt²,    │
         │   d/dt(|a|²)         │
         │   → α(t) ∈ [0,1]     │
         │   Only 2,881 params!  │
         └───────────┬───────────┘
                     │
         z(t) = α(t)·v(t) + (1-α(t))·i(t)
                     │
              ┌──────▼──────┐
              │  Classifier  │
              │  → 27 classes│
              └──────────────┘
```

### Results at a Glance

| Method | Accuracy |
|--------|----------|
| IMU-only | 82.3% |
| Vision-only (Qwen ViT, frozen) | 58.6% |
| Vision-only (ResNet3D, fine-tuned) | **89.3%** |
| Early Fusion | 90.2% |
| **Late Fusion** | **92.3%** |
| Cross-Attention (no gating) | 89.8% |
| PG-MoE (fine-tuned) | 91.4% |
| Oracle upper bound | 98.4% |

### Key Findings

1. **Multimodal fusion works**: All fusion methods beat the best single modality (89.3%)
2. **Simple wins on small data**: Late Fusion (0 learnable params) beats learned fusion on 431 training samples
3. **VLMs ≠ action models**: Qwen2.5-VL encodes semantics (58.6%), ResNet3D encodes motion (89.3%)
4. **Strong complementarity**: Oracle 98.4% shows huge potential — Vision aces tennis serve (+44% over IMU), IMU aces wave (+25% over Vision)
5. **Physics-driven gating is interpretable**: Unlike black-box attention, α(t) is driven by acceleration, jerk, and energy rate

### Journey & Evolution of Ideas

This project evolved significantly from proposal to final report:

| Stage | What we planned | What actually happened |
|-------|----------------|----------------------|
| Proposal | MMTSA-style segment gating | Realized segment-level is too coarse |
| Midterm | Qwen ViT as vision backbone | Discovered VLMs don't encode motion (58.6%) |
| Final | Added ResNet3D, pivoted to physics-driven gating | Late Fusion won on accuracy, but PG-MoE provides interpretability |

**The honest takeaway**: On a small dataset like UTD-MHAD (431 train samples), simple averaging beats learned gating. But the oracle analysis (98.4%) proves the modalities are highly complementary — the right gating mechanism, trained on sufficient data, should dramatically outperform averaging. Physics-driven gating is the right direction; it just needs more data to shine.

---

## 📚 Assignments

| HW | Topic | Key Takeaway |
|----|-------|-------------|
| HW1 | Multimodal Basics | Fundamentals of heterogeneity and alignment |
| HW2 | Fusion Methods | Early vs late vs hybrid fusion tradeoffs |
| HW3 | LoRA Fine-tuning Qwen | Efficient VLM adaptation, used in final project |
| HW4 | GRPO for VLMs | Reinforcement learning for vision-language models |
| HW5 | ... | ... |

---

## 🛠️ Setup & Reproduction

### Requirements
```bash
pip install torch torchvision transformers scipy scikit-learn matplotlib numpy opencv-python Pillow
```

### Data
- Download [UTD-MHAD](http://www.utdallas.edu/~kehtar/UTD-MHAD.html)
- Place RGB videos and Inertial .mat files in `data/utd_mhad/`

### Run
```bash
# 1. Extract vision tokens (requires A100 GPU)
python final_project/code/01_qwen_token_extraction.py

# 2. Train vision expert
python final_project/code/04_resnet3d_vision_expert.py

# 3. Run PG-MoE pipeline
python final_project/code/07_pgmoe_full.py
```

---

## 🙏 Acknowledgments

- Course: MAS.S60/6.S985 Multimodal AI (MIT/Harvard)
- Dataset: UTD-MHAD (Chen et al., 2015)
- Teammate: Xiaoyang Wu (IMU Expert + Phase Arbitrator)
