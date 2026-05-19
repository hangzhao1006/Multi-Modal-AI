# 🧠 Multi-Modal AI — MAS.S60 / 6.S985

> **Hang Zhao** · Harvard 
> **Xiaoyang Wu** · Harvard 
> Spring 2026 · MIT / Harvard Multimodal AI Course

A living portfolio of weekly explorations, homework assignments, and the final research project in multimodal AI.

---

## 📂 Repository Structure

```
Multi-Modal-AI/
│
├── README.md                          ← You are here
├── LICENSE
├── pgmoe_plan.md                      ← Final project planning doc
│
├── assignments/
│   ├── hw1_multimodal_basics/Hang_hw1.ipynb
│   ├── hw2_fusion_methods/Hang_hw2.ipynb
│   ├── hw3_lora_qwen/Hang_hw3.ipynb
│   ├── hw4_grpo_vlm/Hang_hw4.ipynb
│   └── hw5_agent/Hang_hw5.ipynb
│
├── midterm/
│   ├── README.md
│   ├── code/
│   │   ├── 01_baseline_imu_skeleton.py
│   │   ├── 02_rgb_extraction.py
│   │   ├── 03_rgb_resnet3d.py
│   │   ├── 04_crossover_curve.py
│   │   ├── 05_changepoint_detection.py
│   │   ├── 06_temporal_alignment.py
│   │   ├── mmai_pilot.ipynb / mmai_pilot.py
│   │   └── 6.808 Lab 4 - Gesture Recognition via FMCW.ipynb
│   ├── figures/                       ← crossover curves, changepoints, etc.
│   └── report/MMAI_midter_report.pdf
│
└── final_project/
    ├── README.md                      ← Project overview & results
    ├── train.py                       ← PG-MoE training entrypoint
    ├── train_imu.py                   ← IMU expert training
    ├── evaluate.py                    ← Eval / metrics
    ├── save_phase_cell.py
    ├── speaker_script.md              ← 4-slide presentation script
    │
    ├── models/
    │   ├── __init__.py
    │   ├── vision_expert.py           ← ResNet3D backbone
    │   ├── imu_expert.py              ← Deep 1D-CNN
    │   ├── phase_arbitrator.py        ← Physics-driven α(t) gate
    │   ├── cross_attention.py         ← Cross-attn fusion baseline
    │   ├── pgmoe.py                   ← Full PG-MoE model
    │   └── train_loops.py
    │
    ├── notebooks/
    │   ├── hang/                      ← Vision side (Qwen + ResNet3D)
    │   │   ├── 01_qwen_token_extraction.py
    │   │   ├── 02_qwen_token_visualization.py
    │   │   ├── 03_qwen_vision_classification.py
    │   │   ├── 04_resnet3d_vision_expert.py
    │   │   ├── 05_resnet3d_visualization.py
    │   │   ├── 08_fusion_experiments.py
    │   │   ├── 09_presentation_figures.py
    │   │   └── vision_analysis_figures.ipynb
    │   ├── xiaoyang/                  ← IMU side + phase arbitrator
    │   │   ├── train_imu.ipynb
    │   │   ├── build_phase_arbitrator.ipynb
    │   │   ├── save_imu_results.ipynb
    │   │   └── xiaoyang_slides_figures.ipynb
    │   └── joint/                     ← Combined PG-MoE experiments
    │       ├── train_pgmoe.ipynb
    │       ├── train_pgmoe_resnet3d.ipynb
    │       └── pgmoe_analysis_figures.ipynb
    │
    ├── figures/                       ← Result plots
    │   ├── Grad-CAM_ResNet3D_attention.png
    │   ├── t-SNE_ResNet3D_feature_space.png
    │   ├── vision_token_activation_snapshots.png
    │   ├── temporal_feature_dynamics.png
    │   ├── phase_features_throw.png
    │   ├── alpha_untrained.png
    │   ├── imu_confusion_matrix.png
    │   ├── imu_training_curve.png
    │   ├── pre_class_modality_complementarity.png
    │   ├── imu_results.txt
    │   └── phase_results.txt
    │
    ├── reports/                       ← Final write-up
    ├── presentations/MMAI.pdf
    │
    └── process/                       ← Working history (not the final deliverable)
        ├── devlog.md
        ├── handoff_to_hang.md
        ├── code/                      ← Earlier copies of train/eval scripts
        ├── checkpoints/               ← Trained .pt weights
        │   ├── imu_classifier_best.pt
        │   └── imu_expert.pt
        └── devlog/                    ← Screenshots & devlog images
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
python final_project/notebooks/hang/01_qwen_token_extraction.py

# 2. Train vision expert (ResNet3D)
python final_project/notebooks/hang/04_resnet3d_vision_expert.py

# 3. Train IMU expert
python final_project/train_imu.py

# 4. Train full PG-MoE
python final_project/train.py

# 5. Evaluate
python final_project/evaluate.py
```

---

## 🙏 Acknowledgments

- Course: MAS.S60/6.S985 Multimodal AI (MIT/Harvard)
- Dataset: UTD-MHAD (Chen et al., 2015)
- Teammate: Xiaoyang Wu (IMU Expert + Phase Arbitrator)
