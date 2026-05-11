# PG-MoE — Phase-Gated Mixture-of-Experts for HAR

Final report code for *Phase-Gated MoE for Human Activity Recognition*.
Team: Xiaoyang Wu + Hang Zhao.

## Structure

```
final/
├── code/
│   ├── models/
│   │   ├── vision_expert.py      # Hang   — Qwen2-VL-2B vision encoder
│   │   ├── imu_expert.py         # Xiaoyang — Deep 1D-CNN (keeps time axis)
│   │   ├── cross_attention.py    # Hang   — ViLBERT Co-TRM bidirectional
│   │   ├── phase_arbitrator.py   # Xiaoyang — IMU phase -> alpha(t)
│   │   └── pgmoe.py              # together — full model + gated fusion
│   ├── data/
│   │   ├── extract_vision_tokens.py  # Hang  — cache Qwen tokens
│   │   └── dataset.py                # together
│   ├── train.py                  # together — Focal Loss training
│   └── evaluate.py               # together — alpha(t) viz + crossover curve
├── figures/
├── report/
└── README.md
```

See `../../pgmoe_plan.md` for the full plan.
