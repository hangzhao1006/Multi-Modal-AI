# PG-MoE Checkpoints

Trained model weights for the PG-MoE project.

## Files

| File | Size | Purpose |
|---|---|---|
| `imu_expert.pt` | ~6 MB | **Just the encoder** of the IMU expert. Loaded by `pgmoe.py` for joint training (pretrained init, then end-to-end fine-tuned). |
| `imu_classifier_best.pt` | ~6 MB | **Full IMU classifier** (encoder + LayerNorm + Linear head). Reproduces the 82.33% standalone IMU accuracy. |

## How to load

### Just the encoder, for PG-MoE joint training

```python
from models.imu_expert import IMUExpert
model = IMUExpert(d_model=256)
model.load_state_dict(torch.load("project/final/checkpoints/imu_expert.pt"))
```

### Full classifier, to reproduce the 82.33% number

```python
from models.imu_expert import IMUClassifier
model = IMUClassifier(num_classes=27, d_model=256)
model.load_state_dict(torch.load("project/final/checkpoints/imu_classifier_best.pt"))
```

## Training reproducibility

Both files were produced by `code/train_imu.ipynb` (or equivalently `code/train_imu.py`)
running on UTD-MHAD with subjects {1,3,5,7} as train, {2,4,6,8} as test.

- Optimizer: AdamW (lr=1e-3, wd=1e-4)
- Schedule: CosineAnnealingLR(T_max=80)
- Loss: CrossEntropy
- 80 epochs on T4 GPU
- Best test accuracy: **82.33%** (midterm baseline IMUNet: 67.90%)

See `../figures/imu_results.txt` for per-class breakdown.
