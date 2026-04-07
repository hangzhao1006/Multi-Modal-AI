# Midterm Code

## Overview
Phase-wise temporal heterogeneity analysis for multimodal HAR on UTD-MHAD.

## Files

| File | Description | Key Result |
|------|-------------|------------|
| `01_baseline_imu_skeleton.py` | IMU + Skeleton 1D-CNN baselines | IMU: 67.9%, Skeleton: 37.9% |
| `02_rgb_extraction.py` | IMU-aligned RGB frame extraction | 861 videos, 30 frames each |
| `03_rgb_resnet3d.py` | Pretrained ResNet3D fine-tuning | RGB: 72.8% |
| `04_crossover_curve.py` | Phase-wise RGB vs IMU analysis | Crossover at Accel phase (63%) |
| `05_changepoint_detection.py` | IMU + RGB boundary detection | IMU/RGB boundaries align within 3-10% |
| `06_temporal_alignment.py` | RGB/Skeleton/IMU duration analysis | Skeleton 1.0-1.7s shorter than RGB/IMU |

## Key Findings

1. **Temporal Alignment**: RGB and IMU are natively synchronized (<0.5s).
   Skeleton sequences are systematically shorter by 1.0-1.7s (Kinect auto-segments).

2. **Crossover Curve**: RGB (ResNet3D) dominates Prep/Pre-Accel phases (80% vs 54%).
   Both modalities converge at Accel phase (40-60%): both reach 63%.
   IMU drops sharply post-crossover (0.32, 0.29).

3. **Model Capacity Matters**: Simple 3D-CNN gets 18.6%. ResNet3D gets 72.8%.
   IMU Transformer (52.1%) underperforms 1D-CNN (67.9%) due to small dataset size.

## How to Run (in Google Colab)

```python
# 1. Mount Drive and set path
DATA_ROOT = "/content/drive/MyDrive/utd_mhad"

# 2. Run baseline models
exec(open('01_baseline_imu_skeleton.py').read())

# 3. Extract all RGB frames (run once, takes ~20 min)
exec(open('02_rgb_extraction.py').read())
rgb_data = extract_all_videos()

# 4. Train RGB ResNet3D
exec(open('03_rgb_resnet3d.py').read())
model, acc = train_rgb_resnet3d(rgb_data)

# 5. Generate crossover curve
exec(open('04_crossover_curve.py').read())

# 6. Changepoint detection
exec(open('05_changepoint_detection.py').read())
plot_changepoints(rgb_data=rgb_data)

# 7. Temporal alignment analysis
exec(open('06_temporal_alignment.py').read())
analyze_temporal_alignment()
```

## Requirements
```
torch torchvision scipy scikit-learn matplotlib opencv-python numpy wandb
```