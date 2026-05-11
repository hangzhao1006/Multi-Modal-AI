# Phase-Aware Arbitrator (Xiaoyang)
# Extracts IMU physical phase features (acc magnitude, 2nd derivative, energy
# rate) and produces alpha(t) in [0,1] that gates Vision vs IMU.
#
# Phase Encoder:  Linear(F -> 64) -> ReLU -> Linear(64 -> 32) -> ReLU
# Arbitrator MLP: Linear(32 -> 16) -> ReLU -> Linear(16 -> 1) -> Sigmoid
