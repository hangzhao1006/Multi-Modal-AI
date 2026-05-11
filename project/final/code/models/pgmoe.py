# PG-MoE: Phase-Gated Mixture-of-Experts (Xiaoyang + Hang)
# Combines vision_expert, imu_expert, cross_attention, phase_arbitrator
# and a gated fusion head:  z = alpha * f_vision + (1 - alpha) * f_imu
# Output: (B, 27) class logits
