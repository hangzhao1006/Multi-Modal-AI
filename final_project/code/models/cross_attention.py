"""Cross-Modal Attention (ViLBERT Co-TRM style).

Plan ownership: Hang. This file is a working placeholder so we can run the full
PG-MoE pipeline end-to-end before Hang's final version lands. When his version
arrives, swap this module out — the interface is:

    cross_attn(vision, imu) -> (vision_enhanced, imu_enhanced)
    vision: (B, T_v, d), imu: (B, T_i, d)

Architecture (per Lecture 5 / ViLBERT Co-TRM):
    For each layer:
      - vision_attn_imu: Q=vision, K=V=imu     # vision attends to IMU
      - imu_attn_vision: Q=imu,    K=V=vision  # IMU attends to vision
      - Each followed by residual + LayerNorm + FFN + residual + LayerNorm
"""

import torch
import torch.nn as nn


class CoTRMBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.attn_v2i = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_i2v = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_v1 = nn.LayerNorm(d_model)
        self.norm_i1 = nn.LayerNorm(d_model)

        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.ffn_i = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.norm_v2 = nn.LayerNorm(d_model)
        self.norm_i2 = nn.LayerNorm(d_model)

    def forward(self, vision, imu):
        v_attn, _ = self.attn_v2i(query=vision, key=imu, value=imu, need_weights=False)
        i_attn, _ = self.attn_i2v(query=imu, key=vision, value=vision, need_weights=False)
        v = self.norm_v1(vision + v_attn)
        i = self.norm_i1(imu + i_attn)
        v = self.norm_v2(v + self.ffn_v(v))
        i = self.norm_i2(i + self.ffn_i(i))
        return v, i


class CrossModalAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CoTRMBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, vision, imu):
        for layer in self.layers:
            vision, imu = layer(vision, imu)
        return vision, imu
