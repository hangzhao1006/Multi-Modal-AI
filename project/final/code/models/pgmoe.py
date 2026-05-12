"""PG-MoE: Phase-Gated Mixture-of-Experts for HAR.

Combines:
  - VisionProjector  : 2048-d Qwen tokens -> 256-d temporal-aware vision tokens
  - IMUExpert        : raw IMU -> (B, 12, 256) tokens
  - PhaseArbitrator  : raw IMU -> (B, 12) alpha values
  - CrossModalAttention (Co-TRM) : bidirectional vision <-> IMU
  - GatedFusion      : z(t) = alpha(t) * v(t) + (1 - alpha(t)) * i(t)
  - Classification head -> 27 classes

Plus FocalLoss for class imbalance (gamma=2, per pgmoe_plan.md).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .imu_expert import IMUExpert
from .phase_arbitrator import PhaseArbitrator
from .cross_attention import CrossModalAttention


class VisionProjector(nn.Module):
    """Project Qwen vision tokens to the shared d_model=256 space.

    Accepts either:
      (B, T_v, N_patch, d_in)  raw cached Qwen output  ← spatial-pool inside
      (B, T_v, d_in)           already spatial-pooled

    Mirrors Hang's V1 architecture from MMAI_Final.ipynb (LayerNorm + Linear +
    GELU + small temporal transformer), which got 58.6% standalone.
    """

    def __init__(self, d_in=2048, d_model=256, n_frames=60,
                 nhead=8, num_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, vision_tokens):
        if vision_tokens.dim() == 4:
            vision_tokens = vision_tokens.mean(dim=2)
        x = self.proj(vision_tokens)
        x = x + self.pos_embed
        return self.transformer(x)


class GatedFusion(nn.Module):
    """alpha-weighted fusion at the IMU temporal resolution (T_i).

    vision: (B, T_v, d)
    imu:    (B, T_i, d)
    alpha:  (B, T_i)

    Aligns vision -> T_i via adaptive avg pool over time, then mixes.
    """

    def forward(self, vision, imu, alpha):
        T_i = imu.shape[1]
        v_pooled = F.adaptive_avg_pool1d(vision.transpose(1, 2), output_size=T_i).transpose(1, 2)
        a = alpha.unsqueeze(-1)
        return a * v_pooled + (1.0 - a) * imu


class FocalLoss(nn.Module):
    """Focal Loss with class-balanced gamma. gamma=2 per plan section 二.5."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1.0 - pt) ** self.gamma * ce_loss).mean()


class PGMoE(nn.Module):
    """End-to-end PG-MoE classifier.

    forward(vision_tokens, imu, return_alpha=False) -> logits  [, alpha]
    """

    def __init__(self, num_classes=27, d_model=256, T_i=12,
                 vision_n_frames=60, vision_d_in=2048, vision_n_layers=2,
                 attn_n_layers=2, attn_n_heads=4, dropout=0.2):
        super().__init__()
        self.vision_proj = VisionProjector(
            d_in=vision_d_in, d_model=d_model,
            n_frames=vision_n_frames, num_layers=vision_n_layers, dropout=dropout,
        )
        self.imu_expert = IMUExpert(d_model=d_model)
        self.phase_arbitrator = PhaseArbitrator(T_i=T_i)
        self.cross_attn = CrossModalAttention(
            d_model=d_model, n_heads=attn_n_heads, n_layers=attn_n_layers, dropout=dropout,
        )
        self.fusion = GatedFusion()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, vision_tokens, imu, return_alpha=False):
        v = self.vision_proj(vision_tokens)         # (B, T_v, d)
        i = self.imu_expert(imu)                    # (B, T_i, d)
        alpha = self.phase_arbitrator(imu)          # (B, T_i)
        v_enh, i_enh = self.cross_attn(v, i)        # (B, T_v, d), (B, T_i, d)
        z = self.fusion(v_enh, i_enh, alpha)        # (B, T_i, d)
        z_pooled = z.mean(dim=1)                    # (B, d)
        logits = self.head(self.dropout(self.norm(z_pooled)))
        if return_alpha:
            return logits, alpha
        return logits

    def load_pretrained_imu(self, ckpt_path, device="cpu"):
        sd = torch.load(ckpt_path, map_location=device)
        self.imu_expert.load_state_dict(sd)
