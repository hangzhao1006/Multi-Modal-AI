"""
IMU Expert for PG-MoE (Xiaoyang).

Deep 1D-CNN with residual connections and BatchNorm.
NO global pooling — keeps the temporal axis so the tokens can feed into
cross-modal attention and phase-aware gating.

Input:  (B, 6, 192)        6-channel IMU, 192 timesteps
Output: (B, T_i, d_model)  T_i = 12 with the default stride schedule

T_i derivation:  192 -> 96 -> 48 -> 24 -> 12   (four stride-2 layers)
"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel=5, stride=1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, stride=stride, padding=pad),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel, stride=1, padding=pad),
            nn.BatchNorm1d(out_c),
        )
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm1d(out_c),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class IMUExpert(nn.Module):
    def __init__(self, d_model=256, in_channels=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock1D(64, 128, kernel=5, stride=2)
        self.block2 = ResidualBlock1D(128, 256, kernel=5, stride=2)
        self.block3 = ResidualBlock1D(256, d_model, kernel=3, stride=2)

    def forward(self, x):
        x = self.stem(x)      # (B, 64, 96)
        x = self.block1(x)    # (B, 128, 48)
        x = self.block2(x)    # (B, 256, 24)
        x = self.block3(x)    # (B, d_model, 12)
        return x.transpose(1, 2)  # (B, T_i=12, d_model)


class IMUClassifier(nn.Module):
    """Encoder + lightweight head for standalone IMU-only training.

    The head is used only to (a) verify the encoder beats midterm's 67.9% and
    (b) produce pretrained encoder weights for the joint PG-MoE training.
    At joint-training time, drop the head and reuse `self.encoder` directly.
    """

    def __init__(self, num_classes=27, d_model=256):
        super().__init__()
        self.encoder = IMUExpert(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        tokens = self.encoder(x)            # (B, T_i, d)
        pooled = tokens.mean(dim=1)         # temporal mean pool
        return self.head(self.norm(pooled))
