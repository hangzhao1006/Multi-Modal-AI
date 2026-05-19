"""Phase Arbitrator for PG-MoE (Xiaoyang).

Reads raw IMU, extracts physical phase features (acc magnitude, jerk-like
2nd derivative, energy rate), and produces alpha(t) in [0, 1] that gates
vision vs IMU during fusion:

    z(t) = alpha(t) * f_vision(t) + (1 - alpha(t)) * f_imu(t)

Input:  (B, 6, 192)   raw IMU
Output: (B, T_i=12)   alpha values, one per IMU token
"""

import torch
import torch.nn as nn


class PhaseFeatures(nn.Module):
    """No-parameter feature extractor. Uses only the 3 accelerometer channels."""

    def forward(self, x):
        acc = x[:, :3]
        mag = torch.sqrt((acc ** 2).sum(dim=1, keepdim=True) + 1e-8)
        mag_d = torch.diff(mag, dim=2, prepend=mag[:, :, :1])
        mag_dd = torch.diff(mag_d, dim=2, prepend=mag_d[:, :, :1])
        energy = (acc ** 2).sum(dim=1, keepdim=True)
        e_rate = torch.diff(energy, dim=2, prepend=energy[:, :, :1])
        return torch.cat([mag, mag_dd, e_rate], dim=1)


class PhaseEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, out_dim, 1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ArbitratorMLP(nn.Module):
    def __init__(self, in_dim=32, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PhaseArbitrator(nn.Module):
    def __init__(self, T_i=12, feat_dim=3, enc_hidden=64, enc_out=32, arb_hidden=16):
        super().__init__()
        self.features = PhaseFeatures()
        self.pool = nn.AdaptiveAvgPool1d(T_i)
        self.encoder = PhaseEncoder(feat_dim, enc_hidden, enc_out)
        self.arbitrator = ArbitratorMLP(enc_out, arb_hidden)

    def forward(self, imu):
        feats = self.features(imu)
        feats = self.pool(feats)
        h = self.encoder(feats)
        alpha = self.arbitrator(h)
        return alpha.squeeze(1)
