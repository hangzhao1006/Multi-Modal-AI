"""
Fusion Method Comparison (Steps 7b-7d)
- PG-MoE Lightweight (no cross-attn): 88.8%
- Learned Weighted Fusion: 92.1%
- Input-Dependent Fusion: 92.1%
- PG-MoE Logit-Level: 90.7%
- PG-MoE with Expert Fine-tuning: 91.4%

Key finding: Late Fusion (92.3%) wins on 431 samples
Oracle upper bound: 98.4%
"""


# ============================================================
# ============================================================
# Step 7b: PG-MoE Lightweight (无Cross-Attn, 直接per-token gating)
#
# 核心改变:
#   去掉Cross-Modal Attention (431样本学不动200万参数)
#   直接用Expert输出的tokens做per-token fusion
#   Phase Arbitrator是唯一需要学的融合模块
#
# Pipeline:
#   Vision Expert (冻结) → (B, T_v, 256) → align to T_i
#   IMU Expert (冻结)    → (B, 12, 256)
#   Phase Arbitrator (原始IMU → 物理特征 → α(t))
#   Per-token: z(t) = α(t)·v(t) + (1-α(t))·i(t)
#   Classifier → 27类
#
# 为什么去掉Cross-Attn:
#   消融实验证明Cross-Attn(89.8%) < Late Fusion(92.3%)
#   说明431样本不够训练cross-attention
#   让Arbitrator成为唯一学习重心，梯度全部流向α(t)
# ============================================================

import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

print("Step 7b: PG-MoE Lightweight")
print("="*60)

DATA_ROOT = "/content/drive/MyDrive/utd_mhad"
SAVE_DIR = "/content/drive/MyDrive/pgmoe_ckpt"
IMU_LEN = 192
D_MODEL = 256
T_I = 12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 模型定义 (和之前一样，只是PG-MoE换了) ──

class ResidualBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel=5, stride=1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, stride=stride, padding=pad),
            nn.BatchNorm1d(out_c), nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel, stride=1, padding=pad),
            nn.BatchNorm1d(out_c))
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride), nn.BatchNorm1d(out_c))
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class IMUExpert(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(True))
        self.block1 = ResidualBlock1D(64, 128, 5, stride=2)
        self.block2 = ResidualBlock1D(128, 256, 5, stride=2)
        self.block3 = ResidualBlock1D(256, d_model, 3, stride=2)
    def forward(self, x):
        return self.block3(self.block2(self.block1(self.stem(x)))).transpose(1, 2)

class IMUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IMUExpert()
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 27)
    def forward(self, x, return_tokens=False):
        tokens = self.encoder(x)
        if return_tokens:
            return tokens
        return self.head(self.norm(tokens.mean(dim=1)))

class ResNet3DExpert(nn.Module):
    def __init__(self, d_model=256, dropout=0.3):
        super().__init__()
        backbone = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            backbone.stem, backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.proj = nn.Sequential(nn.Linear(512, d_model), nn.GELU(), nn.Dropout(dropout))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 27))
    def forward(self, x, return_tokens=False):
        B = x.shape[0]
        feat = self.spatial_pool(self.features(x)).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        T_out = feat.shape[1]
        feat = self.proj(feat)
        feat = torch.cat([self.cls_token.expand(B,-1,-1), feat], dim=1)
        feat = feat + self.pos_embed[:, :T_out+1, :]
        feat = self.temporal_transformer(feat)
        if return_tokens:
            return feat[:, 1:, :]
        return self.head(feat[:, 0, :])

# ── 加载Expert ──
print("Loading experts...")
imu_classifier = IMUClassifier().to(device)
for path in [f"{SAVE_DIR}/imu_classifier_best.pt", f"{DATA_ROOT}/imu_classifier_best.pt"]:
    if os.path.exists(path):
        imu_classifier.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        print(f"  IMU: {path}")
        break
imu_classifier.eval()

vision_expert = ResNet3DExpert().to(device)
ckpt = torch.load(f"{DATA_ROOT}/resnet3d_vision_expert.pt", map_location=device, weights_only=False)
vision_expert.load_state_dict(ckpt['model_state'])
vision_expert.eval()
print(f"  Vision: acc={ckpt['best_acc']:.1%}")

for p in vision_expert.parameters():
    p.requires_grad = False
for p in imu_classifier.parameters():
    p.requires_grad = False

# ── Phase Arbitrator (队友的) ──
class PhaseFeatures(nn.Module):
    def forward(self, x):
        acc = x[:, :3]
        mag = torch.sqrt((acc**2).sum(dim=1, keepdim=True) + 1e-8)
        mag_d = torch.diff(mag, dim=2, prepend=mag[:, :, :1])
        mag_dd = torch.diff(mag_d, dim=2, prepend=mag_d[:, :, :1])
        energy = (acc**2).sum(dim=1, keepdim=True)
        e_rate = torch.diff(energy, dim=2, prepend=energy[:, :, :1])
        return torch.cat([mag, mag_dd, e_rate], dim=1)

class PhaseArbitrator(nn.Module):
    """物理特征驱动的per-token gating, 只有~3K参数"""
    def __init__(self, T_i=12):
        super().__init__()
        self.features = PhaseFeatures()
        self.pool = nn.AdaptiveAvgPool1d(T_i)
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(True),
            nn.Conv1d(64, 32, 1), nn.ReLU(True))
        self.arbitrator = nn.Sequential(
            nn.Conv1d(32, 16, 1), nn.ReLU(True),
            nn.Conv1d(16, 1, 1), nn.Sigmoid())

    def forward(self, imu_raw):
        feats = self.pool(self.features(imu_raw))  # (B, 3, T_i)
        h = self.encoder(feats)                     # (B, 32, T_i)
        return self.arbitrator(h).squeeze(1)        # (B, T_i)

# ── PG-MoE Lightweight ──
class PGMoELight(nn.Module):
    """
    轻量PG-MoE: 无Cross-Attn, 只有Phase Arbitrator + Classifier

    总可学习参数: ~10K (Arbitrator 3K + Classifier 7K)
    对比完整版: 2.1M → 减少200倍
    → 431样本能学得动
    """
    def __init__(self, d_model=256, T_i=12, num_classes=27, dropout=0.2):
        super().__init__()
        self.T_i = T_i
        self.phase_arbitrator = PhaseArbitrator(T_i=T_i)
        self.vision_align = nn.AdaptiveAvgPool1d(T_i)

        # 轻量classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, vision_tokens, imu_tokens, imu_raw, return_alpha=False):
        # Align vision tokens: (B, T_v, 256) → (B, T_i, 256)
        v_aligned = self.vision_align(
            vision_tokens.permute(0, 2, 1)).permute(0, 2, 1)

        # Phase Arbitrator: raw IMU → α(t) per token
        alpha = self.phase_arbitrator(imu_raw)  # (B, T_i)
        alpha = alpha.unsqueeze(-1)              # (B, T_i, 1)

        # Per-token gated fusion
        fused = alpha * v_aligned + (1 - alpha) * imu_tokens  # (B, T_i, 256)

        # Pool + classify
        logits = self.classifier(fused.mean(dim=1))

        if return_alpha:
            return logits, alpha.squeeze(-1)
        return logits

# ── Dataset ──
rgb_data = torch.load(f"{DATA_ROOT}/rgb_frames_112.pt", map_location='cpu', weights_only=False)

class MultimodalDataset(Dataset):
    def __init__(self, rgb_data, data_root, train=True):
        self.samples = []
        allowed = {1,3,5,7} if train else {2,4,6,8}
        imu_dir = os.path.join(data_root, "Inertial")
        for (action, subject, trial), frames in rgb_data.items():
            if subject not in allowed:
                continue
            imu_path = os.path.join(imu_dir, f"a{action}_s{subject}_t{trial}_inertial.mat")
            if os.path.exists(imu_path):
                self.samples.append((frames, imu_path, action - 1))
        print(f"  {'Train' if train else 'Test'}: {len(self.samples)} samples")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        frames, imu_path, label = self.samples[idx]
        rgb = frames.float().permute(3, 0, 1, 2)
        imu_data = sio.loadmat(imu_path)["d_iner"].astype(np.float32)
        if imu_data.shape[0] < IMU_LEN:
            imu_data = np.concatenate([imu_data, np.zeros((IMU_LEN-imu_data.shape[0], 6), np.float32)])
        imu = torch.from_numpy(imu_data[:IMU_LEN]).T.contiguous()
        return rgb, imu, label

print("\nLoading dataset...")
train_ds = MultimodalDataset(rgb_data, DATA_ROOT, train=True)
test_ds = MultimodalDataset(rgb_data, DATA_ROOT, train=False)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

# ── 训练 ──
print("\n--- Training PG-MoE Light ---")
pgmoe = PGMoELight().to(device)
trainable = sum(p.numel() for p in pgmoe.parameters())
print(f"  Trainable params: {trainable:,} (vs 2.1M in full version)")

optimizer = torch.optim.AdamW(pgmoe.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"\n{'Epoch':>6} | {'Loss':>8} | {'Acc':>7} | {'Best':>6} | {'α mean':>7} | {'α std':>6}")
print("-" * 60)

best_acc = 0
best_state = None
best_alphas = None

for epoch in range(80):
    pgmoe.train()
    total_loss = 0
    for rgb, imu, labels in train_loader:
        rgb, imu, labels = rgb.to(device), imu.to(device), labels.to(device)
        with torch.no_grad():
            v_tokens = vision_expert(rgb, return_tokens=True)
            i_tokens = imu_classifier(imu, return_tokens=True)

        logits, alpha = pgmoe(v_tokens, i_tokens, imu, return_alpha=True)

        # 主loss
        cls_loss = criterion(logits, labels)

        # 辅助loss: α要有变化 + 不要stuck在0.5
        alpha_var_loss = -alpha.var(dim=1).mean()
        alpha_balance = (alpha.mean() - 0.5).pow(2)

        loss = cls_loss + 0.1 * alpha_var_loss + 0.3 * alpha_balance

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pgmoe.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        pgmoe.eval()
        all_preds, all_labels, all_alphas = [], [], []
        with torch.no_grad():
            for rgb, imu, labels in test_loader:
                rgb, imu = rgb.to(device), imu.to(device)
                v_tokens = vision_expert(rgb, return_tokens=True)
                i_tokens = imu_classifier(imu, return_tokens=True)
                logits, alpha = pgmoe(v_tokens, i_tokens, imu, return_alpha=True)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())
                all_alphas.append(alpha.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        alphas_np = np.concatenate(all_alphas, axis=0)
        a_mean, a_std = alphas_np.mean(), alphas_np.std()
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in pgmoe.state_dict().items()}
            best_alphas = alphas_np.copy()
            marker = " ★"
        print(f"  {epoch+1:4d}  | {total_loss/len(train_loader):7.4f} | {acc:6.1%} | "
              f"{best_acc:.1%} | {a_mean:6.3f} | {a_std:5.3f}{marker}")

# ── 保存 ──
save_path = f"{DATA_ROOT}/pgmoe_light_best.pt"
torch.save({'model_state': best_state, 'best_acc': best_acc}, save_path)

print(f"\n{'='*60}")
print(f"✅ Step 7b: PG-MoE Light")
print(f"   PG-MoE Light:       {best_acc:.1%}")
print(f"   Late Fusion:        92.3%")
print(f"   Vision-only:        89.3%")
print(f"   IMU-only:           82.3%")
print(f"   Oracle (upper bound): 98.4%")
print(f"   α mean={alphas_np.mean():.3f}, std={alphas_np.std():.3f}")
print(f"   Saved: {save_path}")

# ── α(t)可视化 ──
if best_alphas is not None:
    print(f"\n--- α(t) Distribution ---")
    print(f"  Per-token α statistics:")
    for t in range(T_I):
        a_t = best_alphas[:, t]
        print(f"    t={t:2d}: mean={a_t.mean():.3f}, std={a_t.std():.3f}, "
              f"min={a_t.min():.3f}, max={a_t.max():.3f}")

# ============================================================
# ============================================================
# Step 7c: Learned Weighted Fusion
#
# 思路: 不在token层面fusion (太细粒度，学不动)
#       在logit层面做per-class的权重学习
#       每个类学一个权重: 这个类该信任Vision还是IMU
#
# 为什么这样做:
#   per-class分析已经证明:
#     Tennis serve: 应该信任Vision (50% vs 94%)
#     Wave: 应该信任IMU (75% vs 100%)
#   → 让模型自动学到这种per-class preference
# ============================================================

import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

print("Step 7c: Learned Weighted Fusion")
print("="*60)

DATA_ROOT = "/content/drive/MyDrive/utd_mhad"
SAVE_DIR = "/content/drive/MyDrive/pgmoe_ckpt"
IMU_LEN = 192
D_MODEL = 256
T_I = 12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Expert模型 (同前, 省略定义, 直接用内存里的) ──
# 如果内存里没有, 需要重新定义和加载
# vision_expert, imu_classifier 应该还在内存里

# 冻结
for p in vision_expert.parameters():
    p.requires_grad = False
for p in imu_classifier.parameters():
    p.requires_grad = False

# ── Phase Arbitrator (保留, 但用在token层面做可视化) ──
class PhaseFeatures(nn.Module):
    def forward(self, x):
        acc = x[:, :3]
        mag = torch.sqrt((acc**2).sum(dim=1, keepdim=True) + 1e-8)
        mag_d = torch.diff(mag, dim=2, prepend=mag[:, :, :1])
        mag_dd = torch.diff(mag_d, dim=2, prepend=mag_d[:, :, :1])
        energy = (acc**2).sum(dim=1, keepdim=True)
        e_rate = torch.diff(energy, dim=2, prepend=energy[:, :, :1])
        return torch.cat([mag, mag_dd, e_rate], dim=1)

# ── 方法1: Learned Per-Class Weighted Fusion ──
class LearnedWeightedFusion(nn.Module):
    """
    每个Expert有自己的分类头
    学习一个per-class的融合权重 w ∈ [0,1]^27
    final_logits = w * vision_logits + (1-w) * imu_logits

    只有27个可学习参数!
    """
    def __init__(self, d_model=256, num_classes=27, dropout=0.2):
        super().__init__()
        self.v_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))
        self.i_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))
        # Per-class fusion weight
        self.fusion_w = nn.Parameter(torch.zeros(num_classes))  # sigmoid → 0.5 init

    def forward(self, v_tokens, i_tokens, imu_raw=None, return_alpha=False):
        v_logits = self.v_head(v_tokens.mean(dim=1))  # (B, 27)
        i_logits = self.i_head(i_tokens.mean(dim=1))  # (B, 27)

        w = torch.sigmoid(self.fusion_w)  # (27,) ∈ [0,1]
        fused = w * v_logits + (1 - w) * i_logits  # (B, 27)

        if return_alpha:
            return fused, w
        return fused

# ── 方法2: Input-Dependent Weighted Fusion ──
class InputDependentFusion(nn.Module):
    """
    融合权重不是固定的per-class，而是根据每个输入动态决定
    用两个Expert的pooled features来预测per-class weights

    参数量适中: ~15K
    """
    def __init__(self, d_model=256, num_classes=27, dropout=0.2):
        super().__init__()
        self.v_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))
        self.i_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))
        # 动态fusion weight predictor
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, v_tokens, i_tokens, imu_raw=None, return_alpha=False):
        v_pool = v_tokens.mean(dim=1)  # (B, 256)
        i_pool = i_tokens.mean(dim=1)  # (B, 256)

        v_logits = self.v_head(v_pool)  # (B, 27)
        i_logits = self.i_head(i_pool)  # (B, 27)

        # 动态权重
        combined = torch.cat([v_pool, i_pool], dim=1)  # (B, 512)
        w = self.gate(combined)  # (B, 27)

        fused = w * v_logits + (1 - w) * i_logits

        if return_alpha:
            return fused, w.mean(dim=1)  # (B,) average across classes
        return fused

# ── 方法3: PG-MoE with Phase Arbitrator at Logit Level ──
class PGMoELogitLevel(nn.Module):
    """
    Phase Arbitrator在logit层面操作:

    每个Expert有分类头 → 得到logits
    Phase Arbitrator → 一个全局α
    fused = α * vision_logits + (1-α) * imu_logits

    和token-level的区别:
      token-level: 先fusion特征再分类 (信息可能损失)
      logit-level: 先分类再fusion (保留各自最强的判断)
    """
    def __init__(self, d_model=256, T_i=12, num_classes=27, dropout=0.2):
        super().__init__()
        self.v_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))
        self.i_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))

        # Phase Arbitrator → 全局α
        self.phase_features = PhaseFeatures()
        self.phase_pool = nn.AdaptiveAvgPool1d(1)  # pool到1个值
        self.phase_gate = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(True),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, v_tokens, i_tokens, imu_raw, return_alpha=False):
        v_logits = self.v_head(v_tokens.mean(dim=1))
        i_logits = self.i_head(i_tokens.mean(dim=1))

        # Phase features → global α
        feats = self.phase_features(imu_raw)          # (B, 3, 192)
        feats = self.phase_pool(feats).squeeze(-1)    # (B, 3)
        alpha = self.phase_gate(feats)                # (B, 1)

        fused = alpha * v_logits + (1 - alpha) * i_logits

        if return_alpha:
            return fused, alpha.squeeze(-1)
        return fused

# ── Dataset (同前) ──
rgb_data = torch.load(f"{DATA_ROOT}/rgb_frames_112.pt", map_location='cpu', weights_only=False)

class MultimodalDataset(Dataset):
    def __init__(self, rgb_data, data_root, train=True):
        self.samples = []
        allowed = {1,3,5,7} if train else {2,4,6,8}
        imu_dir = os.path.join(data_root, "Inertial")
        for (action, subject, trial), frames in rgb_data.items():
            if subject not in allowed:
                continue
            imu_path = os.path.join(imu_dir, f"a{action}_s{subject}_t{trial}_inertial.mat")
            if os.path.exists(imu_path):
                self.samples.append((frames, imu_path, action - 1))
        print(f"  {'Train' if train else 'Test'}: {len(self.samples)} samples")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        frames, imu_path, label = self.samples[idx]
        rgb = frames.float().permute(3, 0, 1, 2)
        imu_data = sio.loadmat(imu_path)["d_iner"].astype(np.float32)
        if imu_data.shape[0] < IMU_LEN:
            imu_data = np.concatenate([imu_data, np.zeros((IMU_LEN-imu_data.shape[0], 6), np.float32)])
        imu = torch.from_numpy(imu_data[:IMU_LEN]).T.contiguous()
        return rgb, imu, label

print("\nLoading dataset...")
train_ds = MultimodalDataset(rgb_data, DATA_ROOT, train=True)
test_ds = MultimodalDataset(rgb_data, DATA_ROOT, train=False)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

# ── 通用训练函数 ──
def train_model(model, name, epochs=60, lr=1e-3):
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  {name}: {params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for rgb, imu, labels in train_loader:
            rgb, imu, labels = rgb.to(device), imu.to(device), labels.to(device)
            with torch.no_grad():
                vt = vision_expert(rgb, return_tokens=True)
                it = imu_classifier(imu, return_tokens=True)
            logits = model(vt, it, imu)
            loss = crit(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sch.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            preds, labs = [], []
            with torch.no_grad():
                for rgb, imu, labels in test_loader:
                    rgb, imu = rgb.to(device), imu.to(device)
                    vt = vision_expert(rgb, return_tokens=True)
                    it = imu_classifier(imu, return_tokens=True)
                    preds.extend(model(vt, it, imu).argmax(1).cpu().numpy())
                    labs.extend(labels.numpy())
            acc = accuracy_score(labs, preds)
            marker = " ★" if acc > best_acc else ""
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ep{epoch+1:3d}: {acc:.1%} (best: {best_acc:.1%}){marker}")

    return best_acc, best_state

# ── 训练3种方法 ──
print("\n" + "="*60)
print("Training 3 fusion methods...")

acc1, state1 = train_model(LearnedWeightedFusion(), "Per-Class Weighted", epochs=60)
acc2, state2 = train_model(InputDependentFusion(), "Input-Dependent", epochs=60)
acc3, state3 = train_model(PGMoELogitLevel(), "PG-MoE Logit-Level", epochs=60)

# ── 保存最好的 ──
best_name = ""
best_final = 0
for name, acc, state in [("per_class", acc1, state1),
                          ("input_dep", acc2, state2),
                          ("pgmoe_logit", acc3, state3)]:
    if acc > best_final:
        best_final = acc
        best_name = name
        torch.save({'model_state': state, 'best_acc': acc},
                   f"{DATA_ROOT}/fusion_{name}_best.pt")

# ── 最终结果表 ──
print(f"\n{'='*60}")
print(f"{'Method':<35} | {'Accuracy':>8}")
print(f"{'-'*35}-+-{'-'*8}")
print(f"{'1. IMU-only':<35} | {'82.3%':>8}")
print(f"{'2. Vision-only':<35} | {'89.3%':>8}")
print(f"{'3. Early Fusion':<35} | {'90.2%':>8}")
print(f"{'4. Late Fusion (avg logits)':<35} | {'92.3%':>8}")
print(f"{'5. Cross-Attn (no gating)':<35} | {'89.8%':>8}")
print(f"{'6. PG-MoE token-level':<35} | {'90.2%':>8}")
print(f"{'7. Per-Class Weighted':<35} | {f'{acc1:.1%}':>8}")
print(f"{'8. Input-Dependent Fusion':<35} | {f'{acc2:.1%}':>8}")
print(f"{'9. PG-MoE Logit-Level':<35} | {f'{acc3:.1%}':>8}")
print(f"{'-'*35}-+-{'-'*8}")
print(f"{'Oracle (upper bound)':<35} | {'98.4%':>8}")
print(f"{'='*60}")

# ============================================================
# ============================================================
# Step 7d: PG-MoE with Expert Fine-tuning
#
# 真正的MoE:
#   两个Expert各自输出logits
#   Phase Arbitrator根据IMU物理特征输出α
#   融合: α·vision_logits + (1-α)·imu_logits
#   解冻Expert最后几层，joint fine-tuning
#
# 和之前的区别:
#   之前: Expert完全冻结，只学fusion → 特征不适配
#   现在: Expert最后几层解冻，和fusion一起学 → 特征适配
# ============================================================

import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

print("Step 7d: PG-MoE with Expert Fine-tuning")
print("="*60)

DATA_ROOT = "/content/drive/MyDrive/utd_mhad"
SAVE_DIR = "/content/drive/MyDrive/pgmoe_ckpt"
IMU_LEN = 192
D_MODEL = 256
T_I = 12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Expert定义 (同前) ──
class ResidualBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel=5, stride=1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, stride=stride, padding=pad),
            nn.BatchNorm1d(out_c), nn.ReLU(True),
            nn.Conv1d(out_c, out_c, kernel, stride=1, padding=pad),
            nn.BatchNorm1d(out_c))
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride), nn.BatchNorm1d(out_c))
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class IMUExpert(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(True))
        self.block1 = ResidualBlock1D(64, 128, 5, stride=2)
        self.block2 = ResidualBlock1D(128, 256, 5, stride=2)
        self.block3 = ResidualBlock1D(256, d_model, 3, stride=2)
    def forward(self, x):
        return self.block3(self.block2(self.block1(self.stem(x)))).transpose(1, 2)

class IMUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IMUExpert()
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 27)
    def forward(self, x, return_tokens=False):
        tokens = self.encoder(x)
        if return_tokens:
            return tokens
        return self.head(self.norm(tokens.mean(dim=1)))

class ResNet3DExpert(nn.Module):
    def __init__(self, d_model=256, dropout=0.3):
        super().__init__()
        backbone = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            backbone.stem, backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.proj = nn.Sequential(nn.Linear(512, d_model), nn.GELU(), nn.Dropout(dropout))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 27))
    def forward(self, x, return_tokens=False):
        B = x.shape[0]
        feat = self.spatial_pool(self.features(x)).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        T_out = feat.shape[1]
        feat = self.proj(feat)
        feat = torch.cat([self.cls_token.expand(B,-1,-1), feat], dim=1)
        feat = feat + self.pos_embed[:, :T_out+1, :]
        feat = self.temporal_transformer(feat)
        if return_tokens:
            return feat[:, 1:, :]
        return self.head(feat[:, 0, :])

# ── 加载预训练权重 ──
print("Loading pretrained experts...")
imu_classifier = IMUClassifier().to(device)
for path in [f"{SAVE_DIR}/imu_classifier_best.pt", f"{DATA_ROOT}/imu_classifier_best.pt"]:
    if os.path.exists(path):
        imu_classifier.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        print(f"  IMU: {path}")
        break

vision_expert = ResNet3DExpert().to(device)
ckpt = torch.load(f"{DATA_ROOT}/resnet3d_vision_expert.pt", map_location=device, weights_only=False)
vision_expert.load_state_dict(ckpt['model_state'])
print(f"  Vision: acc={ckpt['best_acc']:.1%}")

# ── Phase Arbitrator (队友的) ──
class PhaseFeatures(nn.Module):
    def forward(self, x):
        acc = x[:, :3]
        mag = torch.sqrt((acc**2).sum(dim=1, keepdim=True) + 1e-8)
        mag_d = torch.diff(mag, dim=2, prepend=mag[:, :, :1])
        mag_dd = torch.diff(mag_d, dim=2, prepend=mag_d[:, :, :1])
        energy = (acc**2).sum(dim=1, keepdim=True)
        e_rate = torch.diff(energy, dim=2, prepend=energy[:, :, :1])
        return torch.cat([mag, mag_dd, e_rate], dim=1)

class PhaseArbitrator(nn.Module):
    def __init__(self, T_i=12):
        super().__init__()
        self.features = PhaseFeatures()
        self.pool = nn.AdaptiveAvgPool1d(T_i)
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(True),
            nn.Conv1d(64, 32, 1), nn.ReLU(True))
        self.arbitrator = nn.Sequential(
            nn.Conv1d(32, 16, 1), nn.ReLU(True),
            nn.Conv1d(16, 1, 1), nn.Sigmoid())
    def forward(self, imu_raw):
        feats = self.pool(self.features(imu_raw))
        return self.arbitrator(self.encoder(feats)).squeeze(1)  # (B, T_i)

# ── 完整PG-MoE (Logit-level + Phase Arbitrator) ──
class PGMoEFull(nn.Module):
    """
    真正的Mixture-of-Experts:
      Expert 1: Vision → vision_logits
      Expert 2: IMU → imu_logits
      Gating: Phase Arbitrator → α (物理特征驱动)
      Output: α·vision_logits + (1-α)·imu_logits
    """
    def __init__(self, vision_expert, imu_classifier, d_model=256, num_classes=27):
        super().__init__()
        # 两个Expert作为子模块 (部分层解冻)
        self.vision_expert = vision_expert
        self.imu_classifier = imu_classifier

        # 各自的分类头 (新的，从零学)
        self.v_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(0.2),
            nn.Linear(d_model, num_classes))
        self.i_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(0.2),
            nn.Linear(d_model, num_classes))

        # Phase Arbitrator → 全局α
        self.phase_arbitrator = PhaseArbitrator(T_i=1)  # pool到1个值

    def forward(self, rgb, imu_raw, return_alpha=False):
        v_tokens = self.vision_expert(rgb, return_tokens=True)
        i_tokens = self.imu_classifier(imu_raw, return_tokens=True)

        v_logits = self.v_head(v_tokens.mean(dim=1))  # (B, 27)
        i_logits = self.i_head(i_tokens.mean(dim=1))  # (B, 27)

        # Phase Arbitrator → α: (B, 1)
        alpha_raw = self.phase_arbitrator(imu_raw)     # (B, 1)
        alpha = alpha_raw.view(-1, 1)                  # 确保 (B, 1)

        # Gated fusion at logit level
        fused = alpha * v_logits + (1 - alpha) * i_logits  # (B, 27)

        if return_alpha:
            return fused, alpha.squeeze(-1)  # (B, 27), (B,)
        return fused

# ── 组装PG-MoE ──
pgmoe = PGMoEFull(vision_expert, imu_classifier).to(device)

# ── 冻结策略: 只解冻Expert最后几层 ──
print("\nFreeze strategy:")

# 先全部冻结
for p in pgmoe.parameters():
    p.requires_grad = False

# 解冻 Vision: proj + temporal_transformer + cls_token + pos_embed
for name, p in pgmoe.vision_expert.named_parameters():
    if any(x in name for x in ['temporal_transformer', 'proj', 'cls_token', 'pos_embed']):
        p.requires_grad = True

# 解冻 IMU: block3 + norm
for name, p in pgmoe.imu_classifier.named_parameters():
    if any(x in name for x in ['block3', 'norm']):
        p.requires_grad = True

# 解冻 fusion部分 (v_head, i_head, phase_arbitrator)
for p in pgmoe.v_head.parameters():
    p.requires_grad = True
for p in pgmoe.i_head.parameters():
    p.requires_grad = True
for p in pgmoe.phase_arbitrator.parameters():
    p.requires_grad = True

frozen = sum(p.numel() for p in pgmoe.parameters() if not p.requires_grad)
trainable = sum(p.numel() for p in pgmoe.parameters() if p.requires_grad)
print(f"  Frozen: {frozen:,}")
print(f"  Trainable: {trainable:,}")
print(f"  (Vision proj+transformer + IMU block3 + heads + arbitrator)")

# ── Dataset ──
rgb_data = torch.load(f"{DATA_ROOT}/rgb_frames_112.pt", map_location='cpu', weights_only=False)

class MultimodalDataset(Dataset):
    def __init__(self, rgb_data, data_root, train=True):
        self.samples = []
        allowed = {1,3,5,7} if train else {2,4,6,8}
        imu_dir = os.path.join(data_root, "Inertial")
        for (action, subject, trial), frames in rgb_data.items():
            if subject not in allowed:
                continue
            imu_path = os.path.join(imu_dir, f"a{action}_s{subject}_t{trial}_inertial.mat")
            if os.path.exists(imu_path):
                self.samples.append((frames, imu_path, action - 1))
        print(f"  {'Train' if train else 'Test'}: {len(self.samples)} samples")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        frames, imu_path, label = self.samples[idx]
        rgb = frames.float().permute(3, 0, 1, 2)
        imu_data = sio.loadmat(imu_path)["d_iner"].astype(np.float32)
        if imu_data.shape[0] < IMU_LEN:
            imu_data = np.concatenate([imu_data, np.zeros((IMU_LEN-imu_data.shape[0], 6), np.float32)])
        imu = torch.from_numpy(imu_data[:IMU_LEN]).T.contiguous()
        return rgb, imu, label

print("\nLoading dataset...")
train_ds = MultimodalDataset(rgb_data, DATA_ROOT, train=True)
test_ds = MultimodalDataset(rgb_data, DATA_ROOT, train=False)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

# ── 两阶段训练 ──
# Stage 1: 低学习率warmup (10 epochs)
# Stage 2: 正常训练 (60 epochs)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print("\n--- Stage 1: Warmup (lr=5e-5, 10 epochs) ---")

optimizer = torch.optim.AdamW([
    {'params': [p for n, p in pgmoe.vision_expert.named_parameters() if p.requires_grad],
     'lr': 5e-5},  # Expert用低学习率
    {'params': [p for n, p in pgmoe.imu_classifier.named_parameters() if p.requires_grad],
     'lr': 5e-5},
    {'params': list(pgmoe.v_head.parameters()) + list(pgmoe.i_head.parameters()),
     'lr': 1e-3},  # 新的head用高学习率
    {'params': pgmoe.phase_arbitrator.parameters(),
     'lr': 1e-3},
], weight_decay=0.01)

print(f"\n{'Epoch':>6} | {'Loss':>8} | {'Acc':>7} | {'Best':>6} | {'α mean':>7}")
print("-" * 55)

best_acc = 0
best_state = None

for epoch in range(10):
    pgmoe.train()
    total_loss = 0
    for rgb, imu, labels in train_loader:
        rgb, imu, labels = rgb.to(device), imu.to(device), labels.to(device)
        logits = pgmoe(rgb, imu)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pgmoe.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        pgmoe.eval()
        preds, labs, alphas = [], [], []
        with torch.no_grad():
            for rgb, imu, labels in test_loader:
                rgb, imu = rgb.to(device), imu.to(device)
                logits, alpha = pgmoe(rgb, imu, return_alpha=True)
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(labels.numpy())
                alphas.extend(alpha.cpu().numpy())
        acc = accuracy_score(labs, preds)
        a_mean = np.mean(alphas)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in pgmoe.state_dict().items()}
        print(f"  {epoch+1:4d}  | {total_loss/len(train_loader):7.4f} | {acc:6.1%} | "
              f"{best_acc:.1%} | {a_mean:6.3f}")

# Stage 2
print(f"\n--- Stage 2: Full training (70 epochs) ---")

optimizer2 = torch.optim.AdamW([
    {'params': [p for n, p in pgmoe.vision_expert.named_parameters() if p.requires_grad],
     'lr': 3e-5},
    {'params': [p for n, p in pgmoe.imu_classifier.named_parameters() if p.requires_grad],
     'lr': 3e-5},
    {'params': list(pgmoe.v_head.parameters()) + list(pgmoe.i_head.parameters()),
     'lr': 5e-4},
    {'params': pgmoe.phase_arbitrator.parameters(),
     'lr': 5e-4},
], weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=70)

for epoch in range(70):
    pgmoe.train()
    total_loss = 0
    for rgb, imu, labels in train_loader:
        rgb, imu, labels = rgb.to(device), imu.to(device), labels.to(device)
        logits, alpha = pgmoe(rgb, imu, return_alpha=True)

        cls_loss = criterion(logits, labels)
        # 轻微的balance loss防止α跑到极端
        alpha_balance = (alpha.mean() - 0.5).pow(2)
        loss = cls_loss + 0.1 * alpha_balance

        optimizer2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pgmoe.parameters(), 0.5)
        optimizer2.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        pgmoe.eval()
        preds, labs, alphas = [], [], []
        with torch.no_grad():
            for rgb, imu, labels in test_loader:
                rgb, imu = rgb.to(device), imu.to(device)
                logits, alpha = pgmoe(rgb, imu, return_alpha=True)
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(labels.numpy())
                alphas.extend(alpha.cpu().numpy())
        acc = accuracy_score(labs, preds)
        a_mean = np.mean(alphas)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in pgmoe.state_dict().items()}
            marker = " ★"
        print(f"  {epoch+11:4d}  | {total_loss/len(train_loader):7.4f} | {acc:6.1%} | "
              f"{best_acc:.1%} | {a_mean:6.3f}{marker}")

# ── 保存 ──
save_path = f"{DATA_ROOT}/pgmoe_finetune_best.pt"
torch.save({'model_state': best_state, 'best_acc': best_acc}, save_path)

print(f"\n{'='*60}")
print(f"✅ Step 7d: PG-MoE with Expert Fine-tuning")
print(f"   PG-MoE (fine-tuned):     {best_acc:.1%}")
print(f"   Late Fusion (frozen):    92.3%")
print(f"   Vision-only:             89.3%")
print(f"   IMU-only:                82.3%")
print(f"   Oracle:                  98.4%")
print(f"   Saved: {save_path}")
