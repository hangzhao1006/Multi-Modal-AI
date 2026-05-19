"""
Step 5-6: Qwen Vision Token Classification Tests
- Step 5: Global mean pool + MLP → 28.1%
- Step 5: Temporal attention pool → 53.3%  
- Step 6: Temporal Transformer → 58.6% (best)
- Step 6b: Spatial attention (failed)
- Step 6c: V1 + augmentation → 52.8%

Conclusion: Frozen VLM tokens achieve 58.6%,
far below task-specific ResNet3D (89.3%)

Input: vision_tokens_qwen25.pt
"""


# ============================================================
# ============================================================
# Step 4: Vision Token质量验证 - 分类测试
# 目的: 用冻结的Qwen vision tokens + 简单classifier测试分类准确率
# 输入: vision_tokens_qwen25.pt (Step 3生成)
# 期望: 应该 > midterm的ResNet3D 72.8%
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# ── 加载缓存的vision tokens ──
print("Step 4: 加载vision tokens...")
vision_cache = torch.load(
    "/content/drive/MyDrive/utd_mhad/vision_tokens_qwen25.pt",
    map_location='cpu', weights_only=False)
print(f"  加载了 {len(vision_cache)} 个视频")
sample = list(vision_cache.values())[0]
print(f"  每个视频: {sample.shape}")  # (60, 64, 2048)

# ── Dataset ──
class VisionTokenDataset(Dataset):
    def __init__(self, vision_cache, train=True):
        self.samples = []
        train_subjects = {1, 3, 5, 7}
        test_subjects = {2, 4, 6, 8}
        allowed = train_subjects if train else test_subjects

        for (action, subject, trial), tokens in vision_cache.items():
            if subject not in allowed:
                continue
            # tokens: (60, 64, 2048)
            # 方案1: 对每帧的64个tokens取mean → (60, 2048) → 再对60帧取mean → (2048,)
            feat = tokens.mean(dim=1).mean(dim=0)  # (2048,)
            self.samples.append((feat, action - 1))

        print(f"  {'Train' if train else 'Test'}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return feat.float(), label

# ── 分类器: 简单MLP ──
class TokenClassifier(nn.Module):
    def __init__(self, d_in=2048, num_classes=27):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ── 训练 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = VisionTokenDataset(vision_cache, train=True)
test_ds = VisionTokenDataset(vision_cache, train=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = TokenClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()

print(f"\n开始训练 (设备: {device})")
print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

best_acc = 0
for epoch in range(100):
    # Train
    model.train()
    total_loss = 0
    for feats, labels in train_loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # Eval
    if (epoch + 1) % 10 == 0:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, labels in test_loader:
                feats = feats.to(device)
                preds = model(feats).argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
        print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Test Acc: {acc:.1%} | Best: {best_acc:.1%}")

print(f"\n{'='*50}")
print(f"✅ Step 4 结果: Vision Token分类准确率 = {best_acc:.1%}")
print(f"   对比 Midterm ResNet3D: 72.8%")
print(f"   {'🎉 超过了！' if best_acc > 0.728 else '⚠️ 低于预期，可能需要更好的pooling'}")

# ── 方案2测试: 保留时序的Temporal Pooling ──
print(f"\n{'='*50}")
print("测试方案2: Temporal Attention Pooling...")

class TemporalAttentionClassifier(nn.Module):
    """不是简单mean，而是用attention选择重要的帧"""
    def __init__(self, d_in=2048, num_classes=27, n_frames=60):
        super().__init__()
        self.token_pool = nn.Linear(d_in, d_in)  # 64 tokens → 1
        self.temporal_attn = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B, 60, 2048) - 已经对64个token取了mean
        attn_scores = self.temporal_attn(x)  # (B, 60, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, 60, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, 2048)
        return self.classifier(pooled), attn_weights

class VisionTokenDatasetTemporal(Dataset):
    def __init__(self, vision_cache, train=True):
        self.samples = []
        allowed = {1,3,5,7} if train else {2,4,6,8}
        for (action, subject, trial), tokens in vision_cache.items():
            if subject not in allowed:
                continue
            # tokens: (60, 64, 2048)
            feat = tokens.mean(dim=1)  # (60, 2048) - 保留时序！
            self.samples.append((feat, action - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return feat.float(), label

train_ds2 = VisionTokenDatasetTemporal(vision_cache, train=True)
test_ds2 = VisionTokenDatasetTemporal(vision_cache, train=False)
train_loader2 = DataLoader(train_ds2, batch_size=32, shuffle=True)
test_loader2 = DataLoader(test_ds2, batch_size=64, shuffle=False)

model2 = TemporalAttentionClassifier().to(device)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=0.05)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)

best_acc2 = 0
for epoch in range(100):
    model2.train()
    total_loss = 0
    for feats, labels in train_loader2:
        feats, labels = feats.to(device), labels.to(device)
        logits, _ = model2(feats)
        loss = criterion(logits, labels)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        total_loss += loss.item()
    scheduler2.step()

    if (epoch + 1) % 10 == 0:
        model2.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, labels in test_loader2:
                feats = feats.to(device)
                preds, _ = model2(feats)
                preds = preds.argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc2:
            best_acc2 = acc
        print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader2):.4f} | "
              f"Test Acc: {acc:.1%} | Best: {best_acc2:.1%}")

print(f"\n{'='*50}")
print(f"📊 Step 4 最终对比:")
print(f"  方案1 (Global Mean Pool):      {best_acc:.1%}")
print(f"  方案2 (Temporal Attention):     {best_acc2:.1%}")
print(f"  Midterm ResNet3D baseline:      72.8%")
print(f"\n→ 选择更好的方案作为Vision Expert进入Step 5")

# ============================================================
# ============================================================
# Step 6: Vision Temporal Transformer
# 目的: 用temporal self-attention提升vision-only准确率
# 输入: vision_tokens_qwen25.pt (60, 64, 2048)
# 输出模型: 投影2048→256 + temporal transformer → (B, 60, 256)
# 这个模块后续直接作为PG-MoE的Vision Expert
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

print("Step 6: Vision Temporal Transformer")
print("="*50)

# ── 加载tokens ──
vision_cache = torch.load(
    "/content/drive/MyDrive/utd_mhad/vision_tokens_qwen25.pt",
    map_location='cpu', weights_only=False)
print(f"加载了 {len(vision_cache)} 个视频")

D_MODEL = 256  # 统一维度，和IMU对齐

# ── Dataset ──
class VisionTokenDataset(Dataset):
    def __init__(self, vision_cache, train=True):
        self.samples = []
        allowed = {1,3,5,7} if train else {2,4,6,8}
        for (action, subject, trial), tokens in vision_cache.items():
            if subject not in allowed:
                continue
            # (60, 64, 2048) → spatial mean → (60, 2048)
            feat = tokens.mean(dim=1)
            self.samples.append((feat, action - 1))
        print(f"  {'Train' if train else 'Test'}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return feat.float(), label

# ── Vision Temporal Transformer模型 ──
class VisionTemporalTransformer(nn.Module):
    """
    Qwen ViT tokens (60, 2048) → 投影256 → temporal self-attention → 分类

    这个模型有两个用途:
    1. 现在: vision-only分类，验证temporal transformer的效果
    2. 后续: 去掉classifier head，作为PG-MoE的Vision Expert
           输出 (B, 60, 256) 给Cross-Modal Attention
    """
    def __init__(self, d_in=2048, d_model=256, n_frames=60,
                 nhead=8, num_layers=3, dropout=0.2, num_classes=27):
        super().__init__()

        # 投影: 2048 → 256
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames + 1, d_model) * 0.02)

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, return_tokens=False):
        """
        x: (B, 60, 2048)
        return_tokens: if True, 返回 (B, 60, 256) 用于cross-modal attention
        """
        B = x.shape[0]

        # 投影
        x = self.proj(x)  # (B, 60, 256)

        # 加CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 256)
        x = torch.cat([cls, x], dim=1)  # (B, 61, 256)

        # 加positional encoding
        x = x + self.pos_embed

        # Temporal self-attention
        x = self.transformer(x)  # (B, 61, 256)

        if return_tokens:
            # 返回60帧的tokens (去掉CLS)，给后续cross-modal attention用
            return x[:, 1:, :]  # (B, 60, 256)

        # 分类: 用CLS token
        cls_out = x[:, 0, :]  # (B, 256)
        return self.head(cls_out)

# ── 训练 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = VisionTokenDataset(vision_cache, train=True)
test_ds = VisionTokenDataset(vision_cache, train=False)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = VisionTemporalTransformer().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n模型参数: {total_params:,}")
print(f"设备: {device}")
print(f"架构: Qwen ViT (frozen) → proj(2048→256) → 3-layer Transformer → CLS → 27类")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Test Acc':>8} | {'Best':>6}")
print("-" * 45)

best_acc = 0
best_state = None

for epoch in range(150):
    # Train
    model.train()
    total_loss = 0
    for feats, labels in train_loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # Eval every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, labels in test_loader:
                feats = feats.to(device)
                preds = model(feats).argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " ★"

        print(f"  {epoch+1:4d}  | {total_loss/len(train_loader):10.4f} | {acc:7.1%} | {best_acc:.1%}{marker}")

# ── 保存最佳模型 ──
save_path = "/content/drive/MyDrive/utd_mhad/vision_temporal_transformer.pt"
torch.save({
    'model_state': best_state,
    'd_model': D_MODEL,
    'best_acc': best_acc,
}, save_path)

print(f"\n{'='*50}")
print(f"✅ Step 6 完成!")
print(f"   Vision Temporal Transformer: {best_acc:.1%}")
print(f"   对比 Step 5 Temporal Attention: 53.3%")
print(f"   对比 Midterm ResNet3D: 72.8%")
print(f"   模型已保存: {save_path}")
print(f"\n   后续用法:")
print(f"   model.forward(x, return_tokens=True) → (B, 60, 256)")
print(f"   直接作为Vision Expert输入Cross-Modal Attention")

# ============================================================
# ============================================================
# Step 6b: Vision Temporal Transformer (改进版)
# 改进点:
#   1. Spatial Attention Pooling (不再粗暴mean 64 tokens)
#   2. Temporal数据增强 (shift + drop + scale)
#   3. 超参调优 (2层transformer防过拟合, warmup)
#   4. Mixup正则化
# 输入: vision_tokens_qwen25.pt (60, 64, 2048)
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

print("Step 6b: Vision Temporal Transformer (改进版)")
print("="*50)

# ── 加载tokens ──
vision_cache = torch.load(
    "/content/drive/MyDrive/utd_mhad/vision_tokens_qwen25.pt",
    map_location='cpu', weights_only=False)
print(f"加载了 {len(vision_cache)} 个视频")

D_MODEL = 256

# ── Dataset: 保留spatial tokens + 数据增强 ──
class VisionTokenDatasetV2(Dataset):
    def __init__(self, vision_cache, train=True):
        self.train = train
        self.samples = []
        allowed = {1,3,5,7} if train else {2,4,6,8}
        for (action, subject, trial), tokens in vision_cache.items():
            if subject not in allowed:
                continue
            # 保留完整 (60, 64, 2048)，spatial pooling在模型里做
            self.samples.append((tokens, action - 1))
        print(f"  {'Train' if train else 'Test'}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]  # (60, 64, 2048)
        feat = feat.float()

        if self.train:
            # 增强1: 随机temporal shift (±3帧)
            shift = np.random.randint(-3, 4)
            feat = torch.roll(feat, shifts=shift, dims=0)

            # 增强2: 随机drop 5帧 (设为0)
            if np.random.random() > 0.5:
                drop_idx = np.random.choice(60, size=5, replace=False)
                feat[drop_idx] = 0

            # 增强3: 随机drop部分spatial tokens
            if np.random.random() > 0.5:
                drop_spatial = np.random.choice(64, size=16, replace=False)
                feat[:, drop_spatial] = 0

            # 增强4: 高斯噪声
            if np.random.random() > 0.5:
                noise = torch.randn_like(feat) * 0.1
                feat = feat + noise

        return feat, label

# ── Spatial Attention Pooling ──
class SpatialAttentionPool(nn.Module):
    """学习哪些spatial tokens更重要，替代粗暴mean"""
    def __init__(self, d_in=2048):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: (B, 60, 64, 2048)
        scores = self.attn(x)           # (B, 60, 64, 1)
        weights = torch.softmax(scores, dim=2)  # softmax over 64 tokens
        pooled = (x * weights).sum(dim=2)       # (B, 60, 2048)
        return pooled

# ── 改进版Vision Temporal Transformer ──
class VisionTemporalTransformerV2(nn.Module):
    def __init__(self, d_in=2048, d_model=256, n_frames=60,
                 nhead=8, num_layers=2, dropout=0.3, num_classes=27):
        super().__init__()

        # Spatial attention pooling (替代mean)
        self.spatial_pool = SpatialAttentionPool(d_in)

        # 投影: 2048 → 256
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames + 1, d_model) * 0.02)

        # Temporal Transformer (2层，防过拟合)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.4),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, return_tokens=False):
        """
        x: (B, 60, 64, 2048)
        """
        B = x.shape[0]

        # Spatial attention pooling
        x = self.spatial_pool(x)  # (B, 60, 2048)

        # 投影
        x = self.proj(x)  # (B, 60, 256)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 61, 256)
        x = x + self.pos_embed

        # Temporal transformer
        x = self.transformer(x)  # (B, 61, 256)

        if return_tokens:
            return x[:, 1:, :]  # (B, 60, 256)

        cls_out = x[:, 0, :]
        return self.head(cls_out)

# ── Mixup ──
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    B = x.size(0)
    index = torch.randperm(B).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ── 训练 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = VisionTokenDatasetV2(vision_cache, train=True)
test_ds = VisionTokenDatasetV2(vision_cache, train=False)

# batch_size降到16 (因为保留了64个spatial tokens，显存更大)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

model = VisionTemporalTransformerV2().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n模型参数: {total_params:,}")
print(f"设备: {device}")
print(f"改进: Spatial Attn Pool + 数据增强 + Mixup + 2层Transformer")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Test Acc':>8} | {'Best':>6}")
print("-" * 45)

best_acc = 0
best_state = None

for epoch in range(200):
    # Train with mixup
    model.train()
    total_loss = 0
    for feats, labels in train_loader:
        feats, labels = feats.to(device), labels.to(device)

        # Mixup
        feats_mix, y_a, y_b, lam = mixup_data(feats, labels, alpha=0.2)
        logits = model(feats_mix)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # Eval every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, labels in test_loader:
                feats = feats.to(device)
                preds = model(feats).argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " ★"

        print(f"  {epoch+1:4d}  | {total_loss/len(train_loader):10.4f} | {acc:7.1%} | {best_acc:.1%}{marker}")

# ── 保存 ──
save_path = "/content/drive/MyDrive/utd_mhad/vision_temporal_transformer_v2.pt"
torch.save({
    'model_state': best_state,
    'd_model': D_MODEL,
    'best_acc': best_acc,
}, save_path)

print(f"\n{'='*50}")
print(f"✅ Step 6b 完成!")
print(f"   V2 (Spatial Attn + Augment + Mixup): {best_acc:.1%}")
print(f"   V1 (Simple mean pool):               58.6%")
print(f"   Step 5 Temporal Attention:            53.3%")
print(f"   Midterm ResNet3D:                     72.8%")
print(f"   模型已保存: {save_path}")

# ============================================================
# ============================================================
# Step 6c: V1架构 + V2的增强技巧
# 用spatial mean (已验证有效) + 加数据增强和mixup
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

print("Step 6c: V1 + 数据增强 + Mixup")
print("="*50)

vision_cache = torch.load(
    "/content/drive/MyDrive/utd_mhad/vision_tokens_qwen25.pt",
    map_location='cpu', weights_only=False)
print(f"加载了 {len(vision_cache)} 个视频")

D_MODEL = 256

class VisionTokenDatasetAug(Dataset):
    def __init__(self, vision_cache, train=True):
        self.train = train
        self.samples = []
        allowed = {1,3,5,7} if train else {2,4,6,8}
        for (action, subject, trial), tokens in vision_cache.items():
            if subject not in allowed:
                continue
            feat = tokens.mean(dim=1)  # (60, 2048) ← 保持V1的spatial mean
            self.samples.append((feat, action - 1))
        print(f"  {'Train' if train else 'Test'}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        feat = feat.float()

        if self.train:
            # 随机temporal shift
            shift = np.random.randint(-3, 4)
            feat = torch.roll(feat, shifts=shift, dims=0)
            # 随机drop帧
            if np.random.random() > 0.5:
                drop = np.random.choice(60, size=5, replace=False)
                feat[drop] = 0
            # 高斯噪声
            if np.random.random() > 0.5:
                feat = feat + torch.randn_like(feat) * 0.1

        return feat, label

# 用和V1完全一样的模型结构
class VisionTemporalTransformer(nn.Module):
    def __init__(self, d_in=2048, d_model=256, n_frames=60,
                 nhead=8, num_layers=2, dropout=0.3, num_classes=27):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.4),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, return_tokens=False):
        B = x.shape[0]
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        if return_tokens:
            return x[:, 1:, :]
        return self.head(x[:, 0, :])

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1-lam) * x[index], y, y[index], lam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ds = VisionTokenDatasetAug(vision_cache, train=True)
test_ds = VisionTokenDatasetAug(vision_cache, train=False)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = VisionTemporalTransformer().to(device)
print(f"参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Test Acc':>8} | {'Best':>6}")
print("-" * 45)

best_acc = 0
best_state = None

for epoch in range(200):
    model.train()
    total_loss = 0
    for feats, labels in train_loader:
        feats, labels = feats.to(device), labels.to(device)
        feats_mix, y_a, y_b, lam = mixup_data(feats, labels)
        logits = model(feats_mix)
        loss = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feats, labels in test_loader:
                feats = feats.to(device)
                preds = model(feats).argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        acc = accuracy_score(all_labels, all_preds)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " ★"
        print(f"  {epoch+1:4d}  | {total_loss/len(train_loader):10.4f} | {acc:7.1%} | {best_acc:.1%}{marker}")

save_path = "/content/drive/MyDrive/utd_mhad/vision_temporal_transformer_v3.pt"
torch.save({'model_state': best_state, 'd_model': D_MODEL, 'best_acc': best_acc}, save_path)

print(f"\n{'='*50}")
print(f"✅ Step 6c: {best_acc:.1%}")
print(f"   V1 (无增强):  58.6%")
print(f"   V2 (spatial attn): 失败")
print(f"   V3 (V1+增强):  {best_acc:.1%}")
print(f"   ResNet3D:      72.8%")
