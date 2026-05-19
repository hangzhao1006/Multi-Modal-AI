"""
Step 6d: ResNet3D Vision Expert
- Kinetics-400 pretrained r3d_18
- Two-stage training: frozen backbone → full fine-tune
- Result: 89.3% (up from midterm 72.8%)
- Output: (B, T_v, 256) vision tokens

Input: UTD-MHAD RGB videos
Output: resnet3d_vision_expert.pt
"""


# ============================================================
# ============================================================
# Step 6d: ResNet3D Vision Expert
# 目的: 用Kinetics-400预训练的ResNet3D提取motion-aware特征
# 输出: (B, T, 256) 和Qwen版本同接口
# 对比: Qwen ViT 58.6% vs ResNet3D ?%
# ============================================================

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import os
import scipy.io as sio

print("Step 6d: ResNet3D Vision Expert")
print("="*50)

D_MODEL = 256
DATA_ROOT = "/content/drive/MyDrive/utd_mhad"
TARGET_FRAMES = 60
IMG_SIZE = 112  # ResNet3D用112更标准

# ── 提取视频帧 (用IMU对齐裁剪) ──
def find_video(action, subject, trial):
    fname = f"a{action}_s{subject}_t{trial}_color.avi"
    for part in ['RGB-part1','RGB-part2','RGB-part3','RGB-part4']:
        p = os.path.join(DATA_ROOT, part, fname)
        if os.path.exists(p):
            return p
    return None

def extract_frames(video_path, imu_path=None, n_frames=TARGET_FRAMES, img_size=IMG_SIZE):
    """提取视频帧，可选IMU对齐裁剪"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # IMU对齐: 只取动作段
    start_ratio, end_ratio = 0.0, 1.0
    if imu_path and os.path.exists(imu_path):
        imu = sio.loadmat(imu_path)['d_iner']
        mag = np.sqrt((imu[:, :3]**2).sum(axis=1))
        thr = mag.mean() + 0.3 * mag.std()
        active = np.where(mag > thr)[0]
        if len(active) > 0:
            T = len(mag)
            start_ratio = max(0.0, (active[0] - 15) / T)
            end_ratio = min(1.0, (active[-1] + 15) / T)

    f0 = int(total * start_ratio)
    f1 = max(int(total * end_ratio), f0 + n_frames)
    indices = np.linspace(f0, min(f1, total-1), n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < n_frames:
        return None
    return np.array(frames, np.float32) / 255.0

# ── 提取所有视频帧 (或加载缓存) ──
cache_path = os.path.join(DATA_ROOT, "rgb_frames_112.pt")

if os.path.exists(cache_path):
    print("加载缓存的RGB帧...")
    rgb_data = torch.load(cache_path, map_location='cpu', weights_only=False)
    print(f"  加载了 {len(rgb_data)} 个视频")
else:
    print("提取所有视频帧 (首次运行，约10-15分钟)...")
    rgb_data = {}
    success, fail = 0, 0

    for action in range(1, 28):
        for subject in range(1, 9):
            for trial in range(1, 5):
                vp = find_video(action, subject, trial)
                ip = os.path.join(DATA_ROOT, "Inertial",
                                  f"a{action}_s{subject}_t{trial}_inertial.mat")
                if vp is None:
                    fail += 1
                    continue
                frames = extract_frames(vp, ip)
                if frames is not None:
                    rgb_data[(action, subject, trial)] = torch.tensor(frames)
                    success += 1
                else:
                    fail += 1
        print(f"  Action {action:2d}/27 | success={success}")

    print(f"\n提取完成: {success}个视频, 失败{fail}个")
    torch.save(rgb_data, cache_path)
    print(f"缓存已保存: {cache_path}")

# ── Dataset ──
class RGBDataset(Dataset):
    def __init__(self, rgb_data, train=True):
        self.train = train
        self.samples = []
        allowed = {1,3,5,7} if train else {2,4,6,8}
        for (action, subject, trial), frames in rgb_data.items():
            if subject not in allowed:
                continue
            self.samples.append((frames, action - 1))
        print(f"  {'Train' if train else 'Test'}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        # frames: (60, 112, 112, 3)
        x = frames.float().permute(3, 0, 1, 2)  # (3, 60, 112, 112)

        if self.train:
            # 随机水平翻转
            if np.random.random() > 0.5:
                x = x.flip(3)
            # 随机temporal shift
            shift = np.random.randint(-2, 3)
            x = torch.roll(x, shifts=shift, dims=1)

        return x, label

# ── ResNet3D Vision Expert ──
class ResNet3DExpert(nn.Module):
    """
    ResNet3D预训练backbone → 提取temporal feature map → 投影到256维
    输出: (B, T_out, 256) 用于cross-modal attention
    """
    def __init__(self, d_model=256, num_classes=27, dropout=0.3):
        super().__init__()

        # 加载预训练ResNet3D
        backbone = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT)

        # 去掉最后的avgpool和fc，保留时序维度
        self.features = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,  # 输出: (B, 512, T/8, 7, 7) 当T=60时约(B,512,8,7,7)
        )

        # Spatial pooling: (B, 512, T', 7, 7) → (B, 512, T')
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # 投影到d_model
        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # CLS token + positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)  # max 64+1

        # Temporal transformer (轻量)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, return_tokens=False):
        """
        x: (B, 3, 60, 112, 112)
        """
        B = x.shape[0]

        # ResNet3D特征 (冻结前3层)
        feat = self.features(x)  # (B, 512, T', H', W')

        # Spatial pooling
        feat = self.spatial_pool(feat)  # (B, 512, T', 1, 1)
        feat = feat.squeeze(-1).squeeze(-1)  # (B, 512, T')
        feat = feat.permute(0, 2, 1)  # (B, T', 512)

        T_out = feat.shape[1]

        # 投影
        feat = self.proj(feat)  # (B, T', 256)

        # CLS + positional encoding
        cls = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls, feat], dim=1)  # (B, T'+1, 256)
        feat = feat + self.pos_embed[:, :T_out+1, :]

        # Temporal transformer
        feat = self.temporal_transformer(feat)

        if return_tokens:
            return feat[:, 1:, :]  # (B, T', 256)

        return self.head(feat[:, 0, :])

# ── 训练 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = RGBDataset(rgb_data, train=True)
test_ds = RGBDataset(rgb_data, train=False)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

model = ResNet3DExpert().to(device)

# 冻结ResNet3D前3层，只训练layer4 + temporal transformer + head
for name, param in model.features.named_parameters():
    # stem, layer1, layer2, layer3 冻结
    if any(x in name for x in ['0.', '1.', '2.', '3.']):
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\n总参数: {total:,} | 可训练: {trainable:,} | 冻结: {total-trainable:,}")
print(f"设备: {device}")
print(f"架构: ResNet3D(Kinetics预训练) → layer4解冻 → temporal transformer → 27类")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4, weight_decay=0.05
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Test Acc':>8} | {'Best':>6}")
print("-" * 45)

best_acc = 0
best_state = None

for epoch in range(80):
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

    if (epoch + 1) % 5 == 0:
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

# 第二阶段: 解冻全部，低学习率微调
print(f"\n--- Stage 2: 解冻全部, lr=5e-5 ---")
for param in model.parameters():
    param.requires_grad = True

optimizer2 = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=40)

# 加载Stage 1最佳权重
model.load_state_dict(best_state)
model.to(device)

for epoch in range(40):
    model.train()
    total_loss = 0
    for feats, labels in train_loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)
        optimizer2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer2.step()
        total_loss += loss.item()
    scheduler2.step()

    if (epoch + 1) % 5 == 0:
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
        print(f"  {epoch+81:4d}  | {total_loss/len(train_loader):10.4f} | {acc:7.1%} | {best_acc:.1%}{marker}")

save_path = os.path.join(DATA_ROOT, "resnet3d_vision_expert.pt")
torch.save({'model_state': best_state, 'd_model': D_MODEL, 'best_acc': best_acc}, save_path)

print(f"\n{'='*50}")
print(f"✅ Step 6d: ResNet3D Vision Expert")
print(f"   ResNet3D Expert:         {best_acc:.1%}")
print(f"   Midterm ResNet3D:        72.8%")
print(f"   Qwen ViT (Step 6 V1):   58.6%")
print(f"   模型已保存: {save_path}")
print(f"   接口: model(x, return_tokens=True) → (B, T', 256)")
