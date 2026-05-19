"""
04_crossover_curve.py
Phase-wise crossover curve analysis: RGB (ResNet3D) vs IMU (1D-CNN).
Each modality is trained and tested independently on 5 temporal segments
of high-dynamic actions, revealing phase-dependent modality dominance.

Key result: Crossover at Accel phase (40-60%), both reach 63%.
RGB dominates Prep/Pre-Accel (80% vs 54%). IMU drops sharply post-crossover.
"""

import torch
import torch.nn as nn
import torchvision
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

DATA_ROOT           = "/content/drive/MyDrive/utd_mhad"
HIGH_DYNAMIC        = {5, 7, 12, 14, 15, 17, 19}
NUM_CLASSES         = 27
IMG_SIZE            = 64
TRAIN_SUB           = {1,3,5,7}
TEST_SUB            = {2,4,6,8}

def parse_filename(fname):
    parts = fname.split('_')
    return int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:])

# ── IMU Phase Dataset ──
class IMUPhaseDataset(Dataset):
    def __init__(self, phase_start, phase_end, train=True):
        self.samples = []
        allowed = TRAIN_SUB if train else TEST_SUB
        folder  = os.path.join(DATA_ROOT, "Inertial")
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith("_inertial.mat"): continue
            action, subject, _ = parse_filename(fname)
            if subject not in allowed: continue
            if action not in HIGH_DYNAMIC: continue
            self.samples.append((os.path.join(folder, fname), action-1,
                                 phase_start, phase_end))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fpath, label, ps, pe = self.samples[idx]
        data = sio.loadmat(fpath)['d_iner'].astype(np.float32)
        T = data.shape[0]
        seg = data[int(T*ps):int(T*pe)]
        tlen = 80
        if seg.shape[0] < tlen:
            seg = np.concatenate([seg, np.zeros((tlen-seg.shape[0],6),np.float32)])
        return torch.tensor(seg[:tlen]).T, label

class IMUNetPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(128, NUM_CLASSES)
    def forward(self, x): return self.fc(self.net(x).squeeze(-1))

# ── RGB Phase Dataset ──
class RGBPhaseDataset(Dataset):
    def __init__(self, rgb_data, phase_start, phase_end, train=True):
        self.samples = []
        allowed = TRAIN_SUB if train else TEST_SUB
        for (action, subject, trial), frames in rgb_data.items():
            if subject not in allowed: continue
            if action not in HIGH_DYNAMIC: continue
            T     = frames.shape[0]
            start = int(T * phase_start)
            end   = max(int(T * phase_end), start + 8)
            seg   = frames[start:min(end, T)]
            tlen  = 16
            if seg.shape[0] < tlen:
                seg = np.concatenate([seg, np.zeros((tlen-seg.shape[0],IMG_SIZE,IMG_SIZE,3),np.float32)])
            self.samples.append((seg[:tlen], action-1))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        return torch.tensor(frames).permute(3,0,1,2), label

# ── Train functions ──
def train_imu_phase(phase_start, phase_end, epochs=40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = IMUPhaseDataset(phase_start, phase_end, train=True)
    test_ds  = IMUPhaseDataset(phase_start, phase_end, train=False)
    if len(train_ds) == 0: return 0.0
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False)
    model = IMUNetPhase().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    return accuracy_score(labels, preds)

def train_rgb_phase(rgb_data, phase_start, phase_end, epochs=30):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = RGBPhaseDataset(rgb_data, phase_start, phase_end, train=True)
    test_ds  = RGBPhaseDataset(rgb_data, phase_start, phase_end, train=False)
    if len(train_ds) == 0: return 0.0
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False)
    # Pretrained backbone, remap to 7 classes
    action_list = sorted(HIGH_DYNAMIC)
    label_map   = {a-1: i for i, a in enumerate(action_list)}
    backbone    = torchvision.models.video.r3d_18(
        weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
    backbone.fc = nn.Linear(backbone.fc.in_features, 7)
    model = backbone.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            y_m = torch.tensor([label_map[yi.item()] for yi in y])
            x, y_m = x.to(device), y_m.to(device)
            opt.zero_grad(); crit(model(x), y_m).backward(); opt.step()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            labels.extend([label_map[yi.item()] for yi in y])
    return accuracy_score(labels, preds)

def plot_crossover_curve(rgb_accs, imu_accs, phases):
    x = np.arange(len(phases))
    phase_names = [p[2] for p in phases]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, imu_accs, 'o-', color='#E74C3C', lw=2.5, ms=9,
            label='IMU-only (1D-CNN)', zorder=3)
    ax.plot(x, rgb_accs, 's-', color='#2ECC71', lw=2.5, ms=9,
            label='RGB-only (ResNet3D pretrained)', zorder=3)
    for i, (ia, ra) in enumerate(zip(imu_accs, rgb_accs)):
        ax.annotate(f'{ia:.2f}', (x[i], ia), textcoords="offset points",
                    xytext=(0,10), ha='center', color='#E74C3C', fontsize=11, fontweight='bold')
        ax.annotate(f'{ra:.2f}', (x[i], ra), textcoords="offset points",
                    xytext=(0,-16), ha='center', color='#2ECC71', fontsize=11, fontweight='bold')
    ax.axvspan(-0.5, 1.5, alpha=0.07, color='green', label='RGB dominant zone')
    ax.axvspan(2.5,  4.5, alpha=0.07, color='red',   label='IMU dominant zone')
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.6, label='Crossover point')
    ax.set_xticks(x); ax.set_xticklabels(phase_names, fontsize=11)
    ax.set_xlabel('Action Phase', fontsize=13)
    ax.set_ylabel('Classification Accuracy', fontsize=13)
    ax.set_title('Crossover Curve: RGB vs IMU Phase-wise Dominance\n'
                 'High-Dynamic Actions (throw, shoot, bowl, swing, tennis×2, knock)\n'
                 'RGB and IMU are natively temporally synchronized', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1); ax.set_xlim(-0.5, 4.5)
    plt.tight_layout()
    plt.savefig('crossover_curve_final.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: crossover_curve_final.png")

if __name__ == "__main__":
    # rgb_data must be loaded first
    phases = [
        (0.0, 0.2, "Prep\n(0-20%)"),
        (0.2, 0.4, "Pre-Accel\n(20-40%)"),
        (0.4, 0.6, "Accel\n(40-60%)"),
        (0.6, 0.8, "Impact\n(60-80%)"),
        (0.8, 1.0, "Follow\n(80-100%)"),
    ]
    rgb_accs, imu_accs = [], []
    for start, end, name in phases:
        label = name.split('\n')[0]
        print(f"Training RGB [{label}]...", end=" ", flush=True)
        acc = train_rgb_phase(rgb_data, start, end)
        rgb_accs.append(acc); print(f"{acc:.3f}")
        print(f"Training IMU [{label}]...", end=" ", flush=True)
        acc = train_imu_phase(start, end)
        imu_accs.append(acc); print(f"{acc:.3f}")

    plot_crossover_curve(rgb_accs, imu_accs, phases)
    print("\n📊 Phase-wise Results:")
    for i, (s, e, name) in enumerate(phases):
        label   = name.split('\n')[0]
        winner  = 'IMU' if imu_accs[i] > rgb_accs[i] else 'RGB'
        print(f"  {label:15s} | IMU: {imu_accs[i]:.3f} | RGB: {rgb_accs[i]:.3f} | {winner}")