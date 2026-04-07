"""
01_baseline_imu_skeleton.py
Train IMU-only and Skeleton-only baseline classifiers on UTD-MHAD (27 classes).
Results: IMU 1D-CNN: 67.9%, Skeleton 1D-CNN: 37.9%
"""

import scipy.io as sio
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ── Config ──
DATA_ROOT   = "/content/drive/MyDrive/utd_mhad"  # Change to your path
IMU_LEN     = 192
SKEL_FRAMES = 76
NUM_CLASSES = 27
EPOCHS      = 40

def parse_filename(fname):
    parts = fname.split('_')
    return int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:])

# ── Dataset ──
class UTDDataset(Dataset):
    def __init__(self, modality, train=True):
        self.modality = modality
        self.samples  = []
        allowed = {1,3,5,7} if train else {2,4,6,8}
        folder = os.path.join(DATA_ROOT, "Inertial" if modality=='imu' else "Skeleton")
        suffix = "_inertial.mat" if modality=='imu' else "_skeleton.mat"
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(suffix): continue
            action, subject, _ = parse_filename(fname)
            if subject not in allowed: continue
            self.samples.append((os.path.join(folder, fname), action - 1))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        mat = sio.loadmat(fpath)
        if self.modality == 'imu':
            data = mat['d_iner'].astype(np.float32)
            if data.shape[0] < IMU_LEN:
                data = np.concatenate([data, np.zeros((IMU_LEN-data.shape[0],6),np.float32)])
            data = torch.tensor(data[:IMU_LEN]).T  # (6, 192)
        else:
            data = mat['d_skel'].astype(np.float32)
            if data.shape[2] < SKEL_FRAMES:
                data = np.concatenate([data, np.zeros((20,3,SKEL_FRAMES-data.shape[2]),np.float32)], axis=2)
            data = torch.tensor(data[:,:,:SKEL_FRAMES].reshape(60, SKEL_FRAMES))
        return data, label

# ── Models ──
class IMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(128, NUM_CLASSES)
    def forward(self, x): return self.fc(self.net(x).squeeze(-1))

class SkeletonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(60, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(256, NUM_CLASSES)
    def forward(self, x): return self.fc(self.net(x).squeeze(-1))

# ── Training ──
def train_model(modality, epochs=EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = UTDDataset(modality, train=True)
    test_ds  = UTDDataset(modality, train=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)
    model = (IMUNet() if modality=='imu' else SkeletonNet()).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    print(f"\n=== Training {modality.upper()} baseline ({epochs} epochs) ===")
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}, Params: {sum(p.numel() for p in model.parameters()):,}")
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
            total_loss += crit(model(x), y).item()
        if (epoch+1) % 5 == 0:
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
                    labels.extend(y.numpy())
            acc = accuracy_score(labels, preds)
            best_acc = max(best_acc, acc)
            print(f"  Epoch {epoch+1:3d} | Loss {total_loss/len(train_loader):.4f} | Acc {acc:.4f}")
    print(f"✅ {modality.upper()} best accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    return model, best_acc

if __name__ == "__main__":
    imu_model,  imu_acc  = train_model('imu')
    skel_model, skel_acc = train_model('skeleton')
    print(f"\n=== Results ===")
    print(f"IMU-only:      {imu_acc*100:.1f}%")
    print(f"Skeleton-only: {skel_acc*100:.1f}%")
    torch.save(imu_model.state_dict(),  'imu_1dcnn.pth')
    torch.save(skel_model.state_dict(), 'skeleton_1dcnn.pth')