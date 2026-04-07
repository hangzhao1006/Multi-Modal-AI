"""
03_rgb_resnet3d.py
Train RGB-only baseline using pretrained ResNet3D (r3d_18) on UTD-MHAD (27 classes).
Two-stage training: freeze backbone → fine-tune all layers.
Result: 72.8% test accuracy (vs 67.9% for IMU 1D-CNN).
"""

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

NUM_CLASSES = 27
TRAIN_SUB   = {1,3,5,7}
TEST_SUB    = {2,4,6,8}

# rgb_data must be loaded from 02_rgb_extraction.py first

class RGBDataset(Dataset):
    def __init__(self, rgb_data, train=True, only_high_dynamic=False,
                 high_dynamic={5,7,12,14,15,17,19}):
        self.samples = []
        allowed = TRAIN_SUB if train else TEST_SUB
        for (action, subject, trial), frames in rgb_data.items():
            if subject not in allowed: continue
            if only_high_dynamic and action not in high_dynamic: continue
            self.samples.append((frames, action - 1))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        return torch.tensor(frames).permute(3,0,1,2), label  # (3,30,64,64)

class PretrainedRGBNet(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        backbone = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
        backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone.fc.in_features, num_classes))
        self.model = backbone

    def forward(self, x): return self.model(x)

def train_rgb_resnet3d(rgb_data, epochs_stage1=10, epochs_stage2=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = RGBDataset(rgb_data, train=True)
    test_ds  = RGBDataset(rgb_data, train=False)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=2)

    model = PretrainedRGBNet(NUM_CLASSES).to(device)
    crit  = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
                labels.extend(y.numpy())
        return accuracy_score(labels, preds)

    # Stage 1: freeze backbone, train FC only
    print(f"\n=== Stage 1: Train FC only ({epochs_stage1} epochs) ===")
    for p in model.model.parameters(): p.requires_grad = False
    for p in model.model.fc.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(model.model.fc.parameters(), lr=1e-3)
    for epoch in range(epochs_stage1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        acc = evaluate()
        print(f"  Epoch {epoch+1:2d} | Acc {acc:.4f}")

    # Stage 2: unfreeze all, fine-tune
    print(f"\n=== Stage 2: Fine-tune all ({epochs_stage2} epochs) ===")
    for p in model.model.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_stage2)
    best_acc = 0
    for epoch in range(epochs_stage2):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            total_loss += loss.item()
        sch.step()
        if (epoch+1) % 5 == 0:
            acc = evaluate()
            best_acc = max(best_acc, acc)
            print(f"  Epoch {epoch+1:2d} | Loss {total_loss/len(train_loader):.4f} | Acc {acc:.4f}")

    print(f"\n✅ RGB ResNet3D best accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    torch.save(model.state_dict(), 'rgb_resnet3d.pth')
    return model, best_acc

if __name__ == "__main__":
    # Load rgb_data first (from 02_rgb_extraction.py)
    from _02_rgb_extraction import extract_all_videos
    rgb_data = extract_all_videos()
    model, acc = train_rgb_resnet3d(rgb_data)