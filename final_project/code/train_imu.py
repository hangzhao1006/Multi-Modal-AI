"""
Standalone IMU-only training for the PG-MoE IMU expert (Xiaoyang).

Goal: beat midterm baseline (67.9%) and produce `imu_expert.pt` to be loaded
by the joint PG-MoE training run.

Run on Colab:
    !python train_imu.py --data_root /content/drive/MyDrive/utd_mhad \
                         --save_dir  /content/drive/MyDrive/pgmoe_ckpt
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from data.dataset import IMUDataset, NUM_CLASSES
from models.imu_expert import IMUClassifier


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    return accuracy_score(labels, preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = IMUDataset(args.data_root, train=True)
    test_ds = IMUDataset(args.data_root, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = IMUClassifier(num_classes=NUM_CLASSES, d_model=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Train {len(train_ds)} | Test {len(test_ds)} | Params {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0.0, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        sched.step()
        train_loss /= len(train_ds)

        acc = evaluate(model, test_loader, device)
        flag = ""
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "imu_classifier_best.pt"))
            torch.save(model.encoder.state_dict(),
                       os.path.join(args.save_dir, "imu_expert.pt"))
            flag = "  *"
        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
              f"test_acc {acc:.4f} | {time.time()-t0:.1f}s{flag}")

    print(f"\nBest test acc: {best_acc:.4f} ({best_acc*100:.2f}%) at epoch {best_epoch}")
    print(f"Encoder weights saved to {os.path.join(args.save_dir, 'imu_expert.pt')}")


if __name__ == "__main__":
    main()
