"""Reusable training loops for PG-MoE variants.

Putting the loops here (instead of in notebook cells) means future hyperparameter
or loss-function tweaks only need:

    !cd /path/to/repo && git pull
    import importlib, models.train_loops
    importlib.reload(models.train_loops)
    history, best = models.train_loops.train_pgmoe_late(model, ...)

No cell re-pasting, no kernel restart, vision_cache stays in memory.
"""

import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from .pgmoe import FocalLoss, PGMoELate


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for v, i, y in loader:
            logits = model(v.to(device), i.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    return accuracy_score(labels, preds), np.array(preds), np.array(labels)


def train_pgmoe_late(
    model,
    train_loader,
    test_loader,
    save_dir,
    device,
    epochs=30,
    lr_vision_head=1e-3,
    lr_phase=1e-3,
    lr_imu=1e-5,
    freeze_imu=True,
    weight_decay=1e-4,
    focal_gamma=2.0,
    alpha_var_w=0.05,
    ckpt_name="pgmoe_late_best.pt",
    verbose=True,
):
    """Train a PGMoELate model. Returns (history dict, best_acc, best_epoch)."""

    if freeze_imu:
        for p in model.imu_classifier.parameters():
            p.requires_grad = False

    groups = [
        {"params": model.vision_head.parameters(),       "lr": lr_vision_head},
        {"params": model.phase_arbitrator.parameters(),  "lr": lr_phase},
    ]
    if not freeze_imu:
        groups.append({"params": model.imu_classifier.parameters(), "lr": lr_imu})

    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(gamma=focal_gamma)

    if verbose:
        print(f"Training PGMoELate | epochs={epochs} | freeze_imu={freeze_imu} | alpha_var_w={alpha_var_w}")
        for k, g in enumerate(optimizer.param_groups):
            print(f"  group {k}: lr={g['lr']:.1e}, params={sum(p.numel() for p in g['params']):,}")

    best_acc, best_epoch = 0.0, -1
    history = {"cls": [], "var": [], "acc": [], "alpha_mean": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        sum_cls = sum_var = sum_a = 0.0
        n_seen = 0
        for v, i, y in train_loader:
            v, i, y = v.to(device), i.to(device), y.to(device)
            optimizer.zero_grad()
            logits, alpha = model(v, i, return_alpha=True)
            cls_loss = criterion(logits, y)
            var_loss = -alpha.var(dim=1).mean()
            loss = cls_loss + alpha_var_w * var_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bs = v.size(0)
            sum_cls += cls_loss.item() * bs
            sum_var += var_loss.item() * bs
            sum_a   += alpha.mean().item() * bs
            n_seen += bs
        scheduler.step()

        avg_cls, avg_var, avg_a = sum_cls / n_seen, sum_var / n_seen, sum_a / n_seen
        acc, _, _ = evaluate(model, test_loader, device)
        history["cls"].append(avg_cls)
        history["var"].append(avg_var)
        history["acc"].append(acc)
        history["alpha_mean"].append(avg_a)

        flag = ""
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, ckpt_name))
            flag = "  *"
        if verbose:
            print(f"Epoch {epoch:3d} | cls {avg_cls:.4f} | var {avg_var:+.4f} | "
                  f"α̅ {avg_a:.3f} | acc {acc:.4f} | {time.time()-t0:.1f}s{flag}")

    if verbose:
        print(f"\nBest test acc: {best_acc:.4f} ({best_acc*100:.2f}%) at epoch {best_epoch}")
    return history, best_acc, best_epoch


def run_pgmoe_late(
    train_loader,
    test_loader,
    save_dir,
    device,
    imu_clf_ckpt,
    num_classes=27,
    d_model=256,
    T_i=12,
    dropout=0.2,
    **train_kwargs,
):
    """One-call helper: build PGMoELate + load pretrained IMU + train.

    Returns (model, history, best_acc, best_epoch).
    """
    model = PGMoELate(T_i=T_i, d_model=d_model, num_classes=num_classes, dropout=dropout).to(device)
    model.load_pretrained_imu_classifier(imu_clf_ckpt, device=device)
    history, best_acc, best_epoch = train_pgmoe_late(
        model, train_loader, test_loader, save_dir, device, **train_kwargs
    )
    return model, history, best_acc, best_epoch
