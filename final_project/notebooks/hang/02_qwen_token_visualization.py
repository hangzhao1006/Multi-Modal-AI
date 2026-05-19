"""
Step 4: Qwen ViT Token Visualization
- Token activation heatmaps across frames
- Frame-to-frame cosine similarity matrix
- Token energy across different actions
- Patch-level activation overlay
- ViT patch anatomy (grid + heatmap + token vectors)

Input: vision_tokens_qwen25.pt
Output: Multiple visualization figures
"""


# ============================================================
# ============================================================
# Step 4: Vision Token可视化 (English版，报告可用)
# 目的: 理解Qwen ViT tokens的特征质量
# 输入: vision_tokens_qwen25.pt
# 输出: 3张图 → 可直接放进NeurIPS报告
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── 加载tokens ──
print("Step 4: Vision Token Visualization (English)")
vision_cache = torch.load(
    "/content/drive/MyDrive/utd_mhad/vision_tokens_qwen25.pt",
    map_location='cpu', weights_only=False)

# 选一个throw视频
key = (5, 1, 1)  # a5_s1_t1 = throw
tokens = vision_cache[key]  # (60, 64, 2048)
print(f"Video a5_s1_t1 (throw)")
print(f"  Tokens shape: {tokens.shape}")
print(f"  mean={tokens.mean():.4f}, std={tokens.std():.4f}")
print(f"  min={tokens.min():.4f}, max={tokens.max():.4f}")

# ============================================================
# 图1: Token Activation Heatmap (6 key frames)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
frame_indices = [0, 12, 24, 36, 48, 59]
frame_labels = [
    'Frame 0\n(Start)', 'Frame 12\n(Early)', 'Frame 24\n(Mid-early)',
    'Frame 36\n(Mid-late)', 'Frame 48\n(Late)', 'Frame 59\n(End)'
]

for i, (fi, label) in enumerate(zip(frame_indices, frame_labels)):
    ax = axes[i // 3, i % 3]
    data = tokens[fi].numpy()  # (64, 2048)
    # 只显示前100个维度，更清晰
    im = ax.imshow(data[:, :100], aspect='auto', cmap='RdBu_r',
                   norm=TwoSlopeNorm(vmin=data[:,:100].min(), vcenter=0,
                                     vmax=data[:,:100].max()))
    ax.set_title(label, fontsize=12)
    ax.set_ylabel('Token ID (0-63)', fontsize=9)
    ax.set_xlabel('Dimension (first 100)', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('Throw Action: Vision Token Activation Patterns Across Frames\n'
             'Each frame produces 64 tokens × 2048 dimensions from Qwen2.5-VL ViT',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_token_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_token_heatmap.png")

# ============================================================
# 图2: Frame-to-Frame Cosine Similarity Matrix
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

# 每帧: mean over 64 tokens → (60, 2048), 然后计算cosine similarity
frame_feats = tokens.mean(dim=1)  # (60, 2048)
norms = frame_feats.norm(dim=1, keepdim=True)
frame_feats_normed = frame_feats / norms
sim_matrix = (frame_feats_normed @ frame_feats_normed.T).numpy()

im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.80, vmax=1.0)
ax.set_xlabel('Frame Index', fontsize=12)
ax.set_ylabel('Frame Index', fontsize=12)
ax.set_title('Throw Action: Inter-Frame Cosine Similarity of Vision Tokens\n'
             'Diagonal = high similarity (adjacent frames); off-diagonal drops = motion events',
             fontsize=11, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Cosine Similarity', fontsize=11)

# 标注动作区域
ax.annotate('Motion\nevent', xy=(27, 27), fontsize=9, color='white',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

plt.tight_layout()
plt.savefig('fig_frame_similarity.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_frame_similarity.png")

# ============================================================
# 图3: Token Energy (L2 norm) Across Frames for Different Actions
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

action_names = {
    (5,1,1): 'Throw', (6,1,1): 'Basketball Shoot',
    (12,1,1): 'Bowling', (14,1,1): 'Baseball Swing'
}
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E53935']

for (key, name), color in zip(action_names.items(), colors):
    if key not in vision_cache:
        # 尝试其他subject
        for s in range(1, 9):
            alt_key = (key[0], s, 1)
            if alt_key in vision_cache:
                key = alt_key
                break

    if key in vision_cache:
        t = vision_cache[key]  # (60, 64, 2048)
        energy = t.mean(dim=1).norm(dim=1).numpy()  # (60,)
        ax.plot(range(60), energy, label=name, color=color, linewidth=1.5, alpha=0.85)

ax.set_xlabel('Frame Index (0=start, 59=end)', fontsize=12)
ax.set_ylabel('Token Energy (L2 norm)', fontsize=12)
ax.set_title('Vision Token Energy Across Action Duration\n'
             'Energy variation reflects visual change intensity per frame',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 59)

plt.tight_layout()
plt.savefig('fig_token_energy.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_token_energy.png")

print(f"\n{'='*50}")
print("✅ Step 4 完成: 3张可视化已保存")
print("   fig_token_heatmap.png    → 报告Figure X")
print("   fig_frame_similarity.png → 报告Figure X")
print("   fig_token_energy.png     → 报告Figure X")
print("   全部英文标签，可直接用于NeurIPS格式报告")

# ============================================================
# ============================================================
# Step 4b: Patch可视化 - 展示ViT如何把一帧切成64个patch
# 目的: 理解每个token对应图片哪个区域，哪些区域激活最强
# 输入: 一个视频的原始帧 + 对应的vision tokens
# 输出: 原始帧 + patch网格 + 激活热力图叠加
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image as PILImage

print("Step 4b: Patch Visualization")
print("="*50)

# ── resize辅助函数 ──
def resize_heatmap(energy_map_8x8, target_size=224):
    """把8×8的energy map放大到224×224"""
    img = PILImage.fromarray(energy_map_8x8.astype(np.float32))
    img = img.resize((target_size, target_size), PILImage.BICUBIC)
    return np.array(img)

# ── 加载vision tokens ──
vision_cache = torch.load(
    "/content/drive/MyDrive/utd_mhad/vision_tokens_qwen25.pt",
    map_location='cpu', weights_only=False)

# ── 读取原始视频帧 ──
DATA_ROOT = "/content/drive/MyDrive/utd_mhad"
import os

def find_video(action, subject, trial):
    fname = f"a{action}_s{subject}_t{trial}_color.avi"
    for part in ['RGB-part1','RGB-part2','RGB-part3','RGB-part4']:
        p = f"{DATA_ROOT}/{part}/{fname}"
        if os.path.exists(p):
            return p
    return None

def get_frame(video_path, frame_idx, total_frames=60):
    """提取视频的指定帧"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, total_frames, dtype=int)
    target = indices[frame_idx]
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None

# ── 选择throw动作的关键帧 ──
action, subject, trial = 5, 1, 1  # throw
video_path = find_video(action, subject, trial)
tokens = vision_cache[(action, subject, trial)]  # (60, 64, 2048)

frame_indices = [0, 15, 30, 45, 59]
frame_labels = ['Start', 'Preparation', 'Acceleration', 'Impact', 'End']

print(f"Video: a{action}_s{subject}_t{trial} (throw)")
print(f"Tokens shape: {tokens.shape}")
print(f"64 tokens = 8×8 patch grid")

# ============================================================
# 图1: 原始帧 + patch网格叠加
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, (fi, label) in enumerate(zip(frame_indices, frame_labels)):
    frame = get_frame(video_path, fi)
    if frame is None:
        continue

    ax = axes[i]
    frame_resized = cv2.resize(frame, (224, 224))
    ax.imshow(frame_resized)

    patch_size = 224 // 8
    for row in range(1, 8):
        ax.axhline(y=row * patch_size, color='white', linewidth=0.5, alpha=0.6)
    for col in range(1, 8):
        ax.axvline(x=col * patch_size, color='white', linewidth=0.5, alpha=0.6)

    ax.set_title(f'Frame {fi} ({label})\n8×8 = 64 patches', fontsize=11)
    ax.axis('off')

fig.suptitle('Throw Action: Original Frames with ViT Patch Grid (8×8 = 64 tokens per frame)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_patch_grid.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_patch_grid.png")

# ============================================================
# 图2: Patch激活热力图 - 哪些patch区域token激活最强
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, (fi, label) in enumerate(zip(frame_indices, frame_labels)):
    frame = get_frame(video_path, fi)
    if frame is None:
        continue
    frame_resized = cv2.resize(frame, (224, 224))

    axes[0, i].imshow(frame_resized)
    axes[0, i].set_title(f'Frame {fi} ({label})', fontsize=11)
    axes[0, i].axis('off')

    token_energy = tokens[fi].norm(dim=1).numpy()  # (64,)
    energy_map = token_energy.reshape(8, 8)
    energy_resized = resize_heatmap(energy_map)

    axes[1, i].imshow(frame_resized)
    im = axes[1, i].imshow(energy_resized, cmap='jet', alpha=0.5,
                            vmin=np.percentile(token_energy, 10),
                            vmax=np.percentile(token_energy, 90))
    axes[1, i].set_title(f'Token Activation Overlay', fontsize=10)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original Frame', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Token Energy\nOverlay', fontsize=12, fontweight='bold')

fig.suptitle('Throw Action: Patch-level Token Activation Heatmap\n'
             'Brighter = higher L2 norm = stronger visual feature encoding',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_patch_activation.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_patch_activation.png")

# ============================================================
# 图3: 单帧详细展示 - 每个patch的能量值标注
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

fi = 30
frame = get_frame(video_path, fi)
frame_resized = cv2.resize(frame, (224, 224))
token_energy = tokens[fi].norm(dim=1).numpy()
energy_map = token_energy.reshape(8, 8)

# 左: 原始帧 + patch网格
axes[0].imshow(frame_resized)
patch_size = 28
for r in range(1, 8):
    axes[0].axhline(y=r*patch_size, color='white', linewidth=0.8, alpha=0.7)
for c in range(1, 8):
    axes[0].axvline(x=c*patch_size, color='white', linewidth=0.8, alpha=0.7)
for r in range(8):
    for c in range(8):
        pid = r * 8 + c
        if pid % 9 == 0:
            axes[0].text(c*patch_size + patch_size//2, r*patch_size + patch_size//2,
                        f'{pid}', ha='center', va='center', fontsize=7,
                        color='yellow', fontweight='bold')
axes[0].set_title(f'Frame {fi}: Patch Grid (8×8)', fontsize=12)
axes[0].axis('off')

# 中: 能量热力图 (8×8 grid)
im = axes[1].imshow(energy_map, cmap='YlOrRd', interpolation='nearest')
for r in range(8):
    for c in range(8):
        axes[1].text(c, r, f'{energy_map[r,c]:.0f}', ha='center', va='center',
                    fontsize=7, color='black' if energy_map[r,c] < np.median(energy_map) else 'white')
axes[1].set_title(f'Token Energy Map (L2 norm)\nBrighter = stronger activation', fontsize=11)
axes[1].set_xlabel('Patch Column', fontsize=10)
axes[1].set_ylabel('Patch Row', fontsize=10)
plt.colorbar(im, ax=axes[1], shrink=0.8)

# 右: 叠加
energy_resized = resize_heatmap(energy_map)
axes[2].imshow(frame_resized)
axes[2].imshow(energy_resized, cmap='jet', alpha=0.45,
               vmin=np.percentile(token_energy, 5),
               vmax=np.percentile(token_energy, 95))
axes[2].set_title(f'Overlay: Where does ViT focus?', fontsize=12)
axes[2].axis('off')

fig.suptitle(f'Frame {fi} (Acceleration Phase): Detailed Patch Analysis\n'
             f'Qwen2.5-VL ViT encodes each 28×28 pixel patch as one 2048-dim token',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_patch_detail.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_patch_detail.png")

print(f"\n{'='*50}")
print("✅ Step 4b 完成: 3张patch可视化已保存")
print("   fig_patch_grid.png       → 展示ViT的patch切分方式")
print("   fig_patch_activation.png → 5帧的激活热力图对比")
print("   fig_patch_detail.png     → 单帧详细分析")

# ============================================================
# ============================================================
# Step 4e: ViT Patch Anatomy (完整版)
# 上排: 原始帧+patch网格 | 8×8能量热力图 | 激活叠加 | token噪音图(64×160)
# 下排: 3个patch的token向量line plot对比
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import cv2
from PIL import Image as PILImage
import os

print("Step 4e: ViT Patch Anatomy (complete)")
print("="*50)

def resize_heatmap(energy_map_8x8, target_size=224):
    img = PILImage.fromarray(energy_map_8x8.astype(np.float32))
    img = img.resize((target_size, target_size), PILImage.BICUBIC)
    return np.array(img)

vision_cache = torch.load(
    "/content/drive/MyDrive/utd_mhad/vision_tokens_qwen25.pt",
    map_location='cpu', weights_only=False)

DATA_ROOT = "/content/drive/MyDrive/utd_mhad"

def find_video(action, subject, trial):
    fname = f"a{action}_s{subject}_t{trial}_color.avi"
    for part in ['RGB-part1','RGB-part2','RGB-part3','RGB-part4']:
        p = f"{DATA_ROOT}/{part}/{fname}"
        if os.path.exists(p):
            return p
    return None

def get_frame(video_path, frame_idx, total_frames=60):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, total_frames, dtype=int)
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[frame_idx])
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# ── 数据准备 ──
action, subject, trial = 5, 1, 1  # throw
video_path = find_video(action, subject, trial)
tokens = vision_cache[(action, subject, trial)]  # (60, 64, 2048)
fi = 30  # acceleration phase

frame = get_frame(video_path, fi)
frame_resized = cv2.resize(frame, (224, 224))
frame_tokens = tokens[fi]  # (64, 2048)
token_energy = frame_tokens.norm(dim=1).numpy()  # (64,)
energy_map = token_energy.reshape(8, 8)
patch_size = 28

# ── 手动选人体上的3个patch ──
# 从frame 30的throw动作来看:
# 人站在画面中间偏左, 手举起在上方
# row1 col3 ≈ 头部区域 → patch 11
# row2 col3 ≈ 上身/手臂 → patch 19
# row0 col0 ≈ 左上背景(白墙) → patch 0

patches = [
    (19, 'Head', '#E24B4A'),       # row1, col3
    (27, 'Upper body', '#1D9E75'),  # row2, col3
    (0,  'Background', '#378ADD'),  # row0, col0
]

print(f"Frame {fi} of throw (a5_s1_t1)")
for pid, name, _ in patches:
    r, c = pid // 8, pid % 8
    print(f"  {name}: patch #{pid} (row={r}, col={c}), L2={token_energy[pid]:.1f}")

# ============================================================
# 布局: 2行
# 上排: [原始帧+grid] [8×8热力图] [叠加] [token噪音图64×160]
# 下排: [3个patch的token向量 line plot, 占满宽度]
# ============================================================

fig = plt.figure(figsize=(22, 11))
gs = gridspec.GridSpec(2, 4, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3,
                        width_ratios=[1, 1, 1, 1.3])

# ═══════════════════════════════════════════════
# 上排左: 原始帧 + patch网格 + 3个patch标注
# ═══════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(frame_resized)

for r in range(1, 8):
    ax1.axhline(y=r*patch_size, color='white', linewidth=0.5, alpha=0.5)
for c in range(1, 8):
    ax1.axvline(x=c*patch_size, color='white', linewidth=0.5, alpha=0.5)

for pid, name, color in patches:
    r, c = pid // 8, pid % 8
    rect = plt.Rectangle((c*patch_size, r*patch_size), patch_size, patch_size,
                          linewidth=2.5, edgecolor=color, facecolor='none')
    ax1.add_patch(rect)
    # 标签放在patch上方或下方
    ty = r*patch_size - 4 if r > 0 else (r+1)*patch_size + 12
    va = 'bottom' if r > 0 else 'top'
    ax1.text(c*patch_size + patch_size//2, ty, f'{name}',
             ha='center', va=va, fontsize=8, color=color, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor=color))

ax1.set_title('(a) Original frame\n8×8 patch grid', fontsize=11, fontweight='bold')
ax1.axis('off')

# ═══════════════════════════════════════════════
# 上排中左: 8×8能量热力图
# ═══════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(energy_map, cmap='YlOrRd', interpolation='nearest')

for r in range(8):
    for c in range(8):
        val = energy_map[r, c]
        color_txt = 'white' if val > np.percentile(token_energy, 70) else 'black'
        ax2.text(c, r, f'{val:.0f}', ha='center', va='center', fontsize=7, color=color_txt)

for pid, name, color in patches:
    r, c = pid // 8, pid % 8
    rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2.5,
                          edgecolor=color, facecolor='none')
    ax2.add_patch(rect)

ax2.set_title('(b) Token energy map\nL2 norm per patch', fontsize=11, fontweight='bold')
ax2.set_xlabel('Patch column', fontsize=9)
ax2.set_ylabel('Patch row', fontsize=9)
plt.colorbar(im, ax=ax2, shrink=0.8)

# ═══════════════════════════════════════════════
# 上排中右: 热力图叠加
# ═══════════════════════════════════════════════
ax3 = fig.add_subplot(gs[0, 2])
energy_resized = resize_heatmap(energy_map)
ax3.imshow(frame_resized)
ax3.imshow(energy_resized, cmap='jet', alpha=0.45,
           vmin=np.percentile(token_energy, 5),
           vmax=np.percentile(token_energy, 95))

for pid, name, color in patches:
    r, c = pid // 8, pid % 8
    rect = plt.Rectangle((c*patch_size, r*patch_size), patch_size, patch_size,
                          linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
    ax3.add_patch(rect)

ax3.set_title('(c) Activation overlay\non original frame', fontsize=11, fontweight='bold')
ax3.axis('off')

# ═══════════════════════════════════════════════
# 上排右: Token激活噪音图 (64 tokens × 160 dims)
# ═══════════════════════════════════════════════
ax4 = fig.add_subplot(gs[0, 3])

DIMS_SHOW = 160
token_data = frame_tokens[:, :DIMS_SHOW].numpy()  # (64, 160)
vmax = np.percentile(np.abs(token_data), 98)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im4 = ax4.imshow(token_data, aspect='auto', cmap='RdBu_r', norm=norm,
                  interpolation='nearest')

# 标注3个patch在y轴上的位置
for pid, name, color in patches:
    ax4.annotate(name, xy=(-2, pid), xytext=(-25, pid),
                 fontsize=8, color=color, fontweight='bold', va='center', ha='right',
                 arrowprops=dict(arrowstyle='-', color=color, lw=1.5))
    # 水平线标注
    ax4.axhline(y=pid, color=color, linewidth=1, alpha=0.5, linestyle='--')

ax4.set_title('(d) Token activation snapshot\n64 tokens × 160 dims', fontsize=11, fontweight='bold')
ax4.set_xlabel('Feature dimension', fontsize=9)
ax4.set_ylabel('Vision token (patch ID)', fontsize=9)
ax4.set_yticks([0, 16, 32, 48, 63])
ax4.set_xticks([0, 40, 80, 120, 159])
plt.colorbar(im4, ax=ax4, shrink=0.8, label='Activation')

# ═══════════════════════════════════════════════
# 下排: 3个patch的token向量 line plot
# ═══════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, :])

dims_plot = 200
x = np.arange(dims_plot)

for pid, name, color in patches:
    vec = frame_tokens[pid, :dims_plot].numpy()
    ax5.plot(x, vec, color=color, alpha=0.8, linewidth=1.2,
             label=f'{name} patch #{pid} (L2={token_energy[pid]:.0f})')

ax5.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
ax5.set_xlabel('Token dimension (first 200 of 2048)', fontsize=11)
ax5.set_ylabel('Activation value', fontsize=11)
ax5.set_title('(e) Token vector waveform: body patches show stronger and more varied activations',
              fontsize=11, fontweight='bold')
ax5.legend(fontsize=10, loc='upper right')
ax5.set_xlim(0, dims_plot)
ax5.grid(True, alpha=0.15)

fig.suptitle('ViT Patch Anatomy: How Qwen2.5-VL Encodes a Video Frame\n'
             'Frame 30 of throw action (a5_s1_t1)',
             fontsize=14, fontweight='bold', y=0.99)

plt.savefig('fig_patch_anatomy_v2.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_patch_anatomy_v2.png")
