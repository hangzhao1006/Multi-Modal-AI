"""
Generate figures for presentation and report
- Per-class complementarity table
- Combined analysis figure (Grad-CAM + temporal + tokens + t-SNE)
- t-SNE wide format
- Token activation tall format
"""


# ============================================================
# ============================================================
# Slide 4: Per-class complementarity table (精选12类)
# 选最互补的 + 两个都强的 + 两个都弱的
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

action_data = [
    # (name, imu_acc, vis_acc) — 选最有说服力的12类
    # Vision远超IMU的
    ("Tennis serve",   50, 94),
    ("Throw",          25, 56),
    ("Boxing",         69, 100),
    ("Pickup & throw", 69, 100),
    ("Baseball swing", 81, 100),
    # IMU远超Vision的
    ("Wave",          100, 75),
    ("Draw circle CW",100, 75),
    ("Draw circle CCW",100, 75),
    ("Arm curl",       88, 69),
    # 两个都强的
    ("Clap",          100, 100),
    ("Walk",          100, 100),
    ("Squat",         100, 100),
]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axis('off')

# 表头
col_labels = ['Action', 'IMU', 'Vision', 'Gap', 'Stronger']
col_widths = [0.28, 0.12, 0.12, 0.12, 0.18]
col_x = [0.05]
for w in col_widths[:-1]:
    col_x.append(col_x[-1] + w)

# 颜色
header_color = '#1B2A4A'
row_colors = ['#F8FAFC', '#FFFFFF']
vision_color = '#2D6A9F'
imu_color = '#D97706'
same_color = '#64748B'

# 画表头
for j, (label, x, w) in enumerate(zip(col_labels, col_x, col_widths)):
    rect = plt.Rectangle((x, 0.88), w - 0.005, 0.06,
                          facecolor=header_color, edgecolor='none')
    ax.add_patch(rect)
    ax.text(x + w/2, 0.91, label, ha='center', va='center',
            fontsize=11, color='white', fontweight='bold')

# 画数据行
for i, (name, imu, vis) in enumerate(action_data):
    y = 0.82 - i * 0.065
    bg = row_colors[i % 2]
    gap = vis - imu

    if gap > 5:
        stronger = 'Vision'
        gap_color = vision_color
    elif gap < -5:
        stronger = 'IMU'
        gap_color = imu_color
    else:
        stronger = 'Both strong'
        gap_color = same_color

    # 行背景
    for j, (x, w) in enumerate(zip(col_x, col_widths)):
        rect = plt.Rectangle((x, y - 0.025), w - 0.005, 0.06,
                              facecolor=bg, edgecolor='#E2E8F0', linewidth=0.5)
        ax.add_patch(rect)

    # 文字
    ax.text(col_x[0] + col_widths[0]/2, y + 0.005, name,
            ha='center', va='center', fontsize=9.5, color='#0F172A')

    # IMU数值 + 颜色条
    imu_text_color = '#B45309' if imu >= 90 else '#0F172A'
    ax.text(col_x[1] + col_widths[1]/2, y + 0.005, f'{imu}%',
            ha='center', va='center', fontsize=10, color=imu_text_color,
            fontweight='bold' if imu >= 90 else 'normal')

    # Vision数值
    vis_text_color = '#1D4ED8' if vis >= 90 else '#0F172A'
    ax.text(col_x[2] + col_widths[2]/2, y + 0.005, f'{vis}%',
            ha='center', va='center', fontsize=10, color=vis_text_color,
            fontweight='bold' if vis >= 90 else 'normal')

    # Gap
    gap_str = f'+{gap}%' if gap > 0 else f'{gap}%' if gap < 0 else '0%'
    ax.text(col_x[3] + col_widths[3]/2, y + 0.005, gap_str,
            ha='center', va='center', fontsize=10, color=gap_color, fontweight='bold')

    # Stronger
    ax.text(col_x[4] + col_widths[4]/2, y + 0.005, stronger,
            ha='center', va='center', fontsize=9.5, color=gap_color, fontweight='bold')

# Title
ax.text(0.5, 0.97, 'Per-Class Modality Complementarity',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#0F172A',
        transform=ax.transAxes)
ax.text(0.5, 0.935, 'Selected 12 of 27 classes showing strongest complementary patterns',
        ha='center', va='center', fontsize=10, color='#64748B',
        transform=ax.transAxes)

# Oracle callout
ax.text(0.82, 0.07, 'Oracle: 98.4%', ha='center', va='center',
        fontsize=13, color='#0D9488', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0FDF4', edgecolor='#0D9488', linewidth=1.5))
ax.text(0.82, 0.02, '(if perfect gating)', ha='center', va='center',
        fontsize=9, color='#64748B')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('fig_perclass_table.png', dpi=200, bbox_inches='tight')
plt.show()
print("✅ Saved: fig_perclass_table.png")

# ============================================================
# ============================================================
# Slide 6: 一键生成4张分析图 (保存到Drive)
# 需要: rgb_data + resnet3d模型
# ============================================================

import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from sklearn.manifold import TSNE
import os

SAVE_DIR = "/content/drive/MyDrive/utd_mhad/figures/"
DATA_ROOT = "/content/drive/MyDrive/utd_mhad"
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 加载数据和模型 ──
print("Loading data and model...")
rgb_data = torch.load(f"{DATA_ROOT}/rgb_frames_112.pt", map_location='cpu', weights_only=False)

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

model = ResNet3DExpert().to(device)
ckpt = torch.load(f"{DATA_ROOT}/resnet3d_vision_expert.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model loaded (acc={ckpt['best_acc']:.1%})")

def to_uint8(frame):
    f = frame.numpy() if torch.is_tensor(frame) else frame
    if f.max() <= 1.0:
        f = f * 255.0
    return np.clip(f, 0, 255).astype(np.uint8)

# ============================================================
# 图(a): Grad-CAM (layer3, 14×14)
# ============================================================
print("\n(a) Grad-CAM...")

activation, gradient = {}, {}
def fwd_hook(m, inp, out): activation["v"] = out.detach()
def bwd_hook(m, gi, go): gradient["v"] = go[0].detach()

h1 = model.features[3].register_forward_hook(fwd_hook)
h2 = model.features[3].register_full_backward_hook(bwd_hook)

key = (5, 1, 1)
frames = rgb_data[key].float().permute(3, 0, 1, 2).unsqueeze(0).to(device)
frames.requires_grad_(True)
output = model(frames)
pred = output.argmax(1).item()
model.zero_grad()
output[0, pred].backward()

grads = gradient["v"]
acts = activation["v"]
cam = torch.relu((grads.mean(dim=[2,3,4], keepdim=True) * acts).sum(dim=1, keepdim=True))
cam = cam.squeeze().detach().cpu().numpy()
h1.remove(); h2.remove()

T_cam = cam.shape[0]
cam_to_frame = np.linspace(0, 59, T_cam, dtype=int)
cam_indices = np.linspace(0, T_cam-1, 4, dtype=int)

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, ci in enumerate(cam_indices):
    fi = cam_to_frame[ci]
    orig = to_uint8(rgb_data[key][fi])
    cam_s = cam[ci]
    cam_n = (cam_s - cam_s.min()) / (cam_s.max() - cam_s.min() + 1e-8)
    cam_r = np.array(PILImage.fromarray(cam_n.astype(np.float32)).resize((112,112), PILImage.BICUBIC))

    axes[0,i].imshow(orig); axes[0,i].axis('off')
    axes[0,i].set_title(f'F{fi} (t={fi/59:.2f})', fontsize=11)
    axes[1,i].imshow(orig)
    axes[1,i].imshow(cam_r, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1,i].axis('off')

axes[0,0].text(-0.08, 0.5, 'RGB', transform=axes[0,0].transAxes,
               rotation=90, va='center', fontsize=12, fontweight='bold')
axes[1,0].text(-0.08, 0.5, 'CAM', transform=axes[1,0].transAxes,
               rotation=90, va='center', fontsize=12, fontweight='bold')
fig.suptitle(f'Grad-CAM: ResNet3D attention during throw action (pred={pred+1})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}fig_gradcam.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"  ✅ Saved: {SAVE_DIR}fig_gradcam.png")

# ============================================================
# 图(b): Temporal dynamics (energy + cosine similarity)
# ============================================================
print("\n(b) Temporal dynamics...")

action_list = {5: "Throw", 6: "Basketball", 12: "Bowling", 14: "Baseball", 1: "Swipe L"}
colors_map = {"Throw": "#2D6A9F", "Basketball": "#F59E0B", "Bowling": "#0D9488",
              "Baseball": "#EF4444", "Swipe L": "#8B5CF6"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左: temporal energy
for act, name in action_list.items():
    k = (act, 1, 1)
    if k not in rgb_data: continue
    x = rgb_data[k].float().permute(3,0,1,2).unsqueeze(0).to(device)
    with torch.no_grad():
        tok = model(x, return_tokens=True).squeeze().cpu()
    energy = tok.norm(dim=1).numpy()
    e_norm = (energy - energy.mean()) / (energy.std() + 1e-8)
    t = np.linspace(0, 1, len(e_norm))
    axes[0].plot(t, e_norm, label=name, color=colors_map[name], linewidth=1.8, alpha=0.9)

axes[0].set_title('(a) Temporal token energy by action', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Normalized time'); axes[0].set_ylabel('Normalized energy')
axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.2)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

# 右: cosine similarity
k = (5, 1, 1)
x = rgb_data[k].float().permute(3,0,1,2).unsqueeze(0).to(device)
with torch.no_grad():
    tok = model(x, return_tokens=True).squeeze().cpu()
tok_n = tok / (tok.norm(dim=1, keepdim=True) + 1e-8)
sim = (tok_n @ tok_n.T).numpy()

im = axes[1].imshow(sim, cmap='magma', vmin=np.percentile(sim,3), vmax=np.percentile(sim,99.5))
axes[1].set_title('(b) Throw: inter-token cosine similarity', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Token index'); axes[1].set_ylabel('Token index')
axes[1].set_aspect('equal')
plt.colorbar(im, ax=axes[1], shrink=0.85, label='Cosine similarity')

fig.suptitle('Temporal Feature Dynamics (ResNet3D Vision Expert)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}fig_temporal.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"  ✅ Saved: {SAVE_DIR}fig_temporal.png")

# ============================================================
# 图(c): Qwen ViT token activation (如果有缓存)
# ============================================================
print("\n(c) Token activation...")

qwen_path = f"{DATA_ROOT}/vision_tokens_qwen25.pt"
if os.path.exists(qwen_path):
    from matplotlib.colors import TwoSlopeNorm
    vision_cache = torch.load(qwen_path, map_location='cpu', weights_only=False)
    tokens = vision_cache[(5, 1, 1)]  # throw

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    times = [0, 20, 40, 59]
    for i, t in enumerate(times):
        data = tokens[t, :, :160].numpy()
        vmax = np.percentile(np.abs(data), 98)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        axes[i].imshow(data, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')
        axes[i].set_title(f't = {t}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Feature dim.')
        if i == 0:
            axes[i].set_ylabel('Vision token')
            axes[i].set_yticks([0, 32, 63])
        else:
            axes[i].set_yticks([])

    fig.suptitle('Vision-Token Activation Snapshots (Qwen2.5-VL, throw action)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}fig_token_activation.png', dpi=200, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Saved: {SAVE_DIR}fig_token_activation.png")
else:
    print(f"  ⚠️ Qwen tokens not found at {qwen_path}, skipping")

# ============================================================
# 图(d): t-SNE
# ============================================================
print("\n(d) t-SNE... (约1-2分钟)")

features_list, labels_list = [], []
with torch.no_grad():
    for i, ((action, subject, trial), frames) in enumerate(rgb_data.items()):
        x = frames.float().permute(3,0,1,2).unsqueeze(0).to(device)
        tok = model(x, return_tokens=True).squeeze().cpu()
        features_list.append(tok.mean(dim=0).numpy())
        labels_list.append(action)
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(rgb_data)}")

features_np = np.array(features_list)
labels_np = np.array(labels_list)

try:
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                init='pca', learning_rate='auto', max_iter=1000)
except TypeError:
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                init='pca', learning_rate='auto', n_iter=1000)

embedded = tsne.fit_transform(features_np)

action_short = {
    1:'Swipe L', 2:'Swipe R', 3:'Wave', 4:'Clap', 5:'Throw',
    6:'Arm cross', 7:'Shoot', 8:'Draw X', 9:'Draw O',
    10:'Draw tri', 11:'Bowling', 12:'Boxing', 13:'Baseball',
    14:'Tennis', 15:'Arm curl', 16:'Serve', 17:'Push',
    18:'Knock', 19:'Catch', 20:'Pickup', 21:'Sit→Stand',
    22:'Stand→Sit', 23:'Lunge', 24:'Squat', 25:'Kick',
    26:'Walk', 27:'Jog'}

fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.cm.get_cmap('tab20', 27)

for act in range(1, 28):
    mask = labels_np == act
    if mask.sum() == 0: continue
    ax.scatter(embedded[mask,0], embedded[mask,1], s=20, alpha=0.75,
               color=cmap(act-1), edgecolors='none',
               label=f'{act}: {action_short.get(act)}')
    cx, cy = np.median(embedded[mask,0]), np.median(embedded[mask,1])
    ax.text(cx, cy, str(act), fontsize=7, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7))

ax.set_title('t-SNE: ResNet3D Feature Space (89.3% accuracy)', fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE dim. 1'); ax.set_ylabel('t-SNE dim. 2')
ax.legend(fontsize=7, ncol=3, loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=True, framealpha=0.9, edgecolor='#ddd')
ax.grid(True, alpha=0.15)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}fig_tsne.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"  ✅ Saved: {SAVE_DIR}fig_tsne.png")

# ── 最后拼成2×2 ──
print("\n拼接2×2合图...")
import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
panels = [
    (f'{SAVE_DIR}fig_gradcam.png', '(a) Grad-CAM'),
    (f'{SAVE_DIR}fig_temporal.png', '(b) Temporal dynamics'),
    (f'{SAVE_DIR}fig_token_activation.png', '(c) Token activation (Qwen ViT)'),
    (f'{SAVE_DIR}fig_tsne.png', '(d) t-SNE clustering'),
]

for ax, (path, title) in zip(axes.flat, panels):
    if os.path.exists(path):
        ax.imshow(mpimg.imread(path))
    else:
        ax.text(0.5, 0.5, f'{title}\n(not generated)', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')

fig.suptitle('Vision Expert Analysis', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}fig_analysis_combined.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"\n✅ All figures saved to: {SAVE_DIR}")

# ============================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

SAVE_DIR = "/content/drive/MyDrive/utd_mhad/figures/"

# features_np 和 labels_np 应该还在内存里
# 如果不在，需要重新跑提取特征的循环

# 如果embedded还在内存里直接用，否则重新算
try:
    _ = embedded.shape
    print("Using cached t-SNE result")
except:
    print("Recomputing t-SNE...")
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                    init='pca', learning_rate='auto', max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                    init='pca', learning_rate='auto', n_iter=1000)
    embedded = tsne.fit_transform(features_np)

action_short = {
    1:'Swipe L', 2:'Swipe R', 3:'Wave', 4:'Clap', 5:'Throw',
    6:'Arm cross', 7:'Shoot', 8:'Draw X', 9:'Draw O',
    10:'Draw tri', 11:'Bowling', 12:'Boxing', 13:'Baseball',
    14:'Tennis', 15:'Arm curl', 16:'Serve', 17:'Push',
    18:'Knock', 19:'Catch', 20:'Pickup', 21:'Sit→Stand',
    22:'Stand→Sit', 23:'Lunge', 24:'Squat', 25:'Kick',
    26:'Walk', 27:'Jog'}

fig, ax = plt.subplots(figsize=(14, 6))
cmap = plt.cm.get_cmap('tab20', 27)

for act in range(1, 28):
    mask = labels_np == act
    if mask.sum() == 0: continue
    ax.scatter(embedded[mask,0], embedded[mask,1], s=22, alpha=0.75,
               color=cmap(act-1), edgecolors='none',
               label=f'{act}: {action_short.get(act)}')
    cx, cy = np.median(embedded[mask,0]), np.median(embedded[mask,1])
    ax.text(cx, cy, str(act), fontsize=7, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7))

ax.set_title('t-SNE: ResNet3D Feature Space (89.3% accuracy)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE dim. 1', fontsize=11)
ax.set_ylabel('t-SNE dim. 2', fontsize=11)
ax.legend(fontsize=7, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          frameon=True, framealpha=0.9, edgecolor='#ddd',
          handletextpad=0.3, columnspacing=0.8)
ax.grid(True, alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}fig_tsne_wide.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {SAVE_DIR}fig_tsne_wide.png")

# ============================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SAVE_DIR = "/content/drive/MyDrive/utd_mhad/figures/"
DATA_ROOT = "/content/drive/MyDrive/utd_mhad"

vision_cache = torch.load(f"{DATA_ROOT}/vision_tokens_qwen25.pt",
                           map_location='cpu', weights_only=False)
tokens = vision_cache[(5, 1, 1)]

fig, axes = plt.subplots(1, 4, figsize=(10, 4))
times = [0, 20, 40, 59]

for i, t in enumerate(times):
    data = tokens[t, :, :160].numpy()  # (64, 160)
    # 转置: x轴=64 tokens (短), y轴=160 dims (长)
    data_T = data.T  # (160, 64)
    vmax = np.percentile(np.abs(data_T), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    axes[i].imshow(data_T, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')
    axes[i].set_title(f't = {t}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Vision token', fontsize=10)
    if i == 0:
        axes[i].set_ylabel('Feature dim.', fontsize=10)
    else:
        axes[i].set_yticks([])

fig.suptitle('Vision-Token Activation Snapshots (Qwen2.5-VL, throw action)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}fig_token_activation_tall.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {SAVE_DIR}fig_token_activation_tall.png")
