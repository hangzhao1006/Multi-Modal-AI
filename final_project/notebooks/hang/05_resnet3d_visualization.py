"""
ResNet3D Vision Expert Visualization
- Grad-CAM (layer3, 14x14 resolution)
- Temporal feature dynamics (energy profiles + cosine similarity)
- t-SNE feature space clustering (27 classes)
- Multi-action Grad-CAM comparison

Input: resnet3d_vision_expert.pt, rgb_frames_112.pt
Output: Publication-quality figures
"""


# ============================================================
# ============================================================
# Step 4f: ResNet3D Vision Expert Visualization (优化版)
# 优化: 图更大、文字更清晰、间距更合理、标签不重叠
# ============================================================

import os
import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as PILImage
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

print("Step 4f: ResNet3D Vision Expert Visualization (优化版)")
print("=" * 60)

DATA_ROOT = "/content/drive/MyDrive/utd_mhad"
OUT_DIR = "./pogmoe_resnet3d_figures"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

COLORS = {
    "blue": "#3A6EA5",
    "orange": "#C9822B",
    "green": "#4E8D6A",
    "red": "#B85450",
    "purple": "#6F5E9C",
    "gray": "#6C6C6C",
    "lightgray": "#B8B8B8",
    "dark": "#1F1F1F",
}

ACTION_COLORS = {
    "Throw": "#3A6EA5",
    "Basketball": "#C9822B",
    "Bowling": "#4E8D6A",
    "Baseball": "#B85450",
    "Swipe L": "#6F5E9C",
}


def save_figure(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        dpi = 300 if ext == 'png' else None
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08,
                    dpi=dpi if dpi else 'figure')
    print(f"  Saved: {name} (pdf/svg/png)")


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def panel_label(ax, label, x=-0.10, y=1.12):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="left", va="top",
            color=COLORS["dark"])


def smooth_curve(x, window=5):
    if window <= 1:
        return x
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")


def to_uint8_frame(frame):
    frame = frame.detach().cpu().numpy() if torch.is_tensor(frame) else frame
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


gradcam_cmap = LinearSegmentedColormap.from_list(
    "soft_gradcam",
    [(0.0, "#000000"), (0.35, "#3A6EA5"), (0.65, "#F2B56B"), (1.0, "#B2182B")]
)

# ── 加载数据和模型 ──
rgb_data = torch.load(f"{DATA_ROOT}/rgb_frames_112.pt",
                       map_location="cpu", weights_only=False)
print(f"Loaded {len(rgb_data)} videos")


class ResNet3DExpert(nn.Module):
    def __init__(self, d_model=256, num_classes=27, dropout=0.3):
        super().__init__()
        backbone = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            backbone.stem, backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, d_model), nn.GELU(), nn.Dropout(dropout))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x, return_tokens=False):
        B = x.shape[0]
        feat = self.features(x)
        feat = self.spatial_pool(feat).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        T_out = feat.shape[1]
        feat = self.proj(feat)
        cls = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls, feat], dim=1)
        feat = feat + self.pos_embed[:, :T_out + 1, :]
        feat = self.temporal_transformer(feat)
        if return_tokens:
            return feat[:, 1:, :]
        return self.head(feat[:, 0, :])


model = ResNet3DExpert().to(device)
ckpt = torch.load(f"{DATA_ROOT}/resnet3d_vision_expert.pt",
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded. best_acc = {ckpt['best_acc']:.1%}")


# ============================================================
# Figure 1: Grad-CAM (layer3 = 14×14 更精细)
# ============================================================
def make_figure_1_gradcam():
    print("\nFigure 1: Grad-CAM (layer3, 14×14)...")

    activation, gradient = {}, {}

    def fwd_hook(m, inp, out):
        activation["value"] = out.detach()
    def bwd_hook(m, gi, go):
        gradient["value"] = go[0].detach()

    # layer3 = index 3, 输出14×14
    h1 = model.features[3].register_forward_hook(fwd_hook)
    h2 = model.features[3].register_full_backward_hook(bwd_hook)

    key = (5, 1, 1)
    frames = rgb_data[key].float().permute(3, 0, 1, 2).unsqueeze(0).to(device)
    frames.requires_grad_(True)

    output = model(frames)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradient["value"]
    acts = activation["value"]
    weights = grads.mean(dim=[2, 3, 4], keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = cam.squeeze().detach().cpu().numpy()

    h1.remove()
    h2.remove()

    T_cam = cam.shape[0]
    cam_to_frame = np.linspace(0, 59, T_cam, dtype=int)
    cam_indices = np.linspace(0, T_cam - 1, 4, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    fig.subplots_adjust(wspace=0.06, hspace=0.12)

    for i, ci in enumerate(cam_indices):
        fi = cam_to_frame[ci]
        orig = to_uint8_frame(rgb_data[key][fi])

        cam_slice = cam[ci]
        cam_norm = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min() + 1e-8)
        cam_resized = np.array(
            PILImage.fromarray(cam_norm.astype(np.float32)).resize((112, 112), PILImage.BICUBIC))

        # 上排: 原始帧
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f'F{fi}  (t = {fi/59:.2f})', fontsize=11, pad=4)
        axes[0, i].axis("off")

        # 下排: Grad-CAM
        axes[1, i].imshow(orig)
        axes[1, i].imshow(cam_resized, cmap=gradcam_cmap, alpha=0.50, vmin=0, vmax=1)
        axes[1, i].axis("off")

    # 行标签
    axes[0, 0].text(-0.08, 0.5, "RGB", transform=axes[0, 0].transAxes,
                     rotation=90, va="center", ha="center",
                     fontsize=12, fontweight="bold")
    axes[1, 0].text(-0.08, 0.5, "CAM", transform=axes[1, 0].transAxes,
                     rotation=90, va="center", ha="center",
                     fontsize=12, fontweight="bold")

    panel_label(axes[0, 0], "a", x=-0.15, y=1.15)

    fig.suptitle(f'Grad-CAM evidence across throw phases  (predicted class = {pred_class+1})',
                 fontsize=13, fontweight="bold", y=0.98)

    save_figure(fig, "figure1_gradcam")
    plt.show()


# ============================================================
# Figure 2: Temporal feature dynamics
# ============================================================
def make_figure_2_temporal():
    print("\nFigure 2: Temporal dynamics...")

    action_list = {5: "Throw", 6: "Basketball", 12: "Bowling",
                   14: "Baseball", 1: "Swipe L"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.35)

    # ── (a) Temporal energy ──
    ax = axes[0]
    for act, name in action_list.items():
        k = (act, 1, 1)
        if k not in rgb_data:
            continue
        x = rgb_data[k].float().permute(3, 0, 1, 2).unsqueeze(0).to(device)
        with torch.no_grad():
            tok = model(x, return_tokens=True).squeeze().cpu()
        energy = tok.norm(dim=1).numpy()
        energy_norm = (energy - energy.mean()) / (energy.std() + 1e-8)
        energy_smooth = smooth_curve(energy_norm, window=3)
        t_axis = np.linspace(0, 1, len(energy_smooth))
        ax.plot(t_axis, energy_smooth, label=name,
                color=ACTION_COLORS.get(name, COLORS["gray"]),
                linewidth=1.8, alpha=0.9)

    ax.set_title("Temporal token energy by action type", fontsize=12, pad=6)
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Normalized energy (z-score)")
    ax.set_xlim(0, 1)
    ax.grid(True, linewidth=0.3, alpha=0.2)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    clean_axis(ax)
    panel_label(ax, "a")

    # ── (b) Cosine similarity matrix ──
    ax = axes[1]
    k = (5, 1, 1)
    x = rgb_data[k].float().permute(3, 0, 1, 2).unsqueeze(0).to(device)
    with torch.no_grad():
        tok = model(x, return_tokens=True).squeeze().cpu()
    tok_n = tok / (tok.norm(dim=1, keepdim=True) + 1e-8)
    sim = (tok_n @ tok_n.T).numpy()

    im = ax.imshow(sim, cmap="magma",
                    vmin=np.percentile(sim, 3), vmax=np.percentile(sim, 99.5),
                    interpolation="nearest")
    ax.set_title("Throw: inter-token cosine similarity", fontsize=12, pad=6)
    ax.set_xlabel("Token index")
    ax.set_ylabel("Token index")
    ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    clean_axis(ax)
    panel_label(ax, "b")

    fig.suptitle("Temporal feature dynamics (ResNet3D Vision Expert)",
                 fontsize=14, fontweight="bold", y=1.02)

    save_figure(fig, "figure2_temporal")
    plt.show()


# ============================================================
# Figure 3: t-SNE
# ============================================================
def make_figure_3_tsne():
    print("\nFigure 3: t-SNE... (约1-2分钟)")

    features_list, labels_list = [], []
    with torch.no_grad():
        for i, ((action, subject, trial), frames) in enumerate(rgb_data.items()):
            x = frames.float().permute(3, 0, 1, 2).unsqueeze(0).to(device)
            tok = model(x, return_tokens=True).squeeze().cpu()
            features_list.append(tok.mean(dim=0).numpy())
            labels_list.append(action)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(rgb_data)}")

    features_np = np.array(features_list)
    labels_np = np.array(labels_list)
    print(f"  Feature matrix: {features_np.shape}")

    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                     init="pca", learning_rate="auto", max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                     init="pca", learning_rate="auto", n_iter=1000)

    embedded = tsne.fit_transform(features_np)

    action_short = {
        1: "Swipe L", 2: "Swipe R", 3: "Wave", 4: "Clap", 5: "Throw",
        6: "Arm cross", 7: "Shoot", 8: "Draw X", 9: "Draw O",
        10: "Draw tri", 11: "Bowling", 12: "Boxing", 13: "Baseball",
        14: "Tennis", 15: "Arm curl", 16: "Serve", 17: "Push",
        18: "Knock", 19: "Catch", 20: "Pickup", 21: "Sit→Stand",
        22: "Stand→Sit", 23: "Lunge", 24: "Squat", 25: "Kick",
        26: "Walk", 27: "Jog",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20", 27)

    for act in range(1, 28):
        mask = labels_np == act
        if mask.sum() == 0:
            continue
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   s=20, alpha=0.75, color=cmap(act - 1), edgecolors="none",
                   label=f"{act}: {action_short.get(act, f'A{act}')}")

        # 聚类中心标号
        cx, cy = np.median(embedded[mask, 0]), np.median(embedded[mask, 1])
        ax.text(cx, cy, str(act), fontsize=7, fontweight="bold",
                ha="center", va="center", color="black",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))

    ax.set_title("t-SNE of video-level features (ResNet3D, 89.3% accuracy)",
                 fontsize=13, pad=8)
    ax.set_xlabel("t-SNE dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE dimension 2", fontsize=11)
    ax.grid(True, linewidth=0.3, alpha=0.15)
    clean_axis(ax)

    ax.legend(fontsize=7.5, ncol=3, loc="center left",
              bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
              frameon=True, framealpha=0.9, edgecolor="#ddd",
              handletextpad=0.4, columnspacing=0.8)

    panel_label(ax, "c", x=-0.08, y=1.05)

    save_figure(fig, "figure3_tsne")
    plt.show()


# ============================================================
# Run all
# ============================================================
make_figure_1_gradcam()
make_figure_2_temporal()
make_figure_3_tsne()

print("\n" + "=" * 60)
print(f"All figures saved to: {OUT_DIR}")
print("=" * 60)

# ============================================================
# ============================================================
# Step 4g: ResNet3D Extra Visualizations
# Publication-style compact version
#
# Figure 4: Multi-action Grad-CAM comparison
# Figure 5: Subject-level temporal consistency
# ============================================================

import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as PILImage
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

print("Step 4g: ResNet3D Extra Visualizations")
print("=" * 60)

DATA_ROOT = "/content/drive/MyDrive/utd_mhad"
OUT_DIR = "./pogmoe_resnet3d_extra_figures"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Compact publication style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",

    "font.size": 6.6,
    "axes.titlesize": 7.2,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,

    "axes.linewidth": 0.45,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "figure.dpi": 300,
    "savefig.dpi": 600,
})

COLORS = {
    "blue": "#3A6EA5",
    "orange": "#C9822B",
    "green": "#4E8D6A",
    "red": "#B85450",
    "purple": "#6F5E9C",
    "gray": "#6C6C6C",
    "lightgray": "#B8B8B8",
    "dark": "#1F1F1F",
}

SUBJECT_COLORS = {
    1: "#3A6EA5",
    2: "#C9822B",
    3: "#4E8D6A",
    4: "#B85450",
    5: "#6F5E9C",
    6: "#8C6D31",
    7: "#5F6F7A",
    8: "#C05A8A",
}

# Softer than jet, more paper-like
gradcam_cmap = LinearSegmentedColormap.from_list(
    "soft_gradcam",
    [
        (0.00, "#000000"),
        (0.35, "#3A6EA5"),
        (0.65, "#F2B56B"),
        (1.00, "#B2182B"),
    ]
)


def save_figure(fig, name):
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    svg_path = os.path.join(OUT_DIR, f"{name}.svg")
    png_path = os.path.join(OUT_DIR, f"{name}.png")

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.035)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved PNG: {png_path}")


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def panel_label(ax, label, x=-0.10, y=1.12):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLORS["dark"]
    )


def smooth_curve(x, window=3):
    if window <= 1:
        return x
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")


def to_uint8_frame(frame):
    frame = frame.detach().cpu().numpy() if torch.is_tensor(frame) else frame
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def get_existing_key(data, action, trial=1, max_subject=8):
    for subject in range(1, max_subject + 1):
        key = (action, subject, trial)
        if key in data:
            return key
    return None


# ============================================================
# Load data and model
# ============================================================

rgb_data = torch.load(
    f"{DATA_ROOT}/rgb_frames_112.pt",
    map_location="cpu",
    weights_only=False
)

print(f"Loaded {len(rgb_data)} videos")


class ResNet3DExpert(nn.Module):
    def __init__(self, d_model=256, num_classes=27, dropout=0.3):
        super().__init__()

        backbone = torchvision.models.video.r3d_18(
            weights=torchvision.models.video.R3D_18_Weights.DEFAULT
        )

        self.features = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, return_tokens=False):
        B = x.shape[0]

        feat = self.features(x)
        feat = self.spatial_pool(feat).squeeze(-1).squeeze(-1).permute(0, 2, 1)

        T_out = feat.shape[1]

        feat = self.proj(feat)

        cls = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls, feat], dim=1)
        feat = feat + self.pos_embed[:, :T_out + 1, :]

        feat = self.temporal_transformer(feat)

        if return_tokens:
            return feat[:, 1:, :]

        return self.head(feat[:, 0, :])


model = ResNet3DExpert().to(device)

ckpt = torch.load(
    f"{DATA_ROOT}/resnet3d_vision_expert.pt",
    map_location=device,
    weights_only=False
)

model.load_state_dict(ckpt["model_state"])
model.eval()

print("Model loaded.")


# ============================================================
# Grad-CAM utility
# ============================================================

def get_gradcam(model, frames_tensor, layer_idx=3):
    """
    layer_idx:
      3 = layer3, higher spatial resolution
      4 = layer4, more semantic but lower resolution
    """
    activation = {}
    gradient = {}

    def fwd_hook(m, inp, out):
        activation["value"] = out.detach()

    def bwd_hook(m, gi, go):
        gradient["value"] = go[0].detach()

    h1 = model.features[layer_idx].register_forward_hook(fwd_hook)
    h2 = model.features[layer_idx].register_full_backward_hook(bwd_hook)

    x = frames_tensor.clone().requires_grad_(True)

    out = model(x)
    pred = out.argmax(dim=1).item()

    model.zero_grad()
    out[0, pred].backward()

    grads = gradient["value"]
    acts = activation["value"]

    weights = grads.mean(dim=[2, 3, 4], keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = cam.squeeze().detach().cpu().numpy()

    h1.remove()
    h2.remove()

    return cam, pred


# ============================================================
# Figure 4: Multi-action Grad-CAM comparison
# ============================================================

def make_figure_4_multi_action_gradcam():
    print("\nFigure 4: Multi-action Grad-CAM comparison...")

    actions_to_show = [
        (5, "Throw"),
        (12, "Bowling"),
        (6, "Basketball"),
        (1, "Swipe L"),
    ]

    phase_names = ["Start", "Early", "Late", "End"]

    # Compact: 4 actions × 4 phases, overlay only
    fig, axes = plt.subplots(
        len(actions_to_show),
        4,
        figsize=(5.9, 5.2),
        facecolor="white"
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        top=0.88,
        bottom=0.055,
        wspace=0.045,
        hspace=0.12
    )

    for row, (act, act_name) in enumerate(actions_to_show):
        key = get_existing_key(rgb_data, act)

        if key is None:
            print(f"Warning: action {act} not found.")
            continue

        frames = rgb_data[key]
        x = frames.float().permute(3, 0, 1, 2).unsqueeze(0).to(device)

        cam, pred = get_gradcam(model, x, layer_idx=3)

        T_cam = cam.shape[0]
        cam_indices = np.linspace(0, T_cam - 1, 4, dtype=int)
        frame_indices = np.linspace(0, 59, T_cam, dtype=int)

        for col, (ci, phase) in enumerate(zip(cam_indices, phase_names)):
            fi = frame_indices[ci]

            orig = to_uint8_frame(frames[fi])

            cam_slice = cam[ci]
            cam_norm = (cam_slice - cam_slice.min()) / (
                cam_slice.max() - cam_slice.min() + 1e-8
            )

            cam_resized = np.array(
                PILImage.fromarray(cam_norm.astype(np.float32)).resize(
                    (112, 112),
                    PILImage.BICUBIC
                )
            )

            ax = axes[row, col]

            ax.imshow(orig)
            ax.imshow(
                cam_resized,
                cmap=gradcam_cmap,
                alpha=0.52,
                vmin=0,
                vmax=1
            )

            ax.axis("off")

            if row == 0:
                ax.set_title(phase, fontsize=6.8, pad=3)

        # row label
        axes[row, 0].text(
            -0.18,
            0.5,
            f"{act_name}\nP{pred + 1}",
            transform=axes[row, 0].transAxes,
            va="center",
            ha="right",
            fontsize=6.8,
            fontweight="bold",
            color=COLORS["dark"]
        )

    axes[0, 0].text(
        -0.42,
        1.22,
        "a",
        transform=axes[0, 0].transAxes,
        fontsize=9.5,
        fontweight="bold"
    )

    fig.suptitle(
        "Multi-action Grad-CAM comparison",
        y=0.965,
        fontsize=8.8,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.915,
        "Overlayed CAM maps show action-specific spatial evidence",
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["gray"]
    )

    save_figure(fig, "figure4_resnet3d_gradcam_multi_compact")
    plt.show()


# ============================================================
# Figure 5: Same action, different subjects
# ============================================================

def make_figure_5_subject_consistency():
    print("\nFigure 5: Same action, different subjects...")

    actions_compare = [
        (5, "Throw"),
        (12, "Bowling"),
        (14, "Baseball"),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.0, 4.65),
        facecolor="white"
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.91,
        top=0.86,
        bottom=0.105,
        wspace=0.34,
        hspace=0.48
    )

    last_im = None

    for col, (act, act_name) in enumerate(actions_compare):

        # ----------------------------------------------------
        # Top row: temporal energy profiles across subjects
        # ----------------------------------------------------
        ax = axes[0, col]

        for subj in range(1, 9):
            key = (act, subj, 1)

            if key not in rgb_data:
                continue

            x = rgb_data[key].float().permute(3, 0, 1, 2).unsqueeze(0).to(device)

            with torch.no_grad():
                tok = model(x, return_tokens=True).squeeze().cpu()

            energy = tok.norm(dim=1).numpy()

            # Normalize per subject for shape consistency comparison
            energy = (energy - energy.mean()) / (energy.std() + 1e-8)
            energy = smooth_curve(energy, window=3)

            t_axis = np.linspace(0, 1, len(energy))

            ax.plot(
                t_axis,
                energy,
                color=SUBJECT_COLORS[subj],
                linewidth=0.95,
                alpha=0.78,
                label=f"S{subj}"
            )

        ax.set_title(act_name, pad=4)
        ax.set_xlabel("Time")
        if col == 0:
            ax.set_ylabel("Norm. energy")
        else:
            ax.set_ylabel("")

        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        ax.grid(True, linewidth=0.28, alpha=0.18)
        clean_axis(ax)

        # Only show legend once to avoid clutter
        if col == 0:
            ax.legend(
                frameon=False,
                ncol=4,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.08),
                handlelength=1.0,
                columnspacing=0.55,
                borderaxespad=0.0
            )

        # ----------------------------------------------------
        # Bottom row: averaged inter-token similarity
        # ----------------------------------------------------
        ax = axes[1, col]

        all_sims = []

        for subj in range(1, 9):
            key = (act, subj, 1)

            if key not in rgb_data:
                continue

            x = rgb_data[key].float().permute(3, 0, 1, 2).unsqueeze(0).to(device)

            with torch.no_grad():
                tok = model(x, return_tokens=True).squeeze().cpu()

            tok_n = tok / (tok.norm(dim=1, keepdim=True) + 1e-8)
            sim = (tok_n @ tok_n.T).numpy()
            all_sims.append(sim)

        if all_sims:
            min_t = min(s.shape[0] for s in all_sims)
            aligned = [s[:min_t, :min_t] for s in all_sims]
            avg_sim = np.mean(aligned, axis=0)

            vmin = np.percentile(avg_sim, 3)
            vmax = np.percentile(avg_sim, 99.5)

            last_im = ax.imshow(
                avg_sim,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                rasterized=True
            )

        ax.set_xlabel("Token index")
        if col == 0:
            ax.set_ylabel("Token index")
        else:
            ax.set_ylabel("")

        ax.set_title("Avg. similarity", pad=4)

        if all_sims:
            ax.set_xticks([0, min_t // 2, min_t - 1])
            ax.set_yticks([0, min_t // 2, min_t - 1])

        ax.set_box_aspect(1)

    # Panel labels
    panel_label(axes[0, 0], "a", x=-0.18, y=1.20)
    panel_label(axes[1, 0], "b", x=-0.18, y=1.20)

    # Shared colorbar for all bottom heatmaps
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes[1, :],
            fraction=0.025,
            pad=0.035
        )
        cbar.set_label("Cosine", labelpad=4)
        cbar.ax.tick_params(labelsize=5.8, width=0.4, length=2.0)

    fig.suptitle(
        "Subject-level temporal consistency",
        y=0.965,
        fontsize=8.8,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.915,
        "Top: subject-specific energy profiles; bottom: averaged inter-token similarity",
        ha="center",
        va="center",
        fontsize=6.4,
        color=COLORS["gray"]
    )

    save_figure(fig, "figure5_resnet3d_subject_consistency_compact")
    plt.show()


# ============================================================
# Run
# ============================================================

make_figure_4_multi_action_gradcam()
make_figure_5_subject_consistency()

print("\n" + "=" * 70)
print("Finished Step 4g: ResNet3D extra visualizations")
print(f"Figures saved to: {OUT_DIR}")
print("=" * 70)
