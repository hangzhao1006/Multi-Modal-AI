"""
Step 1-3: Qwen2.5-VL Vision Token Extraction
- Load Qwen2.5-VL-3B model
- Extract vision tokens from all 861 UTD-MHAD videos  
- Save to Drive: (60 frames, 64 tokens, 2048 dims) per video
- Output: vision_tokens_qwen25.pt (13.54GB)

Requirements: GPU (A100 recommended), ~42GB VRAM
"""


# ============================================================
# Step 1: 挂载Drive
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
import cv2
import numpy as np
from PIL import Image
!pip install qwen-vl-utils -q
import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gc
from tqdm import tqdm

# ============================================================
# Step 2: 加载模型并缓存到Drive

cache_dir = "/content/drive/MyDrive/models/qwen25vl3b"
os.makedirs(cache_dir, exist_ok=True)

print("加载Qwen2.5-VL-3B（缓存到Drive）...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_dir
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir=cache_dir
)

print("✅ 模型已保存到Drive，下次直接加载不用重新下载")

# Step 3: 测试vision encoder
vision_encoder = model.model.visual
print(f"\nVision Encoder子模块：")
for name, module in vision_encoder.named_children():
    param_count = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {type(module).__name__}, {param_count/1e6:.1f}M params")

# ============================================================
DATA_ROOT = "/content/drive/MyDrive/utd_mhad"

# 读一帧视频
def get_one_frame(action=5, subject=1, trial=1):
    fname = f"a{action}_s{subject}_t{trial}_color.avi"
    for part in ['RGB-part1','RGB-part2','RGB-part3','RGB-part4']:
        p = f"{DATA_ROOT}/{part}/{fname}"
        import os
        if os.path.exists(p):
            cap = cv2.VideoCapture(p)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame)
    return None

img = get_one_frame()
print(f"图片大小: {img.size}")

# 用processor处理
from qwen_vl_utils import process_vision_info

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "describe"}
    ]
}]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt"
).to(model.device)

print(f"\nProcessor输出：")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# 跑vision encoder
with torch.no_grad():
    pixel_values = inputs['pixel_values'].to(model.device)
    grid_thw = inputs.get('image_grid_thw', None)

    output = vision_encoder(pixel_values, grid_thw=grid_thw)

    # 看输出结构
    print(f"输出类型: {type(output)}")
    print(f"输出属性: {[k for k in dir(output) if not k.startswith('_')]}")

    # 试几种方式取tokens
    if hasattr(output, 'last_hidden_state'):
        tokens = output.last_hidden_state
        print(f"\n✅ last_hidden_state: {tokens.shape}")
    if hasattr(output, 'pooler_output'):
        pool = output.pooler_output
        print(f"✅ pooler_output: {pool.shape}")
    if isinstance(output, tuple):
        print(f"\n是tuple，长度={len(output)}")
        for i, o in enumerate(output):
            if hasattr(o, 'shape'):
                print(f"  [{i}]: {o.shape}")

# ============================================================
# Step 3: 提取所有UTD-MHAD视频的vision tokens
# 每个视频 → 60帧 × 224×224 → 每帧64个tokens × 2048维
# 保存到Drive，后续训练直接加载，不用重复提取


DATA_ROOT = "/content/drive/MyDrive/utd_mhad"
TARGET_FRAMES = 60
IMG_SIZE = 224

def find_video(action, subject, trial):
    fname = f"a{action}_s{subject}_t{trial}_color.avi"
    for part in ['RGB-part1','RGB-part2','RGB-part3','RGB-part4']:
        p = os.path.join(DATA_ROOT, part, fname)
        if os.path.exists(p):
            return p
    return None

def extract_video_frames(video_path, n_frames=TARGET_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE))
        frames.append(frame)
    cap.release()
    if len(frames) < n_frames:
        return None
    return frames

def extract_vision_tokens(frames, batch_size=10):
    """分batch过vision encoder，避免OOM"""
    all_tokens = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_tokens = []

        with torch.no_grad():
            for frame in batch_frames:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": "d"}
                    ]
                }]
                text = processor.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs,
                    videos=video_inputs, return_tensors="pt"
                ).to(model.device)

                output = vision_encoder(
                    inputs['pixel_values'].to(model.device),
                    grid_thw=inputs.get('image_grid_thw', None))

                batch_tokens.append(output.pooler_output.cpu())

        all_tokens.extend(batch_tokens)

    # (60, 64, 2048) → 每帧64个tokens
    return torch.stack(all_tokens)

# ── 提取所有视频 ──
print("开始提取所有视频的vision tokens...")
print("每个视频: 60帧 × 64 tokens × 2048维\n")

vision_cache = {}
success, fail = 0, 0

for action in range(1, 28):
    for subject in range(1, 9):
        for trial in range(1, 5):
            vp = find_video(action, subject, trial)
            if vp is None:
                fail += 1
                continue

            frames = extract_video_frames(vp)
            if frames is None:
                fail += 1
                continue

            tokens = extract_vision_tokens(frames)
            vision_cache[(action, subject, trial)] = tokens
            success += 1

    print(f"  Action {action:2d}/27 完成, 成功={success}")

print(f"\n✅ 提取完成: {success}个视频")
print(f"❌ 失败: {fail}个")
print(f"每个视频tokens: {list(vision_cache.values())[0].shape}")

# 保存到Drive
save_path = "/content/drive/MyDrive/utd_mhad/vision_tokens_qwen25.pt"
torch.save(vision_cache, save_path)
print(f"\n💾 已保存到: {save_path}")
print(f"文件大小: {os.path.getsize(save_path)/1e9:.2f}GB")
