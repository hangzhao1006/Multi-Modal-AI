"""
02_rgb_extraction.py
Extract RGB frames from all UTD-MHAD videos using IMU-guided cropping.
Key insight: RGB and IMU are natively synchronized (<0.5s difference),
so IMU active window can be used to crop RGB frames and remove rest periods.
Result: 861 videos extracted successfully (30 frames each, 64x64).
"""

import cv2
import scipy.io as sio
import numpy as np
import os

# ── Config ──
DATA_ROOT     = "/content/drive/MyDrive/utd_mhad"
TARGET_FRAMES = 30
IMG_SIZE      = 64

def find_video(action, subject, trial):
    """Search across 4 RGB part folders."""
    fname = f"a{action}_s{subject}_t{trial}_color.avi"
    for part in ['RGB-part1','RGB-part2','RGB-part3','RGB-part4']:
        p = os.path.join(DATA_ROOT, part, fname)
        if os.path.exists(p):
            return p
    return None

def extract_frames_imu_aligned(video_path, imu_path,
                                n_frames=TARGET_FRAMES, img_size=IMG_SIZE):
    """
    Extract n_frames from video, cropped to IMU-detected active window.
    This leverages RGB-IMU temporal synchronization to remove rest periods.
    """
    # Step 1: find active window from IMU
    imu = sio.loadmat(imu_path)['d_iner']
    mag = np.sqrt((imu[:, :3]**2).sum(axis=1))
    thr = mag.mean() + 0.3 * mag.std()
    active = np.where(mag > thr)[0]
    if len(active):
        T = len(mag)
        s = max(0.0, (active[0]  - 15) / T)
        e = min(1.0, (active[-1] + 15) / T)
    else:
        s, e = 0.0, 1.0

    # Step 2: extract corresponding RGB frames
    cap  = cv2.VideoCapture(video_path)
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0   = int(tot * s)
    f1   = max(int(tot * e), f0 + n_frames)
    idxs = np.linspace(f0, min(f1, tot-1), n_frames, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.resize(fr, (img_size, img_size))
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(fr)
    cap.release()

    if len(frames) < n_frames:
        return None
    return np.array(frames, np.float32) / 255.0  # (30, 64, 64, 3)

def extract_all_videos():
    """Extract all 861 videos and return as dict."""
    print("Extracting all RGB videos (IMU-aligned cropping)...")
    rgb_data = {}
    success, fail = 0, 0

    for action in range(1, 28):
        for subject in range(1, 9):
            for trial in range(1, 5):
                vp = find_video(action, subject, trial)
                ip = f"{DATA_ROOT}/Inertial/a{action}_s{subject}_t{trial}_inertial.mat"
                if vp is None or not os.path.exists(ip):
                    fail += 1; continue
                frames = extract_frames_imu_aligned(vp, ip)
                if frames is not None:
                    rgb_data[(action, subject, trial)] = frames
                    success += 1
                else:
                    fail += 1
        print(f"  Action {action:2d}/27 done, success={success}")

    print(f"\n✅ Extracted: {success} videos")
    print(f"❌ Failed:    {fail} videos")
    print(f"Memory:      {success * TARGET_FRAMES * IMG_SIZE * IMG_SIZE * 3 * 4 / 1e9:.2f} GB")
    return rgb_data

if __name__ == "__main__":
    rgb_data = extract_all_videos()