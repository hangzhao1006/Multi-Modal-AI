"""
UTD-MHAD IMU dataset.

Train: subjects {1, 3, 5, 7}
Test:  subjects {2, 4, 6, 8}
Filename format: aA_sS_t*_inertial.mat   ->  action A, subject S, trial *
"""

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

IMU_LEN = 192
NUM_CLASSES = 27
TRAIN_SUBJECTS = {1, 3, 5, 7}
TEST_SUBJECTS = {2, 4, 6, 8}


def parse_filename(fname):
    parts = fname.split("_")
    return int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:])


def _load_imu(fpath):
    mat = sio.loadmat(fpath)
    data = mat["d_iner"].astype(np.float32)
    if data.shape[0] < IMU_LEN:
        pad = np.zeros((IMU_LEN - data.shape[0], 6), np.float32)
        data = np.concatenate([data, pad], axis=0)
    return torch.from_numpy(data[:IMU_LEN]).T.contiguous()  # (6, 192)


class IMUDataset(Dataset):
    """IMU-only dataset for training/validating the IMU expert standalone."""

    def __init__(self, data_root, train=True):
        self.samples = []
        allowed = TRAIN_SUBJECTS if train else TEST_SUBJECTS
        folder = os.path.join(data_root, "Inertial")
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith("_inertial.mat"):
                continue
            action, subject, _ = parse_filename(fname)
            if subject not in allowed:
                continue
            self.samples.append((os.path.join(folder, fname), action - 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        return _load_imu(fpath), label
