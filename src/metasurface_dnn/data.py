from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetPaths:
    train_npz: str
    val_npz: str
    test_npz: str


class NPZAmplitudeDataset(Dataset):
    """Reads (x,y) from a npz: x=(N,H,W) float32, y=(N,) int64."""

    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path)
        self.x = data["x"].astype(np.float32)
        self.y = data["y"].astype(np.int64)
        if self.x.ndim != 3:
            raise ValueError(f"x must be (N,H,W), got {self.x.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be (N,), got {self.y.shape}")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y length mismatch")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class SyntheticTumorDataset(Dataset):
    """Synthetic binary dataset: blob + optional 'tumor' hot-spot.

    This is ONLY for verifying the simulation/training pipeline.
    """

    def __init__(self, n: int, image_size: int, seed: int = 0):
        self.n = int(n)
        self.image_size = int(image_size)
        rng = np.random.default_rng(seed)

        xs = np.linspace(-1, 1, self.image_size, dtype=np.float32)
        X, Y = np.meshgrid(xs, xs, indexing="ij")

        x_list = []
        y_list = []
        for _ in range(self.n):
            has_tumor = int(rng.random() < 0.5)

            base_sigma = rng.uniform(0.25, 0.45)
            base = np.exp(-(X**2 + Y**2) / (2 * base_sigma**2))

            noise = rng.normal(0.0, 0.03, size=base.shape).astype(np.float32)
            amp = base + noise

            if has_tumor:
                cx = rng.uniform(-0.4, 0.4)
                cy = rng.uniform(-0.4, 0.4)
                sigma = rng.uniform(0.05, 0.12)
                tumor = 0.8 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
                amp = amp + tumor

            amp = np.clip(amp, 0.0, None)
            amp = amp / (amp.max() + 1e-6)

            x_list.append(amp.astype(np.float32))
            y_list.append(has_tumor)

        self.x = np.stack(x_list, axis=0)
        self.y = np.array(y_list, dtype=np.int64)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)
