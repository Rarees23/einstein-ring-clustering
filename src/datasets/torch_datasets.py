"""torch.utils.data.Dataset wrappers over the in-memory (N,C,H,W) stack."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedImageDataset(Dataset):
    """Index into a shared float32 NCHW array; ``indices`` selects train/val/test rows without copying."""

    def __init__(self, images: np.ndarray, indices: Sequence[int] | None = None) -> None:
        if images.ndim != 4:
            raise ValueError(f"Expected NCHW array, got shape {images.shape}")
        if images.dtype != np.float32:
            images = images.astype(np.float32, copy=False)
        self._images = images
        if indices is None:
            self._indices = np.arange(images.shape[0], dtype=np.int64)
        else:
            self._indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = int(self._indices[idx])
        return torch.from_numpy(self._images[row]).float()
