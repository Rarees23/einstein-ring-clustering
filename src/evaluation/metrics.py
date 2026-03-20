from __future__ import annotations

import numpy as np


def masked_mse(output: np.ndarray, target: np.ndarray, mask_thresh: float = 0.01) -> float:
    mask = target > mask_thresh
    if mask.sum() == 0:
        return 0.0
    return float((((output - target) ** 2) * mask).sum() / mask.sum())


def masked_psnr_from_mse(mse_values: np.ndarray) -> np.ndarray:
    return np.array([10.0 * np.log10(1.0 / m) if m > 0 else float("inf") for m in mse_values], dtype=np.float64)

