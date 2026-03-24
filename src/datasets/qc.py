"""Cheap per-stack stats (total flux) for manifests."""

from __future__ import annotations

from typing import Any

import numpy as np


def quality_report(images: np.ndarray) -> dict[str, Any]:
    if images.size == 0:
        return {"count": 0, "empty_count": 0, "near_empty_count": 0}
    signal = images.sum(axis=(1, 2, 3))
    empty_threshold = np.percentile(signal, 5.0)
    near_empty_threshold = np.percentile(signal, 15.0)
    empty_count = int((signal <= empty_threshold).sum())
    near_empty_count = int((signal <= near_empty_threshold).sum())
    return {
        "count": int(images.shape[0]),
        "empty_count": empty_count,
        "near_empty_count": near_empty_count,
        "signal_mean": float(signal.mean()),
        "signal_std": float(signal.std()),
    }
