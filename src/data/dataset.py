from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from src.data.preprocess import find_fits, load_fits_image, preprocess_image
from src.data.records import DatasetLoadRecord


def load_preprocessed_dataset(data_dir: str) -> tuple[np.ndarray, list[str], list[DatasetLoadRecord]]:
    fits_files = find_fits(data_dir)
    images: list[np.ndarray] = []
    filenames: list[str] = []
    records: list[DatasetLoadRecord] = []

    for p in fits_files:
        name = os.path.basename(p)
        if name.startswith("._"):
            records.append(DatasetLoadRecord(path=p, status="rejected", reason="invalid_sidecar"))
            continue
        try:
            raw = load_fits_image(p)
            images.append(preprocess_image(raw))
            rel = os.path.relpath(p, data_dir)
            filenames.append(rel)
            records.append(DatasetLoadRecord(path=p, status="ok", relative_path=rel))
        except Exception as exc:
            records.append(DatasetLoadRecord(path=p, status="rejected", reason=str(exc)))
    if len(images) == 0:
        return np.empty((0, 1, 1, 1), dtype=np.float32), [], records
    return np.array(images, dtype=np.float32), filenames, records


def deterministic_split(
    filenames: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[int]]:
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0:
        raise ValueError("Invalid split ratios")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")
    indices = list(range(len(filenames)))
    indices.sort(key=lambda i: hashlib.sha1(filenames[i].encode("utf-8")).hexdigest())
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def leakage_guard(splits: dict[str, list[int]]) -> None:
    a, b, c = set(splits["train"]), set(splits["val"]), set(splits["test"])
    if a.intersection(b) or a.intersection(c) or b.intersection(c):
        raise RuntimeError("Split leakage detected")


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

