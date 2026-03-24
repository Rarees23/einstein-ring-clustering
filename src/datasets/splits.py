"""Deterministic train/val/test index lists and overlap check."""

from __future__ import annotations

import hashlib


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
