"""
Dataset-layer façade over ``src.data``: load, QC, splits, manifests, PyTorch datasets.

Pipelines and training should depend on this module (or ``src.datasets``), not on
``load_preprocessed_dataset`` / split helpers directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.dataset import deterministic_split, leakage_guard, load_preprocessed_dataset, quality_report
from src.data.records import DatasetLoadRecord, records_to_manifest
from src.datasets.torch_datasets import PreprocessedImageDataset


@dataclass
class PreprocessedCatalog:
    """
    In-memory catalog of preprocessed FITS from a directory.

    Built via :meth:`from_data_dir`, which delegates loading to ``src.data``.
    """

    images: np.ndarray
    filenames: list[str]
    records: list[DatasetLoadRecord]
    data_dir: str

    @classmethod
    def from_data_dir(cls, data_dir: str) -> PreprocessedCatalog:
        images, filenames, records = load_preprocessed_dataset(data_dir)
        if images.size == 0:
            raise RuntimeError(f"No usable FITS images found under {data_dir}")
        return cls(
            images=images,
            filenames=filenames,
            records=records,
            data_dir=data_dir,
        )

    def splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> dict[str, list[int]]:
        sp = deterministic_split(self.filenames, train_ratio=train_ratio, val_ratio=val_ratio)
        leakage_guard(sp)
        return sp

    def manifest_rows(self) -> list[dict[str, Any]]:
        """Rows for ``write_dataset_manifest`` / JSON."""
        return records_to_manifest(self.records)

    def qc(self) -> dict[str, Any]:
        return quality_report(self.images)

    def train_val_datasets(
        self,
        splits: dict[str, list[int]],
    ) -> tuple[PreprocessedImageDataset, PreprocessedImageDataset]:
        val_idx = splits["val"] if splits["val"] else splits["train"]
        train_ds = PreprocessedImageDataset(self.images, splits["train"])
        val_ds = PreprocessedImageDataset(self.images, val_idx)
        return train_ds, val_ds
