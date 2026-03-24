"""In-memory catalog: call a ``PreprocessedDatasetSource``, then splits, QC, manifests, torch views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.dataset import PreprocessedDatasetSource
from src.data.records import DatasetLoadRecord, records_to_manifest
from src.datasets.qc import quality_report
from src.datasets.splits import deterministic_split, leakage_guard
from src.datasets.torch_datasets import PreprocessedImageDataset


@dataclass
class PreprocessedCatalog:
    """Holds ``(N,C,H,W)`` images, paths, and records; use ``load_catalog`` to construct."""

    images: np.ndarray
    filenames: list[str]
    records: list[DatasetLoadRecord]
    data_dir: str

    @classmethod
    def from_data_dir(cls, data_dir: str, *, source: PreprocessedDatasetSource) -> PreprocessedCatalog:
        images, filenames, records = source.load(data_dir)
        if images.size == 0:
            raise RuntimeError(f"No usable images found under {data_dir}")
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
