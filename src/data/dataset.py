"""Load from disk into a (N,C,H,W) stack plus filenames and per-file records."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from src.data.preprocess import find_fits, load_fits_image, preprocess_image
from src.data.records import DatasetLoadRecord


class PreprocessedDatasetSource(ABC):
    """One directory in → images array + paths + load records. Subclass for new formats."""

    @abstractmethod
    def load(self, data_dir: str) -> tuple[np.ndarray, list[str], list[DatasetLoadRecord]]:
        pass


class FitsPreprocessedDatasetSource(PreprocessedDatasetSource):
    """FITS tree → preprocessed tensors (default pipeline)."""

    def load(self, data_dir: str) -> tuple[np.ndarray, list[str], list[DatasetLoadRecord]]:
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
