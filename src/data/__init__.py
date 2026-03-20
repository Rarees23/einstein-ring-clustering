"""
Low-level data: FITS IO, preprocessing, conversion, typed load records, splits, QC.

For pipeline-facing dataset objects (catalogs, PyTorch datasets), use ``src.datasets``.
"""

from src.data.dataset import (
    deterministic_split,
    leakage_guard,
    load_preprocessed_dataset,
    quality_report,
)
from src.data.records import DatasetLoadRecord, records_to_manifest

__all__ = [
    "DatasetLoadRecord",
    "deterministic_split",
    "leakage_guard",
    "load_preprocessed_dataset",
    "quality_report",
    "records_to_manifest",
]
