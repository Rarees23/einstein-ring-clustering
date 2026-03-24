"""Catalogs, PyTorch datasets/loaders, splits, QC — built on top of ``src.data``."""

from src.datasets.catalog_factory import load_catalog, register_catalog_kind, register_dataset_kind
from src.datasets.dataloaders import train_val_dataloaders
from src.datasets.preprocessed_catalog import PreprocessedCatalog
from src.datasets.qc import quality_report
from src.datasets.splits import deterministic_split, leakage_guard
from src.datasets.torch_datasets import PreprocessedImageDataset

__all__ = [
    "PreprocessedCatalog",
    "PreprocessedImageDataset",
    "deterministic_split",
    "leakage_guard",
    "load_catalog",
    "quality_report",
    "register_catalog_kind",
    "register_dataset_kind",
    "train_val_dataloaders",
]
