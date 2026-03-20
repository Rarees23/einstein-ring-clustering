"""
Dataset classes layer — compositions of ``src.data`` used by pipelines and training.

Import from here in orchestration code::

    from src.datasets import PreprocessedCatalog

Lower-level IO and preprocessing stay in ``src.data``.
"""

from src.datasets.preprocessed_catalog import PreprocessedCatalog
from src.datasets.torch_datasets import PreprocessedImageDataset

__all__ = [
    "PreprocessedCatalog",
    "PreprocessedImageDataset",
]
