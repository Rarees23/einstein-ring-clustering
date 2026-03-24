"""Map ``dataset_kind`` to a ``PreprocessedDatasetSource`` and build ``PreprocessedCatalog``."""

from __future__ import annotations

from src.core.runtime import RuntimeConfig
from src.data.dataset import FitsPreprocessedDatasetSource, PreprocessedDatasetSource
from src.datasets.preprocessed_catalog import PreprocessedCatalog

_SOURCE_REGISTRY: dict[str, type[PreprocessedDatasetSource]] = {
    "fits": FitsPreprocessedDatasetSource,
}


def register_dataset_kind(kind: str, source_cls: type[PreprocessedDatasetSource]) -> None:
    _SOURCE_REGISTRY[kind] = source_cls


def register_catalog_kind(kind: str, source_cls: type[PreprocessedDatasetSource]) -> None:
    register_dataset_kind(kind, source_cls)


def load_catalog(cfg: RuntimeConfig, *, data_dir: str | None = None) -> PreprocessedCatalog:
    root = data_dir if data_dir is not None else cfg.data_dir
    try:
        source_cls = _SOURCE_REGISTRY[cfg.dataset_kind]
    except KeyError as e:
        raise ValueError(
            f"Unknown dataset_kind {cfg.dataset_kind!r}. "
            f"Known kinds: {sorted(_SOURCE_REGISTRY)}",
        ) from e
    return PreprocessedCatalog.from_data_dir(root, source=source_cls())
