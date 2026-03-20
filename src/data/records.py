"""Typed records for the data layer (manifest / QC), separate from tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence


@dataclass(frozen=True)
class DatasetLoadRecord:
    """One file seen while building a dataset (ok or rejected)."""

    path: str
    status: Literal["ok", "rejected"]
    relative_path: str | None = None
    reason: str | None = None


def records_to_manifest(records: Sequence[DatasetLoadRecord]) -> list[dict[str, Any]]:
    """Serialize for ``write_dataset_manifest`` (JSON-friendly dicts)."""
    out: list[dict[str, Any]] = []
    for r in records:
        d: dict[str, Any] = {"path": r.path, "status": r.status}
        if r.relative_path is not None:
            d["relative_path"] = r.relative_path
        if r.reason is not None:
            d["reason"] = r.reason
        out.append(d)
    return out
