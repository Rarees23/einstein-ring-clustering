from __future__ import annotations

from typing import Any
import os

from .runtime import RuntimeConfig, write_json


def write_run_manifest(
    run_dir: str,
    cfg: RuntimeConfig,
    stage: str,
    artifacts: dict[str, str] | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "stage": stage,
        "config": cfg.to_dict(),
        "artifacts": artifacts or {},
        "metrics": metrics or {},
        "extra": extra or {},
    }
    path = os.path.join(run_dir, "run_manifest.json")
    write_json(path, payload)
    return path


def write_dataset_manifest(run_dir: str, records: list[dict[str, Any]]) -> str:
    path = os.path.join(run_dir, "dataset_manifest.json")
    write_json(path, {"records": records, "count": len(records)})
    return path

