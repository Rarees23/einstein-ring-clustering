from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import random
from typing import Any

import numpy as np
import torch


@dataclass
class RuntimeConfig:
    repo_root: str
    data_dir: str
    saved_models_dir: str
    results_dir: str
    runs_dir: str
    image_h: int = 128
    image_w: int = 128
    in_channels: int = 1
    latent_dim: int = 64
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    optimizer_kind: str = "adam"
    loss_kind: str = "masked_mse"
    early_stopping_patience: int = 6
    resume_checkpoint_path: str | None = None
    seed: int = 42
    latent_schema_version: str = "pooled-v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_kind: str = "fits"
    model_kind: str = "conv_ae"
    gmm_max_clusters: int = 50

    @staticmethod
    def default() -> "RuntimeConfig":
        src_dir = os.path.dirname(os.path.dirname(__file__))
        repo_root = os.path.abspath(os.path.join(src_dir, ".."))
        return RuntimeConfig(
            repo_root=repo_root,
            data_dir=os.path.join(repo_root, "data"),
            saved_models_dir=os.path.join(repo_root, "saved_models"),
            results_dir=os.path.join(repo_root, "results"),
            runs_dir=os.path.join(repo_root, "results", "runs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_runtime_dirs(cfg: RuntimeConfig) -> None:
    for p in (cfg.saved_models_dir, cfg.results_dir, cfg.runs_dir):
        os.makedirs(p, exist_ok=True)


def create_run_dir(cfg: RuntimeConfig, stage: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(cfg.runs_dir, f"{ts}_{stage}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def set_global_seed(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

