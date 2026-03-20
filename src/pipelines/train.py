from __future__ import annotations

import os

import numpy as np

from src.core.artifacts import build_artifact_paths
from src.core.manifest import write_dataset_manifest, write_run_manifest
from src.core.runtime import RuntimeConfig, create_run_dir, ensure_runtime_dirs, set_global_seed, write_json
from src.datasets import PreprocessedCatalog
from src.models.autoencoder_model import build_autoencoder
from src.training.trainer import train_autoencoder_with_early_stopping


def run_train(cfg: RuntimeConfig | None = None) -> dict[str, float]:
    cfg = cfg or RuntimeConfig.default()
    ensure_runtime_dirs(cfg)
    set_global_seed(cfg.seed, deterministic_cudnn=False)
    run_dir = create_run_dir(cfg, "train")

    print(f"[train] Loading preprocessed FITS from {cfg.data_dir!r} (this can take a while)...", flush=True)
    catalog = PreprocessedCatalog.from_data_dir(cfg.data_dir)
    print(
        f"[train] Loaded {catalog.images.shape[0]} images. Run directory: {run_dir}",
        flush=True,
    )
    splits = catalog.splits()
    qc = catalog.qc()

    write_dataset_manifest(run_dir, catalog.manifest_rows())
    write_json(os.path.join(run_dir, "splits.json"), splits)
    write_json(os.path.join(run_dir, "qc_report.json"), qc)

    train_dataset, val_dataset = catalog.train_val_datasets(splits)
    model = build_autoencoder(cfg.in_channels, cfg.latent_dim, cfg.device)
    artifact = build_artifact_paths(cfg)
    print(
        f"[train] Starting autoencoder on {cfg.device} (epochs≤{cfg.num_epochs}, batch={cfg.batch_size})...",
        flush=True,
    )
    best_val, history_train, history_val = train_autoencoder_with_early_stopping(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=cfg.device,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        best_model_path=artifact.model_path,
        patience=6,
    )

    np.savez(os.path.join(run_dir, "train_curves.npz"), train=np.array(history_train), val=np.array(history_val))
    metrics = {
        "best_val_loss": float(best_val),
        "final_train_loss": float(history_train[-1]),
        "final_val_loss": float(history_val[-1]),
        "num_samples": int(catalog.images.shape[0]),
    }
    write_run_manifest(
        run_dir,
        cfg,
        "train",
        artifacts={"best_model_path": artifact.model_path},
        metrics=metrics,
        extra={"latent_schema_version": cfg.latent_schema_version},
    )
    return metrics


def main() -> None:
    metrics = run_train()
    print("Training complete.", metrics, flush=True)


if __name__ == "__main__":
    main()

