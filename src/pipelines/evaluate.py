from __future__ import annotations

import os

import numpy as np
import torch

from src.core.artifacts import build_artifact_paths
from src.core.manifest import write_dataset_manifest, write_run_manifest
from src.core.runtime import RuntimeConfig, create_run_dir, ensure_runtime_dirs, set_global_seed, write_json
from src.datasets import PreprocessedCatalog
from src.evaluation.metrics import masked_mse, masked_psnr_from_mse
from src.features.latent import LATENT_SCHEMA_VERSION, extract_latents
from src.models.autoencoder_model import build_autoencoder


def run_evaluate(cfg: RuntimeConfig | None = None) -> dict[str, float]:
    cfg = cfg or RuntimeConfig.default()
    ensure_runtime_dirs(cfg)
    set_global_seed(cfg.seed, deterministic_cudnn=False)
    run_dir = create_run_dir(cfg, "evaluate")
    artifact = build_artifact_paths(cfg)

    print(f"[evaluate] Loading data from {cfg.data_dir!r}…", flush=True)
    catalog = PreprocessedCatalog.from_data_dir(cfg.data_dir)
    print(f"[evaluate] Loaded {catalog.images.shape[0]} images. Run directory: {run_dir}", flush=True)
    splits = catalog.splits()
    write_dataset_manifest(run_dir, catalog.manifest_rows())
    write_json(os.path.join(run_dir, "qc_report.json"), catalog.qc())

    filenames = catalog.filenames
    images = catalog.images
    test_idx = splits["test"] if splits["test"] else list(range(len(filenames)))
    test_images = images[test_idx]

    model = build_autoencoder(cfg.in_channels, cfg.latent_dim, cfg.device)
    model.load_state_dict(torch.load(artifact.model_path, map_location=cfg.device))
    model.eval()

    tensor = torch.tensor(test_images, dtype=torch.float32).to(cfg.device)
    with torch.no_grad():
        recon = model(tensor).cpu().numpy()
    mse = np.array([masked_mse(recon[i, 0], test_images[i, 0]) for i in range(test_images.shape[0])])
    psnr = masked_psnr_from_mse(mse)
    lat = extract_latents(model, test_images, cfg.device, cfg.batch_size)

    np.savez(
        artifact.eval_metrics,
        paths=np.array([filenames[i] for i in test_idx], dtype=object),
        mse=mse,
        psnr=psnr,
        latent_vectors=lat,
    )
    metrics = {
        "num_test": int(test_images.shape[0]),
        "mse_mean": float(np.mean(mse)),
        "psnr_mean": float(np.mean(psnr[np.isfinite(psnr)])) if np.isfinite(psnr).any() else float("inf"),
    }
    write_run_manifest(
        run_dir,
        cfg,
        "evaluate",
        artifacts={"eval_metrics": artifact.eval_metrics},
        metrics=metrics,
        extra={"latent_schema_version": LATENT_SCHEMA_VERSION},
    )
    return metrics


def main() -> None:
    metrics = run_evaluate()
    print("Evaluation complete.", metrics, flush=True)


if __name__ == "__main__":
    main()

