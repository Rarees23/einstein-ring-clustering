from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import torch

from src.clustering.service import detect_empty_images, fit_bgmm
from src.core.artifacts import build_artifact_paths
from src.core.manifest import write_dataset_manifest, write_run_manifest
from src.core.runtime import RuntimeConfig, create_run_dir, ensure_runtime_dirs, set_global_seed, write_json
from src.datasets import load_catalog
from src.features.latent import LATENT_SCHEMA_VERSION, extract_latents
from src.models.autoencoder_model import build_autoencoder


def run_cluster(
    max_clusters: int | None = None,
    latent_amplify: float = 5.0,
    empty_percentile: float = 15.0,
    cfg: RuntimeConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or RuntimeConfig.default()
    max_clusters = cfg.gmm_max_clusters if max_clusters is None else max_clusters
    ensure_runtime_dirs(cfg)
    set_global_seed(cfg.seed, deterministic_cudnn=False)
    run_dir = create_run_dir(cfg, "cluster")
    artifact = build_artifact_paths(cfg)

    print(f"[gmm] Loading data from {cfg.data_dir!r}…", flush=True)
    catalog = load_catalog(cfg)
    print(f"[gmm] Loaded {catalog.images.shape[0]} images. Run directory: {run_dir}", flush=True)
    write_dataset_manifest(run_dir, catalog.manifest_rows())
    write_json(os.path.join(run_dir, "qc_report.json"), catalog.qc())

    images, filenames = catalog.images, catalog.filenames
    is_empty = detect_empty_images(images, empty_percentile)

    model = build_autoencoder(cfg)
    print(f"[gmm] Loading weights from {artifact.model_path!r}…", flush=True)
    model.load_state_dict(torch.load(artifact.model_path, map_location=cfg.device))
    print("[gmm] Extracting latents…", flush=True)
    latents = extract_latents(model, images, cfg.device, cfg.batch_size)

    print(f"[gmm] Fitting BGMM (max_clusters={max_clusters})…", flush=True)
    bgmm, scaler, labels = fit_bgmm(
        latents=latents,
        is_empty=is_empty,
        max_clusters=max_clusters,
        latent_amplify=latent_amplify,
        random_state=cfg.seed,
    )

    np.save(artifact.latents, latents)
    np.save(artifact.labels, labels)
    np.save(artifact.filenames, np.array(filenames, dtype=object))
    joblib.dump(bgmm, artifact.gmm_model)
    joblib.dump(scaler, artifact.scaler)
    meta = {
        "latent_amplify": float(latent_amplify),
        "empty_percentile": float(empty_percentile),
        "latent_dim": int(latents.shape[1]),
        "latent_schema_version": LATENT_SCHEMA_VERSION,
    }
    joblib.dump(meta, artifact.gmm_meta)
    write_json(os.path.join(run_dir, "gmm_meta.json"), meta)

    uniq, cnt = np.unique(labels, return_counts=True)
    cluster_sizes = {int(k): int(v) for k, v in zip(uniq, cnt)}
    metrics = {
        "num_images": int(len(images)),
        "num_empty": int(is_empty.sum()),
        "num_clusters_including_empty": int(len(uniq)),
    }
    write_run_manifest(
        run_dir,
        cfg,
        "cluster",
        artifacts={
            "gmm_model": artifact.gmm_model,
            "scaler": artifact.scaler,
            "labels": artifact.labels,
            "latents": artifact.latents,
        },
        metrics=metrics,
        extra={"cluster_sizes": cluster_sizes, "latent_schema_version": LATENT_SCHEMA_VERSION},
    )
    return metrics


def main() -> None:
    _defaults = RuntimeConfig.default()
    parser = argparse.ArgumentParser(description="Cluster pooled latents with Bayesian GMM")
    parser.add_argument("--max_clusters", type=int, default=_defaults.gmm_max_clusters)
    parser.add_argument("--latent_amplify", type=float, default=5.0)
    parser.add_argument("--empty_percentile", type=float, default=15.0)
    args = parser.parse_args()
    metrics = run_cluster(
        max_clusters=args.max_clusters,
        latent_amplify=args.latent_amplify,
        empty_percentile=args.empty_percentile,
    )
    print("Clustering complete.", metrics, flush=True)


if __name__ == "__main__":
    main()

