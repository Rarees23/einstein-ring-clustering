from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import torch

from src.core.artifacts import build_artifact_paths
from src.core.runtime import RuntimeConfig
from src.datasets import PreprocessedCatalog
from src.features.latent import LATENT_SCHEMA_VERSION, extract_latents
from src.models.autoencoder_model import build_autoencoder


def infer_labels(data_dir: str, cfg: RuntimeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RuntimeConfig.default()
    artifact = build_artifact_paths(cfg)
    catalog = PreprocessedCatalog.from_data_dir(data_dir)
    images, filenames = catalog.images, catalog.filenames

    model = build_autoencoder(cfg.in_channels, cfg.latent_dim, cfg.device)
    model.load_state_dict(torch.load(artifact.model_path, map_location=cfg.device))
    model.eval()

    scaler = joblib.load(artifact.scaler)
    gmm = joblib.load(artifact.gmm_model)
    meta = joblib.load(artifact.gmm_meta) if os.path.exists(artifact.gmm_meta) else {}
    if meta.get("latent_schema_version") not in (None, LATENT_SCHEMA_VERSION):
        raise RuntimeError("Latent schema mismatch between inference and cluster artifacts")

    lat = extract_latents(model, images, cfg.device, cfg.batch_size)
    amplify = float(meta.get("latent_amplify", 1.0))
    empty_percentile = meta.get("empty_percentile")
    if empty_percentile is None:
        z_scaled = scaler.transform(lat) * amplify
        labels = gmm.predict(z_scaled)
    else:
        signal = images.sum(axis=(1, 2, 3))
        thr = np.percentile(signal, float(empty_percentile))
        is_empty = signal <= thr
        labels = np.full(lat.shape[0], -1, dtype=int)
        non_empty = ~is_empty
        if np.any(non_empty):
            z_scaled = scaler.transform(lat[non_empty]) * amplify
            labels[non_empty] = gmm.predict(z_scaled)
    return {"images": images, "filenames": filenames, "labels": labels}

