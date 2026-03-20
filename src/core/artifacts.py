from __future__ import annotations

from dataclasses import dataclass
import os

from .runtime import RuntimeConfig


@dataclass
class ArtifactPaths:
    model_path: str
    gmm_dir: str
    gmm_model: str
    scaler: str
    gmm_meta: str
    latents: str
    labels: str
    filenames: str
    eval_dir: str
    eval_metrics: str


def build_artifact_paths(cfg: RuntimeConfig, *, ensure_dirs: bool = True) -> ArtifactPaths:
    gmm_dir = os.path.join(cfg.results_dir, "gmm_output")
    eval_dir = os.path.join(cfg.results_dir, "eval_output")
    if ensure_dirs:
        os.makedirs(gmm_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
    return ArtifactPaths(
        model_path=os.path.join(cfg.saved_models_dir, "best_model.pth"),
        gmm_dir=gmm_dir,
        gmm_model=os.path.join(gmm_dir, "gmm_model.joblib"),
        scaler=os.path.join(gmm_dir, "scaler_latent.joblib"),
        gmm_meta=os.path.join(gmm_dir, "gmm_meta.joblib"),
        latents=os.path.join(gmm_dir, "latents.npy"),
        labels=os.path.join(gmm_dir, "labels.npy"),
        filenames=os.path.join(gmm_dir, "filenames.npy"),
        eval_dir=eval_dir,
        eval_metrics=os.path.join(eval_dir, "eval_metrics.npz"),
    )

