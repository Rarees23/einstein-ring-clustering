"""Save a grid comparing original images and autoencoder reconstructions."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.core.artifacts import build_artifact_paths
from src.core.runtime import RuntimeConfig, create_run_dir, ensure_runtime_dirs, set_global_seed
from src.datasets import load_catalog
from src.models.autoencoder_model import build_autoencoder


def _indices_for_split(splits: dict[str, list[int]], split: str, n_images: int) -> list[int]:
    if split == "train":
        return splits["train"]
    if split == "val":
        return splits["val"] if splits["val"] else splits["train"]
    # empty test split → evaluate on full index set (same as run_evaluate)
    return splits["test"] if splits["test"] else list(range(n_images))


def run_reconstruction_viz(
    cfg: RuntimeConfig | None = None,
    *,
    num_samples: int = 8,
    split: str = "test",
) -> str:
    """Write original vs reconstruction PNG under ``results/runs/``; return its path."""
    cfg = cfg or RuntimeConfig.default()
    ensure_runtime_dirs(cfg)
    set_global_seed(cfg.seed, deterministic_cudnn=False)
    run_dir = create_run_dir(cfg, "recon_viz")
    artifact = build_artifact_paths(cfg)

    if not os.path.isfile(artifact.model_path):
        raise FileNotFoundError(
            f"No trained weights at {artifact.model_path}. Run `python -m src train_ae` first."
        )

    catalog = load_catalog(cfg)
    splits = catalog.splits()
    pool = _indices_for_split(splits, split, len(catalog.filenames))
    if not pool:
        raise RuntimeError(f"No images available for split {split!r}")

    rng = np.random.default_rng(cfg.seed)
    k = min(int(num_samples), len(pool))
    chosen = rng.choice(pool, size=k, replace=False)

    images = catalog.images[chosen].astype(np.float32)
    filenames = [catalog.filenames[i] for i in chosen]

    model = build_autoencoder(cfg)
    model.load_state_dict(torch.load(artifact.model_path, map_location=cfg.device))
    model.eval()

    tensor = torch.tensor(images, dtype=torch.float32).to(cfg.device)
    with torch.no_grad():
        recon = model(tensor).cpu().numpy()

    fig, axes = plt.subplots(k, 2, figsize=(8, 2.2 * k), squeeze=False)
    for row in range(k):
        orig = images[row, 0]
        dec = recon[row, 0]
        name = os.path.basename(filenames[row])
        axes[row, 0].imshow(orig, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(f"{name}\n(original)", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(dec, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title("reconstruction", fontsize=9)
        axes[row, 1].axis("off")

    fig.suptitle(f"Autoencoder reconstructions ({split} split, n={k})", fontsize=11, y=1.002)
    fig.tight_layout()
    out_path = os.path.join(run_dir, "reconstruction_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[recon_viz] Saved figure to {out_path}", flush=True)
    return out_path


def main() -> None:
    path = run_reconstruction_viz()
    print(path, flush=True)


if __name__ == "__main__":
    main()
