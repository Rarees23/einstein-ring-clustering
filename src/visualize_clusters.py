import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from config import GMM_OUTPUT_DIR


def main():
    latents_path = os.path.join(GMM_OUTPUT_DIR, "latents.npy")
    labels_path = os.path.join(GMM_OUTPUT_DIR, "labels.npy")

    if not os.path.exists(latents_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Could not find latents/labels. Expected:\n"
            f"  {latents_path}\n"
            f"  {labels_path}\n"
            f"Run the GMM step first: `python -m src gmm`."
        )

    latents = np.load(latents_path)  # shape: (N, latent_dim)
    labels = np.load(labels_path)    # shape: (N,)

    if latents.ndim != 2:
        raise ValueError(f"Expected latents of shape (N, D), got {latents.shape}")

    # Optional: drop empty images (label -1) from visualization
    mask = labels != -1
    latents_vis = latents[mask]
    labels_vis = labels[mask]

    if latents_vis.shape[0] < 2:
        raise RuntimeError("Not enough non-empty samples to visualize clusters.")

    print(f"Visualizing {latents_vis.shape[0]} samples in 2D using PCA.")

    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(latents_vis)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=labels_vis,
        cmap="tab20",
        s=5,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("GMM clusters in 2D PCA projection")
    plt.tight_layout()

    # Save figure next to GMM outputs
    out_path = os.path.join(GMM_OUTPUT_DIR, "cluster_map_pca.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved cluster map to: {out_path}")

    # Also show interactively if running in a local environment
    try:
        plt.show()
    except Exception:
        # In headless environments, just rely on the saved PNG
        pass


if __name__ == "__main__":
    main()

