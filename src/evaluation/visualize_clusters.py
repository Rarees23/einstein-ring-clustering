import os

import matplotlib.pyplot as plt
import numpy as np
import umap

from src.core.runtime import RuntimeConfig
from src.data.preprocess import load_fits_image, preprocess_image


def main():
    cfg = RuntimeConfig.default()
    gmm_output_dir = os.path.join(cfg.results_dir, "gmm_output")
    latents_path = os.path.join(gmm_output_dir, "latents.npy")
    labels_path = os.path.join(gmm_output_dir, "labels.npy")
    filenames_path = os.path.join(gmm_output_dir, "filenames.npy")

    if not (os.path.exists(latents_path) and os.path.exists(labels_path) and os.path.exists(filenames_path)):
        raise FileNotFoundError(
            f"Could not find latents/labels/filenames in {gmm_output_dir}. "
            f"Run the GMM step first: `python -m src gmm`."
        )

    latents = np.load(latents_path)
    labels = np.load(labels_path)
    filenames = np.load(filenames_path, allow_pickle=True)
    if latents.ndim != 2:
        raise ValueError(f"Expected latents of shape (N, D), got {latents.shape}")

    mask = labels != -1
    latents_vis = latents[mask]
    labels_vis = labels[mask]
    filenames_vis = filenames[mask]
    if latents_vis.shape[0] < 2:
        raise RuntimeError("Not enough non-empty samples to visualize clusters.")

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    latents_2d = reducer.fit_transform(latents_vis)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels_vis, cmap="tab20", s=5, alpha=0.7)
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("GMM clusters in 2D UMAP projection\n(click a point to view its image)")
    fig.tight_layout()

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        diffs = latents_2d - np.array([event.xdata, event.ydata])
        dists = np.einsum("ij,ij->i", diffs, diffs)
        idx = int(np.argmin(dists))
        fname = str(filenames_vis[idx])
        fits_path = os.path.join(cfg.data_dir, fname)
        if not os.path.exists(fits_path):
            print(f"Could not find FITS file for {fname} at {fits_path}")
            return
        try:
            raw = load_fits_image(fits_path)
            img = preprocess_image(raw)[0]
        except Exception as e:
            print(f"Failed to load/preprocess {fits_path}: {e}")
            return
        fig_img, ax_img = plt.subplots()
        ax_img.imshow(img, cmap="gray", origin="lower")
        ax_img.set_title(f"Image for point #{idx}\n{fname}")
        ax_img.axis("off")
        fig_img.tight_layout()
        fig_img.show()

    fig.canvas.mpl_connect("button_press_event", on_click)
    out_path = os.path.join(gmm_output_dir, "cluster_map_umap.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved cluster map to: {out_path}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()

