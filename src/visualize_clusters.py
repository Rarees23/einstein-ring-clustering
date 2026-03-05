import os

import matplotlib.pyplot as plt
import numpy as np
import umap

from config import GMM_OUTPUT_DIR, DATA_DIR
from preprocess import load_fits_image, preprocess_image


def main():
    latents_path = os.path.join(GMM_OUTPUT_DIR, "latents.npy")
    labels_path = os.path.join(GMM_OUTPUT_DIR, "labels.npy")
    filenames_path = os.path.join(GMM_OUTPUT_DIR, "filenames.npy")

    if (
        not os.path.exists(latents_path)
        or not os.path.exists(labels_path)
        or not os.path.exists(filenames_path)
    ):
        raise FileNotFoundError(
            f"Could not find latents/labels/filenames. Expected:\n"
            f"  {latents_path}\n"
            f"  {labels_path}\n"
            f"  {filenames_path}\n"
            f"Run the GMM step first: `python -m src.cluster_gmm`."
        )

    latents = np.load(latents_path)  # shape: (N, latent_dim)
    labels = np.load(labels_path)    # shape: (N,)
    filenames = np.load(filenames_path, allow_pickle=True)  # shape: (N,)

    if latents.ndim != 2:
        raise ValueError(f"Expected latents of shape (N, D), got {latents.shape}")

    # Optional: drop empty images (label -1) from visualization
    mask = labels != -1
    latents_vis = latents[mask]
    labels_vis = labels[mask]
    filenames_vis = filenames[mask]

    if latents_vis.shape[0] < 2:
        raise RuntimeError("Not enough non-empty samples to visualize clusters.")

    print(f"Visualizing {latents_vis.shape[0]} samples in 2D using UMAP.")

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    latents_2d = reducer.fit_transform(latents_vis)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=labels_vis,
        cmap="tab20",
        s=5,
        alpha=0.7,
    )
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("GMM clusters in 2D UMAP projection\n(click a point to view its image)")
    fig.tight_layout()

    # Set up click interaction
    def on_click(event):
        # Only respond to left clicks inside the main axes
        if event.inaxes != ax or event.button != 1:
            return

        # Get click coordinates in data space
        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None:
            return

        # Find nearest point in the 2D embedding
        diffs = latents_2d - np.array([x_click, y_click])
        dists = np.einsum("ij,ij->i", diffs, diffs)
        idx = int(np.argmin(dists))

        fname = str(filenames_vis[idx])
        fits_path = os.path.join(DATA_DIR, fname)

        if not os.path.exists(fits_path):
            print(f"Could not find FITS file for {fname} at {fits_path}")
            return

        try:
            raw = load_fits_image(fits_path)
            img = preprocess_image(raw)[0]  # drop channel dim -> (H, W)
        except Exception as e:
            print(f"Failed to load/preprocess {fits_path}: {e}")
            return

        # Show the image in a new window
        fig_img, ax_img = plt.subplots()
        ax_img.imshow(img, cmap="gray", origin="lower")
        ax_img.set_title(f"Image for point #{idx}\n{fname}")
        ax_img.axis("off")
        fig_img.tight_layout()
        fig_img.show()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Save figure next to GMM outputs
    out_path = os.path.join(GMM_OUTPUT_DIR, "cluster_map_umap.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved cluster map to: {out_path}")

    # Also show interactively if running in a local environment
    try:
        plt.show()
    except Exception:
        # In headless environments, just rely on the saved PNG
        pass


if __name__ == "__main__":
    main()

