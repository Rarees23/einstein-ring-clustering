import os
import joblib
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ConvolutionalAutoencoder import ConvAutoencoder
from config import (
    DATA_DIR,
    SAVED_MODELS_DIR,
    RESULTS_DIR,
    IN_CHANNELS,
    LATENT_DIM,
    BATCH_SIZE,
    DEVICE,
    set_global_seed,
)
from preprocess import find_fits, load_fits_image, preprocess_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive GMM clustering on Einstein ring latent vectors"
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=20,
        help="Maximum number of clusters for Bayesian GMM (upper bound)",
    )
    parser.add_argument(
        "--latent_amplify",
        type=float,
        default=5.0,
        help="Amplify latent differences before clustering",
    )
    parser.add_argument(
        "--empty_percentile",
        type=float,
        default=15.0,
        help="Percentile used to detect empty images",
    )
    args = parser.parse_args()

    MAX_CLUSTERS = args.max_clusters
    LATENT_AMPLIFY = args.latent_amplify
    EMPTY_PERC = args.empty_percentile

    # Best-effort reproducibility
    set_global_seed(42, deterministic_cudnn=False)

    # ---------------- OUTPUT ----------------
    OUT_DIR = os.path.join(RESULTS_DIR, "gmm_output")
    os.makedirs(OUT_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
    META_PATH = os.path.join(OUT_DIR, "gmm_meta.joblib")

    # ---------------- LOAD DATA ----------------
    fits_files = find_fits(DATA_DIR)
    images, filenames = [], []

    for f in tqdm(fits_files, desc="Loading images"):
        try:
            raw = load_fits_image(f)
            images.append(preprocess_image(raw))
            filenames.append(os.path.basename(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")

    images = np.array(images, dtype=np.float32)

    if images.size == 0:
        raise RuntimeError(f"No usable FITS images found under {DATA_DIR}")

    # ---------------- EMPTY IMAGE DETECTION ----------------
    signal = images.sum(axis=(1, 2, 3))
    threshold = np.percentile(signal, EMPTY_PERC)
    is_empty = signal <= threshold
    print(f"Empty images detected: {is_empty.sum()} / {len(images)}")

    # ---------------- LOAD AUTOENCODER ----------------
    model = ConvAutoencoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ---------------- EXTRACT LATENTS ----------------
    ds = TensorDataset(torch.tensor(images, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    latents = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            z = model.encoder(x)
            z = z.mean(dim=(2, 3)) if z.ndim == 4 else z
            latents.append(z.cpu().numpy())

    latents = np.vstack(latents)

    # ---------------- SELECT NON-EMPTY ----------------
    latents_nonempty = latents[~is_empty]
    if latents_nonempty.shape[0] == 0:
        raise RuntimeError("All images appear empty under the chosen percentile.")

    # ---------------- SCALE & AMPLIFY LATENTS ----------------
    scaler_latent = StandardScaler()
    Z_scaled = scaler_latent.fit_transform(latents_nonempty) * LATENT_AMPLIFY

    # ---------------- BAYESIAN GMM ----------------
    bgmm = BayesianGaussianMixture(
        n_components=MAX_CLUSTERS,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-2,
        reg_covar=1e-6,
        max_iter=1000,
        random_state=42,
    )
    bgmm.fit(Z_scaled)
    labels_nonempty = bgmm.predict(Z_scaled)

    # ---------------- MERGE LABELS ----------------
    labels = np.full(len(images), -1, dtype=int)
    labels[~is_empty] = labels_nonempty

    # ---------------- CHECK CLUSTER SIZES ----------------
    unique, counts = np.unique(labels, return_counts=True)
    print("Final cluster sizes:", dict(zip(unique, counts)))
    print("Note: cluster -1 = EMPTY")

    # ---------------- SAVE ----------------
    joblib.dump(bgmm, os.path.join(OUT_DIR, "gmm_model.joblib"))
    joblib.dump(scaler_latent, os.path.join(OUT_DIR, "scaler_latent.joblib"))
    joblib.dump(
        {
            "latent_amplify": float(LATENT_AMPLIFY),
            "empty_percentile": float(EMPTY_PERC),
            "latent_dim": int(LATENT_DIM),
        },
        META_PATH,
    )
    np.save(os.path.join(OUT_DIR, "latents.npy"), latents)
    np.save(os.path.join(OUT_DIR, "labels.npy"), labels)

    print("Bayesian GMM clustering complete. Empty images = -1, max clusters =", MAX_CLUSTERS)
    print(Z_scaled.mean(), Z_scaled.std())


if __name__ == "__main__":
    main()
