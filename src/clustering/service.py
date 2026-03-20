from __future__ import annotations

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler


def detect_empty_images(images: np.ndarray, empty_percentile: float) -> np.ndarray:
    signal = images.sum(axis=(1, 2, 3))
    threshold = np.percentile(signal, empty_percentile)
    return signal <= threshold


def fit_bgmm(
    latents: np.ndarray,
    is_empty: np.ndarray,
    max_clusters: int,
    latent_amplify: float,
    random_state: int,
) -> tuple[BayesianGaussianMixture, StandardScaler, np.ndarray]:
    lat_nonempty = latents[~is_empty]
    if lat_nonempty.shape[0] == 0:
        raise RuntimeError("All images appear empty under the chosen percentile.")
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(lat_nonempty) * latent_amplify
    bgmm = BayesianGaussianMixture(
        n_components=max_clusters,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-2,
        reg_covar=1e-6,
        max_iter=1000,
        random_state=random_state,
    )
    bgmm.fit(z_scaled)
    labels_nonempty = bgmm.predict(z_scaled)
    labels = np.full(latents.shape[0], -1, dtype=int)
    labels[~is_empty] = labels_nonempty
    return bgmm, scaler, labels

