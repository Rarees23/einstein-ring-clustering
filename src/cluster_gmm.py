"""
Train a Gaussian Mixture Model (GMM) to cluster Einstein rings
using latent vectors and a weighted Einstein radius.
The radius contributes but is down-weighted so it doesn't dominate.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from astropy.io import fits
from skimage.transform import resize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import argparse

from ConvolutionalAutoencoder import ConvAutoencoder
from config import DATA_DIR, SAVED_MODELS_DIR, RESULTS_DIR, IMG_H, IMG_W, IN_CHANNELS, LATENT_DIM, BATCH_SIZE, DEVICE

# ---------------- ARGPARSE ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--clusters", "-k", type=int, default=4,
                    help="Number of GMM clusters")
parser.add_argument("--radius_weight", type=float, default=0.2,
                    help="Weight of Einstein radius in clustering (0-1)")
args = parser.parse_args()

K = args.clusters
RADIUS_WEIGHT = args.radius_weight
print(f"Using {K} clusters for GMM with radius weight {RADIUS_WEIGHT}")

# ---------------- PATHS ----------------
OUT_DIR = os.path.join(RESULTS_DIR, "gmm_output")
os.makedirs(OUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pth")

# ---------------- Helper Functions ----------------
def find_fits_and_radius(root):
    fits_paths = sorted(glob.glob(os.path.join(root, "**/*.fits"), recursive=True))
    pairs = []
    for fpath in fits_paths:
        dirn = os.path.dirname(fpath)
        radius_file = os.path.join(dirn, "einstein_radius.txt")
        if not os.path.exists(radius_file):
            radius_file = None
        pairs.append((fpath, radius_file))
    return pairs

def load_radius(path):
    if path is None:
        return np.nan
    try:
        with open(path, "r") as f:
            txt = f.read()
        return float(txt.strip().split()[0])
    except Exception:
        return np.nan

def load_fits_image(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data.ndim > 2:
            idx = tuple(0 for _ in range(data.ndim - 2))
            data = data[idx + (slice(None), slice(None))]
        return np.squeeze(data).astype(np.float32)

def preprocess_image(img, out_h=IMG_H, out_w=IMG_W):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.abs(img)
    threshold = np.percentile(img, 99)
    img[img < threshold] = 0
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    if (img.shape[0], img.shape[1]) != (out_h, out_w):
        img = resize(img, (out_h, out_w), preserve_range=True, anti_aliasing=True)
    return np.expand_dims(img.astype(np.float32), axis=0)

# ---------------- Load Images and Radii ----------------
pairs = find_fits_and_radius(DATA_DIR)
images, radii = [], []

for fits_path, rad_path in tqdm(pairs, desc="Loading images"):
    try:
        img = load_fits_image(fits_path)
        img = preprocess_image(img)
        images.append(img)
        r = load_radius(rad_path)
        radii.append(r)
    except Exception as e:
        print(f"Skipping {fits_path}: {e}")

images = np.array(images, dtype=np.float32)
radii = np.array(radii, dtype=float)

# Fill missing radii
nan_mask = np.isnan(radii)
if nan_mask.any():
    radii[nan_mask] = np.nanmedian(radii)

# ---------------- Load Autoencoder ----------------
model = ConvAutoencoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(DEVICE)
state = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ---------------- Extract Latent Vectors ----------------
ds = TensorDataset(torch.tensor(images, dtype=torch.float32))
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
latents = []

with torch.no_grad():
    for (x_batch,) in loader:
        x_batch = x_batch.to(DEVICE)
        z = model.encoder(x_batch)
        if z.ndim == 4:  
            z = z.mean(axis=(2,3))
        elif z.ndim > 2:
            z = z.reshape(z.shape[0], LATENT_DIM)
        latents.append(z.detach().cpu().numpy())

latents = np.vstack(latents)
print("Latents shape:", latents.shape)

# ---------------- Scale Latents ----------------
scaler_latent = StandardScaler()
Z_scaled = scaler_latent.fit_transform(latents)

# ---------------- Normalize and Weight Radius ----------------
r_norm = (radii - radii.min()) / (radii.max() - radii.min() + 1e-12)
radius_feature = (RADIUS_WEIGHT * r_norm).reshape(-1, 1)

# ---------------- Combine Features ----------------
X_aug = np.hstack([Z_scaled, radius_feature])

# ---------------- Fit GMM ----------------
gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=42, n_init=10)
gmm.fit(X_aug)
labels = gmm.predict(X_aug)
print("GMM training complete.")

# ---------------- Save Models and Data ----------------
joblib.dump(gmm, os.path.join(OUT_DIR, "gmm_weighted_radius_model.joblib"))
joblib.dump(scaler_latent, os.path.join(OUT_DIR, "scaler_latent.joblib"))
np.save(os.path.join(OUT_DIR, "latents.npy"), latents)
np.save(os.path.join(OUT_DIR, "radii.npy"), radii)
np.save(os.path.join(OUT_DIR, "labels.npy"), labels)

print(f"Saved GMM and data to {OUT_DIR}")
