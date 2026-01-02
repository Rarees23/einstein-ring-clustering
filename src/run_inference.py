#!/usr/bin/env python3
"""
Predict clusters for new Einstein ring images using the trained GMM
and display results with radius info. Ensures all clusters are populated.
"""

import os, glob, joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from astropy.io import fits
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm

from ConvolutionalAutoencoder import ConvAutoencoder
from config import SAVED_MODELS_DIR, DATA_DIR, LATENT_DIM, IN_CHANNELS, IMG_H, IMG_W, BATCH_SIZE, DEVICE, RESULTS_DIR

# ---------------- PATHS ----------------
BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
GMM_OUTPUT_DIR = os.path.join(RESULTS_DIR, "gmm_output")
SCALER_PATH = os.path.join(GMM_OUTPUT_DIR, "scaler_latent.joblib")
GMM_MODEL_PATH = os.path.join(GMM_OUTPUT_DIR, "gmm_model.joblib")

# ---------------- HELPERS ----------------
def find_fits_and_radius(root):
    fits_paths = sorted(glob.glob(os.path.join(root, "**/*.fits"), recursive=True))
    return [(f, os.path.join(os.path.dirname(f), "einstein_radius.txt")) for f in fits_paths]

def load_radius(path):
    if not os.path.exists(path):
        return np.nan
    try:
        return float(open(path).read().strip().split()[0])
    except:
        return np.nan

def load_fits_image(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data.ndim > 2:
            idx = tuple(0 for _ in range(data.ndim - 2))
            data = data[idx + (slice(None), slice(None))]
        return np.squeeze(data).astype(np.float32)

def preprocess_image(img, out_h=IMG_H, out_w=IMG_W):
    img = np.nan_to_num(img, nan=0.0)
    img = np.abs(img)
    threshold = np.percentile(img, 99)
    img[img < threshold] = 0
    if img.max() > 0:
        img /= img.max()
    if (img.shape[0], img.shape[1]) != (out_h, out_w):
        img = resize(img, (out_h, out_w), preserve_range=True, anti_aliasing=True)
    return np.expand_dims(img.astype(np.float32), axis=0)

# ---------------- LOAD DATA ----------------
pairs = find_fits_and_radius(DATA_DIR)
images, radii, filenames, folders = [], [], [], []

for fpath, rpath in tqdm(pairs, desc="Loading images"):
    try:
        images.append(preprocess_image(load_fits_image(fpath)))
        folders.append(os.path.basename(os.path.dirname(fpath)))
        filenames.append(os.path.basename(fpath))
        radii.append(load_radius(rpath))
    except Exception as e:
        print(f"Skipping {fpath}: {e}")

images = np.array(images, dtype=np.float32)
radii = np.array(radii, dtype=float)
radii[np.isnan(radii)] = np.nanmedian(radii)

# ---------------- LOAD MODEL ----------------
model = ConvAutoencoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

scaler_latent = joblib.load(SCALER_PATH)
gmm = joblib.load(GMM_MODEL_PATH)
K = gmm.n_components

# ---------------- EXTRACT LATENTS ----------------
ds = TensorDataset(torch.tensor(images, dtype=torch.float32))
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
latents = []

with torch.no_grad():
    for (x,) in loader:
        x = x.to(DEVICE)
        z = model.encoder(x)
        if z.ndim == 4:
            z = z.mean(axis=(2,3))
        elif z.ndim > 2:
            z = z.reshape(z.shape[0], LATENT_DIM)
        latents.append(z.cpu().numpy())

latents = np.vstack(latents)

# ---------------- COMBINE FEATURES ----------------
Z_scaled = scaler_latent.transform(latents)
r_scaled = (radii - radii.min()) / (radii.max() - radii.min() + 1e-12)
X_aug = np.hstack([Z_scaled, r_scaled.reshape(-1,1)])

# ---------------- PREDICT ----------------
probs = gmm.predict_proba(X_aug)
labels = np.argmax(probs, axis=1)

# ---------------- FORCE ALL CLUSTERS POPULATED ----------------
unique_labels = set(labels)
for c in range(K):
    if c not in unique_labels:
        # Assign the farthest point from cluster mean to this empty cluster
        dists = np.linalg.norm(X_aug - gmm.means_[c], axis=1)
        idx = np.argmax(dists)
        labels[idx] = c
        unique_labels.add(c)

# ---------------- DISPLAY ----------------
for fn, folder, r, label, latent in zip(filenames, folders, radii, labels, latents):
    print(f"Image: {fn}")
    print(f"  Folder: {folder}")
    print(f"  Radius: {r:.6f}")
    print(f"  Cluster label: {label}")
    print(f"  Latent norm: {np.linalg.norm(latent):.4f}\n")

# ---------------- VISUALIZE CLUSTERS ----------------
def show_clusters(images, labels, folders, radii, max_per_cluster=30, images_per_row=6):
    for c in np.unique(labels):
        idxs = np.where(labels==c)[0][:max_per_cluster]
        if len(idxs)==0: continue
        n_rows = int(np.ceil(len(idxs)/images_per_row))
        plt.figure(figsize=(images_per_row*2, n_rows*2))
        plt.suptitle(f"Cluster {c}", fontsize=16)
        for i, idx in enumerate(idxs):
            plt.subplot(n_rows, images_per_row, i+1)
            plt.imshow(images[idx][0], origin='lower', cmap='inferno')
            plt.title(f"{folders[idx]} | r={radii[idx]:.3f}", fontsize=8)
            plt.axis('off')
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()

show_clusters(images, labels, folders, radii)
