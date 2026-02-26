import os, glob, joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from astropy.io import fits
from skimage.transform import resize
import streamlit as st

from ConvolutionalAutoencoder import ConvAutoencoder
from config import (
    SAVED_MODELS_DIR, DATA_DIR, LATENT_DIM, IN_CHANNELS,
    IMG_H, IMG_W, BATCH_SIZE, DEVICE, RESULTS_DIR
)

# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(layout="wide")
st.title("Einstein Ring Clusters")

st.sidebar.title("Startup Configuration")

DATA_PATH = st.sidebar.text_input(
    "Path to FITS data folder",
    value=DATA_DIR,
    help="Absolute or relative path to the folder containing FITS images"
)

if not os.path.isdir(DATA_PATH):
    st.error("Invalid data folder path")
    st.stop()

# ---------------- PATHS ----------------
BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
GMM_OUTPUT_DIR = os.path.join(RESULTS_DIR, "gmm_output")
SCALER_PATH = os.path.join(GMM_OUTPUT_DIR, "scaler_latent.joblib")
GMM_MODEL_PATH = os.path.join(GMM_OUTPUT_DIR, "gmm_model.joblib")
GMM_META_PATH = os.path.join(GMM_OUTPUT_DIR, "gmm_meta.joblib")

# ---------------- HELPERS ----------------
def find_fits(root):
    return sorted(glob.glob(os.path.join(root, "**/*.fits"), recursive=True))

def load_fits_image(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data.ndim > 2:
            idx = tuple(0 for _ in range(data.ndim - 2))
            data = data[idx + (slice(None), slice(None))]
        return np.squeeze(data).astype(np.float32)

def preprocess_image(img):
    img = np.nan_to_num(img, nan=0.0)
    img = np.abs(img)

    threshold = np.percentile(img, 99)
    img[img < threshold] = 0

    if img.max() > 0:
        img /= img.max()

    if img.shape != (IMG_H, IMG_W):
        img = resize(
            img,
            (IMG_H, IMG_W),
            preserve_range=True,
            anti_aliasing=True
        )

    return np.expand_dims(img.astype(np.float32), axis=0)

# ---------------- CACHE: LOAD DATA ----------------
@st.cache_data(show_spinner=True)
def load_all_images(data_dir):
    fits_files = find_fits(data_dir)

    images, filenames, folders = [], [], []

    for fpath in fits_files:
        try:
            images.append(preprocess_image(load_fits_image(fpath)))
            filenames.append(os.path.basename(fpath))
            folders.append(os.path.basename(os.path.dirname(fpath)))
        except:
            continue

    images = np.array(images, dtype=np.float32)
    return images, filenames, folders

# ---------------- CACHE: LATENTS + CLUSTERS ----------------
@st.cache_data(show_spinner=True)
def compute_clusters(images):
    model = ConvAutoencoder(
        in_channels=IN_CHANNELS,
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    scaler_latent = joblib.load(SCALER_PATH)
    gmm = joblib.load(GMM_MODEL_PATH)
    meta = joblib.load(GMM_META_PATH) if os.path.exists(GMM_META_PATH) else {}
    latent_amplify = float(meta.get("latent_amplify", 1.0))
    empty_percentile = meta.get("empty_percentile", None)

    is_empty = None
    if empty_percentile is not None:
        signal = images.sum(axis=(1, 2, 3))
        threshold = np.percentile(signal, float(empty_percentile))
        is_empty = signal <= threshold

    ds = TensorDataset(torch.tensor(images))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    latents = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            z = model.encoder(x)

            if z.ndim == 4:
                z = z.mean(dim=(2, 3))
            else:
                z = z.view(z.shape[0], LATENT_DIM)

            latents.append(z.cpu().numpy())

    latents = np.vstack(latents)

    if is_empty is None:
        Z_scaled = scaler_latent.transform(latents) * latent_amplify
        labels = gmm.predict(Z_scaled)
    else:
        labels = np.full(latents.shape[0], -1, dtype=int)
        latents_nonempty = latents[~is_empty]
        Z_scaled = scaler_latent.transform(latents_nonempty) * latent_amplify
        labels[~is_empty] = gmm.predict(Z_scaled)

    unique = np.unique(labels[labels != -1])
    remap = {old: new for new, old in enumerate(unique)}
    labels_mapped = np.array([remap[l] if l != -1 else -1 for l in labels])

    return labels_mapped

# ---------------- LOAD + COMPUTE ----------------
with st.spinner("Loading data and computing clusters..."):
    images, filenames, folders = load_all_images(DATA_PATH)
    labels_mapped = compute_clusters(images)

# ---------------- UI ----------------
st.sidebar.title("Cluster Selection")
clusters = np.unique(labels_mapped)
cluster_choice = st.sidebar.selectbox("Choose cluster", clusters)

selected_idxs = np.where(labels_mapped == cluster_choice)[0]

cols = st.columns(6)
for i, idx in enumerate(selected_idxs):
    col = cols[i % 6]
    col.image(
        images[idx][0],
        width=120,
        clamp=True,
        channels="GRAY"
    )
    col.caption(f"{filenames[idx]}")
