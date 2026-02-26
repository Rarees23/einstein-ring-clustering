import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from astropy.io import fits
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm

from ConvolutionalAutoencoder import ConvAutoencoder
from config import DATA_DIR, SAVED_MODELS_DIR, BEST_MODEL_PATH, IMG_H, IMG_W, IN_CHANNELS, LATENT_DIM, BATCH_SIZE, NUM_EPOCHS, DEVICE

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
def find_fits_files(root):
    return sorted(glob.glob(os.path.join(root, "**/*.fits"), recursive=True))

def load_fits_image(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data.ndim > 2:
            idx = tuple(0 for _ in range(data.ndim - 2))
            data = data[idx + (slice(None), slice(None))]
        return np.squeeze(data).astype(np.float32)

def preprocess_image(img2d, out_h=IMG_H, out_w=IMG_W):
    img = np.nan_to_num(img2d, nan=0.0)
    img = np.abs(img)

    p99 = np.percentile(img, 99.5)
    img = np.clip(img, 0, p99)
    img /= (p99 + 1e-8)

    if (img.shape[0], img.shape[1]) != (out_h, out_w):
        img = resize(img, (out_h, out_w), preserve_range=True, anti_aliasing=True)

    return np.expand_dims(img.astype(np.float32), axis=0)

# ---------------- LOAD DATA ----------------
fits_files = find_fits_files(DATA_DIR)
images = []

for f in tqdm(fits_files, desc="Loading FITS"):
    try:
        images.append(preprocess_image(load_fits_image(f)))
    except Exception as e:
        print("Skipping", f, e)

images = np.array(images, dtype=np.float32)
dataset = TensorDataset(torch.tensor(images))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------- MODEL ----------------
model = ConvAutoencoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------- LOSS ----------------
def mask_mse_loss(output, target):
    mask = 0.1 + 0.9 * (target > 0.01).float()
    return ((output - target) ** 2 * mask).mean()

# ---------------- TRAIN ----------------
best_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    total = 0

    for (x,) in loader:
        x = x.to(DEVICE)
        optimizer.zero_grad()

        x_hat = model(x)
        loss = mask_mse_loss(x_hat, x)

        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)

    avg = total / len(dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg:.6f}")

    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), BEST_MODEL_PATH)

print("Saved best model to:", BEST_MODEL_PATH)
