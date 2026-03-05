import os
import random
from typing import Optional

import numpy as np
import torch

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(__file__)  # src folder
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(REPO_ROOT, "data")           # FITS images go here
SAVED_MODELS_DIR = os.path.join(REPO_ROOT, "saved_models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
GMM_OUTPUT_DIR = os.path.join(RESULTS_DIR, "gmm_output")
EVAL_OUTPUT_DIR = os.path.join(RESULTS_DIR, "eval_output")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GMM_OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pth")

# ---------------- IMAGE / MODEL ----------------
IMG_H, IMG_W = 128, 128
IN_CHANNELS = 1
LATENT_DIM = 64
BATCH_SIZE = 32
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- AUTOENCODER ----------------
RECON_THRESHOLD = 0.05   # optional threshold for reconstruction error
SHOW_FIRST_N = 5         # for visualizing reconstructions during training

# ---------------- GMM ----------------
# cluster_gmm.py uses its own CLI (--max_clusters, --latent_amplify, --empty_percentile)
GMM_MAX_CLUSTERS = 25

# ---------------- FILES ----------------
GMM_MODEL_FILE = os.path.join(GMM_OUTPUT_DIR, "gmm_model.joblib")
SCALER_FILE = os.path.join(GMM_OUTPUT_DIR, "scaler_latent.joblib")
LATENTS_FILE = os.path.join(GMM_OUTPUT_DIR, "latents.npy")
LABELS_FILE = os.path.join(GMM_OUTPUT_DIR, "labels.npy")
FILENAMES_FILE = os.path.join(GMM_OUTPUT_DIR, "filenames.npy")


def set_global_seed(seed: int = 42, deterministic_cudnn: bool = False) -> None:
    """
    Set best-effort global random seed for Python, NumPy, and PyTorch.

    This is not a hard guarantee of full determinism (e.g. on different GPUs),
    but it makes experiments more reproducible by default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
