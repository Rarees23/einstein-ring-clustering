import os
import torch
import argparse

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "../saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
GMM_OUTPUT_DIR = os.path.join(RESULTS_DIR, "gmm_output")
os.makedirs(GMM_OUTPUT_DIR, exist_ok=True)

# ---------------- IMAGE / MODEL ----------------
IMG_H, IMG_W = 128, 128
IN_CHANNELS = 1
LATENT_DIM = 64
BATCH_SIZE = 32
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- AUTOENCODER ----------------
RECON_THRESHOLD = 0.05
SHOW_FIRST_N = 5

# ---------------- GMM ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--clusters", "-k", type=int, default=4,
                    help="Number of GMM clusters")
parser.add_argument("--radius_weight", type=float, default=0.2,
                    help="Weight of Einstein radius in clustering (0-1)")
args = parser.parse_args()

GMM_K = args.clusters
GMM_RADIUS_WEIGHT = args.radius_weight
