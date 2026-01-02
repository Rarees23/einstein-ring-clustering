# Euclid Strong Lens Clustering

Unsupervised discovery and clustering of strong gravitational lens candidates (Einstein rings) from **Euclid FITS images** using deep latent representations and Gaussian Mixture Models.

---

## Overview

This project implements a **research‑grade unsupervised pipeline**:

1. **Convolutional Autoencoder (CAE)** learns compact latent representations of FITS images, focusing on Einstein ring structures.
2. **Latent space extraction** converts images into fixed‑length feature vectors.
3. **Gaussian Mixture Model (GMM)** clusters lenses in latent space, optionally augmented with a physical scalar (Einstein radius / thickness).
4. **Evaluation & visualization** validate reconstruction quality and inspect cluster composition.

The design cleanly separates:

* reusable ML logic (`src/`)
* trained models (`saved_models/`)
* experiment outputs (`results/`)

---

## Project Structure

```text
Euclid/
│
├── src/
│   ├── __main__.py              # CLI entry point
│   ├── autoencoder.py           # Convolutional autoencoder definition
│   ├── train_autoencoder.py     # Autoencoder training
│   ├── gmm_clustering.py        # Latent + radius GMM clustering
│   ├── test_reconstruction.py   # Reconstruction evaluation & metrics
│
├── data/                         # FITS images (not tracked)
├── saved_models/                 # Trained model weights
├── results/                      # Clustering outputs, labels, plots
├── requirements.txt
└── README.md
```

---

## Data Assumptions

* Images are **FITS files** (`.fits`)
* Each image is preprocessed to emphasize **ring‑like structures**
* Optional per‑image scalar files:

  * `einstein_radius.txt` or `normalized_thickness.txt`

Missing scalar values are handled robustly (median imputation).

---

## Installation

```bash
pip install -r requirements.txt
```

Required libraries include:

* PyTorch
* scikit‑learn
* astropy
* scikit‑image
* matplotlib

---

## Usage (CLI)

All workflows are exposed via the project entry point.

Run commands **from the project root**:

### Train the autoencoder

```bash
python -m src train_ae
```

### Cluster latents with GMM

```bash
python -m src gmm -k 4 --radius_weight 0.2
```

### Test reconstructions

```bash
python -m src test
```

> `__main__.py` acts as a thin dispatcher so future UIs (GUI / web) can reuse the same logic.

---

## Why This Architecture

* **Unsupervised**: no labels required
* **Physics‑aware**: optional physical scalars guide clustering
* **Extensible**: easy to add new clustering methods or features
* **UI‑ready**: parameters (`k`, weights) already decoupled from logic

---

## Current Status

* Autoencoder training ✔
* Latent extraction ✔
* GMM clustering ✔
* Reconstruction metrics ✔
* Visualization ✔

Planned:

* Interactive UI (sliders for clusters/weights)
* Experiment tracking
* Paper‑ready analysis

---

## Disclaimer

This project is intended for **research and experimentation** with Euclid‑like data and is not an official ESA pipeline.

---

## Author

**Stan Rareș Constantin**
ICT / Machine Learning
