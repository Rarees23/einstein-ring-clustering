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
│   ├── __main__.py              # CLI entry point / dispatcher
│   ├── autoencoder.py           # Convolutional autoencoder + training loop
│   ├── cluster_gmm.py           # Bayesian GMM clustering in latent space
│   ├── evaluate_autoencoder.py  # Reconstruction evaluation & metrics
│   ├── visualize_clusters.py    # 2D PCA visualization of latent clusters
│   ├── preprocess.py            # Shared FITS loading & preprocessing
│   ├── run_inference.py         # Streamlit UI for interactive cluster browsing
│
├── data/                        # FITS images (not tracked)
├── saved_models/                # Trained model weights
├── results/                     # Clustering outputs, labels, plots
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

### Cluster latents with Bayesian GMM

```bash
python -m src gmm \
  --max_clusters 50 \
  --latent_amplify 5.0 \
  --empty_percentile 15.0
```

Flags:

- **--max_clusters**: maximum number of mixture components (Dirichlet process prior prunes unused ones)
- **--latent_amplify**: global scaling factor for latent space before clustering
- **--empty_percentile**: percentile over total image flux used to tag "empty" images (label `-1`)

### Test reconstructions

```bash
python -m src test
```

### Interactive clustering UI (Streamlit)

```bash
python -m src inference
```

or equivalently:

```bash
streamlit run src/run_inference.py
```

> `__main__.py` acts as a thin dispatcher so future UIs (GUI / web) can reuse the same logic.

---

## Why This Architecture

* **Unsupervised**: no labels required
* **Physics‑aware**: optional physical scalars guide clustering
* **Extensible**: easy to add new clustering methods or features
* **UI‑ready**: clustering hyperparameters (`max_clusters`, `latent_amplify`, etc.) are decoupled from core logic

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
