# Euclid Strong Lens Clustering

Unsupervised discovery and clustering of strong gravitational lens candidates (Einstein rings) from **Euclid FITS images** using deep latent representations and Gaussian Mixture Models.

---

## Overview

This project implements a **research‑grade unsupervised pipeline**:

1. **Convolutional Autoencoder (CAE)** learns compact latent representations of FITS images, focusing on Einstein ring structures.
2. **Latent space extraction** converts images into fixed‑length feature vectors.
3. **Gaussian Mixture Model (GMM)** clusters lenses **purely in latent space** (extension to include physical scalars such as Einstein radius / thickness is planned).
4. **Evaluation & visualization** validate reconstruction quality and inspect cluster composition.

The design cleanly separates:

* reusable ML logic (`src/` — see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for packages and extension points)
* trained models (`saved_models/`)
* experiment outputs (`results/`)

**Contributors:** the layout is meant so teams can change **data**, **model**, **training**, or **clustering** independently; the architecture doc maps each folder to “what to edit when…”.

---

## Project Structure (high level)

```text
├── src/
│   ├── __main__.py          # CLI: python -m src <mode>
│   ├── core/                # Runtime config, paths, manifests
│   ├── data/                # FITS I/O, preprocess, splits, records
│   ├── datasets/            # PreprocessedCatalog + PyTorch Dataset views
│   ├── models/              # Autoencoder
│   ├── training/            # Loss + training loop
│   ├── features/            # Latent extraction (shared contract)
│   ├── clustering/          # GMM + empty-image logic
│   ├── evaluation/          # Metrics & optional viz
│   ├── pipelines/           # Stage orchestration (train, gmm, eval, infer)
│   └── apps/                # Streamlit UI
├── data/                    # FITS images (not tracked)
├── saved_models/            # Weights
├── results/                 # GMM, eval, run manifests
├── requirements.txt
├── docs/ARCHITECTURE.md     # Contributor-oriented package map
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

- **--max_clusters**: upper bound on the number of mixture components; a Dirichlet‑process prior prunes unused ones, so the *effective* number of clusters is learned from the data.
- **--latent_amplify**: global scaling factor for latent space before clustering (helps emphasize structure in the latent space).
- **--empty_percentile**: percentile over total image flux used to tag "empty" images (label `-1` in outputs).

### Test reconstructions

```bash
python -m src test
```

### Interactive clustering UI (Streamlit)

```bash
python -m src inference
```

This runs `streamlit run` on `src/apps/run_inference.py` (requires `streamlit` — see `requirements.txt`).

You can also run the app directly:

```bash
streamlit run src/apps/run_inference.py
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
