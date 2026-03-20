# Architecture (for contributors)

This repo is structured so **many people can work in parallel** and **swap or extend one part** (data loading, model, clustering, UI) **without rewriting the whole pipeline**.

## Principles

1. **Dependencies point inward** — outer layers call inner ones, not the reverse (e.g. `data` does not import `pipelines`).
2. **Orchestration stays thin** — `pipelines/` wires steps; it should not contain heavy algorithms.
3. **One place per concern** — if you’re unsure where something goes, use the table below.

## Package map

| Package | Responsibility | Typical edits |
|--------|----------------|---------------|
| **`src/core/`** | Run config (`RuntimeConfig`), output paths, manifests, seeds | New CLI flags, default hyperparameters, folder layout for artifacts |
| **`src/data/`** | Raw FITS I/O, normalization, Euclid→folder conversion, low-level dataset loading, split/QC helpers | New FITS conventions, different preprocess (clip %, resize), file discovery rules |
| **`src/datasets/`** | Facade over `data`: `PreprocessedCatalog`, PyTorch `Dataset` views | How splits are exposed to training, extra metadata per sample |
| **`src/models/`** | Encoder/decoder architecture, `build_autoencoder` | New backbones, latent size, skip connections |
| **`src/training/`** | Loss, optimizer loop, early stopping | New losses, schedulers, augmentation hooks (if you add them here) |
| **`src/features/`** | Latent extraction contract (`extract_latents`, schema version) | Different pooling, multi-scale latents — **keep in sync** with clustering/infer |
| **`src/clustering/`** | Empty-image handling, BGMM fit/predict | Other clusterers (k-means, HDBSCAN), different priors |
| **`src/evaluation/`** | Reconstruction metrics, optional plots (e.g. UMAP) | New metrics, plots for papers |
| **`src/pipelines/`** | Stage scripts: train, cluster, evaluate, infer | Glue only — call into layers above |
| **`src/apps/`** | Streamlit (or other UIs) | Layout, filters, deployment |
| **`src/__main__.py`** | CLI dispatch to pipelines | New modes, pass-through args |

## “I want to change…”

| Goal | Where to work first |
|------|---------------------|
| How FITS are read or normalized | `src/data/preprocess.py`, `src/data/dataset.py` |
| Train/val/test splitting or QC | `src/data/dataset.py`, `src/datasets/preprocessed_catalog.py` |
| Autoencoder architecture | `src/models/` |
| Training objective or schedule | `src/training/trainer.py` |
| What “latent vector” means for clustering | `src/features/latent.py` (+ bump schema if needed) |
| Clustering algorithm | `src/clustering/service.py` |
| Reconstruction / paper metrics | `src/evaluation/metrics.py` |
| End-to-end stage order or artifacts | `src/pipelines/*.py` |
| CLI commands | `src/__main__.py` |

## Collaboration tips

- **Contract changes** (e.g. latent shape): update `LATENT_SCHEMA_VERSION` / `latent_schema_version` in manifests and re-run cluster + anything that consumes GMM artifacts.
- **Avoid** importing `pipelines` from `data` or `models` — that creates cycles and makes refactors painful.
- **Prefer** adding a small module in the right package over growing a 2000-line pipeline file.

## Runtime layout (defaults)

- Input FITS (preprocessed): `data/` (see `RuntimeConfig.data_dir`)
- Checkpoints: `saved_models/`
- GMM + latents: `results/gmm_output/`
- Eval NPZ: `results/eval_output/`
- Run logs: `results/runs/<timestamp>_<stage>/`

Change paths in **`RuntimeConfig`** / **`build_artifact_paths`** so all stages stay consistent.
