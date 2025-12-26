Einstein Ring Clustering

This project performs unsupervised clustering of Einstein ring images using latent-space embeddings and Gaussian Mixture Models (GMM). All input images are known Einstein rings; the goal is to group them into clusters based on similarity.

Problem

Einstein rings are important astrophysical phenomena caused by strong gravitational lensing. While detection of rings is often studied, this project focuses on grouping pre-identified rings to explore patterns in shape, size, or other features without using labels.

Approach

Feature Extraction - Images are encoded into latent vectors using a convolutional autoencoder. The Einstein radius of each ring is included as a weighted feature to influence clustering.

Clustering - Gaussian Mixture Models (GMM) are applied to the latent-space features to form clusters. Clustering is fully unsupervised; no label information is used.

Analysis - Cluster assignments are analyzed to explore patterns and variations among rings. Visualizations show representative images from each cluster.

Project Structure

einstein-ring-clustering/
├── README.md # Project description
├── requirements.txt # Python dependencies
├── data/ # Images or references (not included)
├── src/
│ ├── autoencoder.py # Defines convolutional autoencoder
│ ├── feature_extraction.py # Generates latent vectors and weighted radius
│ ├── clustering.py # Runs Gaussian Mixture Model
│ └── analysis.py # Visualizes and interprets clusters
└── results/ # Cluster outputs and figures

Dependencies

Python >= 3.8, numpy, pandas, scikit-learn, torch / tensorflow, matplotlib, seaborn, joblib. Install dependencies with:

pip install -r requirements.txt

Usage

Place your Einstein ring images or references in the data/ folder, run feature extraction to create latent vectors and include weighted radius with python src/feature_extraction.py, then run clustering on the extracted features with python src/clustering.py, and finally visualize and analyze the resulting clusters with python src/analysis.py. This sequence reproduces the full pipeline: feature extraction → clustering → visualization.

Notes

All images used are Einstein rings; the project does not perform detection. Clustering is unsupervised; results may vary based on hyperparameters. This repository is intended for research and educational purposes.
