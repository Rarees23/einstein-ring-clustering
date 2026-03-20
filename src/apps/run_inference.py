from __future__ import annotations

import os
import numpy as np
import streamlit as st

from src.core.runtime import RuntimeConfig
from src.pipelines.infer import infer_labels


def main() -> None:
    cfg = RuntimeConfig.default()
    st.set_page_config(layout="wide")
    st.title("Einstein Ring Clusters")
    data_path = st.sidebar.text_input("Path to FITS data folder", value=cfg.data_dir)
    if not os.path.isdir(data_path):
        st.error("Invalid data folder path")
        st.stop()

    with st.spinner("Loading data and computing clusters..."):
        out = infer_labels(data_path, cfg=cfg)

    images = out["images"]
    filenames = out["filenames"]
    labels = out["labels"]

    clusters = np.unique(labels)
    cluster_choice = st.sidebar.selectbox("Choose cluster", clusters)
    selected_idxs = np.where(labels == cluster_choice)[0]
    cols = st.columns(6)
    for i, idx in enumerate(selected_idxs):
        col = cols[i % 6]
        col.image(images[idx][0], width=120, clamp=True, channels="GRAY")
        col.caption(filenames[idx])


if __name__ == "__main__":
    main()

