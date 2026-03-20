"""CLI for ``python -m src`` — thin dispatcher to pipelines."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from src.core.runtime import RuntimeConfig
from src.data.euclid_to_data import main as euclid_to_data_cli_main
from src.pipelines.cluster import run_cluster
from src.pipelines.evaluate import run_evaluate
from src.pipelines.train import run_train


def main() -> None:
    _cfg = RuntimeConfig.default()
    parser = argparse.ArgumentParser(description="Run Euclid strong lens workflows")
    parser.add_argument(
        "mode",
        choices=["train_ae", "gmm", "test", "inference", "euclid_to_data"],
        help="train_ae | gmm | test | inference | euclid_to_data",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=_cfg.gmm_max_clusters,
        help="Max GMM components (gmm mode)",
    )
    parser.add_argument("--latent_amplify", type=float, default=5.0, help="Latent scale factor (gmm mode)")
    parser.add_argument("--empty_percentile", type=float, default=15.0, help="Empty-image percentile (gmm mode)")
    args = parser.parse_args()

    if args.mode == "train_ae":
        run_train()
    elif args.mode == "gmm":
        run_cluster(
            max_clusters=args.max_clusters,
            latent_amplify=args.latent_amplify,
            empty_percentile=args.empty_percentile,
        )
    elif args.mode == "test":
        run_evaluate()
    elif args.mode == "inference":
        # Streamlit must be started with `streamlit run`, not plain Python.
        app = os.path.join(os.path.dirname(__file__), "apps", "run_inference.py")
        print(f"Launching Streamlit ({app})…", flush=True)
        rc = subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app,
                "--browser.gatherUsageStats",
                "false",
            ]
        )
        raise SystemExit(rc)
    elif args.mode == "euclid_to_data":
        argv = sys.argv[:]
        for i, a in enumerate(argv[1:], start=1):
            if a == "euclid_to_data":
                sys.argv = [argv[0]] + argv[i + 1 :]
                break
        try:
            euclid_to_data_cli_main()
        finally:
            sys.argv = argv


if __name__ == "__main__":
    main()
