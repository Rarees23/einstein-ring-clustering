"""
Entry point for the Euclid strong lens project.

Usage:
  python -m src train_ae
  python -m src gmm --clusters 4 --radius_weight 0.2
  python -m src test
  python -m src inference
"""

import argparse
import subprocess
import sys
import os

SRC_DIR = os.path.dirname(__file__)

def main():
    parser = argparse.ArgumentParser(
        description="Run Euclid strong lens workflows"
    )

    parser.add_argument(
        "mode",
        choices=["train_ae", "gmm", "test", "inference"],
        help="train_ae = train autoencoder | gmm = clustering | test = evaluation | inference = run clustering on new images"
    )

    parser.add_argument(
        "--clusters", "-k",
        type=int,
        default=4,
        help="Number of GMM clusters (only for gmm mode)"
    )

    parser.add_argument(
        "--radius_weight",
        type=float,
        default=0.2,
        help="Radius/thickness weight (only for gmm mode)"
    )

    args = parser.parse_args()

    if args.mode == "train_ae":
        script = os.path.join(SRC_DIR, "autoencoder.py")
        if not os.path.exists(script):
            raise FileNotFoundError(f"Autoencoder script not found: {script}")
        subprocess.run([sys.executable, script], check=True)

    elif args.mode == "gmm":
        script = os.path.join(SRC_DIR, "cluster_gmm.py")
        if not os.path.exists(script):
            raise FileNotFoundError(f"GMM clustering script not found: {script}")
        subprocess.run(
            [
                sys.executable,
                script,
                "--clusters", str(args.clusters),
                "--radius_weight", str(args.radius_weight),
            ],
            check=True,
        )

    elif args.mode == "test":
        script = os.path.join(SRC_DIR, "evaluate_autoencoder.py")
        if not os.path.exists(script):
            raise FileNotFoundError(f"Evaluation script not found: {script}")
        subprocess.run([sys.executable, script], check=True)

    elif args.mode == "inference":
        script = os.path.join(SRC_DIR, "run_inference.py")
        if not os.path.exists(script):
            raise FileNotFoundError(f"Inference script not found: {script}")
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
