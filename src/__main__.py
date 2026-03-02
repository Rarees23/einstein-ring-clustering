"""
Entry point for the Euclid strong lens project.

Usage:
  python -m src train_ae
  python -m src gmm [--max_clusters 50] [--latent_amplify 5.0] [--empty_percentile 15.0]
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

    parser.add_argument("--max_clusters", type=int, default=50, help="Max GMM components (gmm mode)")
    parser.add_argument("--latent_amplify", type=float, default=5.0, help="Latent scale factor (gmm mode)")
    parser.add_argument("--empty_percentile", type=float, default=15.0, help="Empty-image percentile (gmm mode)")

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
                "--max_clusters", str(args.max_clusters),
                "--latent_amplify", str(args.latent_amplify),
                "--empty_percentile", str(args.empty_percentile),
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
