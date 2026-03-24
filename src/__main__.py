"""``python -m src``: dispatch to train, gmm, eval, viz, streamlit, euclid_to_data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from src.core.runtime import RuntimeConfig
from src.data.euclid_to_data import main as euclid_to_data_cli_main
from src.pipelines.cluster import run_cluster
from src.evaluation.visualize_reconstructions import run_reconstruction_viz
from src.pipelines.evaluate import run_evaluate
from src.pipelines.train import run_train


def main() -> None:
    _cfg = RuntimeConfig.default()
    parser = argparse.ArgumentParser(description="Run Euclid strong lens workflows")
    parser.add_argument(
        "mode",
        choices=["train_ae", "gmm", "test", "recon_viz", "inference", "euclid_to_data"],
        help="train_ae | gmm | test | recon_viz | inference | euclid_to_data",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=_cfg.gmm_max_clusters,
        help="Max GMM components (gmm mode)",
    )
    parser.add_argument("--latent_amplify", type=float, default=5.0, help="Latent scale factor (gmm mode)")
    parser.add_argument("--empty_percentile", type=float, default=15.0, help="Empty-image percentile (gmm mode)")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="recon_viz: number of random images to show (default 8)",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train"],
        default="test",
        help="recon_viz: which data split to sample from",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="train_ae: learning rate (default 1e-3)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="train_ae: early stopping patience in epochs (default: 6)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="train_ae: load weights from this .pth before training",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        metavar="NAME",
        help="train_ae: adam, adamw, or sgd (default: config)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        metavar="NAME",
        help="train_ae: masked_mse or mse (default: config)",
    )
    args = parser.parse_args()

    if args.mode == "train_ae":
        cfg = RuntimeConfig.default()
        if args.lr is not None:
            cfg.learning_rate = args.lr
        if args.patience is not None:
            cfg.early_stopping_patience = args.patience
        if args.resume is not None:
            cfg.resume_checkpoint_path = os.path.abspath(args.resume)
        if args.optimizer is not None:
            ok_o = {"adam", "adamw", "sgd"}
            if args.optimizer.lower() not in ok_o:
                raise SystemExit(f"--optimizer must be one of {sorted(ok_o)}")
            cfg.optimizer_kind = args.optimizer.lower()
        if args.loss is not None:
            ok_l = {"masked_mse", "mse"}
            if args.loss.lower() not in ok_l:
                raise SystemExit(f"--loss must be one of {sorted(ok_l)}")
            cfg.loss_kind = args.loss.lower()
        run_train(cfg)
    elif args.mode == "gmm":
        run_cluster(
            max_clusters=args.max_clusters,
            latent_amplify=args.latent_amplify,
            empty_percentile=args.empty_percentile,
        )
    elif args.mode == "test":
        metrics = run_evaluate()
        print("Evaluation complete.", metrics, flush=True)
    elif args.mode == "recon_viz":
        run_reconstruction_viz(num_samples=args.num_samples, split=args.split)
    elif args.mode == "inference":
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
