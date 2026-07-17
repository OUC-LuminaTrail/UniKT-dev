"""Evaluate a trained model on the test set.

Usage:
    python evaluate.py --run_dir runs/normal/GIKT_assist09_20260520-232131_fold0_bs128
    python evaluate.py --run_dir ... --checkpoint last_checkpoint.pth
    python evaluate.py --run_dir ... --device cpu
"""

import argparse
import sys
from pathlib import Path

import model  # noqa: F401
from utils.config import load_run_config_archive
from utils.core import TRAINERS, get_logger, seed_everything
from utils.data_process import get_data_source
from utils.experiment_manager import ExperimentManager

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments for evaluation mode."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained KT model on the test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the run directory (contains best_model.pth and run_config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default="best_model.pth",
        help="Checkpoint filename within run_dir (default: best_model.pth)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. 'cpu' or 'cuda' (default: from saved config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size override (default: from saved config)",
    )
    parser.add_argument(
        "--data_base_path",
        default=None,
        help="Data base path override (default: from saved config)",
    )
    return parser.parse_args()


def main():
    """Evaluate a trained model checkpoint on the test set."""
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = run_dir / args.checkpoint
    run_config_path = run_dir / "run_config.yaml"

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not run_config_path.exists():
        logger.error(f"RunConfig archive not found: {run_config_path}")
        sys.exit(1)

    rc = load_run_config_archive(run_config_path)
    model_name = rc.experiment.model_name
    dataset_name = rc.data.dataset

    logger.info(f"Model: {model_name}  Dataset: {dataset_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Override for evaluation mode
    rc.general.swanlab = False
    rc.general.checkpoint_path = None  # weights loaded manually after build
    rc.general.skip_test = True  # prevent TestEvaluationCallback during build
    if args.device is not None:
        rc.general.device = args.device
    if args.batch_size is not None:
        rc.model.batch_size = args.batch_size
    if args.data_base_path is not None:
        rc.data.data_base_path = args.data_base_path

    seed_everything(rc.general.seed, deterministic=not rc.general.no_deterministic)

    exp_manager = ExperimentManager.from_run_dir(run_dir)
    logger.info(f"Loading dataset: {dataset_name}...")
    data_src = get_data_source(rc)

    logger.info(f"Initializing trainer for model: {model_name}...")
    trainer = TRAINERS.get(model_name)(
        rc=rc, data_src=data_src, exp_manager=exp_manager
    )
    trainer.load_weights(str(checkpoint_path))

    logger.info("Running evaluation on test set...")
    trainer.evaluate()


if __name__ == "__main__":
    main()
