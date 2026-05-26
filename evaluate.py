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
from case_analysis import load_model_params
from utils.core import TRAINERS, get_logger
from utils.data_process import get_data_source
from utils.experiment_manager import ExperimentManager

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained KT model on the test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the run directory (contains best_model.pth and hyperparameters.json)",
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
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    checkpoint_path = run_dir / args.checkpoint
    hyperparams_path = run_dir / "hyperparameters.json"

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    if not hyperparams_path.exists():
        logger.error(f"Hyperparameters file not found: {hyperparams_path}")
        sys.exit(1)

    # Step 1: Load hyperparameters to reconstruct args
    model_args, model_name, dataset_name = load_model_params(
        checkpoint_path=str(checkpoint_path),
        hyperparams_path=str(hyperparams_path),
    )

    logger.info(f"Model: {model_name}  Dataset: {dataset_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Step 2: Override args for evaluation mode
    model_args.use_swanlab = False
    model_args.checkpoint_path = None  # We load weights manually after build
    model_args.skip_test = True  # Prevent TestEvaluationCallback during build
    model_args.device = args.device

    if args.batch_size is not None:
        model_args.batch_size = args.batch_size
    if args.data_base_path is not None:
        model_args.data_base_path = args.data_base_path

    # Step 3: Create experiment manager pointing to existing run dir
    exp_manager = ExperimentManager.from_run_dir(run_dir)

    # Step 4: Build data source
    logger.info(f"Loading dataset: {dataset_name}...")
    data_src = get_data_source(dataset_name, model_args)

    # Step 5: Instantiate trainer
    logger.info(f"Initializing trainer for model: {model_name}...")
    trainer_cls = TRAINERS.get(model_name)
    trainer = trainer_cls(
        args=model_args,
        data_src=data_src,
        exp_manager=exp_manager,
    )

    # Step 6: Load saved weights
    trainer.load_weights(str(checkpoint_path))

    # Step 7: Run evaluation
    logger.info("Running evaluation on test set...")
    trainer.evaluate()


if __name__ == "__main__":
    main()
