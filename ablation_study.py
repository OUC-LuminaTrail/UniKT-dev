"""Ablation study runner - main entry point.

Run batch ablation experiments using model variants.
Dataset and fold must be specified via command line arguments.
"""

import argparse

import model  # noqa: F401
from utils.ablation.config_loader import load_config
from utils.ablation.result_formatter import AblationResultFormatter
from utils.ablation.runner import AblationRunner
from utils.core import get_logger

logger = get_logger(__name__)


def main():
    """Run the ablation study workflow."""
    parser = argparse.ArgumentParser(
        description="Run ablation studies using model variants"
    )
    parser.add_argument("--config", help="Path to ablation config JSON (required)")
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=[
            "assistments09",
            "assistments12",
            "assistments17",
            "ednet_kt1",
        ],
        help="Dataset name (required)",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        default=0,
        help="Fold index for K-Fold cross-validation (default: 0)",
    )
    args = parser.parse_args()

    if not args.config:
        parser.error("--config is required")

    # Load config with required dataset and fold parameters
    config = load_config(args.config, dataset=args.dataset, fold=args.fold)

    # Print study info
    logger.info(f"{'=' * 60}")
    logger.info(f"Ablation Study: {config.study_name}")
    logger.info(f"Base Model: {config.base_model}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Fold: {config.fold}")
    logger.info(f"Number of ablations: {len(config.ablations)}")
    logger.info(f"{'=' * 60}")

    # Run experiments
    runner = AblationRunner(config)
    results, exp_base_dir = runner.run_all()

    # Format and print results
    formatter = AblationResultFormatter(results, ranking_metric="auc")
    # Export to CSV in the ablation study's base directory
    csv_path = exp_base_dir / "results.csv"
    formatter.export_to_csv(str(csv_path))
    logger.info(f"Results exported to: {csv_path}")
    formatter.print_console_table()


if __name__ == "__main__":
    main()
