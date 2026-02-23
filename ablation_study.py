"""Ablation study runner - main entry point.

Run batch ablation experiments using model variants.
"""

import argparse

import model  # noqa: F401
from utils.ablation.config_loader import load_config
from utils.ablation.result_formatter import AblationResultFormatter
from utils.ablation.runner import AblationRunner
from utils.core import get_logger
from utils.experiment_manager import ExperimentManager, ExperimentType

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation studies using model variants"
    )
    parser.add_argument("--config", help="Path to ablation config JSON")
    args = parser.parse_args()

    if not args.config:
        parser.error("--config is required")

    # Load config
    config = load_config(args.config)

    # Print study info
    logger.info(f"{'=' * 60}")
    logger.info(f"Ablation Study: {config.study_name}")
    logger.info(f"Base Model: {config.base_model}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Number of ablations: {len(config.ablations)}")
    logger.info(f"{'=' * 60}")

    # Run experiments
    runner = AblationRunner(config)
    results = runner.run_all()

    # Create experiment manager for ablation study
    exp_manager = ExperimentManager(
        exp_type=ExperimentType.ABLATION,
        model_name=config.base_model,
        dataset_name=config.dataset,
        tags=["study"],
    )

    # Format and print results
    try:
        formatter = AblationResultFormatter(results, ranking_metric="auc")
        logger.info("=" * 60)
        logger.info("ABLATION STUDY RESULTS")
        logger.info("=" * 60)
        formatter.print_console_table()

        # Export to CSV
        csv_path = exp_manager.exp_dir / "results.csv"
        formatter.export_to_csv(str(csv_path))
        logger.info(f"Results exported to: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to use result formatter: {e}")
        logger.info("Falling back to simple logging...")

        # Fallback: simple logging
        logger.info("=" * 60)
        logger.info("ABLATION STUDY RESULTS")
        logger.info("=" * 60)
        for result in results:
            logger.info(f"{result['name']}:")
            logger.info(f"  Variant: {result['variant']}")
            metrics = result.get("metrics", {})
            if not metrics:
                logger.info("  No metrics available")
                continue
            for metric_name, metric_value in sorted(metrics.items()):
                if isinstance(metric_value, float):
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
                else:
                    logger.info(f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
