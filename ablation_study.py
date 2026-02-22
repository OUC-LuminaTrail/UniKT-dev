"""Ablation study runner - main entry point.

Run batch ablation experiments using model variants.
"""

import argparse

import model  # noqa: F401
from utils.ablation.config_loader import load_config
from utils.ablation.runner import AblationRunner
from utils.core import get_logger

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

    # Print summary
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
