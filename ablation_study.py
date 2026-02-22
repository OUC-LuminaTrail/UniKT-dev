"""Ablation study runner - main entry point.

Run batch ablation experiments using model variants.
"""

import argparse

import model  # noqa: F401
from utils.ablation.config_loader import load_config
from utils.ablation.runner import AblationRunner
from utils.core import TRAINERS, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation studies using model variants"
    )
    parser.add_argument("--config", help="Path to ablation config JSON")
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List available model variants and exit",
    )
    args = parser.parse_args()

    if args.list_variants:
        print("Available model variants (registered trainers):")
        for name in sorted(TRAINERS.keys()):
            print(f"  - {name}")
        return

    if not args.config:
        parser.error("--config is required unless using --list-variants")

    # Load config
    config = load_config(args.config)

    # Print study info
    print(f"\n{'=' * 60}")
    print(f"Ablation Study: {config.study_name}")
    print(f"Base Model: {config.base_model}")
    print(f"Dataset: {config.dataset}")
    print(f"Number of ablations: {len(config.ablations)}")
    print(f"{'=' * 60}\n")

    # Run experiments
    runner = AblationRunner(config)
    results = runner.run_all()

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Variant: {result['variant']}")
        for metric_name, metric_value in sorted(result["metrics"].items()):
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
