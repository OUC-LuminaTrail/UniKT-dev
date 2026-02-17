#!/usr/bin/env python3
"""Case Analysis Framework for KT Models.

This script provides a command-line interface for:
1. Running inference on trained models and saving predictions
2. Selecting users based on filtering criteria
3. Generating heatmap visualizations for selected users

Usage:
    # Step 1: Run inference (auto-discovery)
    python case_analysis.py inference \\
        -c runs/exp1/best_model.pth \\
        --output_dir outputs/case_analysis/gikt_assist09

    # Step 1: Run inference (explicit hyperparams file)
    python case_analysis.py inference \\
        -c runs/exp1/best_model.pth \\
        --hyperparams /path/to/hyperparameters.json \\
        --output_dir outputs/case_analysis/gikt_assist09

    # Step 2: Select users
    python case_analysis.py select \\
        --predictions_path outputs/case_analysis/gikt_assist09/predictions.parquet \\
        --output_dir outputs/case_analysis/gikt_assist09/diverse \\
        --strategy diverse --num_users 10

    # Step 3: Generate visualizations
    python case_analysis.py plot \\
        --predictions_path outputs/case_analysis/gikt_assist09/predictions.parquet \\
        --selected_users outputs/case_analysis/gikt_assist09/diverse/selected_users.json \\
        --output_dir outputs/case_analysis/gikt_assist09/diverse
"""

import argparse
import json
from pathlib import Path

import model  # noqa: F401
from utils.case_analysis import HeatmapVisualizer
from utils.case_analysis.result_collector import ResultCollector
from utils.core import ANALYZERS, get_logger
from utils.data_process import get_data_source

logger = get_logger(__name__)


def load_model_params(
    checkpoint_path: str, hyperparams_path: str | None = None
) -> tuple[argparse.Namespace, str, str]:
    """Load model parameters from hyperparameter JSON file.

    Priority:
    1. Auto-discover from checkpoint directory (default)
    2. Explicit --hyperparams file (fallback)

    Args:
        checkpoint_path: Path to model checkpoint file
        hyperparams_path: Optional explicit path to hyperparameters JSON

    Returns:
        Tuple of (Namespace with model parameters, model_name, dataset_name)

    Raises:
        FileNotFoundError: If hyperparameter file doesn't exist in either location
        ValueError: If file format is invalid or required parameters are missing
    """
    from utils.hyperparam_manager import HyperparameterManager

    target_path = None

    if hyperparams_path is not None:
        target_path = hyperparams_path
    else:
        checkpoint_dir = Path(checkpoint_path).parent.resolve()
        auto_hyperparams = checkpoint_dir / "hyperparameters.json"
        if auto_hyperparams.exists():
            target_path = str(auto_hyperparams)

    if target_path is None or not Path(target_path).exists():
        checkpoint_dir = Path(checkpoint_path).parent.resolve()
        raise FileNotFoundError(
            f"Hyperparameter file not found. Searched in:\n"
            f"  1. Checkpoint directory: {checkpoint_dir}/hyperparameters.json\n"
            f"  2. Explicit --hyperparams argument: {hyperparams_path}\n"
            f"Please use --hyperparams to specify the file location."
        )

    try:
        manager = HyperparameterManager()
        manager.load(target_path)

        flat_dict = manager._flatten_dict(manager.hyperparams)

        merged_dict = {}
        for key, value in flat_dict.items():
            base_key = key.split(".")[-1] if "." in key else key
            merged_dict[base_key] = value

        params_ns = argparse.Namespace(**merged_dict)

        model_name = manager.metadata.get("model_name")
        dataset_name = manager.metadata.get("dataset_name")

        if not model_name:
            raise ValueError(
                f"Missing 'model_name' in hyperparameter file metadata: {target_path}"
            )
        if not dataset_name:
            raise ValueError(
                f"Missing 'dataset_name' in hyperparameter file metadata: {target_path}"
            )

        return params_ns, model_name, dataset_name

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in hyperparameter file '{target_path}': {e}"
        ) from e
    except Exception as e:
        raise ValueError(
            f"Error loading hyperparameters from '{target_path}': {e}"
        ) from e


def cmd_inference(args):
    """Step 1: Run inference and save predictions."""
    # 1. Load all hyperparameters from file (auto-discover or explicit)
    model_args, model_name, dataset_name = load_model_params(
        checkpoint_path=args.checkpoint, hyperparams_path=args.hyperparams
    )

    logger.info(f"Starting inference for {model_name} on {dataset_name}...")

    # 2. Load data source using parameters from file
    data_src = get_data_source(dataset_name, model_args)

    # 3. Create analyzer (model_args contains all necessary parameters)
    AnalyzerClass = ANALYZERS.get(model_name)
    analyzer = AnalyzerClass(
        args=model_args, data_src=data_src, checkpoint_path=args.checkpoint
    )

    # 4. Run inference
    result_collector = analyzer.run_inference()

    # 5. Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_path = output_path / "predictions.parquet"
    result_collector.save(str(predictions_path))

    # 6. Calculate and save user metrics
    user_metrics = result_collector.calculate_user_metrics()
    metrics_path = output_path / "user_summaries.parquet"
    user_metrics.to_parquet(metrics_path, index=False)

    logger.info("✓ Inference complete!")
    logger.info(f"Predictions saved to: '{predictions_path}'")
    logger.info(f"User metrics saved to: '{metrics_path}'")
    logger.info(f"Total predictions: {len(result_collector.to_dataframe())}")


def cmd_select(args):
    """Step 2: Select users from existing predictions."""
    logger.info("Selecting users from predictions...")

    # 1. Load predictions
    result_collector = ResultCollector.load(args.predictions_path)

    # 2. Select users
    selected_users = result_collector.select_users(
        num_attempts_range=(args.min_seq_len, args.max_seq_len),
        error_rate_range=(args.min_error, args.max_error),
        max_users=args.num_users,
        strategy=args.strategy,
    )

    if len(selected_users) == 0:
        logger.warning("No users selected. Try adjusting the filtering criteria.")
        return

    # 3. Save selection
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    user_metrics = result_collector.calculate_user_metrics()
    selected_metrics = user_metrics[user_metrics["user_id"].isin(selected_users)]

    # Save as JSON
    selected_users_path = output_path / "selected_users.json"
    selected_metrics.to_json(selected_users_path, orient="records", indent=2)

    logger.info(f"Output directory: {output_path}")
    logger.info(f"Selected users saved to: {selected_users_path}")

    # Print summary statistics
    logger.info("Selected users statistics:")
    logger.info(
        f" - Num attempts: {selected_metrics['num_attempts'].min():.0f} - {selected_metrics['num_attempts'].max():.0f}"
    )
    logger.info(
        f" - Error rate: {selected_metrics['error_rate'].min():.3f} - {selected_metrics['error_rate'].max():.3f}"
    )
    logger.info(f" - Accuracy: {selected_metrics['accuracy'].mean():.3f} avg")


def cmd_plot(args):
    """Step 3: Generate visualizations for selected users."""
    logger.info("Generating visualizations...")

    # 1. Load predictions
    result_collector = ResultCollector.load(args.predictions_path)

    # 2. Load selected users
    selected_data = json.loads(Path(args.selected_users).read_text())
    selected_users = [u["user_id"] for u in selected_data]

    logger.info(f"Generating plots for {len(selected_users)} users...")

    # 3. Generate visualizations
    output_path = Path(args.output_dir) / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    visualizer = HeatmapVisualizer()

    # Generate individual user heatmaps
    for user_id in selected_users:
        user_data = result_collector.get_user_sequence(user_id)
        fig = visualizer.plot_user_heatmap(
            user_data,
            user_id,
            output_path=str(output_path / f"user_{user_id}_heatmap.png"),
        )
        # Close figure to free memory
        import matplotlib.pyplot as plt

        plt.close(fig)

    logger.info(f"Generated {len(selected_users)} individual user heatmaps")
    logger.info(f"Figures saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="KT Case Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Run inference (model/dataset auto-detected from hyperparameters.json)
  python case_analysis.py inference \\
      -c runs/exp1/best_model.pth --output_dir outputs/case_analysis/gikt_assist09

  # Step 1: Run inference (explicit hyperparams file)
  python case_analysis.py inference \\
      -c runs/exp1/best_model.pth --hyperparams /path/to/hyperparameters.json \\
      --output_dir outputs/case_analysis/gikt_assist09

  # Step 2: Select diverse users
  python case_analysis.py select \\
      --predictions_path outputs/case_analysis/gikt_assist09/predictions.parquet \\
      --output_dir outputs/case_analysis/gikt_assist09/diverse \\
      --strategy diverse --num_users 10

  # Step 3: Generate visualizations
  python case_analysis.py plot \\
      --predictions_path outputs/case_analysis/gikt_assist09/predictions.parquet \\
      --selected_users outputs/case_analysis/gikt_assist09/diverse/selected_users.json \\
      --output_dir outputs/case_analysis/gikt_assist09/diverse
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: inference
    parser_inference = subparsers.add_parser(
        "inference", help="Run model inference and save predictions"
    )
    parser_inference.add_argument(
        "-c", "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser_inference.add_argument(
        "--hyperparams",
        type=str,
        default=None,
        help="Path to hyperparameters JSON file (default: auto-detect from checkpoint directory)",
    )
    parser_inference.add_argument(
        "--data_base_path", default="./data", help="Data base path (default: ./data)"
    )
    parser_inference.add_argument(
        "--output_dir", required=True, help="Output directory for results"
    )

    # Subcommand: select
    parser_select = subparsers.add_parser(
        "select", help="Select users from predictions based on filtering criteria"
    )
    parser_select.add_argument(
        "--predictions_path", required=True, help="Path to predictions.parquet"
    )
    parser_select.add_argument(
        "--output_dir", required=True, help="Output directory for selected users"
    )
    parser_select.add_argument(
        "--num_users",
        type=int,
        default=10,
        help="Maximum number of users to select (default: 10)",
    )
    parser_select.add_argument(
        "--min_seq_len",
        type=int,
        default=20,
        help="Minimum sequence length (default: 20)",
    )
    parser_select.add_argument(
        "--max_seq_len",
        type=int,
        default=200,
        help="Maximum sequence length (default: 200)",
    )
    parser_select.add_argument(
        "--min_error", type=float, default=0.1, help="Minimum error rate (default: 0.1)"
    )
    parser_select.add_argument(
        "--max_error", type=float, default=0.9, help="Maximum error rate (default: 0.9)"
    )
    parser_select.add_argument(
        "--strategy",
        choices=["diverse", "extreme", "random"],
        default="diverse",
        help="Selection strategy: diverse (sample from error bins), extreme (highest errors), random (default: diverse)",
    )

    # Subcommand: plot
    parser_plot = subparsers.add_parser(
        "plot", help="Generate heatmap visualizations for selected users"
    )
    parser_plot.add_argument(
        "--predictions_path", required=True, help="Path to predictions.parquet"
    )
    parser_plot.add_argument(
        "--selected_users", required=True, help="Path to selected_users.json"
    )
    parser_plot.add_argument(
        "--output_dir", required=True, help="Output directory for figures"
    )

    args = parser.parse_args()

    if args.command == "inference":
        cmd_inference(args)
    elif args.command == "select":
        cmd_select(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
