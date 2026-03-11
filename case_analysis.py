#!/usr/bin/env python3
"""Case Analysis Framework for KT Models.

This script provides a command-line interface for:
1. Running inference on trained models and saving predictions
2. Selecting users based on filtering criteria
3. Generating heatmap visualizations for selected users

Usage:
    # Step 1: Run inference
    python case_analysis.py inference \\
        --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128

    # Step 2: Select users
    python case_analysis.py select \\
        --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128 \\
        --strategy diverse --num_users 10

    # Step 3: Generate visualizations
    python case_analysis.py plot \\
        --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128 \\
        --selected_users diverse
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

    checkpoint_dir = Path(checkpoint_path).parent.resolve()
    target_path = hyperparams_path or str(checkpoint_dir / "hyperparameters.json")

    if not Path(target_path).exists():
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
        merged_dict = {
            k.split(".")[-1] if "." in k else k: v for k, v in flat_dict.items()
        }
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
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = run_dir / "best_model.pth"
    hyperparams_path = (
        Path(args.hyperparams) if args.hyperparams else run_dir / "hyperparameters.json"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_args, model_name, dataset_name = load_model_params(
        checkpoint_path=str(checkpoint_path),
        hyperparams_path=str(hyperparams_path) if hyperparams_path.exists() else None,
    )

    # Optional case-analysis-only sequence sampling length (for heatmap method).
    model_args.case_seq_sample_len = args.case_seq_sample_len
    model_args.case_skill_scope = args.case_skill_scope
    model_args.case_non_relevant_fill = args.case_non_relevant_fill

    logger.info(f"Starting inference for {model_name} on {dataset_name}...")

    data_src = get_data_source(dataset_name, model_args)
    AnalyzerClass = ANALYZERS.get(model_name)
    analyzer = AnalyzerClass(
        args=model_args, data_src=data_src, checkpoint_path=str(checkpoint_path)
    )
    result_collector = analyzer.run_inference()

    output_dir = run_dir / "case_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.parquet"
    result_collector.save(str(predictions_path))

    user_metrics = result_collector.calculate_user_metrics()
    metrics_path = output_dir / "user_summaries.parquet"
    user_metrics.to_parquet(metrics_path, index=False)

    df = result_collector.to_dataframe()
    logger.info("✓ Inference complete!")
    logger.info(f"Predictions saved to: '{predictions_path}'")
    logger.info(f"User metrics saved to: '{metrics_path}'")
    logger.info(f"Total predictions: {len(df)}")


def cmd_select(args):
    """Step 2: Select users from existing predictions."""
    logger.info("Selecting users from predictions...")

    run_dir = Path(args.run_dir).resolve()
    predictions_path = run_dir / "case_analysis" / "predictions.parquet"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}\nPlease run 'inference' command first."
        )

    result_collector = ResultCollector.load(str(predictions_path))

    selected_users = result_collector.select_users(
        min_num_attempts=args.min_seq_len,
        error_rate_range=(args.min_error, args.max_error),
        max_users=args.num_users,
        strategy=args.strategy,
    )

    if not selected_users:
        logger.warning("No users selected. Try adjusting the filtering criteria.")
        return

    output_dir = run_dir / "case_analysis" / args.strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    user_metrics = result_collector.calculate_user_metrics()
    selected_metrics = user_metrics[user_metrics["user_id"].isin(selected_users)]

    selected_users_path = output_dir / "selected_users.json"
    selected_metrics.to_json(selected_users_path, orient="records", indent=2)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Selected users saved to: {selected_users_path}")
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

    run_dir = Path(args.run_dir).resolve()
    predictions_path = run_dir / "case_analysis" / "predictions.parquet"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}\nPlease run 'inference' command first."
        )

    result_collector = ResultCollector.load(str(predictions_path))

    selected_users_path = args.selected_users
    if selected_users_path in ["diverse", "extreme", "random"]:
        selected_users_path = (
            run_dir / "case_analysis" / selected_users_path / "selected_users.json"
        )
    else:
        selected_users_path = Path(selected_users_path)

    if not selected_users_path.exists():
        raise FileNotFoundError(
            f"Selected users file not found: {selected_users_path}\nPlease run 'select' command first or provide a valid path."
        )

    selected_data = json.loads(selected_users_path.read_text())
    selected_users = [u["user_id"] for u in selected_data]

    logger.info(f"Generating plots for {len(selected_users)} users...")

    output_dir = selected_users_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = HeatmapVisualizer()

    import matplotlib.pyplot as plt

    for user_id in selected_users:
        user_data = result_collector.get_user_sequence(user_id)
        if args.max_seq_len and len(user_data) > args.max_seq_len:
            user_data = user_data.head(args.max_seq_len).reset_index(drop=True)
        fig = visualizer.plot_user_heatmap(
            user_data,
            user_id,
            output_path=str(output_dir / f"user_{user_id}_heatmap.png"),
        )
        plt.close(fig)

    logger.info(f"Generated {len(selected_users)} individual user heatmaps")
    logger.info(f"Figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="KT Case Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Run inference
  python case_analysis.py inference \\
      --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128

  # Step 2: Select diverse users
  python case_analysis.py select \\
      --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128 \\
      --strategy diverse --num_users 10

  # Step 3: Generate visualizations
  python case_analysis.py plot \\
      --run_dir runs/normal/GIKT_assistments09_20260217-144913_fold0_bs128 \\
      --selected_users diverse
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_inference = subparsers.add_parser(
        "inference", help="Run model inference and save predictions"
    )
    parser_inference.add_argument(
        "--run_dir",
        required=True,
        help="Path to run directory containing best_model.pth",
    )
    parser_inference.add_argument(
        "--hyperparams",
        type=str,
        default=None,
        help="Path to hyperparameters JSON file (default: auto-detect from run_dir)",
    )
    parser_inference.add_argument(
        "--data_base_path", default="./data", help="Data base path (default: ./data)"
    )
    parser_inference.add_argument(
        "--case_seq_sample_len",
        type=int,
        default=8,
        help="Finite prefix length used to determine relevant skills in case analysis (default: 8)",
    )
    parser_inference.add_argument(
        "--case_skill_scope",
        type=str,
        choices=["relevant_prefix", "relevant_full", "all"],
        default="relevant_prefix",
        help=(
            "Skill scope for knowledge-state inference: "
            "relevant_prefix (skills in first N valid steps), "
            "relevant_full (skills in full valid sequence), "
            "all (all skills in dataset)."
        ),
    )
    parser_inference.add_argument(
        "--case_non_relevant_fill",
        type=str,
        choices=["zero", "nan"],
        default="zero",
        help=(
            "Fill value for non-relevant skills when using relevant_* scope: "
            "zero (legacy behavior) or nan (recommended for heatmap readability)."
        ),
    )

    parser_select = subparsers.add_parser(
        "select", help="Select users from predictions based on filtering criteria"
    )
    parser_select.add_argument("--run_dir", required=True, help="Path to run directory")
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

    parser_plot = subparsers.add_parser(
        "plot", help="Generate heatmap visualizations for selected users"
    )
    parser_plot.add_argument("--run_dir", required=True, help="Path to run directory")
    parser_plot.add_argument(
        "--selected_users",
        required=True,
        help="Strategy name (diverse/extreme/random) or path to selected_users.json",
    )
    parser_plot.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length for plotting (default: None, no truncation)",
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
