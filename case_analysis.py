#!/usr/bin/env python3
r"""Case Analysis Framework for KT Models.

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
import sys
from pathlib import Path

import model  # noqa: F401
from utils.case_analysis import HeatmapVisualizer
from utils.case_analysis.result_collector import ResultCollector
from utils.core import ANALYZERS, add_file_handler, get_logger
from utils.data_process import get_data_source

logger = get_logger(__name__)


def cmd_inference(args):
    """Step 1: Run inference and save predictions."""
    from utils.config import load_run_config_archive

    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = run_dir / "best_model.pth"
    run_config_path = run_dir / "run_config.yaml"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not run_config_path.exists():
        raise FileNotFoundError(f"RunConfig archive not found: {run_config_path}")

    rc = load_run_config_archive(run_config_path)
    model_name = rc.experiment.model_name
    dataset_name = rc.data.dataset

    logger.info(f"Starting inference for {model_name} on {dataset_name}...")

    data_src = get_data_source(rc)
    AnalyzerClass = ANALYZERS.get(model_name)
    analyzer = AnalyzerClass(
        rc=rc, data_src=data_src, checkpoint_path=str(checkpoint_path)
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
    """Run the case analysis workflow (inference, selection, plotting)."""
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

    # --run_dir is declared on each subparser, so a bare `case_analysis.py`
    # invocation has no run_dir attribute; fall back to help as it did before.
    if args.command is None:
        parser.print_help()
        return

    run_dir = Path(args.run_dir).resolve()
    # add_file_handler mkdirs its target parent, so validate first — otherwise a
    # typo'd run_dir spawns a phantom case_analysis/ dir that hides the real error.
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)
    add_file_handler(run_dir / "case_analysis" / "run.log")

    if args.command == "inference":
        cmd_inference(args)
    elif args.command == "select":
        cmd_select(args)
    elif args.command == "plot":
        cmd_plot(args)


if __name__ == "__main__":
    main()
