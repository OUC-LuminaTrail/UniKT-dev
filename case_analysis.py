#!/usr/bin/env python3
r"""Case Analysis Framework for KT Models.

Plugin-driven CLI for analyzing trained KT models at the per-student
level:

1. ``inference``: restore a run, run its registered analyzer, hand the
   output to a sink (default: canonical parquet via DataFrameSink)
2. ``select``: pick representative users with a selector plugin
   (default: diverse/extreme/random over per-user metrics)
3. ``plot``: render selected users with a visualizer plugin
   (default: knowledge-state heatmap)

Usage:
    # Step 1: Run inference
    python case_analysis.py inference \
        --run_dir runs/normal/HDHKT_assistments09_xxx_fold0

    # Step 2: Select users
    python case_analysis.py select \
        --run_dir runs/normal/HDHKT_assistments09_xxx_fold0 \
        --selector diverse --num_users 10

    # Step 3: Generate visualizations
    python case_analysis.py plot \
        --run_dir runs/normal/HDHKT_assistments09_xxx_fold0 \
        --selected_users diverse
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

import pandas as pd

import model  # noqa: F401  (triggers analyzer registry discovery)
from utils.case_analysis import (
    DataFrameSink,
    compute_user_metrics,
    get_user_sequence,
    load_case_results,
)
from utils.core import (
    ANALYZERS,
    CASE_SELECTORS,
    CASE_SINKS,
    CASE_VISUALIZERS,
    get_logger,
    seed_everything,
)
from utils.data_process import get_data_source

logger = get_logger(__name__)


def _filter_supported_options(cls: type, options: dict) -> dict:
    """Drop options the target class's ``select`` method does not accept."""
    params = inspect.signature(cls.select).parameters
    return {k: v for k, v in options.items() if k in params}


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

    seed_everything(rc.general.seed, deterministic=rc.general.deterministic)

    if model_name not in ANALYZERS:
        sys.exit(
            f"Model '{model_name}' has no registered case analyzer. "
            f"Available: {sorted(ANALYZERS.keys())}"
        )
    AnalyzerClass = ANALYZERS.get(model_name)

    logger.info(f"Starting inference for {model_name} on {dataset_name}...")

    data_src = get_data_source(rc)
    if args.sink not in CASE_SINKS:
        sys.exit(f"Unknown sink '{args.sink}'. Available: {sorted(CASE_SINKS.keys())}")
    sink = CASE_SINKS.get(args.sink)()
    analyzer = AnalyzerClass(
        rc=rc,
        data_src=data_src,
        checkpoint_path=str(checkpoint_path),
        sink=sink,
        device=args.device,
        batch_size=args.batch_size,
    )
    result = analyzer.run_inference()

    if not isinstance(result, pd.DataFrame):
        logger.info(
            f"Sink '{args.sink}' produced a non-DataFrame result; "
            "persistence is the sink's own responsibility."
        )
        return

    output_dir = run_dir / "case_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.parquet"
    DataFrameSink.save(result, str(predictions_path))

    user_metrics = compute_user_metrics(result)
    metrics_path = output_dir / "user_summaries.parquet"
    user_metrics.to_parquet(metrics_path, index=False)

    logger.info("✓ Inference complete!")
    logger.info(f"Predictions saved to: '{predictions_path}'")
    logger.info(f"User metrics saved to: '{metrics_path}'")
    logger.info(f"Total predictions: {len(result)}")


def cmd_select(args):
    """Step 2: Select users from existing predictions."""
    logger.info("Selecting users from predictions...")

    run_dir = Path(args.run_dir).resolve()
    predictions_path = run_dir / "case_analysis" / "predictions.parquet"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}\nPlease run 'inference' command first."
        )

    df = load_case_results(str(predictions_path))

    if args.selector not in CASE_SELECTORS:
        sys.exit(
            f"Unknown selector '{args.selector}'. "
            f"Available: {sorted(CASE_SELECTORS.keys())}"
        )
    SelectorClass = CASE_SELECTORS.get(args.selector)

    options = _filter_supported_options(
        SelectorClass,
        {
            "min_seq_len": args.min_seq_len,
            "error_rate_range": (args.min_error, args.max_error),
            "max_users": args.num_users,
        },
    )
    selected_users = SelectorClass().select(df, **options)

    if not selected_users:
        logger.warning("No users selected. Try adjusting the filtering criteria.")
        return

    output_dir = run_dir / "case_analysis" / args.selector
    output_dir.mkdir(parents=True, exist_ok=True)

    user_metrics = compute_user_metrics(df)
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

    df = load_case_results(str(predictions_path))

    selected_users_path = args.selected_users
    if selected_users_path in CASE_SELECTORS:
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

    if args.visualizer not in CASE_VISUALIZERS:
        sys.exit(
            f"Unknown visualizer '{args.visualizer}'. "
            f"Available: {sorted(CASE_VISUALIZERS.keys())}"
        )
    VisualizerClass = CASE_VISUALIZERS.get(args.visualizer)

    logger.info(f"Generating plots for {len(selected_users)} users...")

    output_dir = selected_users_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = VisualizerClass()

    import matplotlib.pyplot as plt

    for user_id in selected_users:
        user_data = get_user_sequence(df, user_id)
        if args.max_seq_len and len(user_data) > args.max_seq_len:
            user_data = user_data.head(args.max_seq_len).reset_index(drop=True)
        fig = visualizer.plot_user(
            user_data,
            user_id,
            output_path=str(output_dir / f"user_{user_id}_heatmap.png"),
        )
        plt.close(fig)

    logger.info(f"Generated {len(selected_users)} individual user visualizations")
    logger.info(f"Figures saved to: {output_dir}")


def main():
    """Run the case analysis workflow (inference, selection, plotting)."""
    parser = argparse.ArgumentParser(
        description="KT Case Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Run inference
  python case_analysis.py inference --run_dir runs/normal/HDHKT_assistments09_xxx_fold0

  # Step 2: Select diverse users
  python case_analysis.py select \\
      --run_dir runs/normal/HDHKT_assistments09_xxx_fold0 \\
      --selector diverse --num_users 10

  # Step 3: Generate visualizations
  python case_analysis.py plot \\
      --run_dir runs/normal/HDHKT_assistments09_xxx_fold0 \\
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
        "--sink",
        default="dataframe",
        help="Case data sink plugin name (default: dataframe)",
    )
    parser_inference.add_argument(
        "--device", default=None, help="Device override (default: from run config)"
    )
    parser_inference.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Inference batch size override (default: from run config)",
    )

    parser_select = subparsers.add_parser(
        "select", help="Select users from predictions via a selector plugin"
    )
    parser_select.add_argument("--run_dir", required=True, help="Path to run directory")
    parser_select.add_argument(
        "--selector",
        default="diverse",
        help="User selector plugin name (default: diverse)",
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
        "--min_error", type=float, default=0.1, help="Minimum error rate (default: 0.1)"
    )
    parser_select.add_argument(
        "--max_error", type=float, default=0.9, help="Maximum error rate (default: 0.9)"
    )

    parser_plot = subparsers.add_parser(
        "plot", help="Generate visualizations for selected users"
    )
    parser_plot.add_argument("--run_dir", required=True, help="Path to run directory")
    parser_plot.add_argument(
        "--selected_users",
        required=True,
        help="Selector name (e.g. diverse) or path to selected_users.json",
    )
    parser_plot.add_argument(
        "--visualizer",
        default="heatmap",
        help="Visualizer plugin name (default: heatmap)",
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
