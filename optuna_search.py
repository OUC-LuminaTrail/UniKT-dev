"""Optuna hyperparameter search runner.

Provides a command-line interface for running Optuna-based hyperparameter
searches on KT models. Supports configurable parameter spaces, multiple
optimization metrics, and trial history export.
"""

import argparse
import os

import model  # noqa: F401
from utils.config import (
    DataParams,
    EarlyStoppingParams,
    GeneralParams,
    get_model_params,
)
from utils.core import TRAINERS, get_logger
from utils.data_process import get_data_source
from utils.experiment_manager import ExperimentManager, ExperimentType
from utils.optuna_utils import (
    OptunaTunerBuilder,
    TrainerObjectiveWrapper,
    direction_for_metric,
    load_config_from_json,
    load_param_space_from_json,
)

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    # Pre-parse model name to dynamically add model-specific arguments
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("-m", "--model", type=str)
    temp_args, _ = temp_parser.parse_known_args()

    model_name = temp_args.model

    # Build the full argument parser
    parser = argparse.ArgumentParser(description="Unified Optuna Hyperparameter Search")

    # Model selection
    available_models = TRAINERS.keys()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=available_models,
        help=f"Model to search hyperparameters for. Available: {', '.join(available_models)}",
    )

    # Optuna configuration
    optuna_group = parser.add_argument_group("Optuna Configuration")
    optuna_group.add_argument(
        "--optuna_config",
        type=str,
        default="./configs/optuna/optuna_config.json",
        help="Path to Optuna config JSON file",
    )
    optuna_group.add_argument(
        "--param_space",
        type=str,
        default=None,
        help="Path to parameter space JSON file (default: ./configs/optuna/param_space_<model>.json)",
    )
    optuna_group.add_argument(
        "--metric",
        type=str,
        choices=["auc", "acc", "rmse", "loss"],
        default="auc",
        help="Metric to optimize",
    )

    # Data parameters
    DataParams.add_args(parser)

    # Base training parameters
    EarlyStoppingParams.add_args(parser)

    # Common parameters
    GeneralParams.add_args(parser)

    # If model is specified, dynamically add model-specific parameters
    if model_name:
        model_params_cls = get_model_params(model_name)
        if model_params_cls:
            model_params_cls.add_args(parser)

    args = parser.parse_args()

    # Set default parameter space path
    if args.param_space is None:
        args.param_space = f"./configs/optuna/param_space_{args.model.lower()}.json"

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Load config first to annotate experiment directory with n_trials
    logger.info(f"Loading Optuna config from: {args.optuna_config}")
    optuna_config = load_config_from_json(args.optuna_config)

    # Create experiment manager
    exp_manager = ExperimentManager(
        exp_type=ExperimentType.HYPERPARAM_SEARCH,
        model_name=args.model,
        dataset_name=args.dataset,
        base_dir="runs",
        tags=[f"n_trials{optuna_config.n_trials}"],
    )

    logger.info(f"Experiment directory: {exp_manager.get_log_dir()}")

    logger.info("=" * 60)
    logger.info(f"{args.model} Optuna Hyperparameter Search")
    logger.info("=" * 60)

    optuna_config.save_dir = exp_manager.get_log_dir()
    # Optimization direction is determined by the metric, overriding the config file's direction
    optuna_config.directions = [direction_for_metric(args.metric)]
    logger.info(f"Optimizing metric '{args.metric}' ({optuna_config.directions[0]})")

    # Load parameter space
    logger.info(f"Loading parameter space from: {args.param_space}")
    param_spaces = load_param_space_from_json(args.param_space)

    # Create data source factory function
    def data_src_factory():
        return get_data_source(dataset_name=args.dataset, args=args)

    # Get trainer class
    trainer_class = TRAINERS.get(args.model)

    # Create objective function wrapper
    objective_wrapper = TrainerObjectiveWrapper(
        trainer_class=trainer_class,
        data_src_fn=data_src_factory,
        base_args=args,
        metric_name=args.metric,
        max_epochs=args.epochs,
        exp_manager=exp_manager,
    )

    # Use the builder to create an OptunaTuner
    tuner = (
        OptunaTunerBuilder()
        .with_config(optuna_config)
        .with_param_spaces(param_spaces)
        .with_objective(objective_wrapper)
        .build()
    )

    # Execute hyperparameter search
    logger.info(
        f"Starting hyperparameter search with {optuna_config.n_trials} trials..."
    )
    best_params = tuner.search()

    # Print results
    tuner.print_summary()

    # Get and save dataframe
    df = tuner.get_dataframe()
    if df is not None:
        log_dir = exp_manager.get_log_dir()
        df_path = os.path.join(log_dir, f"trials_history_{args.model.lower()}.csv")
        df.to_csv(df_path, index=False)
        logger.info(f"Trials history saved to: {df_path}")

    logger.info("=" * 60)
    logger.info("Search completed successfully!")
    logger.info("=" * 60)

    return best_params


if __name__ == "__main__":
    main()
