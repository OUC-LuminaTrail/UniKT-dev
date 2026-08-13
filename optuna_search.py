"""Optuna hyperparameter search runner.

Provides a command-line interface for running Optuna-based hyperparameter
searches on KT models. Supports configurable parameter spaces, multiple
optimization metrics, and trial history export.
"""

import argparse
import os

import model  # noqa: F401
from utils.config import ConfigParser
from utils.core import TRAINERS, get_logger
from utils.data_process import get_data_source
from utils.experiment_manager import ExperimentManager, ExperimentType
from utils.optuna_utils import (
    OptunaTunerBuilder,
    TrainerObjectiveWrapper,
    direction_for_metric,
    load_optuna_config,
    param_spaces_from_model_config,
)

logger = get_logger(__name__)


def main():
    """Main entry point."""
    # Stage 1: optuna-specific args; parse_known_args leaves RunConfig flags in `remaining`.
    opt_parser = argparse.ArgumentParser(add_help=False)
    opt_parser.add_argument(
        "--optuna_config",
        type=str,
        default="./configs/optuna/optuna_config.yaml",
        help="Path to Optuna config yaml file",
    )
    opt_parser.add_argument(
        "--metric",
        type=str,
        choices=["auc", "acc", "rmse", "loss"],
        default="auc",
        help="Metric to optimize",
    )
    opt_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help="Resume an existing search from its run directory (reuses study.db)",
    )
    optuna_args, remaining = opt_parser.parse_known_args()

    # Stage 2: RunConfig via reflective ConfigParser on the remaining argv.
    rc = ConfigParser(
        prog="optuna_search.py", description="Unified Optuna Hyperparameter Search"
    ).parse_args(remaining)
    model_name = rc.experiment.model_name

    # Load config first to annotate experiment directory with n_trials
    logger.info(f"Loading Optuna config from: {optuna_args.optuna_config}")
    optuna_config = load_optuna_config(optuna_args.optuna_config)

    if optuna_args.resume:
        # Reuse the existing run dir so study.db and study_name match the prior
        # run, letting create_study(load_if_exists=True) restore trial history.
        exp_manager = ExperimentManager.from_run_dir(optuna_args.resume)
        logger.info(f"Resuming search in: {exp_manager.get_log_dir()}")
    else:
        exp_manager = ExperimentManager(
            exp_type=ExperimentType.HYPERPARAM_SEARCH,
            model_name=model_name,
            dataset_name=rc.data.dataset,
            base_dir="runs",
            tags=[f"n_trials{optuna_config.n_trials}"],
        )
        logger.info(f"Experiment directory: {exp_manager.get_log_dir()}")

    logger.info("=" * 60)
    logger.info(f"{model_name} Optuna Hyperparameter Search")
    logger.info("=" * 60)

    optuna_config.save_dir = exp_manager.get_log_dir()
    if optuna_args.resume:
        # study_name + db path must match the original run for load_if_exists.
        # n_trials is the number of NEW trials this invocation (Optuna semantics).
        logger.info(
            f"Resume: study_name='{optuna_config.study_name}' (must match the "
            f"original run), db={optuna_config.save_dir}/study.db"
        )
    # Optimization direction is determined by the metric, overriding the config file's direction
    optuna_config.directions = [direction_for_metric(optuna_args.metric)]
    logger.info(
        f"Optimizing metric '{optuna_args.metric}' ({optuna_config.directions[0]})"
    )

    # Search space is derived solely from the model's ModelConfig field metadata.
    param_spaces = param_spaces_from_model_config(model_name)
    if not param_spaces:
        raise ValueError(
            f"No searchable params: {model_name}Config has no fields with 'optuna' metadata."
        )
    logger.info(
        f"Searchable params from {model_name}Config: {[s.name for s in param_spaces]}"
    )

    def data_src_factory():
        return get_data_source(rc)

    trainer_class = TRAINERS.get(model_name)

    objective_wrapper = TrainerObjectiveWrapper(
        trainer_class=trainer_class,
        data_src_fn=data_src_factory,
        base_rc=rc,
        metric_name=optuna_args.metric,
        exp_manager=exp_manager,
    )

    tuner = (
        OptunaTunerBuilder()
        .with_config(optuna_config)
        .with_param_spaces(param_spaces)
        .with_objective(objective_wrapper)
        .build()
    )

    logger.info(
        f"Starting hyperparameter search with {optuna_config.n_trials} trials..."
    )
    best_params = tuner.search()

    tuner.print_summary()

    df = tuner.get_dataframe()
    if df is not None:
        log_dir = exp_manager.get_log_dir()
        df_path = os.path.join(log_dir, f"trials_history_{model_name.lower()}.csv")
        df.to_csv(df_path, index=False)
        logger.info(f"Trials history saved to: {df_path}")

    logger.info("=" * 60)
    logger.info("Search completed successfully!")
    logger.info("=" * 60)

    return best_params


if __name__ == "__main__":
    main()
