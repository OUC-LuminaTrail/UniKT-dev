"""Optuna tuner wrapper and utility tools."""

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import yaml
from dataclasses import asdict, is_dataclass
from optuna.samplers import GridSampler

from utils.core import get_logger

from .config import HyperparameterSpace, OptunaConfig

logger = get_logger(__name__)


class OptunaTuner:
    """Optuna hyperparameter search engine."""

    def __init__(
        self,
        config: OptunaConfig,
        param_space: list[HyperparameterSpace],
        objective_fn: Callable[[optuna.trial.Trial, dict[str, Any]], float],
        objective_kwargs: dict[str, Any] | None = None,
    ):
        """Initialise the Optuna tuner.

        Args:
            config: Optuna search configuration.
            param_space: List of hyperparameter space definitions.
            objective_fn: Objective function to optimise.
            objective_kwargs: Extra keyword arguments for the objective function.
        """
        self.config = config
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.objective_kwargs = objective_kwargs or {}

        # Validate parameter space
        for space in self.param_space:
            space.validate()

        # Create study
        self.study: optuna.Study | None = None
        self._setup_logging()

    def _setup_logging(self):
        """Configure Optuna logging verbosity."""
        if self.config.verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif self.config.verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Optuna objective function wrapper.

        Samples hyperparameters from the search space and invokes the
        user-defined objective function.

        Args:
            trial: Optuna trial object.

        Returns:
            The objective score for this trial.
        """
        # Sample hyperparameters from the search space
        params = {}
        for space in self.param_space:
            params[space.name] = space.suggest(trial)

        # Call user-defined objective function
        score = self.objective_fn(trial, params=params, **self.objective_kwargs)

        return score

    def search(self) -> dict[str, Any]:
        """Execute the hyperparameter search.

        Returns:
            Dictionary of the best found parameters.
        """
        # Create study
        sampler = self.config.get_sampler()
        pruner = self.config.get_pruner()

        storage_url = None
        if self.config.db_url:
            storage_url = self.config.db_url
        elif self.config.save_dir:
            os.makedirs(self.config.save_dir, exist_ok=True)
            storage_url = f"sqlite:///{os.path.join(self.config.save_dir, 'study.db')}"

        study_name = (
            self.config.study_name
            or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        study_kwargs = {
            "sampler": sampler,
            "pruner": pruner,
            "study_name": study_name,
            "storage": storage_url,
            "load_if_exists": True,
        }

        directions = self.config.directions
        if isinstance(directions, list):
            cleaned = [d for d in directions if d]
            if len(cleaned) == 1:
                study_kwargs["direction"] = cleaned[0]
            elif len(cleaned) > 1:
                study_kwargs["directions"] = cleaned
            else:
                raise ValueError("Optuna directions list is empty")
        elif directions:
            study_kwargs["direction"] = directions
        else:
            raise ValueError("Optuna direction configuration missing")

        self.study = optuna.create_study(**study_kwargs)

        # GridSampler exhausts all combinations; no default seed needed
        if not isinstance(sampler, GridSampler):
            self._enqueue_defaults()

        if self.config.n_jobs > 1:
            logger.warning(
                "n_jobs>1 runs trials in threads sharing one GPU/process, which "
                "can cause GPU contention and is unsafe with SwanLab. Recommended "
                "only for CPU models or a dedicated multi-GPU setup."
            )

        # Optimise; failed trials are marked FAIL without stopping the search
        self.study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            catch=(Exception,),
            show_progress_bar=(self.config.verbose > 0),
        )

        # Save results
        if self.config.save_dir:
            self._save_results()

        return self._best_params()

    def _enqueue_defaults(self):
        """Enqueue declared defaults as a startup trial.

        Speeds up sampler convergence. Parameters without declared defaults
        are sampled normally during the trial.
        """
        defaults = {
            space.name: space.default
            for space in self.param_space
            if space.default is not None
        }
        if not defaults:
            return
        self.study.enqueue_trial(defaults)
        logger.info(f"Enqueued {len(defaults)} default params as a startup trial")

    def _best_params(self) -> dict[str, Any]:
        """Return the best parameters.

        Compatible with multi-objective studies and studies with no completed trials.

        Returns:
            Dictionary of best parameters, or empty dict if unavailable.
        """
        if len(self.study.directions) > 1:
            pareto = self.study.best_trials
            return pareto[0].params if pareto else {}
        try:
            return self.study.best_params
        except ValueError:
            logger.warning("No completed trials; cannot determine best params")
            return {}

    def _save_results(self):
        """Save search results to disk as yaml."""
        if not self.study or not self.config.save_dir:
            return

        os.makedirs(self.config.save_dir, exist_ok=True)

        # Best parameters
        best_params_path = os.path.join(self.config.save_dir, "best_params.yaml")
        Path(best_params_path).write_text(
            yaml.safe_dump(self._best_params(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        # Search history
        history_path = os.path.join(self.config.save_dir, "search_history.yaml")
        trials_data = [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            }
            for trial in self.study.trials
        ]
        Path(history_path).write_text(
            yaml.safe_dump(trials_data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        # Configuration echo
        config_path = os.path.join(self.config.save_dir, "optuna_config.yaml")
        Path(config_path).write_text(
            yaml.safe_dump(
                asdict(self.config) if is_dataclass(self.config) else self.config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        logger.info(f"Results saved to {self.config.save_dir}")

    def get_best_trial(self) -> optuna.Trial | None:
        """Retrieve the best trial from the study.

        Returns:
            The best Trial, or None if no study exists.
        """
        if not self.study:
            return None
        return self.study.best_trial

    def print_summary(self):
        """Print a summary of the search results."""
        if not self.study:
            logger.warning("No study found. Run search() first.")
            return

        log = [
            "=" * 60,
            "Optuna Hyperparameter Search Summary",
            "=" * 60,
            f"Study Name: {self.study.study_name}",
            f"Total Trials: {len(self.study.trials)}",
        ]
        if len(self.study.directions) > 1:
            pareto = self.study.best_trials
            log.append(f"Pareto-optimal trials: {len(pareto)}")
            for t in pareto[:5]:
                log.append(f"  trial {t.number}: values={t.values}")
        else:
            try:
                log.append(f"Best Value: {self.study.best_value}")
                log.append("\nBest Parameters:")
                for param, value in self.study.best_params.items():
                    log.append(f"  {param}: {value}")
            except ValueError:
                log.append("No completed trials.")
        logger.info("\n".join(log))

    def get_dataframe(self):
        """Get the trials dataframe (requires pandas).

        Returns:
            DataFrame of trial results, or None if unavailable.
        """
        if not self.study:
            return None
        try:
            return self.study.trials_dataframe()
        except Exception as e:
            logger.error(f"Failed to get dataframe: {e}")
            return None


__all__ = [
    "OptunaTuner",
]
