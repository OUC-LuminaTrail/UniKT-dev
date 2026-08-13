"""Trainer integration with Optuna objective functions."""

import time
import traceback
from collections.abc import Callable
from typing import Any

import optuna

from utils.core import get_logger, seed_everything

from .callback import OptunaTrialCallback
from .config import (
    HyperparameterSpace,
    OptunaConfig,
    direction_for_metric,
    load_optuna_config,
)
from .tuner import OptunaTuner

logger = get_logger(__name__)


class TrainerObjectiveWrapper:
    """Wrapper integrating a Trainer into an Optuna objective function."""

    def __init__(
        self,
        trainer_class: type,
        data_src_fn: Callable[[], Any],
        base_rc: Any,
        metric_name: str | list[str] = "auc",
        max_epochs: int | None = None,
        exp_manager=None,
    ):
        """Initialise the Trainer wrapper.

        Args:
            trainer_class: Trainer class.
            data_src_fn: Data source factory function.
            base_rc: Base RunConfig instance.
            metric_name: Metric name (str) or names (list) to optimise; a list
                enables multi-objective search.
            max_epochs: Maximum number of epochs.
            exp_manager: Experiment manager for creating trial subdirectories.
        """
        self.trainer_class = trainer_class
        self.data_src_fn = data_src_fn
        self.base_rc = base_rc
        self.metric_names = (
            [metric_name] if isinstance(metric_name, str) else list(metric_name)
        )
        self._multi = len(self.metric_names) > 1
        self.maximize = direction_for_metric(self.metric_names[0]) == "maximize"
        self.max_epochs = max_epochs or getattr(base_rc.model, "epochs", 50)
        self.exp_manager = exp_manager

    def __call__(
        self, trial, params: dict[str, Any] | None = None, **kwargs
    ) -> float | list[float]:
        """Execute a single trial with a given hyperparameter combination.

        Args:
            trial: Optuna trial object.
            params: Hyperparameter dictionary for this trial (model-node field
                names; applied as a merge overlay onto ``base_rc.model``).
            **kwargs: Additional keyword arguments.

        Returns:
            The optimised metric value for this trial.
        """
        if params is None:
            params = {}

        start_time = time.perf_counter()

        try:
            trial_rc = self._create_trial_rc(params)

            # Reseed before constructing the trainer so every trial starts from
            # the same RNG state (weight init, data shuffle, training RNG).
            seed_everything(
                trial_rc.general.seed,
                deterministic=not trial_rc.general.no_deterministic,
            )

            trial_exp_manager = None
            if self.exp_manager is not None:
                trial_exp_manager = self.exp_manager.create_sub_experiment(
                    f"trial_{trial.number}"
                )

            # Pruning is single-objective only (trial.report takes one value).
            pruning_cb = None
            if not self._multi:
                pruning_cb = OptunaTrialCallback(
                    trial=trial,
                    metric_name=self.metric_names[0],
                    maximize=self.maximize,
                )

            data_src = self.data_src_fn()
            trainer = self.trainer_class(
                rc=trial_rc, data_src=data_src, exp_manager=trial_exp_manager
            )
            # Register the pruning callback via the trainer's controlled hook:
            # single-stage trainers append to the live list; multi-stage trainers
            # register into _custom_callbacks so every stage includes it.
            if pruning_cb is not None:
                trainer.add_callback(pruning_cb)

            # Record the trial dir before run() so it survives a mid-run failure;
            # OptunaTuner copies the best trial's run_config.yaml from here.
            if trial_exp_manager is not None:
                trial.set_user_attr("trial_dir", trial_exp_manager.get_log_dir())

            trainer.run()

            if pruning_cb is not None and pruning_cb.pruned:
                es = trainer.early_stopping
                best_epoch = es.best_epoch if es is not None else None
                raise optuna.TrialPruned(
                    f"Trial {trial.number} pruned at epoch {best_epoch}"
                )

            value = self._extract_metric(trainer, pruning_cb)

            es = getattr(trainer, "early_stopping", None)
            trial.set_user_attr("best_epoch", es.best_epoch if es is not None else None)
            trial.set_user_attr(
                "duration_sec", round(time.perf_counter() - start_time, 2)
            )
            return value

        except optuna.TrialPruned:
            # Pruning is intentional, not a failure — keep the trial PRUNED.
            raise

        except Exception as e:
            # Record the root cause for debugging; re-raise so
            # study.optimize(catch=...) marks the trial FAIL (not COMPLETE).
            logger.exception(f"Trial {trial.number} failed")
            trial.set_user_attr("error", repr(e))
            trial.set_user_attr("traceback", traceback.format_exc())
            trial.set_user_attr(
                "duration_sec", round(time.perf_counter() - start_time, 2)
            )
            raise

    def _create_trial_rc(self, params: dict[str, Any]) -> Any:
        """Evolve the base RunConfig with a trial's model hyperparameters.

        Trial params are model-node field names; they are applied onto a deep
        copy of ``base_rc.model``. Every trial reseeds from the base seed so only
        the sampled hyperparameters vary across trials.
        """
        import copy

        trial_rc = copy.deepcopy(self.base_rc)
        for name, value in params.items():
            setattr(trial_rc.model, name, value)
        return trial_rc

    def _extract_metric(self, trainer, pruning_cb) -> float | list[float]:
        """Extract the optimised metric value(s) from the trainer.

        Single-objective: prefers the pruning callback's tracked best value,
        falling back to EarlyStopping's best epoch record. Multi-objective: reads
        every objective from EarlyStopping's best-epoch metrics dict.

        Args:
            trainer: The trainer instance.
            pruning_cb: The Optuna trial callback (None for multi-objective).

        Returns:
            The metric value (single-objective) or a list of values
            (multi-objective).
        """
        if self._multi:
            return self._extract_multi(trainer)

        metric_lower = self.metric_names[0].lower()

        # Prefer callback-tracked best value
        if pruning_cb is not None and pruning_cb.best_value is not None:
            return pruning_cb.best_value

        es = getattr(trainer, "early_stopping", None)
        if es is not None:
            if es.best_metrics and es.best_metrics.get(metric_lower) is not None:
                return float(es.best_metrics[metric_lower])
            if es.best_score is not None and es.cfg.monitor.lower() == metric_lower:
                return float(es.best_score)

        raise RuntimeError(
            f"Could not extract metric '{metric_lower}' from trainer "
            f"(pruning_cb.best_value="
            f"{pruning_cb.best_value if pruning_cb is not None else None}, "
            f"early_stopping attached={es is not None}). "
            f"Ensure validation data is provided and the metric name is correct."
        )

    def _extract_multi(self, trainer) -> list[float]:
        """Extract each objective from the best epoch's metrics dict."""
        es = getattr(trainer, "early_stopping", None)
        best_metrics = getattr(es, "best_metrics", None) if es is not None else None
        values: list[float] = []
        for name in self.metric_names:
            value = best_metrics.get(name.lower()) if best_metrics else None
            if value is None:
                available = sorted(best_metrics) if best_metrics else None
                raise RuntimeError(
                    f"Could not extract metric '{name}' from trainer "
                    f"(early_stopping attached={es is not None}, "
                    f"best_metrics keys={available}). "
                    f"Ensure validation data is provided and the metric name is correct."
                )
            values.append(float(value))
        return values


class OptunaTunerBuilder:
    """Optuna tuner builder providing a fluent API."""

    def __init__(self):
        """Initialise the tuner builder with default empty configuration."""
        self.config: OptunaConfig | None = None
        self.param_spaces: list[HyperparameterSpace] = []
        self.objective_fn: Callable | None = None
        self.objective_kwargs: dict[str, Any] = {}

    def from_config_file(self, config_path: str) -> "OptunaTunerBuilder":
        """Load Optuna configuration from a yaml file.

        Args:
            config_path: Path to the yaml configuration file.

        Returns:
            Self for chaining.
        """
        self.config = load_optuna_config(config_path)
        return self

    def with_config(self, config: OptunaConfig) -> "OptunaTunerBuilder":
        """Set the Optuna configuration.

        Args:
            config: An OptunaConfig instance.

        Returns:
            Self for chaining.
        """
        self.config = config
        return self

    def with_param_spaces(
        self, spaces: list[HyperparameterSpace]
    ) -> "OptunaTunerBuilder":
        """Set the parameter space definitions.

        Args:
            spaces: List of HyperparameterSpace instances.

        Returns:
            Self for chaining.
        """
        self.param_spaces = spaces
        return self

    def with_objective(self, fn: Callable) -> "OptunaTunerBuilder":
        """Set the objective function.

        Args:
            fn: Objective function.

        Returns:
            Self for chaining.
        """
        self.objective_fn = fn
        return self

    def with_objective_kwargs(self, **kwargs) -> "OptunaTunerBuilder":
        """Set extra keyword arguments to pass to the objective function.

        Args:
            **kwargs: Keyword arguments for the objective function.

        Returns:
            Self for chaining.
        """
        self.objective_kwargs.update(kwargs)
        return self

    def build(self) -> OptunaTuner:
        """Build and return the OptunaTuner.

        Returns:
            A configured OptunaTuner instance.

        Raises:
            ValueError: If config, param spaces, or objective function are not set.
        """
        if not self.config:
            raise ValueError(
                "OptunaConfig not set. Use from_config_file() or with_config()"
            )
        if not self.param_spaces:
            raise ValueError("Parameter spaces not set. Use with_param_spaces()")
        if not self.objective_fn:
            raise ValueError("Objective function not set. Use with_objective()")

        return OptunaTuner(
            config=self.config,
            param_space=self.param_spaces,
            objective_fn=self.objective_fn,
            objective_kwargs=self.objective_kwargs,
        )


__all__ = [
    "OptunaTunerBuilder",
    "TrainerObjectiveWrapper",
]
