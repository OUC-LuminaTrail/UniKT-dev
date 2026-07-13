"""Trainer integration with Optuna objective functions."""

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
        metric_name: str = "auc",
        max_epochs: int | None = None,
        exp_manager=None,
    ):
        """Initialise the Trainer wrapper.

        Args:
            trainer_class: Trainer class.
            data_src_fn: Data source factory function.
            base_rc: Base RunConfig instance.
            metric_name: Metric name to optimise.
            max_epochs: Maximum number of epochs.
            exp_manager: Experiment manager for creating trial subdirectories.
        """
        self.trainer_class = trainer_class
        self.data_src_fn = data_src_fn
        self.base_rc = base_rc
        self.metric_name = metric_name
        self.maximize = direction_for_metric(metric_name) == "maximize"
        self.max_epochs = max_epochs or getattr(base_rc.model, "epochs", 50)
        self.exp_manager = exp_manager

    def __call__(self, trial, params: dict[str, Any] | None = None, **kwargs) -> float:
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

        trial_rc = self._create_trial_rc(params)

        # Reseed before constructing the trainer so every trial starts from
        # the same RNG state (weight init, data shuffle, training RNG).
        seed_everything(
            trial_rc.general.seed, deterministic=not trial_rc.general.no_deterministic
        )

        trial_exp_manager = None
        if self.exp_manager is not None:
            trial_exp_manager = self.exp_manager.create_sub_experiment(
                f"trial_{trial.number}"
            )

        pruning_cb = OptunaTrialCallback(
            trial=trial, metric_name=self.metric_name, maximize=self.maximize
        )

        data_src = self.data_src_fn()
        trainer = self.trainer_class(
            rc=trial_rc, data_src=data_src, exp_manager=trial_exp_manager
        )
        # trainer.__init__ already calls build(), so the callback list is finalised;
        # append directly to the active list.
        trainer.callback_manager.callbacks.append(pruning_cb)

        trainer.run()

        if pruning_cb.pruned:
            es = trainer.early_stopping
            best_epoch = es.best_epoch if es is not None else None
            raise optuna.TrialPruned(
                f"Trial {trial.number} pruned at epoch {best_epoch}"
            )

        return self._extract_metric(trainer, pruning_cb)

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

    def _worst_value(self) -> float:
        """Return the worst possible target value for the current optimisation direction.

        Used as a fallback for failed trials.

        Returns:
            Negative infinity (maximise) or positive infinity (minimise).
        """
        return float("-inf") if self.maximize else float("inf")

    def _extract_metric(self, trainer, pruning_cb) -> float:
        """Extract the raw metric value from the trainer.

        Prefers the callback's tracked best value, falling back to
        EarlyStopping's best epoch record.

        Args:
            trainer: The trainer instance.
            pruning_cb: The Optuna trial callback.

        Returns:
            The metric value.
        """
        metric_lower = self.metric_name.lower()

        # Prefer callback-tracked best value
        if pruning_cb.best_value is not None:
            return pruning_cb.best_value

        es = getattr(trainer, "early_stopping", None)
        if es is not None:
            if es.best_metrics and es.best_metrics.get(metric_lower) is not None:
                return float(es.best_metrics[metric_lower])
            if es.best_score is not None and es.cfg.monitor.lower() == metric_lower:
                return float(es.best_score)

        logger.warning(f"Could not extract metric '{metric_lower}' from trainer")
        return self._worst_value()


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
