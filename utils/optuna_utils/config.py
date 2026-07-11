"""Optuna configuration and parameter space utilities."""

import json
from dataclasses import dataclass, field
from typing import Any

from optuna.pruners import (
    BasePruner,
    MedianPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
)
from optuna.samplers import (
    BaseSampler,
    CmaEsSampler,
    GridSampler,
    RandomSampler,
    TPESampler,
)
from optuna.trial import Trial

# auc/acc are maximise, rmse/loss are minimise
_METRIC_DIRECTIONS: dict[str, str] = {
    "auc": "maximize",
    "acc": "maximize",
    "rmse": "minimize",
    "loss": "minimize",
}


def direction_for_metric(metric_name: str) -> str:
    """Return the Optuna optimisation direction for a given metric name.

    Args:
        metric_name: Name of the metric (e.g. 'auc', 'acc', 'rmse', 'loss').

    Returns:
        'maximize' or 'minimize'.

    Raises:
        ValueError: If the metric name is not recognised.
    """
    direction = _METRIC_DIRECTIONS.get(metric_name.lower())
    if direction is None:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. "
            f"Expected one of: {sorted(_METRIC_DIRECTIONS)}"
        )
    return direction


@dataclass
class HyperparameterSpace:
    """Definition of a single hyperparameter search space."""

    name: str
    type: str  # 'int', 'float', 'categorical'
    low: float | None = None
    high: float | None = None
    log: bool | None = None  # Logarithmic sampling for numeric params
    step: float | None = None  # Step size for integer params
    choices: list[Any] | None = None  # Choices for categorical params
    default: Any | None = None

    def validate(self):
        """Validate the parameter space configuration completeness."""
        if self.type == "int":
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Integer parameter '{self.name}' requires 'low' and 'high'"
                )
            if self.low >= self.high:
                raise ValueError(f"Parameter '{self.name}': low must be less than high")
        elif self.type == "float":
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Float parameter '{self.name}' requires 'low' and 'high'"
                )
            if self.low >= self.high:
                raise ValueError(f"Parameter '{self.name}': low must be less than high")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError(
                    f"Categorical parameter '{self.name}' requires 'choices'"
                )
        else:
            raise ValueError(f"Unsupported parameter type: {self.type}")

        if self.default is not None:
            if self.type in ("int", "float"):
                if not (self.low <= self.default <= self.high):
                    raise ValueError(
                        f"Parameter '{self.name}': default {self.default} "
                        f"out of range [{self.low}, {self.high}]"
                    )
            elif self.type == "categorical" and self.default not in self.choices:
                raise ValueError(
                    f"Parameter '{self.name}': default {self.default} "
                    f"not in choices {self.choices}"
                )

    def suggest(self, trial: Trial) -> Any:
        """Sample a parameter value from the Optuna trial.

        Args:
            trial: Optuna trial object.

        Returns:
            The sampled parameter value.
        """
        self.validate()

        if self.type == "int":
            return trial.suggest_int(
                self.name,
                low=int(self.low),
                high=int(self.high),
                step=int(self.step) if self.step else 1,
                log=self.log or False,
            )
        elif self.type == "float":
            return trial.suggest_float(
                self.name,
                low=float(self.low),
                high=float(self.high),
                log=self.log or False,
            )
        elif self.type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)


@dataclass
class OptunaConfig:
    """Optuna search configuration."""

    # Sampler configuration
    sampler: str = "tpe"  # 'tpe', 'random', 'grid', 'cmaes'
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int = (
        42  # Seed for stochastic samplers (tpe/random/cmaes); grid is exhaustive
    )

    # Pruner configuration
    pruner: str = "median"  # 'median', 'percentile', 'successive_halving', None
    pruner_kwargs: dict[str, Any] = field(default_factory=dict)

    # Search configuration
    n_trials: int = 100
    n_jobs: int = 1  # Number of parallel jobs
    timeout: int | None = None  # Timeout in seconds
    directions: list[str] = field(
        default_factory=lambda: ["maximize"]
    )  # 'maximize' or 'minimize'

    # Storage and logging
    study_name: str | None = None
    db_url: str | None = None  # Persistence DB URL, e.g. "sqlite:///study.db"
    save_dir: str | None = None
    verbose: int = 1  # 0=quiet, 1=normal, 2=verbose

    def get_sampler(self) -> BaseSampler:
        """Create a sampler based on the configuration.

        Returns:
            An Optuna BaseSampler instance.

        Raises:
            ValueError: If the sampler name is not recognised.
        """
        sampler_name = self.sampler.lower()
        kwargs = self.sampler_kwargs.copy()
        # Stochastic samplers are seeded for reproducibility (grid is exhaustive)
        if sampler_name != "grid" and "seed" not in kwargs:
            kwargs["seed"] = self.seed

        if sampler_name == "tpe":
            return TPESampler(**kwargs)
        elif sampler_name == "random":
            return RandomSampler(**kwargs)
        elif sampler_name == "grid":
            if "search_space" not in kwargs:
                raise ValueError(
                    "GridSampler requires 'search_space' in sampler_kwargs, "
                    'e.g. {"lr": [1e-3, 1e-4], "layers": [1, 2]}'
                )
            return GridSampler(**kwargs)
        elif sampler_name == "cmaes":
            return CmaEsSampler(**kwargs)
        else:
            raise ValueError(f"Unsupported sampler: {sampler_name}")

    def get_pruner(self) -> BasePruner | None:
        """Create a pruner based on the configuration.

        Returns:
            An Optuna BasePruner instance, or None if no pruner is configured.

        Raises:
            ValueError: If the pruner name is not recognised.
        """
        if self.pruner is None:
            return None

        pruner_name = self.pruner.lower()
        kwargs = self.pruner_kwargs.copy()

        if pruner_name == "median":
            return MedianPruner(**kwargs)
        elif pruner_name == "percentile":
            if "percentile" not in kwargs:
                raise ValueError(
                    "PercentilePruner requires 'percentile' (0-100) in pruner_kwargs"
                )
            return PercentilePruner(**kwargs)
        elif pruner_name == "successive_halving":
            return SuccessiveHalvingPruner(**kwargs)
        else:
            raise ValueError(f"Unsupported pruner: {pruner_name}")


def load_config_from_json(config_path: str) -> OptunaConfig:
    """Load an OptunaConfig from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        An OptunaConfig instance.
    """
    with open(config_path) as f:
        config_dict = json.load(f)
    return OptunaConfig(**config_dict)


def load_param_space_from_json(space_path: str) -> list[HyperparameterSpace]:
    """Load hyperparameter space definitions from a JSON file.

    Args:
        space_path: Path to the JSON parameter space file.

    Returns:
        A list of HyperparameterSpace instances.
    """
    with open(space_path) as f:
        spaces_dict = json.load(f)

    param_spaces = []
    for space_config in spaces_dict:
        if isinstance(space_config, dict):
            param_spaces.append(HyperparameterSpace(**space_config))

    return param_spaces


__all__ = [
    "HyperparameterSpace",
    "OptunaConfig",
    "direction_for_metric",
    "load_config_from_json",
    "load_param_space_from_json",
]
