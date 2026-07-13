"""Unified run configuration tree — the single source of truth.

One typed dataclass tree: the CLI is auto-derived from field metadata by
:class:`ConfigParser`; yaml archives round-trip via OmegaConf. Per-model
hyperparameters live in :class:`ModelConfig` subclasses (registered via
``@register_model_config``); framework-level knobs live in the fixed
sub-configs below.

Runtime ``rc`` is an OmegaConf ``DictConfig`` (mutable, dot-access,
yaml-native). The polymorphic ``model`` node is always the concrete subclass
at schema-build time (see ``build_run_config_schema``), never structured on
the :class:`ModelConfig` base directly.

The training knobs (epochs/batch_size/learning_rate/weight_decay) are declared
on :class:`ModelConfig` as a contract; each subclass overrides their defaults.
"""

from dataclasses import dataclass, field


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration.

    Attributes:
        monitor: Metric to monitor, one of 'auc', 'acc', 'rmse', 'loss'
        mode: Optimization mode, 'max' for auc/acc, 'min' for rmse/loss
        patience: Number of epochs to tolerate without improvement
        min_delta: Minimum improvement threshold
    """

    monitor: str = field(
        default="auc",
        metadata={
            "help": "Metric to monitor for early stopping (choices: auc, acc, rmse, loss, default: auc)",
            "short": "esm",
            "choices": ["auc", "acc", "rmse", "loss"],
        },
    )
    mode: str = field(
        default="max",
        metadata={
            "help": "Optimization mode for monitored metric (default: max)",
            "short": "esmo",
        },
    )
    patience: int = field(
        default=10,
        metadata={
            "help": "Number of epochs with no improvement before stopping (default: 10)",
            "short": "esp",
        },
    )
    min_delta: float = field(
        default=0.0,
        metadata={
            "help": "Minimum change to qualify as improvement (default: 0.0)",
            "short": "esd",
        },
    )


@dataclass
class GeneralConfig:
    """Framework-level general knobs: logging, device, seed, tracking."""

    log_dir: str | None = field(
        default=None,
        metadata={
            "help": "Directory to save logs and models (default: runs/<timestamp>)",
            "short": "ld",
        },
    )
    checkpoint_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to model checkpoint for resuming training",
            "short": "cp",
        },
    )
    device: str | None = field(
        default=None,
        metadata={
            "help": "Device to use: 'cuda' or 'cpu' (default: auto-detect)",
            "short": "dev",
        },
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed for reproducibility (default: 42)",
            "short": "s",
        },
    )
    no_deterministic: bool = field(
        default=False,
        metadata={
            "help": "Disable deterministic algorithms (deterministic is enabled by default)"
        },
    )
    no_swanlab: bool = field(
        default=False,
        metadata={
            "help": "Disable SwanLab experiment tracking (default: SwanLab on)",
            "short": "nsl",
        },
    )
    log_batch_metrics: bool = field(
        default=False,
        metadata={
            "help": "Log per-batch loss to batch_metrics_<phase>.csv (default: False)",
            "short": "lbm",
        },
    )
    skip_test: bool = field(
        default=False,
        metadata={
            "help": "Skip test set evaluation after training (default: False)",
            "short": "st",
        },
    )
    cache: bool = field(
        default=False,
        metadata={
            "help": "Enable disk cache for model data preparation (default: False)",
            "short": "ca",
        },
    )


@dataclass
class CompileConfig:
    """``torch.compile`` execution knobs (framework-level, model-agnostic)."""

    compile: bool = field(
        default=False,
        metadata={
            "help": "Enable torch.compile for model optimization (default: False)"
        },
    )
    compile_mode: str = field(
        default="default",
        metadata={
            "help": "Compilation mode (default: default)",
            "choices": [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ],
        },
    )
    compile_fullgraph: bool = field(
        default=False,
        metadata={
            "help": "Require the entire function be capturable into a single graph (default: False)"
        },
    )
    compile_dynamic: bool | None = field(
        default=None,
        metadata={
            "help": "Use dynamic shape tracing (default: None, PyTorch auto-detects)"
        },
    )
    compile_backend: str = field(
        default="inductor",
        metadata={"help": "Compilation backend (default: inductor)"},
    )


@dataclass
class ExperimentConfig:
    """Experiment identity used for run naming and archive lookup."""

    model_name: str = field(default="", metadata={"help": "Model name", "short": "m"})
    dataset_name: str = field(default="", metadata={"help": "Dataset name"})


@dataclass
class RunDataConfig:
    """Dataset selection, split, sequence bounds, and sampling."""

    dataset: str = field(
        default="",
        metadata={"help": "Dataset name", "short": "d"},
    )
    data_base_path: str = field(
        default="./data",
        metadata={"help": "Path to the data files (default: ./data)", "short": "dbp"},
    )
    fold: int = field(
        default=0,
        metadata={
            "help": "Index of fold for K-Fold cross-validation (default: 0)",
            "short": "f",
        },
    )
    kfold: int = field(
        default=5,
        metadata={
            "help": "Number of folds for K-Fold cross-validation (>=2 to enable, default: 5)"
        },
    )
    test_ratio: float = field(
        default=0.2,
        metadata={"help": "Ratio for test set (default: 0.2)"},
    )
    min_seq_len: int = field(
        default=3,
        metadata={"help": "Minimum sequence length (default: 3)"},
    )
    max_seq_len: int = field(
        default=200,
        metadata={"help": "Maximum sequence length (default: 200)"},
    )
    sample_size: int | None = field(
        default=None,
        metadata={"help": "Absolute sample count (None to disable)"},
    )
    sample_ratio: float | None = field(
        default=None,
        metadata={"help": "Sample ratio 0.0-1.0 (overrides sample_size)"},
    )
    sample_strategy: str = field(
        default="random",
        metadata={
            "help": "Sampling strategy (default: random)",
            "choices": ["random", "stratified", "time"],
        },
    )
    sample_attempts_bins: list[int] = field(
        default_factory=lambda: [20, 100],
        metadata={
            "help": "Attempt count bin edges (e.g. 20 100 for low/medium/high)",
            "nargs": "+",
        },
    )
    sample_correct_bins: list[float] = field(
        default_factory=lambda: [0.4, 0.8],
        metadata={
            "help": "Correct rate bin edges (e.g. 0.4 0.8 for low/medium/high)",
            "nargs": "+",
        },
    )


@dataclass
class ModelConfig:
    """Base class for per-model hyperparameter configs.

    Each model subclasses this and registers via ``@register_model_config``.
    The training knobs below are declared here as a contract (so the framework
    can always read ``rc.model.epochs`` / ``rc.model.batch_size`` / etc.);
    subclasses override the defaults with their model-specific values.
    """

    epochs: int = field(
        default=150, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    batch_size: int = field(
        default=32, metadata={"help": "Batch size for training", "short": "bs"}
    )
    learning_rate: float = field(
        default=1e-3, metadata={"help": "Learning rate", "short": "lr"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay", "short": "wd"}
    )


@dataclass
class RunConfig:
    """Top-level run configuration tree.

    The ``model`` node is polymorphic: the runtime schema is assembled with
    the concrete :class:`ModelConfig` subclass, never structured on this base.
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: RunDataConfig = field(default_factory=RunDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# Fixed framework nodes; ``model`` is filled per model_name at build time.
_FRAMEWORK_NODES: dict[str, type] = {
    "general": GeneralConfig,
    "compile": CompileConfig,
    "early_stopping": EarlyStoppingConfig,
    "experiment": ExperimentConfig,
    "data": RunDataConfig,
}


def build_run_config_schema(model_name: str) -> dict[str, type]:
    """Return ``{node_name: dataclass_cls}`` for the concrete model's tree.

    The polymorphic ``model`` node is bound to the concrete registered
    :class:`ModelConfig` subclass; OmegaConf validates against it.
    """
    from ..core import MODEL_CONFIGS  # lazy: avoid any import-time cycle

    model_cls = MODEL_CONFIGS.get(model_name)
    if model_cls is None:
        available = ", ".join(sorted(MODEL_CONFIGS.keys())) or "none yet"
        raise KeyError(
            f"No ModelConfig registered for '{model_name}'. Available: {available}"
        )
    return {**_FRAMEWORK_NODES, "model": model_cls}


__all__ = [
    "CompileConfig",
    "EarlyStoppingConfig",
    "ExperimentConfig",
    "GeneralConfig",
    "ModelConfig",
    "RunConfig",
    "RunDataConfig",
    "build_run_config_schema",
]
