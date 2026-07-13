"""Unified configuration management module.

The RunConfig tree is the single source of truth; ConfigParser derives the CLI
from it reflectively, and archive.py round-trips it to yaml. EarlyStopping
(algorithm) and the runtime containers (TrainingConfig/DataConfig/...) live in
utils.training, not here.
"""

from .archive import (
    load_run_config_archive,
    load_run_metadata,
    save_run_config_archive,
)
from .config_parser import ConfigParser, register_config_group
from .data_config import (
    DataLoaderConfig,
    create_optimized_dataloader,
)
from .run_config import (
    CompileConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    GeneralConfig,
    ModelConfig,
    RunConfig,
    RunDataConfig,
)

__all__ = [
    "CompileConfig",
    "ConfigParser",
    "DataLoaderConfig",
    "EarlyStoppingConfig",
    "ExperimentConfig",
    "GeneralConfig",
    "ModelConfig",
    "RunConfig",
    "RunDataConfig",
    "create_optimized_dataloader",
    "load_run_config_archive",
    "load_run_metadata",
    "register_config_group",
    "save_run_config_archive",
]
