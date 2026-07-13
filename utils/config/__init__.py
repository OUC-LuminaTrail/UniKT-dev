"""Unified configuration management module.

The RunConfig tree is the single source of truth; :class:`ConfigParser` (built on
``jsonargparse``) derives the CLI from it reflectively, and :mod:`archive`
round-trips it to yaml. EarlyStopping (algorithm) and the runtime containers
(TrainingConfig/DataConfig/...) live in utils.training, not here.
"""

from .archive import (
    load_run_config_archive,
    load_run_metadata,
    save_run_config_archive,
)
from .config_parser import ConfigParser
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
    config_to_dict,
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
    "config_to_dict",
    "create_optimized_dataloader",
    "load_run_config_archive",
    "load_run_metadata",
    "save_run_config_archive",
]
