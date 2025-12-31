"""统一配置管理模块

提供配置基类、参数配置、数据配置和训练配置。
"""

from .base import BaseConfig
from .param_config import (
    BaseParamConfig,
    DataParams,
    EarlyStoppingParams,
    GeneralParams,
    register_model_params,
    get_model_params,
    list_models,
)
from .data_config import (
    DataLoaderConfig,
    KFoldDataLoaderConfig,
    optimize_dataloader,
    create_optimized_dataloader,
    optimize_kfold_dataloaders,
)
from .training_config import EarlyStoppingConfig, EarlyStopping

__all__ = [
    # Base
    "BaseConfig",
    # Param Config
    "BaseParamConfig",
    "DataParams",
    "EarlyStoppingParams",
    "GeneralParams",
    "register_model_params",
    "get_model_params",
    "list_models",
    # Data Config
    "DataLoaderConfig",
    "KFoldDataLoaderConfig",
    "optimize_dataloader",
    "create_optimized_dataloader",
    "optimize_kfold_dataloaders",
    # Training Config
    "EarlyStoppingConfig",
    "EarlyStopping",
]
