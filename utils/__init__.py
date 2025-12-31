"""Utils模块 - 统一导入接口

提供清晰的、分层的导入路径。
"""

# ============================================================================
# 核心模块
# ============================================================================
from .core import (
    MODELS,
    TRAINERS,
    DATA_SOURCES,
    COMPONENTS,
    get_logger,
    seed_everything,
)

# ============================================================================
# 配置管理
# ============================================================================
from .config import (
    BaseConfig,
    EarlyStoppingConfig,
    EarlyStopping,
    DataLoaderConfig,
    BaseParamConfig,
    DataParams,
    EarlyStoppingParams,
    GeneralParams,
    register_model_params,
    get_model_params,
    list_models,
)

# ============================================================================
# 训练相关
# ============================================================================
from .training import (
    BaseTrainer,
    MetricsAccumulator,
    Callback,
    CallbackManager,
)

# ============================================================================
# 向后兼容的别名
# ============================================================================
# Trainer = BaseTrainer  # 如果需要完全兼容

__all__ = [
    # Core
    "MODELS",
    "TRAINERS",
    "DATA_SOURCES",
    "COMPONENTS",
    "get_logger",
    "seed_everything",
    # Config
    "BaseConfig",
    "EarlyStoppingConfig",
    "EarlyStopping",
    "DataLoaderConfig",
    "BaseParamConfig",
    "DataParams",
    "EarlyStoppingParams",
    "GeneralParams",
    "register_model_params",
    "get_model_params",
    "list_models",
    # Training
    "BaseTrainer",
    "MetricsAccumulator",
    "Callback",
    "CallbackManager",
]
