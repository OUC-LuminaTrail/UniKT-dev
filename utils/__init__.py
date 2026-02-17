"""Utils模块 - 统一导入接口

提供清晰的、分层的导入路径。
"""

# ============================================================================
# 核心模块
# ============================================================================
# ============================================================================
# 配置管理
# ============================================================================
from .config import (
    BaseParamConfig,
    DataLoaderConfig,
    DataParams,
    EarlyStopping,
    EarlyStoppingConfig,
    EarlyStoppingParams,
    GeneralParams,
    get_model_params,
    list_models,
    register_model_params,
)
from .core import (
    COMPONENTS,
    DATA_SOURCES,
    MODELS,
    TRAINERS,
    get_logger,
    seed_everything,
)

# ============================================================================
# 训练相关
# ============================================================================
from .training import (
    BaseTrainer,
    Callback,
    CallbackManager,
    MetricsAccumulator,
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
