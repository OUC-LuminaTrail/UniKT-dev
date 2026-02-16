"""用于修改模型行为的消融策略。

此包包含用于进行消融研究的各种策略。
"""

from .base import (
    BaseAblationStrategy,
    IdentityModule,
    apply_ablation,
)
from .feature_zero import FeatureZeroStrategy
from .module_disable import ModuleDisableStrategy
from .module_replace import ModuleReplaceStrategy
from .module_zero import ModuleZeroStrategy
from .parameter_freeze import ParameterFreezeStrategy

__all__ = [
    "BaseAblationStrategy",
    "IdentityModule",
    "apply_ablation",
    "ModuleDisableStrategy",
    "ModuleZeroStrategy",
    "FeatureZeroStrategy",
    "ParameterFreezeStrategy",
    "ModuleReplaceStrategy",
]
