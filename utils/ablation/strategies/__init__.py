"""用于修改模型行为的消融策略。

此包包含用于进行消融研究的各种策略。
"""

from .base import (
    BaseAblationStrategy,
    IdentityModule,
    apply_ablation,
)

from .module_disable import ModuleDisableStrategy
from .module_zero import ModuleZeroStrategy
from .feature_zero import FeatureZeroStrategy
from .parameter_freeze import ParameterFreezeStrategy
from .module_replace import ModuleReplaceStrategy

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
