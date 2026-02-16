"""消融研究框架。

一个灵活、非侵入式的框架，用于对 PyTorch 模型进行消融研究。
使用 Python 的高级特性（上下文管理器、钩子、动态属性）来
修改模型行为而无需更改源代码。

Usage:
    from utils.ablation import AblationExperiment, load_ablation_config

    config = load_ablation_config("configs/ablation/gikt_ablation.json")
    experiment = AblationExperiment(
        base_trainer=GIKTTrainer,
        config=config,
        args=args,
        data_src=data_src
    )
    results = experiment.run_all()
"""

from ..core import ABLATION_STRATEGIES
from .config import (
    AblationConfig,
    AblationModification,
    AblationStudyConfig,
    load_ablation_config,
    save_ablation_config,
    validate_ablation_config,
)
from .experiment import (
    AblationExperiment,
    AblationResult,
    ExperimentSummary,
)

# Import strategies to register them
from .strategies import (
    FeatureZeroStrategy,
    ModuleDisableStrategy,
    ModuleReplaceStrategy,
    ModuleZeroStrategy,
    ParameterFreezeStrategy,
    apply_ablation,
)


# Convenience functions for strategy registry
def register_strategy(name, strategy_cls=None):
    """注册消融策略。

    可以作为装饰器使用：
        @register_strategy("my_strategy")
        class MyStrategy(BaseAblationStrategy):
            pass

    或直接调用：
        register_strategy("my_strategy", MyStrategy)
    """

    def decorator(cls):
        return ABLATION_STRATEGIES.register(name)(cls)

    if strategy_cls is not None:
        return decorator(strategy_cls)
    return decorator


def get_strategy(name):
    """获取注册的策略类。"""
    return ABLATION_STRATEGIES.get(name)


def list_strategies():
    """列出所有已注册的策略名称。"""
    return ABLATION_STRATEGIES.list_names()


__all__ = [
    # Config
    "AblationConfig",
    "AblationModification",
    "AblationStudyConfig",
    "load_ablation_config",
    "save_ablation_config",
    "validate_ablation_config",
    # Registry
    "ABLATION_STRATEGIES",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    # Experiment
    "AblationExperiment",
    "AblationResult",
    "ExperimentSummary",
    # Strategies
    "ModuleDisableStrategy",
    "ModuleZeroStrategy",
    "FeatureZeroStrategy",
    "ParameterFreezeStrategy",
    "ModuleReplaceStrategy",
    "apply_ablation",
]
