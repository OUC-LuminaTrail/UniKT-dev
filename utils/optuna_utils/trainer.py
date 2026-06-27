"""与 Optuna 目标函数集成的 Trainer。"""

from argparse import Namespace
from collections.abc import Callable
from typing import Any

import optuna

from utils.core import get_logger

from .callback import OptunaTrialCallback
from .config import (
    HyperparameterSpace,
    OptunaConfig,
    direction_for_metric,
    load_config_from_json,
    load_param_space_from_json,
)
from .tuner import OptunaTuner

logger = get_logger(__name__)


class TrainerObjectiveWrapper:
    """
    将Trainer集成到Optuna目标函数的包装器
    """

    def __init__(
        self,
        trainer_class: type,
        data_src_fn: Callable[[], Any],
        base_args: Namespace,
        metric_name: str = "auc",
        max_epochs: int | None = None,
        exp_manager=None,
    ):
        """
        初始化Trainer包装器

        Args:
            trainer_class: 训练器类
            data_src_fn: 数据源工厂函数
            base_args: 基础参数
            metric_name: 优化指标名称
            max_epochs: 最大epoch数
            exp_manager: 实验管理器（用于创建trial子目录）
        """
        self.trainer_class = trainer_class
        self.data_src_fn = data_src_fn
        self.base_args = base_args
        self.metric_name = metric_name
        self.maximize = direction_for_metric(metric_name) == "maximize"
        self.max_epochs = max_epochs or getattr(base_args, "epochs", 50)
        self.exp_manager = exp_manager

    def __call__(self, trial, params: dict[str, Any] = None, **kwargs) -> float:
        """执行一次超参数组合的训练。"""
        if params is None:
            params = {}

        args = self._create_trial_args(params)

        trial_exp_manager = None
        if self.exp_manager is not None:
            trial_exp_manager = self.exp_manager.create_sub_experiment(
                f"trial_{trial.number}"
            )

        pruning_cb = OptunaTrialCallback(
            trial=trial, metric_name=self.metric_name, maximize=self.maximize
        )

        try:
            data_src = self.data_src_fn()
            trainer = self.trainer_class(
                args=args, data_src=data_src, exp_manager=trial_exp_manager
            )
            # trainer.__init__ 末尾已 build()，回调列表已定型，需直接追加到活跃列表
            trainer.callback_manager.callbacks.append(pruning_cb)

            trainer.run()

            if pruning_cb.pruned:
                raise optuna.TrialPruned(
                    f"Trial {trial.number} pruned at epoch "
                    f"{getattr(trainer, '_best_epoch', None)}"
                )

            return self._extract_metric(trainer, pruning_cb)

        except optuna.TrialPruned:
            raise
        except Exception as e:
            import traceback

            logger.error(
                f"Trial {trial.number} failed: {str(e)}\n{traceback.format_exc()}"
            )
            return self._worst_value()

    def _create_trial_args(self, params: dict[str, Any]) -> Namespace:
        """根据trial参数创建新的args"""
        import copy

        args = copy.deepcopy(self.base_args)

        # 特殊处理batch_size（可能需要重新创建DataLoader）
        if "batch_size" in params:
            args.batch_size = params["batch_size"]

        # 更新其他参数
        for key, value in params.items():
            if key == "batch_size":
                continue  # 已在上方处理
            setattr(args, key, value)

        return args

    def _worst_value(self) -> float:
        """当前优化方向下的最差目标值（失败 trial 使用）。"""
        return float("-inf") if self.maximize else float("inf")

    def _extract_metric(self, trainer, pruning_cb) -> float:
        """提取优化指标的原始值（方向由 study direction 决定，此处不取负）。"""
        metric_lower = self.metric_name.lower()

        # 优先取回调追踪的最佳值，回退到 EarlyStopping 最佳 epoch 的记录
        if pruning_cb.best_value is not None:
            return pruning_cb.best_value

        es = getattr(trainer, "early_stopping", None)
        if es is not None:
            if es.best_metrics and es.best_metrics.get(metric_lower) is not None:
                return float(es.best_metrics[metric_lower])
            if es.best_score is not None and es.cfg.monitor.lower() == metric_lower:
                return float(es.best_score)

        logger.warning(f"Could not extract metric '{metric_lower}' from trainer")
        return self._worst_value()


class OptunaTunerBuilder:
    """
    Optuna调优器构建器，提供流畅的API
    """

    def __init__(self):
        self.config: OptunaConfig | None = None
        self.param_spaces: list[HyperparameterSpace] = []
        self.objective_fn: Callable | None = None
        self.objective_kwargs: dict[str, Any] = {}

    def from_config_file(self, config_path: str) -> "OptunaTunerBuilder":
        """从JSON配置文件加载Optuna配置"""
        self.config = load_config_from_json(config_path)
        return self

    def from_param_space_file(self, space_path: str) -> "OptunaTunerBuilder":
        """从JSON文件加载参数空间"""
        self.param_spaces = load_param_space_from_json(space_path)
        return self

    def with_config(self, config: OptunaConfig) -> "OptunaTunerBuilder":
        """设置Optuna配置"""
        self.config = config
        return self

    def with_param_spaces(
        self, spaces: list[HyperparameterSpace]
    ) -> "OptunaTunerBuilder":
        """设置参数空间"""
        self.param_spaces = spaces
        return self

    def with_objective(self, fn: Callable) -> "OptunaTunerBuilder":
        """设置目标函数"""
        self.objective_fn = fn
        return self

    def with_objective_kwargs(self, **kwargs) -> "OptunaTunerBuilder":
        """设置传递给目标函数的额外参数"""
        self.objective_kwargs.update(kwargs)
        return self

    def build(self) -> OptunaTuner:
        """构建OptunaTuner"""
        if not self.config:
            raise ValueError(
                "OptunaConfig not set. Use from_config_file() or with_config()"
            )
        if not self.param_spaces:
            raise ValueError(
                "Parameter spaces not set. Use from_param_space_file() or with_param_spaces()"
            )
        if not self.objective_fn:
            raise ValueError("Objective function not set. Use with_objective()")

        return OptunaTuner(
            config=self.config,
            param_space=self.param_spaces,
            objective_fn=self.objective_fn,
            objective_kwargs=self.objective_kwargs,
        )


__all__ = [
    "TrainerObjectiveWrapper",
    "OptunaTunerBuilder",
]
