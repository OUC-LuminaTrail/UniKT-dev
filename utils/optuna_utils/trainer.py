"""与 Optuna 目标函数集成的 Trainer。"""

from argparse import Namespace
from collections.abc import Callable
from typing import Any

from utils.core import get_logger

from .config import (
    HyperparameterSpace,
    OptunaConfig,
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
        self.max_epochs = max_epochs or getattr(base_args, "epochs", 50)
        self.exp_manager = exp_manager

        # 验证metric_name
        if metric_name.lower() not in ["auc", "acc", "rmse", "loss"]:
            raise ValueError(f"Invalid metric_name: {metric_name}")

    def __call__(self, trial, params: dict[str, Any] = None, **kwargs) -> float:
        """
        执行一次超参数组合的训练
        """
        if params is None:
            params = {}

        # 创建副本，避免修改原始args
        args = self._create_trial_args(params)

        # 为这个trial创建子实验管理器
        trial_exp_manager = None
        if self.exp_manager is not None:
            trial_exp_manager = self.exp_manager.create_sub_experiment(
                f"trial_{trial.number}"
            )

        try:
            # 加载数据
            data_src = self.data_src_fn()

            # 初始化trainer
            trainer = self.trainer_class(
                args=args, data_src=data_src, exp_manager=trial_exp_manager
            )

            # 运行训练
            trainer.run()

            # 获取最佳指标
            metric_value = self._extract_metric(trainer)

            # 用于修剪的报告
            self._report_intermediate_values(trial, trainer)

            return metric_value

        except Exception as e:
            import traceback

            logger.error(
                f"Trial {trial.number} failed: {str(e)}\n{traceback.format_exc()}"
            )
            # 无论指标是什么，我们都在最大化目标函数（对于loss是最大化 -loss）
            # 因此失败时应返回 -inf，表示该次尝试无效且性能极差
            return float("-inf")

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

    def _extract_metric(self, trainer) -> float:
        """从trainer中提取优化指标"""
        metric_lower = self.metric_name.lower()

        # 优先尝试从 EarlyStopping 获取最佳指标
        # 前提是 EarlyStopping 正在监控我们关心的同一个指标
        if getattr(trainer, "early_stopping", None) is not None:
            es_monitor = trainer.early_stopping.cfg.monitor.lower()
            # 如果 Optuna 优化的指标与 EarlyStopping 监控的指标一致
            if es_monitor == metric_lower:
                best_score = trainer.early_stopping.best_score
                if best_score is not None:
                    # 根据指标类型决定是否取反
                    # Optuna 默认最大化
                    # AUC, ACC: 越大越好 -> 直接返回
                    # RMSE, Loss: 越小越好 -> 取反返回
                    if metric_lower in ["rmse", "loss"]:
                        return -float(best_score)
                    return float(best_score)

        # 尝试从最后的验证指标中获取
        if hasattr(trainer, "_last_val_metrics"):
            metrics = trainer._last_val_metrics
            if metric_lower == "auc" and metrics.get("auc") is not None:
                return float(metrics["auc"])
            elif metric_lower == "acc" and metrics.get("acc") is not None:
                return float(metrics["acc"])
            elif metric_lower == "rmse" and metrics.get("rmse") is not None:
                # 最小化RMSE，返回负值
                return -float(metrics["rmse"])
            elif metric_lower == "loss":
                # 如果有loss则返回负值（因为我们要最大化）
                return -float(getattr(trainer, "_last_val_loss", 0))

        # 回退策略
        logger.warning(f"Could not extract metric '{metric_lower}' from trainer")
        return float("-inf")

    def _report_intermediate_values(self, trial, trainer):
        """报告中间值用于修剪（可选）"""
        # 这是一个扩展点，如果需要更精细的修剪策略可以在这里实现
        pass


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
