"""Optuna 调优器包装器和辅助工具。"""

import os
import json
from typing import Any, Callable, Dict, List, Optional
from dataclasses import asdict
from datetime import datetime

import optuna

from .config import HyperparameterSpace, OptunaConfig
from utils.core import get_logger

logger = get_logger(__name__)


class OptunaTuner:
    """
    Optuna超参数搜索器
    """

    def __init__(
        self,
        config: OptunaConfig,
        param_space: List[HyperparameterSpace],
        objective_fn: Callable[[optuna.trial.Trial, Dict[str, Any]], float],
        objective_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化Optuna调优器
        """
        self.config = config
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.objective_kwargs = objective_kwargs or {}

        # 验证参数空间
        for space in self.param_space:
            space.validate()

        # 创建学习目标
        self.study: Optional[optuna.Study] = None
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        if self.config.verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif self.config.verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """Optuna目标函数包装"""
        # 从参数空间中采样超参数
        params = {}
        for space in self.param_space:
            params[space.name] = space.suggest(trial)

        # 调用用户定义的目标函数
        score = self.objective_fn(trial, params=params, **self.objective_kwargs)

        return score

    def search(self) -> Dict[str, Any]:
        """
        执行超参数搜索
        """
        # 创建Study
        sampler = self.config.get_sampler()
        pruner = self.config.get_pruner()

        storage_url = None
        if self.config.db_url:
            storage_url = self.config.db_url
        elif self.config.save_dir:
            os.makedirs(self.config.save_dir, exist_ok=True)
            storage_url = f"sqlite:///{os.path.join(self.config.save_dir, 'study.db')}"

        study_name = (
            self.config.study_name
            or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        study_kwargs = {
            "sampler": sampler,
            "pruner": pruner,
            "study_name": study_name,
            "storage": storage_url,
            "load_if_exists": True,
        }

        directions = self.config.directions
        if isinstance(directions, list):
            cleaned = [d for d in directions if d]
            if len(cleaned) == 1:
                study_kwargs["direction"] = cleaned[0]
            elif len(cleaned) > 1:
                study_kwargs["directions"] = cleaned
            else:
                raise ValueError("Optuna directions list is empty")
        elif directions:
            study_kwargs["direction"] = directions
        else:
            raise ValueError("Optuna direction configuration missing")

        self.study = optuna.create_study(**study_kwargs)

        # 优化
        self.study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            show_progress_bar=(self.config.verbose > 0),
        )

        # 保存结果
        if self.config.save_dir:
            self._save_results()

        # 返回最佳参数
        return self.study.best_params

    def _save_results(self):
        """保存搜索结果"""
        if not self.study or not self.config.save_dir:
            return

        os.makedirs(self.config.save_dir, exist_ok=True)

        # 保存最佳参数
        best_params_path = os.path.join(self.config.save_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(self.study.best_params, f, indent=2)

        # 保存搜索历史
        history_path = os.path.join(self.config.save_dir, "search_history.json")
        trials_data = []
        for trial in self.study.trials:
            trials_data.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
            )
        with open(history_path, "w") as f:
            json.dump(trials_data, f, indent=2)

        # 保存配置
        config_path = os.path.join(self.config.save_dir, "optuna_config.json")
        config_dict = asdict(self.config)
        config_dict["sampler_kwargs"] = str(config_dict.get("sampler_kwargs", {}))
        config_dict["pruner_kwargs"] = str(config_dict.get("pruner_kwargs", {}))
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Results saved to {self.config.save_dir}")

    def get_best_trial(self) -> Optional[optuna.Trial]:
        """获取最佳trial"""
        if not self.study:
            return None
        return self.study.best_trial

    def print_summary(self):
        """打印搜索结果摘要"""
        if not self.study:
            logger.warning("No study found. Run search() first.")
            return

        log = [
            "=" * 60,
            "Optuna Hyperparameter Search Summary",
            "=" * 60,
            f"Study Name: {self.study.study_name}",
            f"Total Trials: {len(self.study.trials)}",
            f"Best Value: {self.study.best_value}",
            "\nBest Parameters:",
        ]
        for param, value in self.study.best_params.items():
            log.append(f"  {param}: {value}")
        logger.info("\n".join(log))

    def get_dataframe(self):
        """获取试验数据框（需要pandas）"""
        if not self.study:
            return None
        try:
            return self.study.trials_dataframe()
        except Exception as e:
            logger.error(f"Failed to get dataframe: {e}")
            return None


__all__ = [
    "OptunaTuner",
]
