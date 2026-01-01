"""消融实验执行和结果管理。

处理运行单个消融实验和聚合结果。
"""

import json
import time
import copy
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Type

from .config import AblationConfig, AblationModification, AblationStudyConfig
from .strategies import apply_ablation
from utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class AblationResult:
    """单个消融实验的结果。

    Attributes:
        name: 实验名称
        description: 实验描述
        metrics: 指标字典（acc、auc、rmse 等）
        modifications: 应用的修改列表
        training_time: 训练时间（秒）
        timestamp: 实验时间戳
    """

    name: str
    description: str
    metrics: Dict[str, float] = field(default_factory=dict)
    modifications: List[AblationModification] = field(default_factory=list)
    training_time: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示。"""
        return {
            "name": self.name,
            "description": self.description,
            "metrics": self.metrics,
            "modifications": [m.to_dict() for m in self.modifications],
            "training_time": self.training_time,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AblationResult":
        """从字典表示创建。"""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            metrics=data.get("metrics", {}),
            modifications=[
                AblationModification.from_dict(m) for m in data.get("modifications", [])
            ],
            training_time=data.get("training_time", 0.0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class ExperimentSummary:
    """消融研究的摘要。

    Attributes:
        model_name: 模型名称
        baseline: 基线结果
        ablations: 消融结果列表
        comparisons: 性能比较
    """

    model_name: str
    baseline: AblationResult
    ablations: List[AblationResult] = field(default_factory=list)
    comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def compute_comparisons(self, metric: str = "auc") -> None:
        """计算每个消融相对于基线的性能下降。

        Args:
            metric: 要比较的指标（默认："auc"）
        """
        baseline_value = self.baseline.metrics.get(metric, 0.0)

        for ablation in self.ablations:
            ablation_value = ablation.metrics.get(metric, 0.0)
            drop = baseline_value - ablation_value

            self.comparisons[ablation.name] = {
                "baseline": baseline_value,
                "ablation": ablation_value,
                "drop": drop,
                "drop_percentage": (drop / baseline_value * 100)
                if baseline_value > 0
                else 0.0,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_name": self.model_name,
            "baseline": self.baseline.to_dict(),
            "ablations": [a.to_dict() for a in self.ablations],
            "comparisons": self.comparisons,
        }

    def print_summary(self, metric: str = "auc") -> None:
        """
        Print formatted summary of ablation study.

        Args:
            metric: Metric to display (default: "auc")
        """
        log = [f"Ablation Study Summary: {self.model_name}"]

        log.append(f"Baseline ({self.baseline.name}):")
        for key, value in self.baseline.metrics.items():
            log.append(f"  {key.upper()}: {value:.4f}")

        log.append("\nAblations:")
        log.append(f"{'Name':<30} {'Value':<10} {'Drop':<10} {'Drop %':<10}")
        log.append("-" * 60)
        for ablation in self.ablations:
            value = ablation.metrics.get(metric, 0.0)
            baseline_value = self.baseline.metrics.get(metric, 0.0)
            drop = baseline_value - value
            drop_pct = (drop / baseline_value * 100) if baseline_value > 0 else 0.0

            log.append(
                f"{ablation.name:<30} {value:<10.4f} {drop:<10.4f} {drop_pct:<10.2f}%"
            )

        logger.info("\n".join(log))


class AblationExperiment:
    """
    Orchestrates ablation experiments.

    Manages running baseline and ablation experiments,
    collecting results, and generating summaries.
    """

    def __init__(
        self,
        base_trainer: Type,
        config: AblationStudyConfig,
        args: Any,
        data_src: Any,
        exp_manager,
    ):
        """
        Initialize ablation experiment.

        Args:
            base_trainer: Trainer class to use for experiments
            config: Ablation study configuration
            args: Training arguments
            data_src: Data source
            exp_manager: ExperimentManager instance
        """
        self.base_trainer = base_trainer
        self.config = config
        self.args = args
        self.data_src = data_src
        self.exp_manager = exp_manager
        self.output_dir = Path(exp_manager.get_log_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[AblationResult] = []

    def run_baseline(self) -> AblationResult:
        """
        Run baseline experiment (no modifications).

        Returns:
            AblationResult: Baseline results
        """
        logger.info(f"Running baseline: {self.config.baseline.name}")

        start_time = time.time()

        # Create a copy of args to avoid modifying the original
        run_args = copy.deepcopy(self.args)

        # Create sub-experiment manager for baseline
        baseline_exp_manager = self.exp_manager.create_sub_experiment(
            self.config.baseline.name
        )

        # Create trainer and run
        trainer = self.base_trainer(
            args=run_args, data_src=self.data_src, exp_manager=baseline_exp_manager
        )
        trainer.run()

        # Collect metrics
        metrics = self._collect_metrics(trainer)

        training_time = time.time() - start_time

        result = AblationResult(
            name=self.config.baseline.name,
            description=self.config.baseline.description,
            metrics=metrics,
            modifications=self.config.baseline.modifications,
            training_time=training_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.results.append(result)
        return result

    def run_ablation(self, ablation_config: AblationConfig) -> AblationResult:
        """
        Run a single ablation experiment.

        Args:
            ablation_config: Ablation configuration

        Returns:
            AblationResult: Ablation results
        """
        logger.info(f"Running ablation: {ablation_config.name}")
        logger.info(f"Description: {ablation_config.description}")

        start_time = time.time()

        # Create a copy of args to avoid modifying the original
        run_args = copy.deepcopy(self.args)

        # Create sub-experiment manager for this ablation
        ablation_exp_manager = self.exp_manager.create_sub_experiment(
            ablation_config.name
        )

        # Create trainer
        trainer = self.base_trainer(
            args=run_args, data_src=self.data_src, exp_manager=ablation_exp_manager
        )
        model = trainer.model

        # Apply ablation strategies using ExitStack to manage multiple context managers
        with ExitStack() as stack:
            for mod in ablation_config.modifications:
                stack.enter_context(
                    apply_ablation(
                        model=model,
                        strategy_type=mod.type,
                        target=mod.target,
                        params=mod.params,
                    )
                )

            # Run training with ablations applied
            trainer.run()

        # Collect metrics
        metrics = self._collect_metrics(trainer)

        training_time = time.time() - start_time

        result = AblationResult(
            name=ablation_config.name,
            description=ablation_config.description,
            metrics=metrics,
            modifications=ablation_config.modifications,
            training_time=training_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.results.append(result)
        return result

    def run_all(self) -> ExperimentSummary:
        """
        Run all experiments (baseline + ablations).

        Returns:
            ExperimentSummary: Complete study summary
        """
        # Run baseline
        baseline = self.run_baseline()

        # Run ablations
        ablations = []
        for ablation_config in self.config.ablations:
            result = self.run_ablation(ablation_config)
            ablations.append(result)

        # Create summary
        summary = ExperimentSummary(
            model_name=self.config.model_name,
            baseline=baseline,
            ablations=ablations,
        )

        # Compute comparisons
        summary.compute_comparisons()

        # Print summary
        summary.print_summary()

        # Save results
        self._save_results(summary)

        return summary

    def _collect_metrics(self, trainer: Any) -> Dict[str, float]:
        """
        Collect metrics from trainer.

        Args:
            trainer: Trainer instance

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Get best metrics from early stopping if available
        if hasattr(trainer, "early_stopping") and trainer.early_stopping is not None:
            if trainer.early_stopping.best_score is not None:
                monitor = trainer._monitor_name()
                metrics[monitor] = float(trainer.early_stopping.best_score)

        # Get last validation metrics if available
        if hasattr(trainer, "_last_val_metrics"):
            for key, value in trainer._last_val_metrics.items():
                if value is not None:
                    metrics[key] = float(value)

        return metrics

    def _save_results(self, summary: ExperimentSummary) -> None:
        """
        Save experiment results to JSON file.

        Args:
            summary: Experiment summary to save
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = self.output_dir / f"{self.config.model_name}_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")


__all__ = [
    "AblationResult",
    "ExperimentSummary",
    "AblationExperiment",
]
