"""实验配置模块

提供实验配置的数据类定义。
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path

from utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """实验配置类

    用于序列化和存储实验的元数据配置。

    Attributes:
        exp_type: 实验类型（normal/ablation/hyperparam_search）
        model_name: 模型名称
        dataset_name: 数据集名称
        base_dir: 基础目录
        tags: 标签列表
        description: 实验描述（可选）

    Example:
        >>> config = ExperimentConfig(
        ...     exp_type="normal",
        ...     model_name="GIKT",
        ...     dataset_name="assist09",
        ...     tags=["fold0", "bs64"]
        ... )
        >>> config_dict = config.to_dict()
        >>> json_str = config.to_json()
    """

    exp_type: str
    model_name: str
    dataset_name: str
    base_dir: str = "runs"
    tags: Optional[List[str]] = field(default_factory=list)
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典

        Returns:
            包含所有配置字段的字典
        """
        return {
            "exp_type": self.exp_type,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "base_dir": self.base_dir,
            "tags": self.tags,
            "description": self.description,
        }

    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串

        Args:
            indent: JSON缩进空格数

        Returns:
            JSON格式的字符串
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, filepath: str):
        """保存配置到文件

        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

        logger.info(f"Experiment config saved to: {filepath}")

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """从字典创建配置

        Args:
            data: 配置字典

        Returns:
            ExperimentConfig 实例
        """
        return cls(
            exp_type=data.get("exp_type", "normal"),
            model_name=data.get("model_name", ""),
            dataset_name=data.get("dataset_name", ""),
            base_dir=data.get("base_dir", "runs"),
            tags=data.get("tags", []),
            description=data.get("description"),
        )

    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """从文件加载配置

        Args:
            filepath: 配置文件路径

        Returns:
            ExperimentConfig 实例
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Experiment config loaded from: {filepath}")
        return cls.from_dict(data)

    def validate(self) -> bool:
        """验证配置的有效性

        Returns:
            配置是否有效
        """
        valid_exp_types = ["normal", "ablation", "hyperparam_search"]

        if self.exp_type not in valid_exp_types:
            logger.error(
                f"Invalid exp_type: {self.exp_type}. Must be one of {valid_exp_types}"
            )
            return False

        if not self.model_name:
            logger.error("model_name is required")
            return False

        if not self.dataset_name:
            logger.error("dataset_name is required")
            return False

        return True

    def __str__(self) -> str:
        """字符串表示"""
        tags_str = f", tags={self.tags}" if self.tags else ""
        desc_str = f", description={self.description}" if self.description else ""
        return f"ExperimentConfig({self.exp_type}, {self.model_name}, {self.dataset_name}{tags_str}{desc_str})"


__all__ = ["ExperimentConfig"]
