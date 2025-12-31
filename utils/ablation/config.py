"""消融研究的配置管理。

处理消融配置文件的加载、验证和解析。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


@dataclass
class AblationModification:
    """表示对模型的单个修改。

    Attributes:
        type: 策略类型（module_disable、feature_zero、parameter_freeze、module_replace）
        target: 目标模块名称（例如 "conv"、"history_review"）
        params: 策略的额外参数
    """

    type: Literal[
        "module_disable",
        "module_zero",
        "feature_zero",
        "parameter_freeze",
        "module_replace",
    ]
    target: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示。"""
        return {
            "type": self.type,
            "target": self.target,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AblationModification":
        """从字典表示创建。"""
        return cls(
            type=data["type"],
            target=data["target"],
            params=data.get("params", {}),
        )


@dataclass
class AblationConfig:
    """单个消融实验的配置。

    Attributes:
        name: 实验名称
        description: 可读的描述
        modifications: 要应用的修改列表
    """

    name: str
    description: str
    modifications: List[AblationModification] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示。"""
        return {
            "name": self.name,
            "description": self.description,
            "modifications": [m.to_dict() for m in self.modifications],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AblationConfig":
        """从字典表示创建。"""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            modifications=[
                AblationModification.from_dict(m) for m in data.get("modifications", [])
            ],
        )


@dataclass
class AblationStudyConfig:
    """消融研究的完整配置。

    Attributes:
        model_name: 要消融的模型名称
        baseline: 基线配置（无修改）
        ablations: 消融配置列表
    """

    model_name: str
    baseline: AblationConfig
    ablations: List[AblationConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示。"""
        return {
            "model_name": self.model_name,
            "baseline": self.baseline.to_dict(),
            "ablations": [a.to_dict() for a in self.ablations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AblationStudyConfig":
        """从字典表示创建。"""
        return cls(
            model_name=data["model_name"],
            baseline=AblationConfig.from_dict(data["baseline"]),
            ablations=[AblationConfig.from_dict(a) for a in data.get("ablations", [])],
        )


def validate_ablation_config(
    config: Dict[str, Any], model: Optional[Any] = None
) -> bool:
    """验证消融配置结构。

    Args:
        config: 要验证的配置字典
        model: 可选的模型实例，用于验证目标

    Returns:
        如果有效则返回 True

    Raises:
        ValueError: 如果配置无效
    """
    required_keys = ["model_name", "baseline"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    valid_strategies = [
        "module_disable",
        "module_zero",
        "feature_zero",
        "parameter_freeze",
        "module_replace",
    ]

    # Validate baseline
    baseline = config["baseline"]
    if "name" not in baseline:
        raise ValueError("Baseline configuration must have a 'name'")

    # Validate ablations
    for ablation in config.get("ablations", []):
        if "name" not in ablation:
            raise ValueError("Ablation configuration must have a 'name'")

        # Check for duplicate ablation names
        ablation_names = [a["name"] for a in config.get("ablations", [])]
        if ablation_names.count(ablation["name"]) > 1:
            raise ValueError(f"Duplicate ablation name: {ablation['name']}")

        for mod in ablation.get("modifications", []):
            if "type" not in mod:
                raise ValueError("Modification must have a 'type'")
            if mod["type"] not in valid_strategies:
                raise ValueError(
                    f"Invalid strategy type: {mod['type']}. "
                    f"Valid types: {valid_strategies}"
                )
            if "target" not in mod:
                raise ValueError("Modification must have a 'target'")

            # Additional validation for specific strategies
            if mod["type"] == "feature_zero":
                if "params" not in mod or "indices" not in mod["params"]:
                    raise ValueError("Feature zero strategy requires 'params.indices'")
                if not mod["params"]["indices"]:
                    raise ValueError("Feature zero strategy requires non-empty indices")

            elif mod["type"] == "module_replace":
                if "params" not in mod or "replacement_module" not in mod["params"]:
                    raise ValueError(
                        "Module replace strategy requires 'params.replacement_module'"
                    )

    return True


def load_ablation_config(config_path: str) -> AblationStudyConfig:
    """从 JSON 文件加载消融配置。

    Args:
        config_path: 配置文件路径

    Returns:
        AblationStudyConfig: 解析的配置

    Raises:
        FileNotFoundError: 如果配置文件不存在
        ValueError: 如果配置无效
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_ablation_config(data)
    return AblationStudyConfig.from_dict(data)


def save_ablation_config(
    config: AblationStudyConfig, output_path: str, indent: int = 2
) -> None:
    """将消融配置保存到 JSON 文件。

    Args:
        config: 要保存的配置
        output_path: 保存配置的路径
        indent: JSON 缩进级别
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=indent, ensure_ascii=False)


__all__ = [
    "AblationModification",
    "AblationConfig",
    "AblationStudyConfig",
    "validate_ablation_config",
    "load_ablation_config",
    "save_ablation_config",
]
