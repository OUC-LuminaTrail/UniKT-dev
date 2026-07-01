"""
通用超参数管理模块
支持超参数的保存、加载和验证
"""

import json
import os
from argparse import Namespace
from datetime import datetime
from typing import Any

import torch
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.core import get_logger

logger = get_logger(__name__)


class HyperparameterManager:
    """
    超参数管理器

    功能：
    1. 保存超参数到JSON/YAML文件
    2. 从文件加载超参数
    3. 验证超参数的完整性
    4. 生成超参数摘要
    5. 支持版本控制和实验追踪
    """

    def __init__(self, save_dir: str | None = None):
        """
        初始化超参数管理器

        Args:
            save_dir: 保存目录，如果为None则使用默认的runs目录
        """
        self.save_dir = save_dir
        self.hyperparams: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
        }
        # 摘要渲染专用终端，与日志 RichHandler 共享宽度/颜色探测
        self._console = Console()

    def get_hyperparameters_dict(self) -> dict[str, Any]:
        """
        获取当前超参数字典

        Returns:
            超参数字典
        """
        return self.hyperparams

    def add_hyperparams(self, params: dict | Namespace, group: str | None = None):
        """
        添加超参数（自动识别并序列化所有传入的参数）

        Args:
            params: 超参数字典或argparse.Namespace对象
            group: 参数组名称（如'model', 'training', 'data'）。如果为None，参数将添加到根级别
        """
        # 将Namespace转换为字典
        if isinstance(params, Namespace):
            params = vars(params)

        # 序列化所有参数
        serialized_params = self._serialize_params(params)

        if group:
            # 添加到指定组
            if group not in self.hyperparams:
                self.hyperparams[group] = {}
            self.hyperparams[group].update(serialized_params)
        else:
            # 添加到根级别
            self.hyperparams.update(serialized_params)

    def add_metadata(self, key: str, value: Any):
        """
        添加元数据信息

        Args:
            key: 元数据键
            value: 元数据值
        """
        self.metadata[key] = self._serialize_value(value)

    def _serialize_value(self, value: Any) -> Any:
        """
        序列化单个值，使其可以JSON化

        Args:
            value: 要序列化的值

        Returns:
            序列化后的值
        """
        # 处理None
        if value is None:
            return None
        # 处理基本类型
        elif isinstance(value, (int, float, str, bool)):
            return value
        # 处理PyTorch张量
        elif isinstance(value, torch.Tensor):
            return value.tolist()
        # 处理PyTorch设备
        elif isinstance(value, torch.device):
            return str(value)
        # 处理Namespace
        elif isinstance(value, Namespace):
            return vars(value)
        # 处理列表和元组
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        # 处理字典
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        # 处理集合
        elif isinstance(value, set):
            return list(value)
        # 处理Path对象
        elif hasattr(value, "__fspath__"):  # pathlib.Path
            return str(value)
        # 处理可调用对象（函数、类等）
        elif callable(value):
            if hasattr(value, "__name__"):
                return f"<callable: {value.__name__}>"
            else:
                return f"<callable: {type(value).__name__}>"
        # 其他对象尝试转换为字符串
        else:
            try:
                # 尝试直接转换
                return str(value)
            except Exception:
                # 如果失败，返回类型信息
                return f"<{type(value).__name__} object>"

    def _serialize_params(self, params: dict) -> dict:
        """
        序列化参数字典

        Args:
            params: 参数字典

        Returns:
            序列化后的参数字典
        """
        return {k: self._serialize_value(v) for k, v in params.items()}

    def save(self, filename: str = "hyperparameters.json", format: str = "json"):
        """
        保存超参数到文件

        Args:
            filename: 文件名
            format: 保存格式 ('json' 或 'yaml')
        """
        if self.save_dir is None:
            raise ValueError("Save directory not set. Please set save_dir first.")

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        filepath = os.path.join(self.save_dir, filename)

        # 准备保存的数据
        data = {"metadata": self.metadata, "hyperparameters": self.hyperparams}

        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        elif format == "yaml":
            try:
                import yaml

                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            except ImportError:
                logger.warning("PyYAML not installed. Saving as JSON instead.")
                with open(
                    filepath.replace(".yaml", ".json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")

        logger.info(f"Hyperparameters saved to: {filepath}")

    def load(self, filepath: str) -> dict:
        """
        从文件加载超参数

        Args:
            filepath: 文件路径

        Returns:
            加载的超参数字典
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Hyperparameter file not found: {filepath}")

        # 根据文件扩展名判断格式
        _, ext = os.path.splitext(filepath)

        if ext == ".json":
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        elif ext in [".yaml", ".yml"]:
            try:
                import yaml

                with open(filepath, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML not installed. Cannot load YAML file.")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        self.metadata = data.get("metadata", {})
        self.hyperparams = data.get("hyperparameters", {})

        logger.info(f"Hyperparameters loaded from: {filepath}")
        return self.hyperparams

    def get_summary(self) -> str:
        """
        生成超参数摘要

        Returns:
            格式化的超参数摘要字符串
        """
        lines = []

        # 元数据
        lines.append("[Metadata]")
        for key, value in self.metadata.items():
            lines.append(f"  {key}: {value}")

        # 超参数
        lines.append("\n[Hyperparameters]")
        self._format_params(self.hyperparams, lines, indent=1)

        return "\n".join(lines)

    def render_summary(self) -> Panel:
        """渲染超参数摘要为 rich Panel：外层卡片 + 分组 + 双列 key/value。

        与 get_summary() 的纯文本不同，此方法产出带边框与着色的可视卡片，
        供训练启动时直观展示，避免多行纯文本刷屏。
        """
        sections: list = []

        # 元数据压缩为一行（model_name/dataset_name 已上移到标题，created_at 过长隐藏）
        meta_pairs = [
            f"{k}={v}"
            for k, v in self.metadata.items()
            if k not in ("model_name", "dataset_name", "created_at")
        ]
        if meta_pairs:
            sections.append(Text("   ".join(meta_pairs), style="dim"))
            sections.append(Text(""))

        for group, params in self.hyperparams.items():
            if not params:
                continue
            sections.append(Text(f"■ {group.capitalize()}", style="bold magenta"))
            sections.append(self._param_grid(params))
            sections.append(Text(""))

        return Panel(
            Group(*sections),
            title=self._build_title(),
            title_align="left",
            border_style="blue",
            padding=(0, 1),
            expand=False,
        )

    def print_summary(self, console: Console | None = None) -> None:
        """打印 rich 超参数摘要卡片。"""
        (console or self._console).print(self.render_summary())

    def _param_grid(self, params: dict) -> Table:
        """把组内参数排成双列 key/value 网格，行数减半。"""
        items = list(params.items())
        half = (len(items) + 1) // 2
        left, right = items[:half], items[half:]

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="cyan", no_wrap=True)
        grid.add_column()
        grid.add_column(style="cyan", no_wrap=True)
        grid.add_column()

        for i in range(half):
            lk, lv = left[i]
            if i < len(right):
                rk, rv = right[i]
                grid.add_row(str(lk), str(lv), str(rk), str(rv))
            else:
                grid.add_row(str(lk), str(lv), "", "")
        return grid

    def _build_title(self) -> str:
        """构造卡片标题，拼接可用的 model/dataset 元信息。"""
        model = self.metadata.get("model_name")
        dataset = self.metadata.get("dataset_name")
        if model and dataset:
            return f"Hyperparameters · {model} @ {dataset}"
        if model:
            return f"Hyperparameters · {model}"
        if dataset:
            return f"Hyperparameters · @{dataset}"
        return "Hyperparameters"

    def _format_params(self, params: dict, lines: list, indent: int = 0):
        """
        递归格式化参数字典

        Args:
            params: 参数字典
            lines: 输出行列表
            indent: 缩进级别
        """
        indent_str = "  " * indent
        for key, value in params.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                self._format_params(value, lines, indent + 1)
            else:
                lines.append(f"{indent_str}{key}: {value}")

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """
        将嵌套字典展平

        Args:
            d: 要展平的字典
            parent_key: 父键名
            sep: 分隔符

        Returns:
            展平后的字典
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def to_namespace(self, group: str | None = None) -> Namespace:
        """
        将超参数转换为Namespace对象

        Args:
            group: 如果指定，只转换特定组的参数

        Returns:
            Namespace对象
        """
        params = (
            self.hyperparams.get(group, {})
            if group
            else self._flatten_dict(self.hyperparams)
        )

        return Namespace(**params)

    def validate_required(self, required_params: list) -> bool:
        """
        验证必需的参数是否存在

        Args:
            required_params: 必需参数列表（支持点号表示法，如'model.hidden_dim'）

        Returns:
            是否所有必需参数都存在
        """
        flat_params = self._flatten_dict(self.hyperparams)
        missing = []

        for param in required_params:
            if param not in flat_params:
                missing.append(param)

        if missing:
            logger.warning(f"Missing required parameters: {missing}")
            return False

        return True


def create_hyperparameter_manager(
    args: dict | Namespace,
    save_dir: str,
    model_name: str | None = None,
    dataset_name: str | None = None,
    auto_group: bool = True,
) -> HyperparameterManager:
    """
    便捷函数：创建并配置超参数管理器

    Args:
        args: 超参数（字典或Namespace）
        save_dir: 保存目录
        model_name: 模型名称
        dataset_name: 数据集名称
        auto_group: 是否自动分组参数（默认True）。如果为False，则不分组直接保存所有参数

    Returns:
        配置好的HyperparameterManager实例
    """
    manager = HyperparameterManager(save_dir=save_dir)

    # 添加元数据
    if model_name:
        manager.add_metadata("model_name", model_name)
    if dataset_name:
        manager.add_metadata("dataset_name", dataset_name)

    # 转换为字典（如果是Namespace）
    args_dict = vars(args) if isinstance(args, Namespace) else args

    # 如果需要自动分组
    if auto_group:
        # 自动分组（基于常见的命名模式）
        model_params = {}
        training_params = {}
        data_params = {}
        other_params = {}

        for key, value in args_dict.items():
            # 模型相关参数
            if any(
                kw in key.lower()
                for kw in [
                    "hidden",
                    "embedding",
                    "layer",
                    "dropout",
                    "dim",
                    "hop",
                    "top_k",
                    "head",
                    "attention",
                ]
            ):
                model_params[key] = value
            # 训练相关参数
            elif any(
                kw in key.lower()
                for kw in [
                    "epoch",
                    "batch",
                    "lr",
                    "optimizer",
                    "loss",
                    "weight_decay",
                    "momentum",
                ]
            ):
                training_params[key] = value
            # 数据相关参数
            elif any(
                kw in key.lower() for kw in ["data", "dataset", "sequence", "max_len"]
            ):
                data_params[key] = value
            else:
                other_params[key] = value

        # 添加所有组（即使为空也不会影响）
        if model_params:
            manager.add_hyperparams(model_params, group="model")
        if training_params:
            manager.add_hyperparams(training_params, group="training")
        if data_params:
            manager.add_hyperparams(data_params, group="data")
        if other_params:
            manager.add_hyperparams(other_params, group="general")
    else:
        # 不分组，直接添加所有参数
        manager.add_hyperparams(args_dict)

    return manager


__all__ = ["HyperparameterManager", "create_hyperparameter_manager"]
