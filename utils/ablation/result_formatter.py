"""消融实验结果格式化模块。

此模块提供统一的消融实验结果输出格式化功能，支持控制台表格输出和 CSV 文件导出。
主要用于将 AblationRunner.run_all() 返回的结果格式化为易读的形式。

Usage:
    from utils.ablation.result_formatter import AblationResultFormatter

    # 假设 results 是 AblationRunner.run_all() 返回的结果列表
    formatter = AblationResultFormatter(results, ranking_metric='auc')
    formatter.print_console_table()
    formatter.export_to_csv(output_path='results.csv')

Expected Data Structure:
    results 是一个字典列表，每个字典包含：
    {
        "name": str,          # 实验名称
        "variant": str,       # 模型变体名称
        "metrics": dict       # 评估指标字典，如 {"acc": 0.85, "auc": 0.82, "rmse": 0.35}
    }

Output Formats:
    - Console: 使用 Rich 库生成彩色表格，按指标排序
    - CSV: 使用标准 csv 模块导出，包含所有指标和汇总统计

Note:
    - 支持的指标：acc（越高越好）、auc（越高越好）、rmse（越低越好）
    - 表格会自动对齐列宽，突出显示最佳结果
    - CSV 文件包含原始数据和汇总统计（均值、标准差）
    - 基线对比自动计算 delta 值（acc/auc: variant - baseline, rmse: baseline - variant）
"""

import csv
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


class AblationResultFormatter:
    """消融实验结果格式化器

    将 AblationRunner 返回的结果列表格式化为美观的表格并导出为 CSV。
    支持基线对比、按指定指标排序和最佳模型高亮。

    Attributes:
        results: 结果列表，每个元素包含 'name', 'variant', 'metrics' 键
        ranking_metric: 用于排序和查找最佳模型的指标名称
        HIGHER_IS_BETTER: 值越大越好的指标列表
        LOWER_IS_BETTER: 值越小越好的指标列表
    """

    # 定义指标方向性
    HIGHER_IS_BETTER = ["acc", "auc"]
    LOWER_IS_BETTER = ["rmse"]

    def __init__(
        self, results: list[dict[str, Any]], ranking_metric: str = "auc"
    ) -> None:
        """初始化格式化器

        Args:
            results: AblationRunner 返回的结果列表
            ranking_metric: 用于排序和查找最佳模型的指标名称，默认 'auc'
        """
        self.results = results
        self.ranking_metric = ranking_metric
        self.console = Console()

    def _get_baseline_metrics(self) -> dict[str, float]:
        """获取基线指标值

        从第一个结果中提取指标作为基线。

        Returns:
            基线指标字典，包含 'acc', 'auc', 'rmse'

        Raises:
            ValueError: 如果 results 为空或没有指标数据
        """
        if not self.results:
            raise ValueError("Results list is empty")

        first_result = self.results[0]
        metrics = first_result.get("metrics", {})

        if not metrics:
            raise ValueError("No metrics found in baseline result")

        return metrics

    def _calculate_delta(
        self, value: float, baseline_value: float, metric_name: str
    ) -> float:
        """计算指标值相对于基线的差异

        对于 acc/auc: delta = value - baseline (越大越好)
        对于 rmse: delta = baseline - value (越小越好)

        Args:
            value: 当前指标值
            baseline_value: 基线指标值
            metric_name: 指标名称

        Returns:
            差异值（可为正或负）
        """
        if metric_name in self.HIGHER_IS_BETTER:
            return value - baseline_value
        elif metric_name in self.LOWER_IS_BETTER:
            return baseline_value - value
        else:
            # 未知指标，默认使用 "越大越好" 的计算方式
            return value - baseline_value

    def _find_best_variant(self) -> str:
        """查找具有最佳 ranking_metric 值的变体名称

        Returns:
            最佳变体的 'name' 字段值

        Raises:
            ValueError: 如果 results 为空或没有指标数据
        """
        if not self.results:
            raise ValueError("Results list is empty")

        best_result = None
        best_value = None

        for result in self.results:
            metrics = result.get("metrics", {})
            if self.ranking_metric not in metrics:
                continue

            value = metrics[self.ranking_metric]

            # 根据指标方向性确定最佳值
            if self.ranking_metric in self.HIGHER_IS_BETTER:
                is_better = best_value is None or value > best_value
            elif self.ranking_metric in self.LOWER_IS_BETTER:
                is_better = best_value is None or value < best_value
            else:
                # 未知指标，默认使用 "越大越好"
                is_better = best_value is None or value > best_value

            if is_better:
                best_value = value
                best_result = result

        if best_result is None:
            raise ValueError(f"No results found with metric '{self.ranking_metric}'")

        return best_result.get("name", "")

    def export_to_csv(self, output_path: str) -> None:
        """将消融实验结果导出为 CSV 文件

        导出的 CSV 文件包含排序后的结果、指标值和相对于基线的差异值。
        按指定的 ranking_metric 排序。

        Args:
            output_path: 输出 CSV 文件路径

        Raises:
            ValueError: 如果 results 为空或没有指标数据
        """
        if not self.results:
            raise ValueError("Results list is empty")

        # 获取基线指标
        baseline_metrics = self._get_baseline_metrics()

        # 创建输出目录（如果不存在）
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # 按 ranking_metric 排序结果
        sorted_results = self._sort_results_by_metric()

        # 定义 CSV 列
        metrics_to_export = ["acc", "auc", "rmse"]
        delta_metrics = ["delta_acc", "delta_auc", "delta_rmse"]
        all_columns = (
            ["rank", "variant", "description"] + metrics_to_export + delta_metrics
        )

        # 写入 CSV 文件
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

            # 写入表头
            writer.writerow(all_columns)

            # 写入数据行
            for rank, result in enumerate(sorted_results, start=1):
                row = []
                metrics = result.get("metrics", {})
                description = result.get("description", "")
                variant = result.get("name", "")

                # 排名
                row.append(rank)

                # 变体名称
                row.append(variant)

                # 描述
                row.append(description)

                # 原始指标
                for metric in metrics_to_export:
                    if metric in metrics:
                        row.append(metrics[metric])
                    else:
                        row.append("null")

                # Delta 指标
                for i, metric in enumerate(metrics_to_export):
                    if metric in metrics and metric in baseline_metrics:
                        delta = self._calculate_delta(
                            metrics[metric], baseline_metrics[metric], metric
                        )
                        row.append(delta)
                    else:
                        row.append("null")

                writer.writerow(row)

    def _sort_results_by_metric(self) -> list[dict[str, Any]]:
        """按 ranking_metric 对结果进行排序

        根据 ranking_metric 的方向性（越高越好或越低越好）排序结果。

        Returns:
            排序后的结果列表

        Raises:
            ValueError: 如果 results 为空或没有指标数据
        """
        if not self.results:
            raise ValueError("Results list is empty")

        def get_sort_value(result: dict[str, Any]) -> float:
            metrics = result.get("metrics", {})
            if self.ranking_metric not in metrics:
                # 如果缺少指标，返回极值使其排在最后
                if self.ranking_metric in self.HIGHER_IS_BETTER:
                    return float("-inf")
                else:
                    return float("inf")
            return metrics[self.ranking_metric]

        # 根据 ranking_metric 的方向性决定升序还是降序
        reverse = self.ranking_metric in self.HIGHER_IS_BETTER
        return sorted(self.results, key=get_sort_value, reverse=reverse)

    def print_console_table(self) -> None:
        """打印控制台表格

        使用 Rich 库生成美观的表格，显示所有消融实验结果。
        表格包括排名、变体名称、描述、指标值和相对于基线的差异。
        最佳模型行将使用粗体和绿色高亮显示。

        Raises:
            ValueError: 如果 results 为空或没有指标数据
        """
        if not self.results:
            raise ValueError("Results list is empty")

        # 获取基线指标
        baseline_metrics = self._get_baseline_metrics()

        # 按 ranking_metric 排序结果
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get("metrics", {}).get(self.ranking_metric, 0),
            reverse=(self.ranking_metric in self.HIGHER_IS_BETTER),
        )

        # 查找所有最佳变体（支持并列）
        best_value = None
        best_variant_names = []

        for result in self.results:
            metrics = result.get("metrics", {})
            if self.ranking_metric not in metrics:
                continue

            value = metrics[self.ranking_metric]

            # 根据指标方向性确定最佳值
            if self.ranking_metric in self.HIGHER_IS_BETTER:
                is_best = best_value is None or value > best_value
            elif self.ranking_metric in self.LOWER_IS_BETTER:
                is_best = best_value is None or value < best_value
            else:
                # 未知指标，默认使用 "越大越好"
                is_best = best_value is None or value > best_value

            if is_best:
                best_value = value
                best_variant_names = [result.get("name", "")]
            elif value == best_value:
                # 并列最佳，添加到列表
                best_variant_names.append(result.get("name", ""))

        # 创建表格
        table = Table(title="Ablation Study Results")
        table.add_column("Rank", style="bold", justify="right")
        table.add_column("Variant", style="bold")
        table.add_column("Description", style="italic")

        for metric in ["acc", "auc", "rmse"]:
            table.add_column(f"{metric.upper()}", style="cyan", justify="right")
            table.add_column(f"Delta_{metric.upper()}", justify="right")

        # 添加数据行
        for rank, result in enumerate(sorted_results, start=1):
            name = result.get("name", "N/A")
            description = result.get("description", "")
            metrics = result.get("metrics", {})

            row = [str(rank), name, description]

            # 为每个指标添加值和差值
            for metric in ["acc", "auc", "rmse"]:
                value = metrics.get(metric)
                baseline_value = baseline_metrics.get(metric, 0)

                # 格式化指标值
                if value is not None:
                    formatted_value = f"{value:.4f}"
                    delta_value = self._calculate_delta(value, baseline_value, metric)
                    formatted_delta = (
                        f"{'+' if delta_value >= 0 else ''}{delta_value:.4f}"
                    )
                else:
                    formatted_value = "N/A"
                    formatted_delta = "N/A"

                row.extend([formatted_value, formatted_delta])

            # 如果是最佳变体，使用粗体绿色样式高亮
            if name in best_variant_names:
                table.add_row(*row, style="bold green")
            else:
                table.add_row(*row)

        # 打印表格
        self.console.print(table)


__all__ = ["AblationResultFormatter"]
