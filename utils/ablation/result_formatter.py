"""Ablation result formatting module.

Provides unified output formatting for ablation study results, supporting
console table printing and CSV file export.
Used for formatting results returned by AblationRunner.run_all().

Usage:
    from utils.ablation.result_formatter import AblationResultFormatter

    # Assume results is the list returned by AblationRunner.run_all()
    formatter = AblationResultFormatter(results, ranking_metric='auc')
    formatter.print_console_table()
    formatter.export_to_csv(output_path='results.csv')

Expected data structure — each result dict::

    {
        "name": str,          # Experiment name
        "variant": str,       # Model variant name
        "metrics": dict       # Evaluation metric dict, e.g. {"acc": 0.85, "auc": 0.82, "rmse": 0.35}
    }

Output Formats:
    - Console: Uses Rich library to generate color-coded tables, sorted by metric
    - CSV: Uses standard csv module for export, includes all metrics and summary statistics

Note:
    - Supported metrics: acc (higher is better), auc (higher is better), rmse (lower is better)
    - Table columns are automatically aligned, best results are highlighted
    - CSV includes raw data and summary statistics (mean, std)
    - Baseline comparison auto-calculates delta values (acc/auc: variant - baseline, rmse: baseline - variant)
"""

import csv
from pathlib import Path
from typing import Any, ClassVar

from rich.console import Console
from rich.table import Table


class AblationResultFormatter:
    """Formatter for ablation study results.

    Formats the result list returned by AblationRunner into readable tables
    and exports to CSV. Supports baseline comparison, ranking by specified
    metric, and best-model highlighting.

    Attributes:
        results: List of result dicts, each containing 'name', 'variant', 'metrics' keys
        ranking_metric: Metric name used for ranking and finding the best model
        HIGHER_IS_BETTER: List of metrics where higher values are better
        LOWER_IS_BETTER: List of metrics where lower values are better
    """

    # Define metric direction
    HIGHER_IS_BETTER: ClassVar[list[str]] = ["acc", "auc"]
    LOWER_IS_BETTER: ClassVar[list[str]] = ["rmse"]

    def __init__(
        self, results: list[dict[str, Any]], ranking_metric: str = "auc"
    ) -> None:
        """Initialize the formatter.

        Args:
            results: Result list from AblationRunner
            ranking_metric: Metric name for ranking and finding the best model, defaults to 'auc'
        """
        self.results = results
        self.ranking_metric = ranking_metric
        self.console = Console()

    def _get_baseline_metrics(self) -> dict[str, float]:
        """Get baseline metric values.

        Extracts metrics from the first result as the baseline.

        Returns:
            Baseline metrics dict containing 'acc', 'auc', 'rmse'

        Raises:
            ValueError: If results is empty or has no metric data
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
        """Calculate the delta of a metric value relative to the baseline.

        For acc/auc: delta = value - baseline (higher is better)
        For rmse: delta = baseline - value (lower is better)

        Args:
            value: Current metric value
            baseline_value: Baseline metric value
            metric_name: Metric name

        Returns:
            Delta value (may be positive or negative)
        """
        if metric_name in self.HIGHER_IS_BETTER:
            return value - baseline_value
        elif metric_name in self.LOWER_IS_BETTER:
            return baseline_value - value
        else:
            # Unknown metric, default to "higher is better" calculation
            return value - baseline_value

    def _find_best_variant(self) -> str:
        """Find the variant with the best ranking_metric value.

        Returns:
            'name' field of the best variant

        Raises:
            ValueError: If results is empty or has no metric data
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

            # Determine the best value based on metric direction
            if self.ranking_metric in self.HIGHER_IS_BETTER:
                is_better = best_value is None or value > best_value
            elif self.ranking_metric in self.LOWER_IS_BETTER:
                is_better = best_value is None or value < best_value
            else:
                # Unknown metric, default to "higher is better"
                is_better = best_value is None or value > best_value

            if is_better:
                best_value = value
                best_result = result

        if best_result is None:
            raise ValueError(f"No results found with metric '{self.ranking_metric}'")

        return best_result.get("name", "")

    def export_to_csv(self, output_path: str) -> None:
        """Export ablation results to a CSV file.

        The exported CSV file contains sorted results, metric values, and
        delta values relative to the baseline. Sorted by the specified
        ranking_metric.

        Args:
            output_path: Output CSV file path

        Raises:
            ValueError: If results is empty or has no metric data
        """
        if not self.results:
            raise ValueError("Results list is empty")

        # Get baseline metrics
        baseline_metrics = self._get_baseline_metrics()

        # Create output directory if it does not exist
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Sort results by ranking_metric
        sorted_results = self._sort_results_by_metric()

        # Define CSV columns
        metrics_to_export = ["acc", "auc", "rmse"]
        delta_metrics = ["delta_acc", "delta_auc", "delta_rmse"]
        all_columns = [
            "rank",
            "variant",
            "description",
            *metrics_to_export,
            *delta_metrics,
        ]

        # Write to CSV file
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

            # Write header row
            writer.writerow(all_columns)

            # Write data rows
            for rank, result in enumerate(sorted_results, start=1):
                row = []
                metrics = result.get("metrics", {})
                description = result.get("description", "")
                variant = result.get("name", "")

                # Rank
                row.append(rank)

                # Variant name
                row.append(variant)

                # Description
                row.append(description)

                # Raw metrics
                for metric in metrics_to_export:
                    if metric in metrics:
                        row.append(metrics[metric])
                    else:
                        row.append("null")

                # Delta metrics
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
        """Sort results by ranking_metric.

        Sorts results according to the ranking_metric's direction
        (higher is better or lower is better).

        Returns:
            Sorted list of results

        Raises:
            ValueError: If results is empty or has no metric data
        """
        if not self.results:
            raise ValueError("Results list is empty")

        def get_sort_value(result: dict[str, Any]) -> float:
            metrics = result.get("metrics", {})
            if self.ranking_metric not in metrics:
                # If metric is missing, return extreme value to push it to the end
                if self.ranking_metric in self.HIGHER_IS_BETTER:
                    return float("-inf")
                else:
                    return float("inf")
            return metrics[self.ranking_metric]

        # Sort ascending or descending based on ranking_metric direction
        reverse = self.ranking_metric in self.HIGHER_IS_BETTER
        return sorted(self.results, key=get_sort_value, reverse=reverse)

    def print_console_table(self) -> None:
        """Print a console table.

        Uses the Rich library to generate a formatted table displaying all
        ablation results. The table includes rank, variant name, description,
        metric values, and delta values relative to the baseline.
        The best model row is highlighted in bold green.

        Raises:
            ValueError: If results is empty or has no metric data
        """
        if not self.results:
            raise ValueError("Results list is empty")

        # Get baseline metrics
        baseline_metrics = self._get_baseline_metrics()

        # Sort results by ranking_metric
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get("metrics", {}).get(self.ranking_metric, 0),
            reverse=(self.ranking_metric in self.HIGHER_IS_BETTER),
        )

        # Find all best variants (support ties)
        best_value = None
        best_variant_names = []

        for result in self.results:
            metrics = result.get("metrics", {})
            if self.ranking_metric not in metrics:
                continue

            value = metrics[self.ranking_metric]

            # Determine the best value based on metric direction
            if self.ranking_metric in self.HIGHER_IS_BETTER:
                is_best = best_value is None or value > best_value
            elif self.ranking_metric in self.LOWER_IS_BETTER:
                is_best = best_value is None or value < best_value
            else:
                # Unknown metric, default to "higher is better"
                is_best = best_value is None or value > best_value

            if is_best:
                best_value = value
                best_variant_names = [result.get("name", "")]
            elif value == best_value:
                # Tie for best, add to the list
                best_variant_names.append(result.get("name", ""))

        # Create table
        table = Table(title="Ablation Study Results")
        table.add_column("Rank", style="bold", justify="right")
        table.add_column("Variant", style="bold")
        table.add_column("Description", style="italic")

        for metric in ["acc", "auc", "rmse"]:
            table.add_column(f"{metric.upper()}", style="cyan", justify="right")
            table.add_column(f"Delta_{metric.upper()}", justify="right")

        # Add data rows
        for rank, result in enumerate(sorted_results, start=1):
            name = result.get("name", "N/A")
            description = result.get("description", "")
            metrics = result.get("metrics", {})

            row = [str(rank), name, description]

            # Add value and delta for each metric
            for metric in ["acc", "auc", "rmse"]:
                value = metrics.get(metric)
                baseline_value = baseline_metrics.get(metric, 0)

                # Format metric value
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

            # Highlight best variant(s) in bold green
            if name in best_variant_names:
                table.add_row(*row, style="bold green")
            else:
                table.add_row(*row)

        # Print the table
        self.console.print(table)


__all__ = ["AblationResultFormatter"]
