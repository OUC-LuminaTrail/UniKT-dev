"""Metric logging backend module.

Provides a unified metric logging abstraction (MetricLogger) consistent
with the project's registry + ABC + factory pattern (see DataSource /
DATA_SOURCES / get_data_source).

Backends:
- LocalMetricLogger: Local CSV logging, always enabled.
- SwanLabMetricLogger: SwanLab remote logging, enabled by default,
  disabled via ``--no_swanlab``.
"""

import csv
import os
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import IO, Any

from ..core import METRIC_LOGGERS, get_logger, register_metric_logger

logger = get_logger(__name__)


class MetricLogger(ABC):
    """Abstract base class for metric logging backends.

    All methods must be safe to call repeatedly. Backend initialization
    (e.g. swanlab.init) is deferred to ``init_run``; ``log_*`` methods
    should silently skip when not initialized.
    """

    @abstractmethod
    def init_run(
        self,
        *,
        log_dir: str,
        experiment_name: str,
        group: str,
        tags: list[str],
        config: dict[str, Any],
    ) -> None:
        """Initialize the logging backend for a new run.

        Args:
            log_dir: Directory for storing logs.
            experiment_name: Name of this experiment.
            group: Experiment group for grouping runs.
            tags: Tags for categorizing the run.
            config: Experiment configuration dict.
        """

    @abstractmethod
    def log_metrics(
        self,
        *,
        phase: str,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log epoch-level aggregated metrics.

        Args:
            phase: Phase name (e.g. ``"train"``, ``"val"``).
            metrics: Metric name to value mapping.
            step: Global step number.
            epoch: Current epoch number.
            stage: Stage name for multi-stage training (optional).
        """

    @abstractmethod
    def log_early_stopping(
        self,
        *,
        phase: str,
        best_score: float | None,
        num_bad_epochs: int,
        best_metrics: dict[str, float] | None,
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log early stopping trajectory.

        Args:
            phase: Phase name.
            best_score: Best monitored score so far.
            num_bad_epochs: Number of epochs without improvement.
            best_metrics: Best metric values (optional).
            step: Global step number.
            epoch: Current epoch number.
            stage: Stage name (optional).
        """

    @abstractmethod
    def log_batch(
        self,
        *,
        phase: str,
        global_step: int,
        epoch: int,
        batch_idx: int,
        loss: float,
        stage: str | None = None,
    ) -> None:
        """Log per-batch loss.

        Args:
            phase: Phase name.
            global_step: Global step number.
            epoch: Current epoch number.
            batch_idx: Batch index within the epoch.
            loss: Loss value for this batch.
            stage: Stage name (optional).
        """

    @abstractmethod
    def log_final(self, *, metrics: dict[str, float], step: int) -> None:
        """Log final summary metrics.

        Args:
            metrics: Final metric values.
            step: Global step number.
        """

    @abstractmethod
    def finish(self) -> None:
        """Clean up and tear down the logging backend."""


@register_metric_logger("local")
class LocalMetricLogger(MetricLogger):
    """Local CSV metric logger.

    Writes CSV files under ``log_dir`` per phase:
      ``metrics_{phase}.csv``           Epoch-level aggregated metrics.
      ``early_stopping[_{stage}].csv``   Early stopping trajectory.
      ``batch_metrics_{phase}.csv``      Per-batch loss (when enabled).
      ``metrics_final.csv``              Final summary metrics.

    Multi-stage runs distinguish series via ``metrics_{stage}_{phase}.csv``.
    Headers are written lazily and extended in-place when new metric
    columns appear.
    """

    def __init__(self, *, log_dir: str, log_batch_metrics: bool = False):
        """Initialize the local CSV metric logger.

        Args:
            log_dir: Directory for output CSV files.
            log_batch_metrics: Whether to log per-batch loss.
        """
        self._log_dir = log_dir
        self._log_batch = log_batch_metrics
        os.makedirs(log_dir, exist_ok=True)
        self._csv_files: dict[str, IO] = {}
        self._csv_headers: dict[str, list[str]] = {}

    @staticmethod
    def _series(phase: str, stage: str | None) -> str:
        """Build a series identifier from phase and optional stage."""
        return f"{stage}_{phase}" if stage else phase

    def init_run(self, **kwargs) -> None:
        """No-op: log_dir was already set at construction time."""
        pass

    def _write_row(
        self, path: str, leading: list[tuple[str, Any]], values: dict[str, Any]
    ) -> None:
        """Write a single CSV row.

        ``leading`` columns are fixed prefix pairs ``(name, value)``,
        ``values`` are dynamic metric columns. The header is determined
        on first write; subsequent rows align to existing columns, with
        missing values left blank.

        Args:
            path: CSV file path.
            leading: Ordered list of ``(column_name, value)`` prefix pairs.
            values: Dynamic metric column name to value mapping.
        """
        f = self._csv_files.get(path)
        if f is None:
            header = [c for c, _ in leading] + sorted(values.keys())
            f = open(path, "a", newline="")  # noqa: SIM115
            self._csv_files[path] = f
            self._csv_headers[path] = header
            csv.writer(f).writerow(header)
        metric_cols = self._csv_headers[path][len(leading) :]
        row = [v for _, v in leading] + [values.get(c, "") for c in metric_cols]
        csv.writer(f).writerow(row)
        f.flush()

    def log_metrics(self, *, phase, metrics, step, epoch, stage=None) -> None:
        """Log epoch-level metrics to a CSV file.

        Args:
            phase: Phase name (e.g. ``"train"``, ``"val"``).
            metrics: Metric name to value mapping.
            step: Global step (unused in CSV logging).
            epoch: Current epoch number.
            stage: Stage name (optional).
        """
        values = {k: v for k, v in metrics.items() if v is not None}
        path = os.path.join(self._log_dir, f"metrics_{self._series(phase, stage)}.csv")
        self._write_row(path, [("epoch", epoch)], values)

    def log_early_stopping(
        self,
        *,
        phase,
        best_score,
        num_bad_epochs,
        best_metrics,
        step,
        epoch,
        stage=None,
    ) -> None:
        """Log early stopping trajectory to a CSV file.

        Args:
            phase: Phase name.
            best_score: Best monitored score.
            num_bad_epochs: Consecutive epochs without improvement.
            best_metrics: Best metric values (optional).
            step: Global step (unused in CSV logging).
            epoch: Current epoch number.
            stage: Stage name (optional).
        """
        values: dict[str, Any] = {
            "best_score": best_score if best_score is not None else "",
            "num_bad_epochs": num_bad_epochs,
        }
        if best_metrics:
            values.update({f"best_{k}": v for k, v in best_metrics.items()})
        filename = f"early_stopping_{stage}.csv" if stage else "early_stopping.csv"
        path = os.path.join(self._log_dir, filename)
        self._write_row(path, [("epoch", epoch)], values)

    def log_batch(
        self, *, phase, global_step, epoch, batch_idx, loss, stage=None
    ) -> None:
        """Log per-batch loss to a CSV file.

        Only writes when ``log_batch_metrics`` was enabled at init.

        Args:
            phase: Phase name.
            global_step: Global step number.
            epoch: Current epoch number.
            batch_idx: Batch index within the epoch.
            loss: Loss value for this batch.
            stage: Stage name (optional).
        """
        if not self._log_batch:
            return
        path = os.path.join(
            self._log_dir, f"batch_metrics_{self._series(phase, stage)}.csv"
        )
        self._write_row(
            path,
            [
                ("global_step", global_step),
                ("epoch", epoch),
                ("batch_idx", batch_idx),
                ("loss", loss),
            ],
            {},
        )

    def log_final(self, *, metrics, step) -> None:
        """Log final summary metrics to a CSV file.

        Args:
            metrics: Final metric values.
            step: Global step number.
        """
        if not metrics:
            return
        path = os.path.join(self._log_dir, "metrics_final.csv")
        self._write_row(path, [("step", step)], dict(metrics))

    def finish(self) -> None:
        """Close all open CSV file handles."""
        for f in self._csv_files.values():
            with suppress(Exception):
                f.close()
        self._csv_files.clear()
        self._csv_headers.clear()


@register_metric_logger("swanlab")
class SwanLabMetricLogger(MetricLogger):
    """SwanLab metric logging backend.

    ``swanlab`` is lazily imported inside each method to keep it an
    optional dependency.
    """

    def __init__(self):
        """Initialize the SwanLab logger with uninitialized state."""
        self._initialized = False

    def init_run(self, *, log_dir, experiment_name, group, tags, config) -> None:
        """Initialize the SwanLab run.

        Args:
            log_dir: Log directory (passed through to swanlab).
            experiment_name: Name of this experiment.
            group: Experiment group.
            tags: Tags for the run.
            config: Experiment configuration dict.
        """
        import swanlab
        from dotenv import load_dotenv
        from swanlab.plugin.notification import LarkCallback

        load_dotenv()
        callbacks = []
        webhook = os.getenv("LARK_WEBHOOK_URL")
        secret = os.getenv("LARK_SECRET")
        if webhook:
            callbacks.append(LarkCallback(webhook_url=webhook, secret=secret))

        swanlab.init(
            workspace=os.getenv("SWANLAB_WORKSPACE", None),
            project_name="kt-exp-graph",
            experiment_name=f"Run_{experiment_name}",
            config=config,
            callbacks=callbacks,
            group=group,
            tags=tags,
            settings=swanlab.Settings(),
        )
        self._initialized = True

    @staticmethod
    def _prefix(phase: str, stage: str | None) -> str:
        """Build a SwanLab metric prefix from phase and optional stage."""
        return (
            f"{stage.upper()}/{phase.capitalize()}/"
            if stage
            else f"{phase.capitalize()}/"
        )

    def log_metrics(self, *, phase, metrics, step, epoch, stage=None) -> None:
        """Log epoch-level metrics to SwanLab.

        Args:
            phase: Phase name.
            metrics: Metric name to value mapping.
            step: Global step number.
            epoch: Current epoch number (unused in SwanLab).
            stage: Stage name (optional).
        """
        if not self._initialized:
            return
        import swanlab

        prefix = self._prefix(phase, stage)
        payload = {
            f"{prefix}{name.upper()}-epoch": v
            for name, v in metrics.items()
            if v is not None
        }
        if payload:
            swanlab.log(payload, step=step)

    def log_early_stopping(
        self,
        *,
        phase,
        best_score,
        num_bad_epochs,
        best_metrics,
        step,
        epoch,
        stage=None,
    ) -> None:
        """Log early stopping trajectory to SwanLab.

        Args:
            phase: Phase name.
            best_score: Best monitored score.
            num_bad_epochs: Consecutive epochs without improvement.
            best_metrics: Best metric values (optional).
            step: Global step number.
            epoch: Current epoch number (unused in SwanLab).
            stage: Stage name (optional).
        """
        if not self._initialized:
            return
        import swanlab

        sp = f"{stage.upper()}/" if stage else ""
        data = {
            f"{sp}ES/Best": best_score,
            f"{sp}ES/Num_Bad_Epochs": num_bad_epochs,
        }
        if best_metrics:
            data.update(
                {f"{sp}ES/Best_{k.upper()}": v for k, v in best_metrics.items()}
            )
        swanlab.log(data, step=step)

    def log_batch(self, **kwargs) -> None:
        """No-op: SwanLab does not log per-batch metrics to avoid noise."""

    def log_final(self, *, metrics, step) -> None:
        """Log final summary metrics to SwanLab.

        Args:
            metrics: Final metric values.
            step: Global step number.
        """
        if not self._initialized or not metrics:
            return
        import swanlab

        swanlab.log(metrics, step=step)

    def finish(self) -> None:
        """Finish the SwanLab run and clean up."""
        if not self._initialized:
            return
        import swanlab

        swanlab.finish()
        self._initialized = False


class MetricLoggerComposite(MetricLogger):
    """Composite wrapper that fans out to multiple metric logger backends.

    Each method call is forwarded to every backend; exceptions from
    individual backends are caught and logged.
    """

    def __init__(self, loggers: list[MetricLogger]):
        """Initialize the composite logger.

        Args:
            loggers: List of MetricLogger instances to fan out to.
        """
        self._loggers = loggers

    def _fanout(self, method: str, **kwargs) -> None:
        """Call a method on all wrapped loggers, isolating exceptions."""
        for lg in self._loggers:
            try:
                getattr(lg, method)(**kwargs)
            except Exception as e:
                logger.warning(f"{type(lg).__name__}.{method} failed: {e}")

    def init_run(self, **kwargs) -> None:
        """Initialize all backends."""
        self._fanout("init_run", **kwargs)

    def log_metrics(self, **kwargs) -> None:
        """Log metrics to all backends."""
        self._fanout("log_metrics", **kwargs)

    def log_early_stopping(self, **kwargs) -> None:
        """Log early stopping to all backends."""
        self._fanout("log_early_stopping", **kwargs)

    def log_batch(self, **kwargs) -> None:
        """Log batch metrics to all backends."""
        self._fanout("log_batch", **kwargs)

    def log_final(self, **kwargs) -> None:
        """Log final metrics to all backends."""
        self._fanout("log_final", **kwargs)

    def finish(self) -> None:
        """Finish logging on all backends."""
        self._fanout("finish")


def get_metric_logger(name: str, **kwargs) -> MetricLogger:
    """Instantiate a registered metric logger backend by name.

    Args:
        name: Backend registration name (e.g. ``"local"``, ``"swanlab"``).
        **kwargs: Arguments forwarded to the backend constructor.

    Returns:
        An instance of the requested MetricLogger subclass.
    """
    cls = METRIC_LOGGERS.get(name)
    return cls(**kwargs)


def build_default_metric_loggers(
    *, log_dir: str, log_batch_metrics: bool, no_swanlab: bool
) -> MetricLogger:
    """Build the default metric logger composite.

    Local CSV logging is always enabled; SwanLab is included unless
    ``no_swanlab`` is set.

    Args:
        log_dir: Directory for local CSV logs.
        log_batch_metrics: Whether to log per-batch loss.
        no_swanlab: If True, skip SwanLab backend.

    Returns:
        A MetricLoggerComposite instance.
    """
    loggers: list[MetricLogger] = [
        get_metric_logger("local", log_dir=log_dir, log_batch_metrics=log_batch_metrics)
    ]
    if not no_swanlab:
        loggers.append(get_metric_logger("swanlab"))
    return MetricLoggerComposite(loggers)


def resolve_metric_logging_flags(experiment_config, hyperparams) -> tuple[bool, bool]:
    """Resolve ``no_swanlab`` and ``log_batch_metrics`` flags.

    Explicitly passed values take precedence; otherwise fall back to
    CLI arguments in ``hyperparams``. This ensures ``--no_swanlab``
    and ``--log_batch_metrics`` work for all model trainers.

    Args:
        experiment_config: ExperimentConfig instance.
        hyperparams: Hyperparameter object or namespace (optional).

    Returns:
        Tuple of ``(no_swanlab: bool, log_batch_metrics: bool)``.
    """
    no_swanlab = experiment_config.no_swanlab or bool(
        getattr(hyperparams, "no_swanlab", False)
    )
    log_batch_metrics = experiment_config.log_batch_metrics or bool(
        getattr(hyperparams, "log_batch_metrics", False)
    )
    return no_swanlab, log_batch_metrics


__all__ = [
    "LocalMetricLogger",
    "MetricLogger",
    "MetricLoggerComposite",
    "SwanLabMetricLogger",
    "build_default_metric_loggers",
    "get_metric_logger",
    "resolve_metric_logging_flags",
]
