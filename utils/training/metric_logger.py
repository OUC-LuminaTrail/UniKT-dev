"""指标记录后端模块

提供统一的指标记录抽象（MetricLogger），与项目的注册表+ABC+工厂模式一致
（参照 DataSource / DATA_SOURCES / get_data_source）。

后端：
- LocalMetricLogger：本地 CSV 记录，始终启用。
- SwanLabMetricLogger：SwanLab 记录，默认启用，可通过 --no_swanlab 关闭。
"""

import csv
import os
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import IO, Any

from ..core import METRIC_LOGGERS, get_logger, register_metric_logger

logger = get_logger(__name__)


class MetricLogger(ABC):
    """指标记录后端抽象基类。

    所有方法须可安全重复调用。后端初始化（如 swanlab.init）在 init_run 中完成，
    未初始化时各 log_* 方法应静默跳过。
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
    ) -> None: ...

    @abstractmethod
    def log_metrics(
        self,
        *,
        phase: str,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    @abstractmethod
    def log_final(self, *, metrics: dict[str, float], step: int) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


@register_metric_logger("local")
class LocalMetricLogger(MetricLogger):
    """本地 CSV 指标记录器。

    在 log_dir 下按 phase 写入：
      metrics_{phase}.csv           每 epoch 的聚合指标（train/val/test）
      early_stopping[_{stage}].csv  早停轨迹（best_score / num_bad_epochs / best_*）
      batch_metrics_{phase}.csv     每 batch 的 loss（仅 log_batch_metrics 开启时）
      metrics_final.csv             最终摘要指标

    多阶段场景以 stage 区分 series：metrics_{stage}_{phase}.csv。
    表头惰性写入，后续出现新指标列时原地扩展表头。
    """

    def __init__(self, *, log_dir: str, log_batch_metrics: bool = False):
        self._log_dir = log_dir
        self._log_batch = log_batch_metrics
        os.makedirs(log_dir, exist_ok=True)
        self._csv_files: dict[str, IO] = {}
        self._csv_headers: dict[str, list[str]] = {}

    @staticmethod
    def _series(phase: str, stage: str | None) -> str:
        return f"{stage}_{phase}" if stage else phase

    def init_run(self, **kwargs) -> None:
        # log_dir 已在构造时设置，无需额外初始化
        pass

    def _write_row(
        self, path: str, leading: list[tuple[str, Any]], values: dict[str, Any]
    ) -> None:
        """写入一行。leading 为固定前缀列 [(列名, 值)]，values 为动态指标列。

        表头在首次写入时确定；后续行按已存表头对齐，缺失列留空。
        同一 series 的指标列在实际训练中保持稳定，故无需扩展表头。
        """
        f = self._csv_files.get(path)
        if f is None:
            # 句柄在整个 run 期间复用，按需 flush 以保证崩溃安全
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
        if not metrics:
            return
        path = os.path.join(self._log_dir, "metrics_final.csv")
        self._write_row(path, [("step", step)], dict(metrics))

    def finish(self) -> None:
        for f in self._csv_files.values():
            with suppress(Exception):
                f.close()
        self._csv_files.clear()
        self._csv_headers.clear()


@register_metric_logger("swanlab")
class SwanLabMetricLogger(MetricLogger):
    """SwanLab 指标记录后端。swanlab 仅在方法内惰性导入，保持其为可选依赖。"""

    def __init__(self):
        self._initialized = False

    def init_run(self, *, log_dir, experiment_name, group, tags, config) -> None:
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
        return (
            f"{stage.upper()}/{phase.capitalize()}/"
            if stage
            else f"{phase.capitalize()}/"
        )

    def log_metrics(self, *, phase, metrics, step, epoch, stage=None) -> None:
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
        # SwanLab 不记录 batch 级别 loss（量级过大），仅本地后端记录
        pass

    def log_final(self, *, metrics, step) -> None:
        if not self._initialized or not metrics:
            return
        import swanlab

        swanlab.log(metrics, step=step)

    def finish(self) -> None:
        if not self._initialized:
            return
        import swanlab

        swanlab.finish()
        self._initialized = False


class MetricLoggerComposite(MetricLogger):
    """组合多个指标记录后端，统一 fan-out 并隔离单个后端的异常。"""

    def __init__(self, loggers: list[MetricLogger]):
        self._loggers = loggers

    def _fanout(self, method: str, **kwargs) -> None:
        for lg in self._loggers:
            try:
                getattr(lg, method)(**kwargs)
            except Exception as e:
                logger.warning(f"{type(lg).__name__}.{method} failed: {e}")

    def init_run(self, **kwargs) -> None:
        self._fanout("init_run", **kwargs)

    def log_metrics(self, **kwargs) -> None:
        self._fanout("log_metrics", **kwargs)

    def log_early_stopping(self, **kwargs) -> None:
        self._fanout("log_early_stopping", **kwargs)

    def log_batch(self, **kwargs) -> None:
        self._fanout("log_batch", **kwargs)

    def log_final(self, **kwargs) -> None:
        self._fanout("log_final", **kwargs)

    def finish(self) -> None:
        self._fanout("finish")


def get_metric_logger(name: str, **kwargs) -> MetricLogger:
    """按名称实例化已注册的指标记录后端。"""
    cls = METRIC_LOGGERS.get(name)
    return cls(**kwargs)


def build_default_metric_loggers(
    *, log_dir: str, log_batch_metrics: bool, no_swanlab: bool
) -> MetricLogger:
    """构建默认指标记录组合：本地始终启用，SwanLab 除非 no_swanlab。"""
    loggers: list[MetricLogger] = [
        get_metric_logger("local", log_dir=log_dir, log_batch_metrics=log_batch_metrics)
    ]
    if not no_swanlab:
        loggers.append(get_metric_logger("swanlab"))
    return MetricLoggerComposite(loggers)


def resolve_metric_logging_flags(experiment_config, hyperparams) -> tuple[bool, bool]:
    """解析 no_swanlab / log_batch_metrics：显式传入优先，否则回退到 CLI 参数（hyperparams）。

    多数训练器子类调用 with_experiment 时不透传这两个参数，故统一在此回退读取，
    使 --no_swanlab / --log_batch_metrics 对所有模型生效。
    """
    no_swanlab = experiment_config.no_swanlab or bool(
        getattr(hyperparams, "no_swanlab", False)
    )
    log_batch_metrics = experiment_config.log_batch_metrics or bool(
        getattr(hyperparams, "log_batch_metrics", False)
    )
    return no_swanlab, log_batch_metrics


__all__ = [
    "MetricLogger",
    "LocalMetricLogger",
    "SwanLabMetricLogger",
    "MetricLoggerComposite",
    "get_metric_logger",
    "build_default_metric_loggers",
    "resolve_metric_logging_flags",
]
