"""Environment metadata + background resource sampling."""

import contextlib
import os
import platform
import statistics
import threading
from dataclasses import dataclass, field

import torch

from utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class EnvironmentInfo:
    """硬件/软件环境快照（一次性采集）。"""

    python_version: str = ""
    torch_version: str = ""
    numpy_version: str = ""
    cuda_available: bool = False
    cuda_version: str | None = None
    cudnn_version: int | None = None
    device_type: str = "cpu"
    gpu_name: str | None = None
    gpu_count: int = 0
    gpu_total_memory_mib: float | None = None
    gpu_capability: str | None = None
    gpu_max_sm_clock_mhz: int | None = None
    cpu_logical_cores: int | None = None
    cpu_physical_cores: int | None = None
    cpu_model: str | None = None
    total_ram_gib: float | None = None
    platform: str = ""
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False
    deterministic_algorithms: bool = False
    cuda_matmul_allow_tf32: bool = False
    cudnn_allow_tf32: bool = False
    float32_matmul_precision: str | None = None
    model_dtype: str = ""


def collect_environment(device: torch.device, model: torch.nn.Module | None = None) -> EnvironmentInfo:
    """Collect one-shot environment metadata; subitems fall back to defaults on failure."""
    import sys

    info = EnvironmentInfo(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        device_type=device.type,
        cuda_available=torch.cuda.is_available(),
        platform=platform.platform(),
    )

    try:
        import numpy

        info.numpy_version = numpy.__version__
    except Exception:
        pass

    info.cpu_logical_cores = os.cpu_count()

    try:
        import psutil

        info.cpu_physical_cores = psutil.cpu_count(logical=False)
        info.total_ram_gib = psutil.virtual_memory().total / 1024**3
    except Exception:
        pass

    with contextlib.suppress(Exception):
        import cpuinfo

        info.cpu_model = cpuinfo.get_cpu_info().get("brand_raw")

    if torch.backends.cudnn.is_available():
        info.cudnn_version = int(torch.backends.cudnn.version())
    info.cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    info.cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    with contextlib.suppress(Exception):
        info.deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

    # TF32 changes matmul/cuDNN effective throughput; record to explain latency vs logical FLOPs.
    info.cuda_matmul_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    info.cudnn_allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    with contextlib.suppress(Exception):
        info.float32_matmul_precision = torch.get_float32_matmul_precision()
    if model is not None:
        with contextlib.suppress(Exception):
            info.model_dtype = str(next(model.parameters()).dtype)

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        info.cuda_version = torch.version.cuda
        info.gpu_name = torch.cuda.get_device_name(idx)
        info.gpu_count = torch.cuda.device_count()
        try:
            props = torch.cuda.get_device_properties(idx)
            info.gpu_total_memory_mib = props.total_memory / 1024**2
            info.gpu_capability = f"{props.major}.{props.minor}"
        except Exception:
            pass
        # Max SM clock (device capability) for normalizing sampled clock.
        with contextlib.suppress(Exception):
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                info.gpu_max_sm_clock_mhz = int(
                    pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
                )
            finally:
                pynvml.nvmlShutdown()

    return info


@dataclass
class ResourceSummary:
    """单指标聚合：mean/peak/min/p50 + 样本数。"""

    mean: float | None = None
    peak: float | None = None
    min: float | None = None
    p50: float | None = None
    n: int = 0


@dataclass
class ResourceStats:
    """后台资源采样聚合结果。"""

    cpu_percent: ResourceSummary = field(default_factory=ResourceSummary)
    process_rss_mib: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_util_pct: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_power_w: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_mem_used_mib: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_temp_c: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_sm_clock_mhz: ResourceSummary = field(default_factory=ResourceSummary)


class ResourceSampler:
    """Background thread sampling CPU/RAM/GPU utilization and power per stage.

    psutil/pynvml queries block (1-5ms each), kept off the latency timing path
    via a daemon thread; they release the GIL and overlap GPU kernels.
    """

    def __init__(self, device: torch.device, interval_s: float = 0.05) -> None:
        self.device = device
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Per-stage sample buckets; _current routes each sample to its stage.
        self._buckets: dict[str, dict[str, list[float]]] = {}
        self._current: str | None = None
        self._proc = None
        self._nv_handle = None
        self._nv_ok = False
        self._nvml_inited = False

    def start(self) -> None:
        try:
            import psutil

            self._proc = psutil.Process(os.getpid())
            # First cpu_percent call has no baseline (returns 0); prime it here.
            self._proc.cpu_percent(interval=None)
            for child in self._proc.children(recursive=True):
                child.cpu_percent(interval=None)
        except Exception as e:
            logger.warning(
                f"[Setup] psutil unavailable, CPU/RAM sampling disabled: {e}"
            )
            self._proc = None

        if self.device.type == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                # Mark NVML initialized so stop() pairs the shutdown even if GetHandle fails.
                self._nvml_inited = True
                idx = self.device.index or 0
                self._nv_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._nv_ok = True
            except Exception as e:
                logger.warning(
                    f"[Setup] pynvml unavailable, GPU sampling disabled: {e}"
                )
                self._nv_ok = False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def begin_stage(self, name: str) -> None:
        """Route subsequent samples to the named stage's bucket."""
        if name not in self._buckets:
            self._buckets[name] = _new_bucket()
        self._current = name

    def end_stage(self) -> None:
        """Stop attributing samples until the next stage begins."""
        self._current = None

    def _sample_once(self) -> None:
        # Snapshot _current once; end_stage() on the main thread can flip it to
        # None between two attribute reads, so a second load risks KeyError.
        current = self._current
        if current is None:
            return
        bucket = self._buckets.get(current)
        if bucket is None:
            return
        if self._proc is not None:
            try:
                import psutil

                cpu = self._proc.cpu_percent(interval=None)
                rss = self._proc.memory_info().rss / 1024**2
                # Include DataLoader child processes
                for child in self._proc.children(recursive=True):
                    try:
                        cpu += child.cpu_percent(interval=None)
                        rss += child.memory_info().rss / 1024**2
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                bucket["cpu"].append(cpu)
                bucket["rss"].append(rss)
            except Exception:
                pass

        if self._nv_ok and self._nv_handle is not None:
            try:
                import pynvml

                util = pynvml.nvmlDeviceGetUtilizationRates(self._nv_handle)
                bucket["gpu_util"].append(float(util.gpu))
                bucket["gpu_mem"].append(float(util.memory))
                # Power returns mW
                bucket["gpu_power"].append(
                    float(pynvml.nvmlDeviceGetPowerUsage(self._nv_handle)) / 1000.0
                )
                bucket["gpu_temp"].append(
                    float(
                        pynvml.nvmlDeviceGetTemperature(
                            self._nv_handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                )
                bucket["gpu_sm_clock"].append(
                    float(
                        pynvml.nvmlDeviceGetClockInfo(
                            self._nv_handle, pynvml.NVML_CLOCK_SM
                        )
                    )
                )
            except Exception:
                pass

    def stop(self) -> dict[str, ResourceStats]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml_inited:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass

        return {
            name: ResourceStats(
                cpu_percent=_summarize(b["cpu"]),
                process_rss_mib=_summarize(b["rss"]),
                gpu_util_pct=_summarize(b["gpu_util"]),
                gpu_power_w=_summarize(b["gpu_power"]),
                gpu_mem_used_mib=_summarize(b["gpu_mem"]),
                gpu_temp_c=_summarize(b["gpu_temp"]),
                gpu_sm_clock_mhz=_summarize(b["gpu_sm_clock"]),
            )
            for name, b in self._buckets.items()
        }


def _new_bucket() -> dict[str, list[float]]:
    """Empty per-stage sample container."""
    return {
        "cpu": [],
        "rss": [],
        "gpu_util": [],
        "gpu_power": [],
        "gpu_mem": [],
        "gpu_temp": [],
        "gpu_sm_clock": [],
    }


def _summarize(xs: list[float]) -> ResourceSummary:
    """聚合样本为 mean/peak/min/p50。"""
    if not xs:
        return ResourceSummary()
    return ResourceSummary(
        mean=sum(xs) / len(xs),
        peak=max(xs),
        min=min(xs),
        p50=statistics.median(xs),
        n=len(xs),
    )
