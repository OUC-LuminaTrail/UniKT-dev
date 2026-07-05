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
    gpu_total_memory_gib: float | None = None
    gpu_capability: str | None = None
    cpu_logical_cores: int | None = None
    cpu_physical_cores: int | None = None
    cpu_model: str | None = None
    total_ram_gib: float | None = None
    platform: str = ""
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False
    deterministic_algorithms: bool = False


def collect_environment(device: torch.device) -> EnvironmentInfo:
    """采集一次性环境元数据。任何子项失败退化为 None，绝不抛错。"""
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

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        info.cuda_version = torch.version.cuda
        info.gpu_name = torch.cuda.get_device_name(idx)
        info.gpu_count = torch.cuda.device_count()
        try:
            props = torch.cuda.get_device_properties(idx)
            info.gpu_total_memory_gib = props.total_memory / 1024**3
            info.gpu_capability = f"{props.major}.{props.minor}"
        except Exception:
            pass

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


class ResourceSampler:
    """后台线程按固定间隔采样 CPU/RAM/GPU 利用率与功耗。

    psutil/pynvml 查询是阻塞的（每样本 1-5ms），不能落在延迟计时关键路径上，
    因此放 daemon 线程。这些调用释放 GIL，可与 GPU kernel 并发。
    """

    def __init__(self, device: torch.device, interval_s: float = 0.05) -> None:
        self.device = device
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu: list[float] = []
        self._rss: list[float] = []
        self._gpu_util: list[float] = []
        self._gpu_power: list[float] = []
        self._gpu_mem: list[float] = []
        self._gpu_temp: list[float] = []
        self._proc = None
        self._nv_handle = None
        self._nv_ok = False
        self._nvml_inited = False

    def start(self) -> None:
        try:
            import psutil

            self._proc = psutil.Process(os.getpid())
            # 首次 cpu_percent 无基线返回 0，先预热
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
                # 标记 NVML 已初始化，stop 据此 shutdown——即使后续 GetHandle 失败也要配对
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

    def _sample_once(self) -> None:
        if self._proc is not None:
            try:
                import psutil

                cpu = self._proc.cpu_percent(interval=None)
                rss = self._proc.memory_info().rss / 1024**2
                # 包含 DataLoader 子进程
                for child in self._proc.children(recursive=True):
                    try:
                        cpu += child.cpu_percent(interval=None)
                        rss += child.memory_info().rss / 1024**2
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self._cpu.append(cpu)
                self._rss.append(rss)
            except Exception:
                pass

        if self._nv_ok and self._nv_handle is not None:
            try:
                import pynvml

                util = pynvml.nvmlDeviceGetUtilizationRates(self._nv_handle)
                self._gpu_util.append(float(util.gpu))
                self._gpu_mem.append(float(util.memory))
                # 功耗返回 mW
                self._gpu_power.append(
                    float(pynvml.nvmlDeviceGetPowerUsage(self._nv_handle)) / 1000.0
                )
                self._gpu_temp.append(
                    float(
                        pynvml.nvmlDeviceGetTemperature(
                            self._nv_handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                )
            except Exception:
                pass

    def stop(self) -> ResourceStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml_inited:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass

        return ResourceStats(
            cpu_percent=_summarize(self._cpu),
            process_rss_mib=_summarize(self._rss),
            gpu_util_pct=_summarize(self._gpu_util),
            gpu_power_w=_summarize(self._gpu_power),
            gpu_mem_used_mib=_summarize(self._gpu_mem),
            gpu_temp_c=_summarize(self._gpu_temp),
        )


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
