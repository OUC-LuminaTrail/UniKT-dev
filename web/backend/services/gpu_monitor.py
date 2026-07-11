"""GPU monitoring service with NVML-based status and system-level metrics.

Provides a GpuMonitor class that periodically queries NVIDIA GPU devices via
pynvml and system resources via psutil, with configurable caching.
"""

import contextlib
import os
import threading
import time

import psutil
import pynvml
from schemas import GpuInfo, GpuStatusResponse, SystemStatusResponse


class GpuMonitor:
    """Monitors GPU and system status with NVML and psutil.

    Caches GPU status for a configurable interval to reduce NVML query frequency.

    Args:
        cache_seconds: How long to cache GPU status results (default 2.0).
    """

    def __init__(self, cache_seconds: float = 2.0):
        """Initialize the GPU monitor and attempt NVML initialisation.

        Args:
            cache_seconds: Cache lifetime in seconds for GPU status.
        """
        self._cache_seconds = cache_seconds
        self._cached: GpuStatusResponse | None = None
        self._last_update: float = 0
        self._lock = threading.Lock()
        self._nvml_initialized = False
        self._device_count = 0
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._device_count = pynvml.nvmlDeviceGetCount()
        except (pynvml.NVMLError, Exception):
            pass

    @property
    def device_count(self) -> int:
        """Stable GPU count captured at init.

        Used for scheduling-lane sizing so a transient NVML miss in the cached
        ``get_status`` (which returns ``[]`` on any query error) cannot collapse
        all GPU lanes to the CPU lane mid-dispatch.
        """
        return self._device_count

    def get_status(self) -> GpuStatusResponse:
        """Return the current GPU status, using cached data if still fresh.

        Returns:
            A GpuStatusResponse with per-GPU metrics.
        """
        with self._lock:
            now = time.time()
            if self._cached and (now - self._last_update) < self._cache_seconds:
                return self._cached

        gpus = self._query_nvml()
        response = GpuStatusResponse(
            gpus=gpus,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        with self._lock:
            self._cached = response
            self._last_update = time.time()
        return response

    def _query_nvml(self) -> list[GpuInfo]:
        """Query all NVIDIA GPUs with NVML.

        Returns:
            A list of GpuInfo objects, or an empty list if NVML is unavailable.
        """
        if not self._nvml_initialized:
            return []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except pynvml.NVMLError:
                    power = 0.0
                gpus.append(
                    GpuInfo(
                        index=i,
                        name=name,
                        utilization_percent=float(util.gpu),
                        memory_used_mb=round(mem.used / (1024 * 1024), 1),
                        memory_total_mb=round(mem.total / (1024 * 1024), 1),
                        temperature_c=float(temp),
                        power_usage_w=round(power, 1),
                        processes=[],
                    )
                )
            return gpus
        except (pynvml.NVMLError, Exception):
            return []

    def get_system_status(self) -> SystemStatusResponse:
        """Return system-level CPU, memory and GPU aggregate status.

        Returns:
            A SystemStatusResponse with CPU, memory, load, and GPU metrics.
        """
        cpu_percent = psutil.cpu_percent(interval=0.3)
        mem = psutil.virtual_memory()
        gpus = self.get_status()
        gpu_util = 0.0
        gpu_mem_percent = 0.0
        if gpus.gpus:
            gpu_util = gpus.gpus[0].utilization_percent
            g = gpus.gpus[0]
            gpu_mem_percent = (
                (g.memory_used_mb / g.memory_total_mb * 100)
                if g.memory_total_mb > 0
                else 0
            )
        load1, load5, load15 = os.getloadavg()
        return SystemStatusResponse(
            cpu_percent=cpu_percent,
            memory_used_gb=round(mem.used / (1024**3), 1),
            memory_total_gb=round(mem.total / (1024**3), 1),
            memory_percent=mem.percent,
            gpu_utilization=gpu_util,
            gpu_memory_percent=round(gpu_mem_percent, 1),
            load_1m=round(load1, 2),
            load_5m=round(load5, 2),
            load_15m=round(load15, 2),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def shutdown(self):
        """Release NVML resources.

        Safe to call multiple times; suppresses errors.
        """
        if self._nvml_initialized:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
