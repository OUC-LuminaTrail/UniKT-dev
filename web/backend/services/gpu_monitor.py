import contextlib
import os
import time

import psutil
import pynvml
from schemas import GpuInfo, GpuStatusResponse, SystemStatusResponse


class GpuMonitor:
    def __init__(self, cache_seconds: float = 2.0):
        self._cache_seconds = cache_seconds
        self._cached: GpuStatusResponse | None = None
        self._last_update: float = 0

    def get_status(self) -> GpuStatusResponse:
        now = time.time()
        if self._cached and (now - self._last_update) < self._cache_seconds:
            return self._cached

        gpus = self._query_nvml()
        self._cached = GpuStatusResponse(
            gpus=gpus,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._last_update = now
        return self._cached

    def _query_nvml(self) -> list[GpuInfo]:
        try:
            pynvml.nvmlInit()
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
            pynvml.nvmlShutdown()
            return gpus
        except (pynvml.NVMLError, Exception):
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
            return []

    def get_system_status(self) -> SystemStatusResponse:
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
