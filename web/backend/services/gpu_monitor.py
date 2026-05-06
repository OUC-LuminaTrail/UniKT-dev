import os
import subprocess
import time

import psutil
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

        gpus = self._query_nvidia_smi()
        self._cached = GpuStatusResponse(
            gpus=gpus,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._last_update = now
        return self._cached

    def _query_nvidia_smi(self) -> list[GpuInfo]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                gpus.append(
                    GpuInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        utilization_percent=float(parts[2]),
                        memory_used_mb=float(parts[3]),
                        memory_total_mb=float(parts[4]),
                        temperature_c=float(parts[5]),
                        power_usage_w=float(parts[6]),
                        processes=[],
                    )
                )
            return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
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
