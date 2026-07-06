"""GPU router — real-time GPU and system status endpoints.

Returns cached GPU status (utilization, memory, temperature) and system-level
status (CPU, memory, load averages).
"""

from dependencies import get_gpu_monitor
from fastapi import APIRouter, Depends
from schemas import GpuStatusResponse, SystemStatusResponse
from services.gpu_monitor import GpuMonitor

router = APIRouter(prefix="/api/gpu", tags=["gpu"])


@router.get("/status", response_model=GpuStatusResponse)
def gpu_status(monitor: GpuMonitor = Depends(get_gpu_monitor)):
    """Return current GPU status (utilization, memory, temperature).

    Args:
        monitor: Injected GpuMonitor singleton.

    Returns:
        A GpuStatusResponse with per-GPU information.
    """
    return monitor.get_status()


@router.get("/system", response_model=SystemStatusResponse)
def system_status(monitor: GpuMonitor = Depends(get_gpu_monitor)):
    """Return current system status (CPU, memory, load, GPU aggregate).

    Args:
        monitor: Injected GpuMonitor singleton.

    Returns:
        A SystemStatusResponse with CPU, memory, and GPU metrics.
    """
    return monitor.get_system_status()
