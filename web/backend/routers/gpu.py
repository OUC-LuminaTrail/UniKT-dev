from dependencies import get_gpu_monitor
from fastapi import APIRouter, Depends
from schemas import GpuStatusResponse, SystemStatusResponse
from services.gpu_monitor import GpuMonitor

router = APIRouter(prefix="/api/gpu", tags=["gpu"])


@router.get("/status", response_model=GpuStatusResponse)
def gpu_status(monitor: GpuMonitor = Depends(get_gpu_monitor)):
    return monitor.get_status()


@router.get("/system", response_model=SystemStatusResponse)
def system_status(monitor: GpuMonitor = Depends(get_gpu_monitor)):
    return monitor.get_system_status()
