from fastapi import APIRouter, Depends
from main import get_gpu_monitor
from schemas import GpuStatusResponse
from services.gpu_monitor import GpuMonitor

router = APIRouter(prefix="/api/gpu", tags=["gpu"])


@router.get("/status", response_model=GpuStatusResponse)
def gpu_status(monitor: GpuMonitor = Depends(get_gpu_monitor)):
    return monitor.get_status()
