import subprocess
import threading

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/system", tags=["system"])


class CapabilitiesResponse(BaseModel):
    has_gpu: bool
    gpu_count: int
    gpu_names: list[str]


_cached_capabilities: CapabilitiesResponse | None = None
_capabilities_lock = threading.Lock()


def _detect_gpu() -> CapabilitiesResponse:
    global _cached_capabilities
    if _cached_capabilities is not None:
        return _cached_capabilities
    with _capabilities_lock:
        if _cached_capabilities is not None:
            return _cached_capabilities
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                _cached_capabilities = CapabilitiesResponse(
                    has_gpu=False, gpu_count=0, gpu_names=[]
                )
                return _cached_capabilities
            names = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            _cached_capabilities = CapabilitiesResponse(
                has_gpu=len(names) > 0, gpu_count=len(names), gpu_names=names
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            _cached_capabilities = CapabilitiesResponse(
                has_gpu=False, gpu_count=0, gpu_names=[]
            )
    return _cached_capabilities


@router.get("/capabilities", response_model=CapabilitiesResponse)
def get_capabilities():
    return _detect_gpu()
