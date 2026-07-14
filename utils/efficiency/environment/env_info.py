"""One-shot hardware/software environment snapshot."""

import contextlib
import os
import platform
from dataclasses import dataclass

import torch


@dataclass
class EnvironmentInfo:
    """Hardware/software environment snapshot (one-shot collection)."""

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

    def determinism_dict(self) -> dict:
        """Determinism-related flags, for the report's ``determinism`` block."""
        return {
            "cudnn_benchmark": self.cudnn_benchmark,
            "cudnn_deterministic": self.cudnn_deterministic,
            "deterministic_algorithms": self.deterministic_algorithms,
        }


def collect_environment(
    device: torch.device, model: torch.nn.Module | None = None
) -> EnvironmentInfo:
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
