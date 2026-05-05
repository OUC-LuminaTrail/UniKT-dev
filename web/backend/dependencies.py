from services.process_manager import ProcessManager
from services.gpu_monitor import GpuMonitor

process_manager: ProcessManager | None = None
gpu_monitor: GpuMonitor | None = None


def get_process_manager() -> ProcessManager:
    assert process_manager is not None
    return process_manager


def get_gpu_monitor() -> GpuMonitor:
    assert gpu_monitor is not None
    return gpu_monitor
