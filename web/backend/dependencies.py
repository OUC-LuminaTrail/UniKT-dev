from services.gpu_monitor import GpuMonitor
from services.preprocess_manager import PreprocessManager
from services.process_manager import ProcessManager

process_manager: ProcessManager | None = None
gpu_monitor: GpuMonitor | None = None


def get_process_manager() -> ProcessManager:
    assert process_manager is not None
    return process_manager


def get_gpu_monitor() -> GpuMonitor:
    assert gpu_monitor is not None
    return gpu_monitor


preprocess_manager: PreprocessManager | None = None


def get_preprocess_manager() -> PreprocessManager:
    assert preprocess_manager is not None
    return preprocess_manager
