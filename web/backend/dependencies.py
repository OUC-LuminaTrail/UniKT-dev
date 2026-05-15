from services.gpu_monitor import GpuMonitor
from services.preprocess_manager import PreprocessManager
from services.process_manager import ProcessManager
from services.python_env import PythonEnvManager
from services.settings_manager import SettingsManager

process_manager: ProcessManager | None = None
gpu_monitor: GpuMonitor | None = None
python_env_manager: PythonEnvManager | None = None
settings_manager: SettingsManager | None = None
preprocess_manager: PreprocessManager | None = None


def get_process_manager() -> ProcessManager:
    assert process_manager is not None
    return process_manager


def get_gpu_monitor() -> GpuMonitor:
    assert gpu_monitor is not None
    return gpu_monitor


def get_python_env_manager() -> PythonEnvManager:
    assert python_env_manager is not None
    return python_env_manager


def get_preprocess_manager() -> PreprocessManager:
    assert preprocess_manager is not None
    return preprocess_manager


def get_settings_manager() -> SettingsManager:
    assert settings_manager is not None
    return settings_manager
