"""FastAPI dependency injection singletons.

Holds module-level references to ProcessManager, GpuMonitor,
PythonEnvManager, SettingsManager, and PreprocessManager instances.
Each getter function asserts the instance is initialized before
returning it, allowing lifespan startup to set them up.
"""

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
    """Return the global ProcessManager singleton.

    Returns:
        The initialized ProcessManager instance.

    Raises:
        AssertionError: If the manager has not been initialized.
    """
    assert process_manager is not None
    return process_manager


def get_gpu_monitor() -> GpuMonitor:
    """Return the global GpuMonitor singleton.

    Returns:
        The initialized GpuMonitor instance.

    Raises:
        AssertionError: If the monitor has not been initialized.
    """
    assert gpu_monitor is not None
    return gpu_monitor


def get_python_env_manager() -> PythonEnvManager:
    """Return the global PythonEnvManager singleton.

    Returns:
        The initialized PythonEnvManager instance.

    Raises:
        AssertionError: If the manager has not been initialized.
    """
    assert python_env_manager is not None
    return python_env_manager


def get_preprocess_manager() -> PreprocessManager:
    """Return the global PreprocessManager singleton.

    Returns:
        The initialized PreprocessManager instance.

    Raises:
        AssertionError: If the manager has not been initialized.
    """
    assert preprocess_manager is not None
    return preprocess_manager


def get_settings_manager() -> SettingsManager:
    """Return the global SettingsManager singleton.

    Returns:
        The initialized SettingsManager instance.

    Raises:
        AssertionError: If the manager has not been initialized.
    """
    assert settings_manager is not None
    return settings_manager
