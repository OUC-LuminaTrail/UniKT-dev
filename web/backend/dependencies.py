"""FastAPI dependency injection singletons.

Holds module-level references to ProcessManager, GpuMonitor,
PythonEnvManager, SettingsManager, and PreprocessManager instances.
Each getter raises if the instance is not initialized before
returning it, allowing lifespan startup to set them up.
"""

from typing import TypeVar

from services.gpu_monitor import GpuMonitor
from services.line_render import LineRenderCache
from services.preprocess_manager import PreprocessManager
from services.process_manager import ProcessManager
from services.python_env import PythonEnvManager
from services.schema_extractor import SchemaExtractor
from services.settings_manager import SettingsManager

process_manager: ProcessManager | None = None
gpu_monitor: GpuMonitor | None = None
python_env_manager: PythonEnvManager | None = None
schema_extractor: SchemaExtractor | None = None
settings_manager: SettingsManager | None = None
preprocess_manager: PreprocessManager | None = None
line_cache: LineRenderCache | None = None

T = TypeVar("T")


def _require(instance: T | None, name: str) -> T:
    """Return *instance*, or raise if lifespan startup has not set it yet.

    A plain ``assert`` would be stripped under ``python -O``, leaving the
    getters to return None and fail later with an opaque AttributeError.
    """
    if instance is None:
        raise RuntimeError(f"{name} is not initialized")
    return instance


def get_process_manager() -> ProcessManager:
    """Return the global ProcessManager singleton.

    Returns:
        The initialized ProcessManager instance.

    Raises:
        RuntimeError: If the manager has not been initialized.
    """
    return _require(process_manager, "ProcessManager")


def get_gpu_monitor() -> GpuMonitor:
    """Return the global GpuMonitor singleton.

    Returns:
        The initialized GpuMonitor instance.

    Raises:
        RuntimeError: If the monitor has not been initialized.
    """
    return _require(gpu_monitor, "GpuMonitor")


def get_python_env_manager() -> PythonEnvManager:
    """Return the global PythonEnvManager singleton.

    Returns:
        The initialized PythonEnvManager instance.

    Raises:
        RuntimeError: If the manager has not been initialized.
    """
    return _require(python_env_manager, "PythonEnvManager")


def get_preprocess_manager() -> PreprocessManager:
    """Return the global PreprocessManager singleton.

    Returns:
        The initialized PreprocessManager instance.

    Raises:
        RuntimeError: If the manager has not been initialized.
    """
    return _require(preprocess_manager, "PreprocessManager")


def get_settings_manager() -> SettingsManager:
    """Return the global SettingsManager singleton.

    Returns:
        The initialized SettingsManager instance.

    Raises:
        RuntimeError: If the manager has not been initialized.
    """
    return _require(settings_manager, "SettingsManager")


def get_line_cache() -> LineRenderCache:
    """Return the global LineRenderCache singleton.

    Shared by the log routers (HTTP/WS reads) and both task managers (PTY feeds).
    """
    return _require(line_cache, "LineRenderCache")


def get_schema_extractor() -> SchemaExtractor:
    """Return the global SchemaExtractor singleton.

    Shared by the schemas API and ProcessManager so the model schema is
    extracted once and its field->node route map is reused at task creation.

    Returns:
        The initialized SchemaExtractor instance.

    Raises:
        RuntimeError: If the extractor has not been initialized.
    """
    return _require(schema_extractor, "SchemaExtractor")
