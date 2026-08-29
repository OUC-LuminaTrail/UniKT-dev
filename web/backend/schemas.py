"""Pydantic request/response models for the web API.

Defines all data transfer objects (DTOs) used by the FastAPI endpoints,
including task, environment, GPU, schema, and experiment schemas.
"""

from datetime import datetime

from pydantic import BaseModel


class TaskCreate(BaseModel):
    """Request model for creating a new experiment task.

    Attributes:
        name: A human-readable name for the task.
        env_id: Identifier of the Python environment to use.
        custom_python_path: An optional custom Python interpreter path.
        model_name: The model name to train.
        params: Additional training parameters as a dictionary.
        gpu: Requested GPU index for pinning, or None for auto-assignment.
    """

    name: str = ""
    env_id: str
    custom_python_path: str | None = None
    model_name: str
    params: dict
    gpu: int | None = None


class TaskResponse(BaseModel):
    """Response model representing a task's full state.

    Attributes:
        id: Unique task identifier.
        name: Task name.
        command: The command string that was launched.
        model_name: Model being trained.
        dataset_name: Dataset used for training.
        env_type: Environment type (e.g. pixi, conda, custom).
        env_name: Environment name within its type.
        python_path: Python interpreter path used.
        status: Current task status (pending, running, completed, etc.).
        pid: Process ID of the running task, or None.
        started_at: When the task started, or None.
        finished_at: When the task finished, or None.
        exit_code: Process exit code, or None.
        created_at: When the task record was created.
        tags: JSON-encoded list of tags.
        extra_params: JSON-encoded dictionary of extra parameters.
        gpu_request: Requested GPU index (None = auto), set at creation.
        gpu_assigned: Actual GPU index the task was dispatched to, or None.
    """

    id: int
    name: str
    command: str
    model_name: str
    dataset_name: str
    env_type: str
    env_name: str
    python_path: str | None = None
    status: str
    pid: int | None
    started_at: datetime | None
    finished_at: datetime | None
    exit_code: int | None
    created_at: datetime
    tags: str
    extra_params: str
    gpu_request: int | None = None
    gpu_assigned: int | None = None

    model_config = {"from_attributes": True}


class EnvironmentInfo(BaseModel):
    """Information about a detected Python environment.

    Attributes:
        id: Unique environment identifier (e.g. "pixi:default").
        type: Environment type (pixi, conda, custom).
        name: Short name of the environment.
        display_name: Human-readable display label.
        python_path: Optional Python interpreter path.
    """

    id: str
    type: str
    name: str
    display_name: str
    python_path: str | None = None


class EnvHealthCheckRequest(BaseModel):
    """Request model for performing an environment health check.

    Attributes:
        env_id: The environment identifier to check.
        custom_python_path: An optional custom Python path override.
    """

    env_id: str
    custom_python_path: str | None = None


class EnvHealthResult(BaseModel):
    """Result of an environment health check.

    Attributes:
        env_id: The checked environment identifier.
        python_available: Whether Python is reachable in the environment.
        python_version: Python version string, or None.
        torch_available: Whether PyTorch is installed.
        torch_version: PyTorch version string, or None.
        error: Error message if the check failed, or None.
    """

    env_id: str
    python_available: bool
    python_version: str | None = None
    torch_available: bool
    torch_version: str | None = None
    error: str | None = None


class ParamField(BaseModel):
    """Metadata for a single CLI parameter.

    Attributes:
        type: Parameter type string (e.g. "str", "int", "float").
        default: Default value.
        help: Help text describing the parameter.
        required: Whether the parameter is required.
        choices: Allowed choices for the parameter, or None.
        short: Short flag alias, or None.
        nargs: Number of arguments consumed, or None.
    """

    type: str
    default: object = None
    help: str = ""
    required: bool = False
    choices: list | None = None
    short: str | None = None
    nargs: str | None = None
    optuna: dict | None = None


class ParamGroup(BaseModel):
    """A named group of related CLI parameters.

    Attributes:
        group_name: Display name for the parameter group.
        node: RunConfig node key the group routes to (general/compile/
            early_stopping/data/model), used by the backend to route params
            without importing torch. None when unknown.
        params: Mapping of parameter names to their metadata.
    """

    group_name: str
    node: str | None = None
    params: dict[str, ParamField]


class ModelSchemaResponse(BaseModel):
    """Response model for a complete model schema definition.

    Attributes:
        model_name: The model name.
        param_groups: List of parameter groups for the model.
    """

    model_name: str
    param_groups: list[ParamGroup]


class ExperimentInfo(BaseModel):
    """Summary information about an experiment.

    Attributes:
        name: Experiment name.
        path: Filesystem path to the experiment directory.
        model_name: Model used, or None.
        dataset_name: Dataset used, or None.
        timestamp: Experiment timestamp, or None.
        type: Experiment type label.
    """

    name: str
    path: str
    model_name: str | None = None
    dataset_name: str | None = None
    timestamp: str | None = None
    type: str


class ExperimentDetail(BaseModel):
    """Detailed information about an experiment.

    Attributes:
        name: Experiment name.
        path: Filesystem path to the experiment directory.
        files: List of files in the experiment directory.
        hyperparams: Dictionary of hyperparameters, or None.
    """

    name: str
    path: str
    files: list[str]
    hyperparams: dict | None = None


class GpuInfo(BaseModel):
    """Information about a single GPU device.

    Attributes:
        index: GPU device index.
        name: GPU model name.
        utilization_percent: GPU utilization percentage.
        memory_used_mb: Used GPU memory in megabytes.
        memory_total_mb: Total GPU memory in megabytes.
        temperature_c: GPU temperature in Celsius.
        power_usage_w: Power usage in watts.
        processes: List of running processes on this GPU.
    """

    index: int
    name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_usage_w: float
    processes: list[dict]


class GpuStatusResponse(BaseModel):
    """Response model for GPU status endpoint.

    Attributes:
        gpus: List of GPU information entries.
        updated_at: Timestamp of the status snapshot.
    """

    gpus: list[GpuInfo]
    updated_at: str


class SystemStatusResponse(BaseModel):
    """Response model for system status (CPU, memory, GPU).

    Attributes:
        cpu_percent: Overall CPU usage percentage.
        memory_used_gb: Used system memory in gigabytes.
        memory_total_gb: Total system memory in gigabytes.
        memory_percent: Memory usage percentage.
        gpu_utilization: Primary GPU utilization percentage.
        gpu_memory_percent: Primary GPU memory usage percentage.
        load_1m: System load average over 1 minute.
        load_5m: System load average over 5 minutes.
        load_15m: System load average over 15 minutes.
        updated_at: Timestamp of the snapshot.
    """

    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory_percent: float
    load_1m: float
    load_5m: float
    load_15m: float
    updated_at: str


class ResourceSnapshot(BaseModel):
    """Latest scalar gauges accompanying a resource history response.

    Attributes:
        cpu_percent: Overall CPU usage percentage.
        cpu_cores: Per-core usage percentages of the latest sample.
        load_1m: System load average over 1 minute.
        load_5m: System load average over 5 minutes.
        load_15m: System load average over 15 minutes.
        memory_used_gb: Used system memory in gigabytes.
        memory_total_gb: Total system memory in gigabytes.
        memory_percent: Memory usage percentage.
        swap_used_gb: Used swap space in gigabytes.
        swap_total_gb: Total swap space in gigabytes.
        swap_percent: Swap usage percentage.
    """

    cpu_percent: float
    cpu_cores: list[float]
    load_1m: float
    load_5m: float
    load_15m: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    swap_used_gb: float
    swap_total_gb: float
    swap_percent: float


class GpuHistorySeries(BaseModel):
    """Per-GPU utilization and memory percent series aligned to history timestamps.

    Attributes:
        index: GPU device index.
        name: GPU model name.
        utilization_percent: Utilization samples; null marks a missed query.
        memory_percent: Memory usage percent samples; null marks a missed query.
    """

    index: int
    name: str
    utilization_percent: list[float | None]
    memory_percent: list[float | None]


class ResourceHistoryResponse(BaseModel):
    """Column-oriented metric history with a shared epoch-millisecond axis.

    Attributes:
        timestamps: Epoch-millisecond timestamps, strictly increasing.
        cpu_percent: Overall CPU usage percentage per sample.
        memory_percent: RAM usage percentage per sample.
        swap_percent: Swap usage percentage per sample.
        net_up_bps: Network send rate in bytes per second.
        net_down_bps: Network receive rate in bytes per second.
        disk_read_bps: Disk read rate in bytes per second.
        disk_write_bps: Disk write rate in bytes per second.
        gpus: Per-GPU utilization and memory history series.
        snapshot: Latest scalar gauges (always the global latest, unfiltered).
        interval_seconds: Sampler interval in seconds.
    """

    timestamps: list[int]
    cpu_percent: list[float]
    memory_percent: list[float]
    swap_percent: list[float]
    net_up_bps: list[float]
    net_down_bps: list[float]
    disk_read_bps: list[float]
    disk_write_bps: list[float]
    gpus: list[GpuHistorySeries]
    snapshot: ResourceSnapshot
    interval_seconds: float


class SearchCreate(BaseModel):
    """Request model for creating a hyperparameter search task.

    Attributes:
        name: A human-readable name for the search.
        env_id: Identifier of the Python environment to use.
        custom_python_path: An optional custom Python interpreter path.
        model_name: The model to search over.
        dataset: The dataset to search on.
        gpu: Requested GPU index, or None for auto-assignment.
        runconfig_params: Flat RunConfig knobs (epochs/fold/batch_size/...) used
            as the base configuration for every trial.
        optuna_config: Optuna study knobs (metric/n_trials/sampler/pruner/...).
            ``metric`` (auc/acc/rmse/loss) selects the optimised objective.
    """

    name: str = ""
    env_id: str
    custom_python_path: str | None = None
    model_name: str
    dataset: str
    gpu: int | None = None
    runconfig_params: dict = {}
    optuna_config: dict = {}


class SearchTrialInfo(BaseModel):
    """One trial row in the search progress table."""

    number: int
    state: str
    value: float | None = None
    params: dict = {}
    datetime_start: str | None = None
    datetime_complete: str | None = None


class SearchStudyResponse(BaseModel):
    """Aggregated trial-progress summary read from ``study.db``."""

    total: int
    completed: int
    running: int
    pruned: int
    failed: int
    direction: str
    best_trial: dict | None = None
    trials: list[SearchTrialInfo] = []


class SearchStudyPathResponse(BaseModel):
    """The ``study.db`` location and a ready-to-copy optuna-dashboard command."""

    study_db_path: str | None = None
    dashboard_command: str | None = None
    exists: bool = False
