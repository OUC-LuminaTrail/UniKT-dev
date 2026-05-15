from datetime import datetime

from pydantic import BaseModel


class TaskCreate(BaseModel):
    name: str = ""
    env_id: str
    custom_python_path: str | None = None
    model_name: str
    params: dict


class TaskResponse(BaseModel):
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
    exp_dir: str
    started_at: datetime | None
    finished_at: datetime | None
    exit_code: int | None
    created_at: datetime
    tags: str
    extra_params: str

    model_config = {"from_attributes": True}


class EnvironmentInfo(BaseModel):
    id: str
    type: str
    name: str
    display_name: str
    python_path: str | None = None


class EnvHealthCheckRequest(BaseModel):
    env_id: str
    custom_python_path: str | None = None


class EnvHealthResult(BaseModel):
    env_id: str
    python_available: bool
    python_version: str | None = None
    torch_available: bool
    torch_version: str | None = None
    error: str | None = None


class ParamField(BaseModel):
    type: str
    default: object = None
    help: str = ""
    required: bool = False
    choices: list | None = None
    short: str | None = None
    nargs: str | None = None


class ParamGroup(BaseModel):
    group_name: str
    params: dict[str, ParamField]


class ModelSchemaResponse(BaseModel):
    model_name: str
    param_groups: list[ParamGroup]


class ExperimentInfo(BaseModel):
    name: str
    path: str
    model_name: str | None = None
    dataset_name: str | None = None
    timestamp: str | None = None
    type: str


class ExperimentDetail(BaseModel):
    name: str
    path: str
    files: list[str]
    hyperparams: dict | None = None


class GpuInfo(BaseModel):
    index: int
    name: str
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_usage_w: float
    processes: list[dict]


class GpuStatusResponse(BaseModel):
    gpus: list[GpuInfo]
    updated_at: str


class SystemStatusResponse(BaseModel):
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
