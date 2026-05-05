from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import LOG_DIR
from database import init_db
from routers import tasks, logs, environments, schemas_api, experiments, gpu
from services.process_manager import ProcessManager
from services.gpu_monitor import GpuMonitor

process_manager: ProcessManager | None = None
gpu_monitor: GpuMonitor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global process_manager, gpu_monitor
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    process_manager = ProcessManager()
    process_manager.recover_tasks()
    gpu_monitor = GpuMonitor()
    yield
    if process_manager:
        process_manager.shutdown()


app = FastAPI(title="KT Experiment Manager", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router)
app.include_router(logs.router)
app.include_router(environments.router)
app.include_router(schemas_api.router)
app.include_router(experiments.router)
app.include_router(gpu.router)


def get_process_manager() -> ProcessManager:
    assert process_manager is not None
    return process_manager


def get_gpu_monitor() -> GpuMonitor:
    assert gpu_monitor is not None
    return gpu_monitor
