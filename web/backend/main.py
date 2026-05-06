from contextlib import asynccontextmanager

from database import init_db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import (
    datasets,
    environments,
    experiments,
    gpu,
    logs,
    preprocess,
    schemas_api,
    settings_api,
    tasks,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import dependencies as deps

    init_db()
    deps.process_manager = __import__(
        "services.process_manager", fromlist=["ProcessManager"]
    ).ProcessManager()
    deps.process_manager.recover_tasks()
    deps.gpu_monitor = __import__(
        "services.gpu_monitor", fromlist=["GpuMonitor"]
    ).GpuMonitor()
    deps.preprocess_manager = __import__(
        "services.preprocess_manager", fromlist=["PreprocessManager"]
    ).PreprocessManager()
    yield
    if deps.process_manager:
        deps.process_manager.shutdown()


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
app.include_router(preprocess.router)
app.include_router(datasets.router)
app.include_router(settings_api.router)
