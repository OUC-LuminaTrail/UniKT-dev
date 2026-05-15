from contextlib import asynccontextmanager

from database import init_db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_problem.handler import add_exception_handler, new_exception_handler
from middleware import MessageMiddleware
from routers import (
    datasets,
    environments,
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
    deps.settings_manager = __import__(
        "services.settings_manager", fromlist=["SettingsManager"]
    ).SettingsManager()
    yield
    if deps.process_manager:
        deps.process_manager.shutdown()


app = FastAPI(title="KT Experiment Manager", lifespan=lifespan)

eh = new_exception_handler()
add_exception_handler(app, eh)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Messages"],
)

app.add_middleware(MessageMiddleware)

app.include_router(tasks.router)
app.include_router(logs.router)
app.include_router(environments.router)
app.include_router(schemas_api.router)
app.include_router(gpu.router)
app.include_router(preprocess.router)
app.include_router(datasets.router)
app.include_router(settings_api.router)
