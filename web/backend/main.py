from contextlib import asynccontextmanager
from pathlib import Path

from database import init_db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_pagination import add_pagination
from fastapi_pagination.api import set_page
from fastapi_problem.handler import add_exception_handler, new_exception_handler
from middleware import MessageMiddleware
from pagination import Page
from routers import (
    capabilities,
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
    deps.settings_manager = __import__(
        "services.settings_manager", fromlist=["SettingsManager"]
    ).SettingsManager()
    deps.python_env_manager = __import__(
        "services.python_env", fromlist=["PythonEnvManager"]
    ).PythonEnvManager(settings_manager=deps.settings_manager)
    deps.process_manager = __import__(
        "services.process_manager", fromlist=["ProcessManager"]
    ).ProcessManager(env_manager=deps.python_env_manager)
    deps.process_manager.recover_tasks()
    deps.gpu_monitor = __import__(
        "services.gpu_monitor", fromlist=["GpuMonitor"]
    ).GpuMonitor()
    deps.preprocess_manager = __import__(
        "services.preprocess_manager", fromlist=["PreprocessManager"]
    ).PreprocessManager(env_manager=deps.python_env_manager)
    yield
    if deps.preprocess_manager:
        deps.preprocess_manager.shutdown()
    if deps.process_manager:
        deps.process_manager.shutdown()
    if deps.gpu_monitor:
        deps.gpu_monitor.shutdown()


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

set_page(Page)
add_pagination(app)

app.include_router(tasks.router)
app.include_router(logs.router)
app.include_router(environments.router)
app.include_router(schemas_api.router)
app.include_router(gpu.router)
app.include_router(preprocess.router)
app.include_router(datasets.router)
app.include_router(settings_api.router)
app.include_router(capabilities.router)

dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")
