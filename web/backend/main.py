from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import LOG_DIR
from database import init_db
from dependencies import process_manager, gpu_monitor
from routers import tasks, logs, environments, schemas_api, experiments, gpu


@asynccontextmanager
async def lifespan(app: FastAPI):
    import dependencies as deps

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    deps.process_manager = __import__("services.process_manager", fromlist=["ProcessManager"]).ProcessManager()
    deps.process_manager.recover_tasks()
    deps.gpu_monitor = __import__("services.gpu_monitor", fromlist=["GpuMonitor"]).GpuMonitor()
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
