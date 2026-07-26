"""FastAPI application entry point for the KT Experiment Manager.

Configures the FastAPI app with CORS, error handling, pagination, and registers all API routers.
"""

import logging
from contextlib import asynccontextmanager

from database import init_db
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination
from fastapi_pagination.api import set_page
from fastapi_problem.handler import add_exception_handler, new_exception_handler
from middleware import MessageMiddleware
from pagination import Page
from routers import (
    capabilities,
    datasets,
    environments,
    events,
    gpu,
    logs,
    preprocess,
    registry,
    schemas_api,
    settings_api,
    tasks,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    Initializes the database, creates manager/dependency singletons, and
    recovers interrupted tasks on startup. Shuts down all managers on exit.
    """
    import asyncio

    import dependencies as deps
    from services import app_lock, event_bus

    app_lock.acquire()
    try:
        init_db()
        event_bus.set_loop(asyncio.get_running_loop())

        deps.settings_manager = __import__(
            "services.settings_manager", fromlist=["SettingsManager"]
        ).SettingsManager()
        deps.python_env_manager = __import__(
            "services.python_env", fromlist=["PythonEnvManager"]
        ).PythonEnvManager(settings_manager=deps.settings_manager)
        deps.gpu_monitor = __import__(
            "services.gpu_monitor", fromlist=["GpuMonitor"]
        ).GpuMonitor()
        deps.schema_extractor = __import__(
            "services.schema_extractor", fromlist=["SchemaExtractor"]
        ).SchemaExtractor(env_manager=deps.python_env_manager)
        deps.process_manager = __import__(
            "services.process_manager", fromlist=["ProcessManager"]
        ).ProcessManager(
            env_manager=deps.python_env_manager,
            gpu_monitor=deps.gpu_monitor,
            schema_extractor=deps.schema_extractor,
        )
        deps.process_manager.gpu_slots = deps.settings_manager.get_gpu_slots()
        deps.process_manager.recover_tasks()
        deps.preprocess_manager = __import__(
            "services.preprocess_manager", fromlist=["PreprocessManager"]
        ).PreprocessManager(
            env_manager=deps.python_env_manager,
            schema_extractor=deps.schema_extractor,
        )
        deps.preprocess_manager.recover_tasks()
        yield
    finally:
        # Must run even when the lifespan task is cancelled or the server exits
        # with an exception, or running subprocesses keep their GPUs forever.
        for manager in (
            deps.preprocess_manager,
            deps.process_manager,
            deps.gpu_monitor,
        ):
            if manager:
                try:
                    manager.shutdown()
                except Exception:
                    logger.exception("shutdown failed for %s", type(manager).__name__)
        app_lock.release()


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
app.include_router(events.router)
app.include_router(environments.router)
app.include_router(schemas_api.router)
app.include_router(registry.router)
app.include_router(gpu.router)
app.include_router(preprocess.router)
app.include_router(datasets.router)
app.include_router(settings_api.router)
app.include_router(capabilities.router)
