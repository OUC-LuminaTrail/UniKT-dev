from dependencies import get_python_env_manager
from fastapi import APIRouter, Depends
from schemas import EnvHealthCheckRequest
from services.python_env import PythonEnvManager

router = APIRouter(prefix="/api/environments", tags=["environments"])


@router.get("")
def list_environments(mgr: PythonEnvManager = Depends(get_python_env_manager)):
    return mgr.discover()


@router.post("/health-check")
async def health_check(
    body: EnvHealthCheckRequest,
    mgr: PythonEnvManager = Depends(get_python_env_manager),
):
    return await mgr.health_check(body.env_id, body.custom_python_path)
