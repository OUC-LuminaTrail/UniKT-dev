from fastapi import APIRouter
from schemas import EnvironmentInfo
from services.environment_resolver import EnvironmentResolver

router = APIRouter(prefix="/api/environments", tags=["environments"])


@router.get("", response_model=list[EnvironmentInfo])
def list_environments():
    return EnvironmentResolver().discover()
