from fastapi import APIRouter, Depends

from dependencies import get_process_manager
from services.process_manager import ProcessManager
from pydantic import BaseModel

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsResponse(BaseModel):
    max_concurrent: int


class SettingsUpdate(BaseModel):
    max_concurrent: int


@router.get("")
def get_settings(pm: ProcessManager = Depends(get_process_manager)):
    return SettingsResponse(max_concurrent=pm.max_concurrent)


@router.put("")
def update_settings(body: SettingsUpdate, pm: ProcessManager = Depends(get_process_manager)):
    pm.max_concurrent = body.max_concurrent
    return SettingsResponse(max_concurrent=pm.max_concurrent)
