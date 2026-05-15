from dependencies import get_process_manager, get_settings_manager
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from services.process_manager import ProcessManager
from services.settings_manager import SettingsManager

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsResponse(BaseModel):
    max_concurrent: int


class SettingsUpdate(BaseModel):
    max_concurrent: int


@router.get("")
def get_settings(pm: ProcessManager = Depends(get_process_manager)):
    return SettingsResponse(max_concurrent=pm.max_concurrent)


@router.put("")
def update_settings(
    body: SettingsUpdate, pm: ProcessManager = Depends(get_process_manager)
):
    pm.max_concurrent = body.max_concurrent
    return SettingsResponse(max_concurrent=pm.max_concurrent)


class DefaultEnvUpdate(BaseModel):
    env_id: str


@router.get("/default-env")
def get_default_env(sm: SettingsManager = Depends(get_settings_manager)):
    return {"default_env_id": sm.get_default_env()}


@router.post("/default-env")
def set_default_env(
    body: DefaultEnvUpdate, sm: SettingsManager = Depends(get_settings_manager)
):
    sm.set_default_env(body.env_id)
    return {"ok": True, "default_env_id": body.env_id}


@router.get("/initialized")
def check_initialized(sm: SettingsManager = Depends(get_settings_manager)):
    return {"initialized": sm.is_setup_completed()}
