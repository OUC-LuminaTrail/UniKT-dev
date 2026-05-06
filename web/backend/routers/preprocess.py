from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from dependencies import get_preprocess_manager
from services.log_watcher import LogWatcher
from services.preprocess_manager import PreprocessManager
from pydantic import BaseModel

router = APIRouter(prefix="/api/preprocess", tags=["preprocess"])


class PreprocessStartRequest(BaseModel):
    action: str
    dataset: str
    params: dict = {}


@router.post("", status_code=201)
def start_preprocess(body: PreprocessStartRequest, pm: PreprocessManager = Depends(get_preprocess_manager)):
    if body.action not in ("download", "process"):
        raise HTTPException(400, "action must be 'download' or 'process'")
    if not body.dataset:
        raise HTTPException(400, "dataset is required")
    task = pm.start(body.action, body.dataset, body.params)
    return {
        "id": task.id,
        "command": " ".join(task.command),
        "status": task.status,
        "started_at": task.started_at.isoformat(),
    }


@router.get("/{task_id}")
def get_preprocess(task_id: int, pm: PreprocessManager = Depends(get_preprocess_manager)):
    task = pm.get(task_id)
    if not task:
        raise HTTPException(404, "Preprocess task not found")
    return {
        "id": task.id,
        "command": " ".join(task.command),
        "status": task.status,
        "exit_code": task.exit_code,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
    }


@router.post("/{task_id}/stop")
def stop_preprocess(task_id: int, pm: PreprocessManager = Depends(get_preprocess_manager)):
    if not pm.stop(task_id):
        raise HTTPException(400, "Cannot stop task")
    return {"status": "stopping"}


@router.websocket("/api-preprocess/{task_id}/logs/stream")
async def stream_preprocess_logs(websocket: WebSocket, task_id: int, from_offset: int = Query(0), pm: PreprocessManager = Depends(get_preprocess_manager)):
    await websocket.accept()
    task = pm.get(task_id)
    if not task:
        await websocket.send_json({"type": "error", "content": "Task not found"})
        await websocket.close()
        return

    def check_alive():
        t = pm.get(task_id)
        return t is not None and t.status == "running"

    watcher = LogWatcher()
    try:
        await watcher.stream_log(
            task.log_path,
            websocket,
            check_alive=check_alive,
            from_offset=from_offset,
        )
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
