"""Preprocess router — data download/processing tasks.

Provides CRUD endpoints for launching and managing preprocess tasks (download
or process actions), including WebSocket log streaming and PTY resize.
"""

from dependencies import get_preprocess_manager
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel
from services.log_watcher import LogWatcher
from services.preprocess_manager import PreprocessManager

from utils.core import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/preprocess", tags=["preprocess"])


class PreprocessStartRequest(BaseModel):
    """Request model for starting a preprocess task.

    Attributes:
        action: The action to perform (``download`` or ``process``).
        dataset: Name of the dataset to preprocess.
        params: Additional parameters for the action.
        env_id: Optional environment identifier.
        custom_python_path: Optional custom Python interpreter path.
    """

    action: str
    dataset: str
    params: dict = {}
    env_id: str | None = None
    custom_python_path: str | None = None


class ResizeRequest(BaseModel):
    """Request model for resizing a PTY terminal.

    Attributes:
        cols: Number of terminal columns.
        rows: Number of terminal rows.
    """

    cols: int
    rows: int


@router.post("", status_code=201)
def start_preprocess(
    body: PreprocessStartRequest,
    pm: PreprocessManager = Depends(get_preprocess_manager),
):
    """Start a new preprocess task (download or process).

    Args:
        body: The preprocess start request.
        pm: Injected PreprocessManager singleton.

    Returns:
        A dict with the task ``id``, ``command``, ``status``, and ``started_at``.

    Raises:
        HTTPException: 400 if the action or dataset is invalid.
    """
    if body.action not in ("download", "process"):
        raise HTTPException(400, "action must be 'download' or 'process'")
    if not body.dataset:
        raise HTTPException(400, "dataset is required")
    task = pm.start(
        body.action, body.dataset, body.params, body.env_id, body.custom_python_path
    )
    return {
        "id": task.id,
        "command": " ".join(task.command),
        "status": task.status,
        "started_at": task.started_at.isoformat(),
    }


@router.get("")
def list_preprocess(pm: PreprocessManager = Depends(get_preprocess_manager)):
    """List all preprocess tasks.

    Args:
        pm: Injected PreprocessManager singleton.

    Returns:
        A list of task dicts with ``id``, ``command``, ``status``, etc.
    """
    tasks = []
    for t in pm.list_all():
        tasks.append(
            {
                "id": t.id,
                "command": " ".join(t.command),
                "status": t.status,
                "exit_code": t.exit_code,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "finished_at": t.finished_at.isoformat() if t.finished_at else None,
            }
        )
    return tasks


@router.get("/{task_id}")
def get_preprocess(
    task_id: int, pm: PreprocessManager = Depends(get_preprocess_manager)
):
    """Return details for a specific preprocess task.

    Args:
        task_id: The preprocess task identifier.
        pm: Injected PreprocessManager singleton.

    Returns:
        A task dict with ``id``, ``command``, ``status``, etc.

    Raises:
        HTTPException: 404 if the task does not exist.
    """
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
def stop_preprocess(
    task_id: int, pm: PreprocessManager = Depends(get_preprocess_manager)
):
    """Stop a running preprocess task.

    Args:
        task_id: The preprocess task identifier.
        pm: Injected PreprocessManager singleton.

    Returns:
        A dict with ``status`` set to ``stopping``.

    Raises:
        HTTPException: 400 if the task cannot be stopped.
    """
    if not pm.stop(task_id):
        raise HTTPException(400, "Cannot stop task")
    return {"status": "stopping"}


@router.delete("/{task_id}")
def delete_preprocess(
    task_id: int,
    pm: PreprocessManager = Depends(get_preprocess_manager),
):
    """Delete a preprocess task and its log chunks.

    Args:
        task_id: The preprocess task identifier.
        pm: Injected PreprocessManager singleton.

    Returns:
        A dict with ``status`` set to ``deleted``.

    Raises:
        HTTPException: 400 if the task cannot be deleted.
    """
    if not pm.delete(task_id):
        raise HTTPException(400, "Cannot delete preprocess task")
    return {"status": "deleted"}


@router.post("/{task_id}/resize")
def resize_preprocess(
    task_id: int,
    body: ResizeRequest,
    pm: PreprocessManager = Depends(get_preprocess_manager),
):
    """Resize the PTY terminal for a running preprocess task.

    Args:
        task_id: The preprocess task identifier.
        body: The resize request (cols, rows).
        pm: Injected PreprocessManager singleton.

    Returns:
        A dict with ``ok`` set to ``True``.
    """
    pm.resize_pty(task_id, body.cols, body.rows)
    return {"ok": True}


@router.websocket("/{task_id}/logs/stream")
async def stream_preprocess_logs(
    websocket: WebSocket,
    task_id: int,
    from_offset: int = Query(0),
    pm: PreprocessManager = Depends(get_preprocess_manager),
):
    """Stream preprocess task logs live over a WebSocket connection.

    Args:
        websocket: The WebSocket connection.
        task_id: The preprocess task identifier.
        from_offset: Starting byte offset for the log stream.
        pm: Injected PreprocessManager singleton.
    """
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
            "preprocess",
            task_id,
            websocket,
            check_alive=check_alive,
            from_offset=from_offset,
        )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Preprocess WebSocket stream error")
