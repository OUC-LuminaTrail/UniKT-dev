"""Logs router — task log retrieval and WebSocket streaming (file-backed)."""

import logging

from config import TASK_LOGS_DIR
from database import SessionLocal
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from models import Task
from services.log_reader import read_log_text, stream_log

logger = logging.getLogger(__name__)

router = APIRouter(tags=["logs"])


@router.get("/api/tasks/{task_id}/logs")
def get_logs(
    task_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    """Fetch log content for a task from its log file.

    Args:
        task_id: The task identifier.
        offset: Byte offset to start reading from.
        limit: Maximum number of 64KB chunks to return (1-500).

    Returns:
        A dict with ``content`` (decoded text) and ``total_bytes``.

    Raises:
        HTTPException: 404 if the task does not exist.
    """
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        if not task:
            raise HTTPException(404, "Task not found")
    return read_log_text(TASK_LOGS_DIR / f"{task_id}.log", offset, limit)


@router.websocket("/api/tasks/{task_id}/logs/stream")
async def stream_logs(
    websocket: WebSocket, task_id: int, from_offset: int = Query(0, ge=0)
):
    """Stream task logs live over a WebSocket from the task's log file.

    Args:
        websocket: The WebSocket connection.
        task_id: The task identifier.
        from_offset: Starting byte offset for the log stream.
    """
    await websocket.accept()
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        if not task:
            await websocket.send_json({"type": "error", "content": "Task not found"})
            await websocket.close()
            return

    def check_alive():
        with SessionLocal() as session:
            t = session.get(Task, task_id)
            if not t:
                return False
            # A queued task has no pid yet but its output is still to come;
            # reporting it dead closes the stream before it ever starts.
            if t.status == "pending":
                return True
            return t.pid is not None and t.status in ("running", "stopping")

    try:
        await stream_log(
            TASK_LOGS_DIR / f"{task_id}.log",
            websocket,
            check_alive=check_alive,
            from_offset=from_offset,
        )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket stream error")
