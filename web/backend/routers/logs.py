"""Logs router — task log retrieval and WebSocket streaming.

Provides a REST endpoint for fetching paginated log chunks by task ID and a
WebSocket endpoint for live log streaming with offset tracking.
"""

from database import SessionLocal
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from models import LogChunk, Task
from services.log_watcher import LogWatcher
from sqlalchemy import asc

from utils.core import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["logs"])


@router.get("/api/tasks/{task_id}/logs")
def get_logs(task_id: int, offset: int = 0, limit: int = Query(10000, ge=1, le=100000)):
    """Fetch paginated log content for a given task.

    Args:
        task_id: The task identifier.
        offset: Byte offset to start reading from.
        limit: Maximum number of log chunks to return (1-100000).

    Returns:
        A dict with ``content`` (decoded text) and ``total_lines`` (chunk count).

    Raises:
        HTTPException: 404 if the task does not exist.
    """
    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        rows = (
            session.query(LogChunk.raw_data, LogChunk.byte_offset)
            .filter(LogChunk.source == "task", LogChunk.source_id == task_id)
            .order_by(asc(LogChunk.byte_offset))
            .offset(offset)
            .limit(limit)
            .all()
        )
        if not rows:
            return {"content": "", "total_lines": 0}
        chunks = b"".join(row[0] for row in rows)
        try:
            text = chunks.decode("utf-8")
        except UnicodeDecodeError:
            text = chunks.decode("utf-8", errors="replace")
        total = (
            session.query(LogChunk)
            .filter(LogChunk.source == "task", LogChunk.source_id == task_id)
            .count()
        )
        return {"content": text, "total_lines": total}


@router.websocket("/api/tasks/{task_id}/logs/stream")
async def stream_logs(websocket: WebSocket, task_id: int, from_offset: int = Query(0)):
    """Stream task logs live over a WebSocket connection.

    Sends JSON messages of type ``data`` with decoded text and offset tracking,
    and a final ``done`` message when the stream ends.

    Args:
        websocket: The WebSocket connection.
        task_id: The task identifier.
        from_offset: Starting byte offset for the log stream.
    """
    await websocket.accept()
    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        if not task:
            await websocket.send_json({"type": "error", "content": "Task not found"})
            await websocket.close()
            return

    def check_alive():
        with SessionLocal() as session:
            t = session.query(Task).get(task_id)
            if not t:
                return False
            return t.pid is not None and t.status in ("running", "stopping")

    watcher = LogWatcher()
    try:
        await watcher.stream_log(
            "task",
            task_id,
            websocket,
            check_alive=check_alive,
            from_offset=from_offset,
        )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket stream error")
