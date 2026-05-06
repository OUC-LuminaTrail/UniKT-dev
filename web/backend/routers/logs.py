from sqlalchemy import asc

from database import SessionLocal
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from models import LogChunk, Task
from services.log_watcher import LogWatcher

router = APIRouter(tags=["logs"])


@router.get("/api/tasks/{task_id}/logs")
def get_logs(task_id: int, offset: int = 0, limit: int = 10000):
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
        pass
