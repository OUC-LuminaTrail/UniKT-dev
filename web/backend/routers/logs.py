from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from database import SessionLocal
from models import Task
from services.log_watcher import LogWatcher
from main import get_process_manager

router = APIRouter(tags=["logs"])


@router.get("/api/tasks/{task_id}/logs")
def get_logs(task_id: int, offset: int = 0, limit: int = 10000):
    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        log_path = Path(task.log_file_path)
        if not log_path.exists():
            return {"content": "", "total_lines": 0}
        with open(log_path, "r") as f:
            lines = f.readlines()
        total = len(lines)
        selected = lines[offset : offset + limit]
        return {"content": "".join(selected), "total_lines": total}


@router.websocket("/api/tasks/{task_id}/logs/stream")
async def stream_logs(websocket: WebSocket, task_id: int):
    await websocket.accept()
    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        if not task:
            await websocket.send_json({"type": "error", "content": "Task not found"})
            await websocket.close()
            return
        log_path = task.log_file_path
        task_pid = task.pid

    def is_alive():
        return task_pid is not None

    watcher = LogWatcher()
    try:
        await watcher.stream_log(log_path, websocket, check_alive=is_alive)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
