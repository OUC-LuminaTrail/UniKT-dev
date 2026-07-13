"""Tasks router — CRUD and queue management for experiment tasks.

Provides endpoints to create, list, get, stop, kill, delete, and resize tasks,
as well as managing the task execution queue (list and reorder).
"""

import contextlib
import json
import logging
from datetime import datetime

from config import TASK_LOGS_DIR
from database import SessionLocal
from dependencies import get_process_manager
from fastapi import APIRouter, Depends, HTTPException
from models import LogChunk, Task
from pagination import Page, Params
from pydantic import BaseModel
from schemas import TaskCreate, TaskResponse
from services.process_manager import ProcessManager
from services.task_state import transition
from sqlalchemy import desc, select

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

logger = logging.getLogger(__name__)


@router.post("", response_model=TaskResponse, status_code=201)
def create_task(body: TaskCreate, pm: ProcessManager = Depends(get_process_manager)):
    """Create a new experiment task and enqueue it for execution.

    Args:
        body: The task creation request.
        pm: Injected ProcessManager singleton.

    Returns:
        The created Task record.

    Raises:
        HTTPException: 500 if the task failed to launch.
    """
    with SessionLocal() as session:
        dataset_name = body.params.get("dataset", "")
        task = Task(
            name=body.name,
            command="",
            model_name=body.model_name,
            dataset_name=dataset_name,
            env_type="",
            env_name="",
            status="pending",
            tags="[]",
            extra_params=json.dumps(body.params),
            gpu_request=body.gpu,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

    try:
        pm.launch_task(
            task_id=task_id,
            model_name=body.model_name,
            params=body.params,
            env_id=body.env_id,
            custom_python_path=body.custom_python_path,
        )
    except Exception:
        logger.exception("Failed to launch task %s (%s)", task_id, body.model_name)
        with SessionLocal() as session:
            transition(
                session,
                Task,
                task_id,
                "pending",
                "failed",
                finished_at=datetime.now(),
            )
        raise HTTPException(500, "Failed to launch task")

    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        return task


@router.get("", response_model=Page[TaskResponse])
def list_tasks(
    status: str | None = None,
    params: Params = Depends(),
):
    """List tasks with optional status filter and pagination.

    Active tasks (null finished_at) appear first, then most recently
    finished tasks.

    Args:
        status: Optional status string filter.
        params: Pagination parameters.

    Returns:
        A paginated Page of TaskResponse items.
    """
    from fastapi_pagination.ext.sqlalchemy import paginate

    with SessionLocal() as session:
        stmt = select(Task).order_by(
            Task.finished_at.is_(None).desc(),
            desc(Task.finished_at),
        )
        if status:
            stmt = stmt.where(Task.status == status)
        return paginate(session, stmt, params=params)


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: int):
    """Return a single task by its ID.

    Args:
        task_id: The task identifier.

    Returns:
        The Task record.

    Raises:
        HTTPException: 404 if the task does not exist.
    """
    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        return task


@router.post("/{task_id}/stop")
def stop_task(task_id: int, pm: ProcessManager = Depends(get_process_manager)):
    """Request a graceful stop of a running task.

    Args:
        task_id: The task identifier.
        pm: Injected ProcessManager singleton.

    Returns:
        A dict with ``status`` set to ``stopping``.

    Raises:
        HTTPException: 400 if the task cannot be stopped.
    """
    if not pm.stop_task(task_id):
        raise HTTPException(400, "Cannot stop task")
    return {"status": "stopping"}


@router.post("/{task_id}/kill")
def kill_task(task_id: int, pm: ProcessManager = Depends(get_process_manager)):
    """Force-kill a running task.

    Args:
        task_id: The task identifier.
        pm: Injected ProcessManager singleton.

    Returns:
        A dict with ``status`` set to ``killed``.

    Raises:
        HTTPException: 400 if the task cannot be killed.
    """
    if not pm.kill_task(task_id):
        raise HTTPException(400, "Cannot kill task")
    return {"status": "killed"}


class ResizeRequest(BaseModel):
    """Request model for resizing a task terminal.

    Attributes:
        cols: Number of terminal columns.
        rows: Number of terminal rows.
    """

    cols: int
    rows: int


@router.post("/{task_id}/resize")
def resize_terminal(
    task_id: int, body: ResizeRequest, pm: ProcessManager = Depends(get_process_manager)
):
    """Resize the PTY terminal for a running task.

    Args:
        task_id: The task identifier.
        body: The resize dimensions.
        pm: Injected ProcessManager singleton.

    Returns:
        A dict with ``ok`` set to ``True``.
    """
    pm.resize_pty(task_id, body.cols, body.rows)
    return {"ok": True}


@router.delete("/{task_id}")
def delete_task(task_id: int, pm: ProcessManager = Depends(get_process_manager)):
    """Delete a task and its associated logs.

    Args:
        task_id: The task identifier.
        pm: Injected ProcessManager singleton.

    Returns:
        A dict with ``status`` set to ``deleted``.

    Raises:
        HTTPException: 404 if the task does not exist,
            400 if the task is still running.
    """
    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        if task.status == "running":
            raise HTTPException(400, "Cannot delete running task")
        if task.status == "pending":
            pm.remove_from_queue(task_id)
        session.query(LogChunk).filter_by(source="task", source_id=task_id).delete()
        session.delete(task)
        session.commit()

    log_path = TASK_LOGS_DIR / f"{task_id}.log"
    if log_path.is_file():
        with contextlib.suppress(OSError):
            log_path.unlink()
    return {"status": "deleted"}


@router.get("/queue/list")
def get_queue(pm: ProcessManager = Depends(get_process_manager)):
    """Return the ordered task execution queue with task details.

    Args:
        pm: Injected ProcessManager singleton.

    Returns:
        A list of task detail dicts in queue order.
    """
    task_ids = pm.get_queue()
    with SessionLocal() as session:
        tasks = (
            session.query(Task).filter(Task.id.in_(task_ids)).all() if task_ids else []
        )
        order = {tid: i for i, tid in enumerate(task_ids)}
        tasks.sort(key=lambda t: order.get(t.id, 999))
        return [
            {
                "id": t.id,
                "name": t.name,
                "model_name": t.model_name,
                "dataset_name": t.dataset_name,
                "env_name": t.env_name,
                "status": t.status,
                "gpu_request": t.gpu_request,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tasks
        ]


class ReorderRequest(BaseModel):
    """Request model for reordering the task queue.

    Attributes:
        task_ids: The desired order of task IDs in the queue.
    """

    task_ids: list[int]


@router.put("/queue/reorder")
def reorder_queue(
    body: ReorderRequest, pm: ProcessManager = Depends(get_process_manager)
):
    """Reorder the task execution queue.

    Args:
        body: The reorder request with the desired task ID order.
        pm: Injected ProcessManager singleton.

    Returns:
        A dict with ``ok`` set to ``True``.
    """
    pm.reorder_queue(body.task_ids)
    return {"ok": True}
