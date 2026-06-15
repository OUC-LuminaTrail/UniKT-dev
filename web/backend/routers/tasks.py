import json

from database import SessionLocal
from dependencies import get_process_manager
from fastapi import APIRouter, Depends, HTTPException
from models import Task
from pagination import Page, Params
from pydantic import BaseModel
from schemas import TaskCreate, TaskResponse
from services.process_manager import ProcessManager
from sqlalchemy import desc, select

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse, status_code=201)
def create_task(body: TaskCreate, pm: ProcessManager = Depends(get_process_manager)):
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
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

    pm.launch_task(
        task_id=task_id,
        model_name=body.model_name,
        params=body.params,
        env_id=body.env_id,
        custom_python_path=body.custom_python_path,
    )

    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        return task


@router.get("", response_model=Page[TaskResponse])
def list_tasks(
    status: str | None = None,
    params: Params = Depends(),
):
    from fastapi_pagination.ext.sqlalchemy import paginate

    with SessionLocal() as session:
        stmt = select(Task).order_by(
            Task.finished_at.is_(None).desc(),  # active tasks (null finished_at) first
            desc(Task.finished_at),             # then by completion time, most recent first
        )
        if status:
            stmt = stmt.where(Task.status == status)
        return paginate(session, stmt, params=params)


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: int):
    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        return task


@router.post("/{task_id}/stop")
def stop_task(task_id: int, pm: ProcessManager = Depends(get_process_manager)):
    if not pm.stop_task(task_id):
        raise HTTPException(400, "Cannot stop task")
    return {"status": "stopping"}


@router.post("/{task_id}/kill")
def kill_task(task_id: int, pm: ProcessManager = Depends(get_process_manager)):
    if not pm.kill_task(task_id):
        raise HTTPException(400, "Cannot kill task")
    return {"status": "killed"}


class ResizeRequest(BaseModel):
    cols: int
    rows: int


@router.post("/{task_id}/resize")
def resize_terminal(
    task_id: int, body: ResizeRequest, pm: ProcessManager = Depends(get_process_manager)
):
    pm.resize_pty(task_id, body.cols, body.rows)
    return {"ok": True}


@router.delete("/{task_id}")
def delete_task(task_id: int, pm: ProcessManager = Depends(get_process_manager)):
    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        if task.status == "running":
            raise HTTPException(400, "Cannot delete running task")
        if task.status == "pending":
            pm.remove_from_queue(task_id)
        session.delete(task)
        session.commit()
    return {"status": "deleted"}


@router.get("/queue/list")
def get_queue(pm: ProcessManager = Depends(get_process_manager)):
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
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tasks
        ]


class ReorderRequest(BaseModel):
    task_ids: list[int]


@router.put("/queue/reorder")
def reorder_queue(
    body: ReorderRequest, pm: ProcessManager = Depends(get_process_manager)
):
    pm.reorder_queue(body.task_ids)
    return {"ok": True}
