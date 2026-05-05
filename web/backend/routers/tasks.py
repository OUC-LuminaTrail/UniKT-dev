import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc

from database import SessionLocal
from main import get_process_manager
from models import Task
from schemas import TaskCreate, TaskResponse
from services.process_manager import ProcessManager
from config import LOG_DIR

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
            log_file_path=str(LOG_DIR / f"task_{0}.log"),
            tags="[]",
            extra_params=json.dumps(body.params),
        )
        session.add(task)
        session.flush()
        task.log_file_path = str(LOG_DIR / f"task_{task.id}.log")
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


@router.get("", response_model=list[TaskResponse])
def list_tasks(
    status: str | None = None,
    page: int = 1,
    page_size: int = 20,
):
    with SessionLocal() as session:
        q = session.query(Task).order_by(desc(Task.created_at))
        if status:
            q = q.filter(Task.status == status)
        tasks = q.offset((page - 1) * page_size).limit(page_size).all()
        return tasks


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


@router.delete("/{task_id}")
def delete_task(task_id: int):
    with SessionLocal() as session:
        task = session.query(Task).get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        if task.status == "running":
            raise HTTPException(400, "Cannot delete running task")
        session.delete(task)
        session.commit()
    return {"status": "deleted"}
