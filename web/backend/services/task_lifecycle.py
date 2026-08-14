"""Shared Task lifecycle handlers for the tasks and search routers.

Both routers operate on the same ``tasks`` table through the same
ProcessManager, so the stop/kill/delete bodies are factored out here to keep
them from drifting. The search router reuses them unchanged and only injects
its own artifact cleanup hook on delete.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

from database import SessionLocal
from errors import AppError
from models import Task

from services.line_render import LineRenderCache
from services.process_manager import ProcessManager


def stop_task_handler(pm: ProcessManager, task_id: int) -> None:
    """Request a graceful stop; raise AppError if the task cannot be stopped."""
    if not pm.stop_task(task_id):
        raise AppError("cannot_stop_task")


def kill_task_handler(pm: ProcessManager, task_id: int) -> None:
    """Force-kill; raise AppError if the task cannot be killed."""
    if not pm.kill_task(task_id):
        raise AppError("cannot_kill_task")


def delete_task_handler(
    pm: ProcessManager,
    cache: LineRenderCache,
    task_id: int,
    log_dir: Path,
    post_cleanup: Callable[[int], None] | None = None,
) -> dict:
    """Delete a task row, its log, and any router-specific artifacts.

    Args:
        pm: ProcessManager singleton (interrupted/pending cleanup).
        cache: LineRenderCache singleton (log eviction).
        task_id: The task identifier.
        log_dir: Directory holding the per-task ``{id}.log`` file.
        post_cleanup: Optional ``(task_id) -> None`` invoked after the row is
            deleted (the search router drops its persisted optuna YAML here).

    Returns:
        ``{"status": "deleted"}``.

    Raises:
        AppError: 404 if the task does not exist, 400 if still active.
    """
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        if not task:
            raise AppError("task_not_found", 404)
        if task.status in ("running", "stopping"):
            raise AppError("cannot_delete_active_task")
        if task.status == "interrupted":
            # Kill the orphan pid and drop the recover monitor before the row
            # goes away, so neither outlives the delete.
            pm.force_cleanup_interrupted(task_id)
        if task.status == "pending":
            pm.remove_from_queue(task_id)
        session.delete(task)
        session.commit()

    if post_cleanup is not None:
        post_cleanup(task_id)

    log_path = log_dir / f"{task_id}.log"
    if log_path.is_file():
        with contextlib.suppress(OSError):
            log_path.unlink()
    cache.evict(log_path)
    return {"status": "deleted"}
