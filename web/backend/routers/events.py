"""Events router — SSE stream of task/preprocess status changes.

On connect, sends a snapshot of every task and preprocess task status, then
streams incremental status events published via the event bus.
"""

import asyncio
import json
import logging

from database import SessionLocal
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models import PreprocessTask, Task
from services import event_bus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["events"])

KEEPALIVE_SECONDS = 15.0


@router.get("/api/events")
async def events():
    """Server-sent events stream of status changes."""

    async def gen():
        # Subscribe before snapshotting: events published in between are queued
        # rather than lost. Duplicate snapshots are harmless — clients overwrite
        # by id; a missed terminal status is not.
        q = event_bus.subscribe()
        try:
            with SessionLocal() as session:
                snapshot = [
                    {
                        "type": "task_status",
                        "id": t.id,
                        "status": t.status,
                        "pid": t.pid,
                    }
                    for t in session.query(Task).all()
                ] + [
                    {"type": "preprocess_status", "id": p.id, "status": p.status}
                    for p in session.query(PreprocessTask).all()
                ]
            for event in snapshot:
                yield f"data: {json.dumps(event)}\n\n"

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            event_bus.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")
