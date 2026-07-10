"""In-process event bus bridging sync worker threads to async SSE clients.

Sync code (the scheduler loop, request handlers, task_state transitions) calls
:func:`publish` from any thread; it wakes the running asyncio loop via
``call_soon_threadsafe`` and fans the event out to every registered async
subscriber queue. SSE connections consume those queues.
"""

import asyncio
import contextlib
import logging

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None
_listeners: set[asyncio.Queue] = set()


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Capture the running loop (called once from lifespan startup)."""
    global _loop
    _loop = loop


def publish(event: dict) -> None:
    """Publish an event to all subscribers (safe from any thread)."""
    loop = _loop
    if loop is None or not loop.is_running():
        return
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(_fanout, event)


def _fanout(event: dict) -> None:
    for q in list(_listeners):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("event listener queue full, dropping event")


def subscribe() -> asyncio.Queue:
    """Register an async queue that will receive published events."""
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _listeners.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """Remove a previously registered subscriber queue."""
    _listeners.discard(q)
