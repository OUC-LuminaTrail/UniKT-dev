"""Shared pid identity check for orphan recovery.

Both managers reattach to live orphan processes by pid after a restart, and a
pid can have been recycled by an unrelated process in the meantime.
"""

from datetime import datetime

import psutil


def pid_reused(proc: psutil.Process, started_at: datetime | None) -> bool:
    """Return True when a live pid's process started long after the task did."""
    try:
        if started_at and proc.create_time() > started_at.timestamp() + 60:
            return True
    except (psutil.NoSuchProcess, OSError):
        return True
    return False
