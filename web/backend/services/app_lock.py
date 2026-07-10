"""Single-instance startup lock.

Acquires an exclusive flock on ``data/app.lock`` at startup so a second backend
process (e.g. an accidental second ``web-backend`` or a ``--reload`` reloader
child) exits instead of fighting one database and two in-memory schedulers.
"""

import logging
import sys

from config import DATA_DIR

logger = logging.getLogger(__name__)

_LOCK_PATH = DATA_DIR / "app.lock"
_lock_file = None


def acquire() -> None:
    """Acquire the exclusive lock or exit if another instance holds it."""
    global _lock_file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import fcntl

    _lock_file = open(_LOCK_PATH, "w")  # noqa: SIM115 - held open to keep the flock
    try:
        fcntl.flock(_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.error("Another backend instance holds %s; exiting.", _LOCK_PATH)
        sys.exit(1)


def release() -> None:
    """Release the lock (idempotent)."""
    global _lock_file
    if _lock_file is not None:
        import contextlib

        with contextlib.suppress(Exception):
            import fcntl

            fcntl.flock(_lock_file.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            _lock_file.close()
        _lock_file = None
