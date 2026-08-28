"""Preprocess task manager — data download/processing lifecycle.

Manages PTY-backed subprocesses for dataset download and processing. Task state
is persisted in the ``preprocess_tasks`` table (so it survives restarts) and
output is appended to per-task ``.log`` files. Every status write goes through
the CAS state machine.
"""

import contextlib
import fcntl
import json
import logging
import os
import pty
import signal
import struct
import subprocess
import termios
import threading
from dataclasses import dataclass
from datetime import datetime

import psutil
from config import PREPROCESS_LOGS_DIR, PROJECT_ROOT
from database import SessionLocal
from models import PreprocessTask

from services.cli_builder import build_param_flags
from services.line_render import LineRenderCache
from services.pid_utils import pid_reused
from services.python_env import PythonEnvManager
from services.schema_extractor import SchemaExtractor
from services.task_state import transition

logger = logging.getLogger(__name__)


@dataclass
class PreprocessTaskInfo:
    """Detached snapshot of a preprocess task returned to callers."""

    id: int
    command: str
    status: str
    exit_code: int | None
    started_at: datetime | None
    finished_at: datetime | None


def _snapshot(row: PreprocessTask) -> PreprocessTaskInfo:
    return PreprocessTaskInfo(
        id=row.id,
        command=row.command,
        status=row.status,
        exit_code=row.exit_code,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


class PreprocessManager:
    """Manages lifecycle of preprocess subprocesses."""

    def __init__(
        self,
        env_manager: PythonEnvManager,
        schema_extractor: SchemaExtractor,
        line_cache: LineRenderCache,
    ):
        """Initialize the preprocess manager."""
        self._env_manager = env_manager
        self._schema_extractor = schema_extractor
        self._line_cache = line_cache
        self._procs: dict[int, subprocess.Popen] = {}
        self._master_fds: dict[int, int] = {}
        self._readers: dict[int, threading.Thread] = {}
        self._monitors: dict[int, threading.Thread] = {}
        self._lock = threading.RLock()

    def start(
        self,
        action: str,
        dataset: str,
        params: dict,
        env_id: str | None = None,
        custom_python_path: str | None = None,
    ) -> PreprocessTaskInfo:
        """Create a preprocess task row, spawn its subprocess, and return a snapshot."""
        # env_id wins; when unset, resolve_command falls back to the
        # wizard-configured default env (or raises EnvironmentNotConfigured).
        base = self._env_manager.resolve_command(env_id, custom_python_path)
        command = base + self._build_command(action, dataset, params)

        with SessionLocal() as session:
            row = PreprocessTask(
                action=action,
                dataset=dataset,
                command=" ".join(command),
                env_type=(env_id.split(":", 1)[0] if env_id else ""),
                env_name=(env_id.split(":", 1)[1] if env_id else ""),
                python_path=custom_python_path or "",
                status="running",
                started_at=datetime.now(),
                params=json.dumps(params),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            task_id = row.id

        launched = self._spawn(task_id, command)
        with SessionLocal() as session:
            row = session.get(PreprocessTask, task_id)
            if not launched and row and row.status == "running":
                transition(
                    session,
                    PreprocessTask,
                    task_id,
                    "running",
                    "failed",
                    finished_at=datetime.now(),
                )
                row = session.get(PreprocessTask, task_id)
            return (
                _snapshot(row)
                if row
                else PreprocessTaskInfo(
                    id=task_id,
                    command=" ".join(command),
                    status="failed",
                    exit_code=None,
                    started_at=None,
                    finished_at=None,
                )
            )

    def _spawn(self, task_id: int, command: list[str]) -> bool:
        try:
            master_fd, slave_fd = pty.openpty()
        except OSError:
            logger.exception("preprocess openpty failed for task %s", task_id)
            return False

        try:
            # Matches the pyte emulator columns so rich wraps exactly as
            # rendered downstream; the frontend no longer resizes the PTY.
            winsize = struct.pack("HHHH", 24, 80, 0, 0)
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
            env = os.environ.copy()
            env["TERM"] = "xterm-256color"
            env["FORCE_COLOR"] = "1"
            proc = subprocess.Popen(
                command,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
                env=env,
            )
            os.close(slave_fd)
        except Exception:
            logger.exception("preprocess spawn failed for task %s", task_id)
            with contextlib.suppress(OSError):
                os.close(slave_fd)
            with contextlib.suppress(OSError):
                os.close(master_fd)
            return False

        # Register the handle before anything can observe the pid, so a
        # concurrent stop() either sees the handle (and terminates cleanly) or
        # ran before the pid write — and the status check below then reaps the
        # fresh process instead of orphaning it behind a "stopped" row.
        with self._lock:
            self._procs[task_id] = proc
            self._master_fds[task_id] = master_fd

        try:
            with SessionLocal() as session:
                row = session.get(PreprocessTask, task_id)
                lost_race = row is not None and row.status != "running"
                if row is not None and not lost_race:
                    row.pid = proc.pid
                    session.commit()
        except Exception:
            # The registered handle must not outlive a failed pid write:
            # without a monitor the process would hang on a full PTY buffer
            # until the backend restarts. Kill and unregister instead.
            logger.exception("pid write failed for preprocess %s; killing", task_id)
            self._kill_group(proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=3)
            with self._lock:
                self._procs.pop(task_id, None)
                fd = self._master_fds.pop(task_id, None)
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            return False

        if lost_race:
            # stop()/kill() won the race while we were spawning and owns the
            # row's final state; kill the fresh process group and unregister.
            logger.info(
                "preprocess %s spawn lost the stop race (status in DB); killing",
                task_id,
            )
            self._kill_group(proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=3)
            with self._lock:
                self._procs.pop(task_id, None)
                fd = self._master_fds.pop(task_id, None)
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            return False

        reader = threading.Thread(
            target=self._read_pty, args=(task_id, master_fd), daemon=True
        )
        monitor = threading.Thread(target=self._monitor, args=(task_id,), daemon=True)
        # Register before start: an instantly-exiting process would otherwise
        # have _monitor's _cleanup run against not-yet-registered threads.
        with self._lock:
            self._readers[task_id] = reader
            self._monitors[task_id] = monitor
        reader.start()
        monitor.start()
        return True

    def _read_pty(self, task_id: int, master_fd: int) -> None:
        path = PREPROCESS_LOGS_DIR / f"{task_id}.log"
        try:
            with open(path, "ab") as f:
                while True:
                    try:
                        data = os.read(master_fd, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    try:
                        f.write(data)
                        f.flush()
                        self._line_cache.feed(path)
                    except OSError:
                        logger.warning("log write failed for preprocess %s", task_id)
                        break
        except Exception:
            # Don't transition — closing master_fd below SIGPIPEs the subprocess
            # and _monitor records the real exit code via the normal path.
            logger.exception("reader thread fatal error for preprocess %s", task_id)
        # Close only if we still own the registration: _cleanup may already
        # have popped (and closed) the fd, and a second blind close could hit
        # an fd number since reused by another task's openpty.
        with self._lock:
            fd = self._master_fds.pop(task_id, None)
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)

    def _monitor(self, task_id: int) -> None:
        with self._lock:
            proc = self._procs.get(task_id)
        if proc is None:
            return
        exit_code = -1
        try:
            proc.wait()
            exit_code = proc.returncode
        except OSError:
            pass

        self._cleanup(task_id)
        with SessionLocal() as session:
            to = "completed" if exit_code == 0 else "failed"
            transition(
                session,
                PreprocessTask,
                task_id,
                "running",
                to,
                exit_code=exit_code,
                finished_at=datetime.now(),
                pid=None,
            )

    def _cleanup(self, task_id: int) -> None:
        # Join the reader before closing master_fd so the PTY kernel buffer
        # drains fully and the final log bytes are not lost.
        reader = self._readers.pop(task_id, None)
        if reader and reader.is_alive():
            reader.join(timeout=5)
        master_fd = self._master_fds.pop(task_id, None)
        if master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(master_fd)
        with self._lock:
            self._procs.pop(task_id, None)
            self._monitors.pop(task_id, None)

    def _terminate(self, task_id: int) -> bool:
        with self._lock:
            proc = self._procs.get(task_id)
        if proc is None:
            return False
        pid = proc.pid
        with contextlib.suppress(Exception):
            self._kill_group(pid, signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                self._kill_group(pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=3)
        self._cleanup(task_id)
        return True

    def _kill_group(self, pid: int, sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(pid, sig)

    def get(self, task_id: int) -> PreprocessTaskInfo | None:
        """Return a snapshot of a preprocess task, or None if missing."""
        with SessionLocal() as session:
            row = session.get(PreprocessTask, task_id)
            return _snapshot(row) if row else None

    def list_all(self) -> list[PreprocessTaskInfo]:
        """Return snapshots of all preprocess tasks, newest first."""
        with SessionLocal() as session:
            rows = (
                session.query(PreprocessTask).order_by(PreprocessTask.id.desc()).all()
            )
            return [_snapshot(r) for r in rows]

    def stop(self, task_id: int) -> bool:
        """Gracefully stop a running preprocess task via SIGINT."""
        with SessionLocal() as session:
            row = session.get(PreprocessTask, task_id)
            status = row.status if row else None
        if status != "running":
            return False
        with SessionLocal() as session:
            claimed = transition(
                session, PreprocessTask, task_id, "running", "stopping"
            )
        if not claimed:
            return True
        if not self._terminate(task_id):
            # Spawn raced us (handle not yet registered): fall back to the row
            # pid so the subprocess cannot outlive a "stopped" row.
            with SessionLocal() as session:
                row = session.get(PreprocessTask, task_id)
                fallback_pid = row.pid if row else None
            if fallback_pid:
                self._kill_group(fallback_pid, signal.SIGKILL)
        with SessionLocal() as session:
            transition(
                session,
                PreprocessTask,
                task_id,
                "stopping",
                "stopped",
                finished_at=datetime.now(),
                pid=None,
            )
        return True

    def delete(self, task_id: int) -> bool:
        """Delete a finished preprocess task and its log file."""
        with SessionLocal() as session:
            row = session.get(PreprocessTask, task_id)
            if not row:
                return False
            if row.status in ("running", "stopping"):
                return False
            # An interrupted row may still own a live orphan process and a
            # recover-monitor thread; both must be reclaimed with the row.
            orphan = (row.pid, row.started_at) if row.status == "interrupted" else None
            session.delete(row)
            session.commit()
        if orphan and orphan[0]:
            pid, started_at = orphan
            try:
                proc = psutil.Process(pid)
                reused = pid_reused(proc, started_at)
            except psutil.Error:
                reused = True
            if not reused:
                self._kill_group(pid, signal.SIGKILL)
        with self._lock:
            self._monitors.pop(task_id, None)

        log_path = PREPROCESS_LOGS_DIR / f"{task_id}.log"
        if log_path.is_file():
            with contextlib.suppress(OSError):
                log_path.unlink()
        self._line_cache.evict(log_path)
        return True

    def recover_tasks(self) -> None:
        """Reattach live orphan preprocess processes at startup."""
        with SessionLocal() as session:
            running = (
                session.query(
                    PreprocessTask.id,
                    PreprocessTask.pid,
                    PreprocessTask.status,
                    PreprocessTask.started_at,
                )
                .filter(
                    PreprocessTask.status.in_(["running", "stopping", "interrupted"])
                )
                .all()
            )
            for task_id, pid, prior_status, started_at in running:
                if pid and psutil.pid_exists(pid):
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running() and not pid_reused(proc, started_at):
                            if prior_status != "interrupted":
                                transition(
                                    session,
                                    PreprocessTask,
                                    task_id,
                                    prior_status,
                                    "interrupted",
                                )
                            t = threading.Thread(
                                target=self._recover_monitor,
                                args=(task_id, pid),
                                daemon=True,
                            )
                            t.start()
                            with self._lock:
                                self._monitors[task_id] = t
                            continue
                    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                        pass
                transition(
                    session,
                    PreprocessTask,
                    task_id,
                    prior_status,
                    "failed",
                    finished_at=datetime.now(),
                    pid=None,
                )

    def _recover_monitor(self, task_id: int, pid: int) -> None:
        exit_code = -1
        try:
            proc = psutil.Process(pid)
            # psutil.Process has no returncode attribute; wait() returns the
            # exit status, or None for non-child pids (recovered orphans are
            # never children). None keeps the unknown outcome honest; the
            # status below still resolves conservatively to failed.
            exit_code = proc.wait()
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            OSError,
        ):
            pass
        with SessionLocal() as session:
            row = session.get(PreprocessTask, task_id)
            if row and row.status in ("running", "interrupted"):
                to = "completed" if exit_code == 0 else "failed"
                transition(
                    session,
                    PreprocessTask,
                    task_id,
                    row.status,
                    to,
                    exit_code=exit_code,
                    finished_at=datetime.now(),
                    pid=None,
                )
        with self._lock:
            self._monitors.pop(task_id, None)

    def _build_command(
        self,
        action: str,
        dataset: str,
        params: dict,
    ) -> list[str]:
        # Env prefix is NOT included here — preview shows just data_process.py
        # (mirrors train.py in the task-launch preview); start() prepends the
        # resolved env command for the actual subprocess.
        cmd = ["data_process.py", action, "-d", dataset]
        # Routes/defaults come from the same schema reflection that drives the
        # frontend form; allow_flat_node yields --force/--extra (node "").
        routes = self._schema_extractor.get_preprocess_field_routes(action)
        defaults = self._schema_extractor.get_preprocess_field_defaults(action)
        cmd.extend(build_param_flags(params, routes, defaults, allow_flat_node=True))
        return cmd

    def preview_command(
        self,
        action: str,
        dataset: str,
        params: dict,
    ) -> str:
        """Build the command string without launching (for UI preview).

        Excludes the env prefix, mirroring ProcessManager.preview_command
        (which returns ``train.py ...`` without the pixi/conda prefix).
        """
        return " ".join(self._build_command(action, dataset, params))

    def shutdown(self) -> None:
        """Terminate all running preprocess subprocesses."""
        for task_id in list(self._procs):
            self._terminate(task_id)
