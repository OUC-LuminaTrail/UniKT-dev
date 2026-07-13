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
from models import LogChunk, PreprocessTask

from services.python_env import PythonEnvManager
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

    def __init__(self, env_manager: PythonEnvManager):
        """Initialize the preprocess manager."""
        self._env_manager = env_manager
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
        command = self._build_command(
            action, dataset, params, env_id, custom_python_path
        )

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
            row = session.query(PreprocessTask).get(task_id)
            if not launched and row and row.status == "running":
                transition(
                    session,
                    PreprocessTask,
                    task_id,
                    "running",
                    "failed",
                    finished_at=datetime.now(),
                )
                row = session.query(PreprocessTask).get(task_id)
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
            return False

        with SessionLocal() as session:
            row = session.query(PreprocessTask).get(task_id)
            if row:
                row.pid = proc.pid
                session.commit()

        with self._lock:
            self._procs[task_id] = proc
            self._master_fds[task_id] = master_fd

        reader = threading.Thread(
            target=self._read_pty, args=(task_id, master_fd), daemon=True
        )
        reader.start()
        monitor = threading.Thread(target=self._monitor, args=(task_id,), daemon=True)
        monitor.start()
        with self._lock:
            self._readers[task_id] = reader
            self._monitors[task_id] = monitor
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
                    except OSError:
                        logger.warning("log write failed for preprocess %s", task_id)
                        break
        except Exception:
            logger.exception("reader thread fatal error for preprocess %s", task_id)
            with SessionLocal() as session:
                transition(
                    session,
                    PreprocessTask,
                    task_id,
                    "running",
                    "failed",
                    finished_at=datetime.now(),
                )
        with contextlib.suppress(OSError):
            os.close(master_fd)
        with self._lock:
            self._master_fds.pop(task_id, None)

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
        master_fd = self._master_fds.pop(task_id, None)
        if master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(master_fd)
        reader = self._readers.pop(task_id, None)
        if reader and reader.is_alive():
            reader.join(timeout=5)
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
            row = session.query(PreprocessTask).get(task_id)
            return _snapshot(row) if row else None

    def list_all(self) -> list[PreprocessTaskInfo]:
        """Return snapshots of all preprocess tasks, newest first."""
        with SessionLocal() as session:
            rows = (
                session.query(PreprocessTask).order_by(PreprocessTask.id.desc()).all()
            )
            return [_snapshot(r) for r in rows]

    def resize_pty(self, task_id: int, cols: int, rows: int) -> bool:
        """Resize a running preprocess task's PTY; return True on success."""
        master_fd = self._master_fds.get(task_id)
        if master_fd is None:
            return False
        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
            return True
        except OSError:
            return False

    def stop(self, task_id: int) -> bool:
        """Gracefully stop a running preprocess task via SIGINT."""
        with SessionLocal() as session:
            row = session.query(PreprocessTask).get(task_id)
            status = row.status if row else None
        if status != "running":
            return False
        with SessionLocal() as session:
            claimed = transition(
                session, PreprocessTask, task_id, "running", "stopping"
            )
        if not claimed:
            return True
        self._terminate(task_id)
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
            row = session.query(PreprocessTask).get(task_id)
            if not row:
                return False
            if row.status in ("running", "stopping"):
                return False
            session.query(LogChunk).filter_by(
                source="preprocess", source_id=task_id
            ).delete()
            session.delete(row)
            session.commit()

        log_path = PREPROCESS_LOGS_DIR / f"{task_id}.log"
        if log_path.is_file():
            with contextlib.suppress(OSError):
                log_path.unlink()
        return True

    def recover_tasks(self) -> None:
        """Reattach live orphan preprocess processes at startup."""
        with SessionLocal() as session:
            running = (
                session.query(
                    PreprocessTask.id, PreprocessTask.pid, PreprocessTask.status
                )
                .filter(PreprocessTask.status.in_(["running", "stopping"]))
                .all()
            )
            for task_id, pid, prior_status in running:
                if pid and psutil.pid_exists(pid):
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
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
                    except psutil.NoSuchProcess:
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
            proc.wait()
            exit_code = proc.returncode
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            OSError,
        ):
            pass
        with SessionLocal() as session:
            row = session.query(PreprocessTask).get(task_id)
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
        env_id: str | None,
        custom_python_path: str | None,
    ) -> list[str]:
        # env_id wins; when unset, resolve_command falls back to the
        # wizard-configured default env (or raises EnvironmentNotConfigured).
        base = self._env_manager.resolve_command(env_id, custom_python_path)
        cmd = [*base, "data_process.py", action, "-d", dataset]
        if action == "download":
            if params.get("force"):
                cmd.append("--force")
            if params.get("max_retries") is not None:
                cmd.extend(["--max_retries", str(params["max_retries"])])
            if params.get("num_threads") is not None:
                cmd.extend(["--num_threads", str(params["num_threads"])])
        elif action == "process":
            # data_process.py registers RunDataConfig + GeneralConfig via
            # register_config_group, so these are dot-path flags.
            for key in (
                "min_seq_len",
                "max_seq_len",
                "kfold",
                "sample_size",
                "sample_ratio",
            ):
                if params.get(key) is not None:
                    cmd.extend([f"--data.{key}", str(params[key])])
            # nargs "+" list fields: spread the list into multiple tokens.
            for key in ("sample_attempts_bins", "sample_correct_bins"):
                val = params.get(key)
                if val is None:
                    continue
                vals = val if isinstance(val, list) else [val]
                cmd.append(f"--data.{key}")
                cmd.extend(str(v) for v in vals)
            if params.get("sample_strategy"):
                cmd.extend(["--data.sample_strategy", params["sample_strategy"]])
            if params.get("seed") is not None:
                cmd.extend(["--general.seed", str(params["seed"])])
            if params.get("extra"):
                cmd.extend(["--extra", str(params["extra"])])
        return cmd

    def shutdown(self) -> None:
        """Terminate all running preprocess subprocesses."""
        for task_id in list(self._procs):
            self._terminate(task_id)
