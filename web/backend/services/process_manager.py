"""Process manager — experiment task lifecycle with a multi-GPU scheduler.

One background thread owns the queue, the running-process table, and the PTY
file descriptors. Each GPU is a scheduling lane with ``gpu_slots`` concurrent
slots; tasks request either a specific GPU or auto-assignment and are
dispatched once a lane has a free slot. Per-lane occupancy is derived from the
DB (rows whose status is running/stopping/interrupted, grouped by
``gpu_assigned``) rather than a hand counter, so lifecycle code never touches
slot accounting. With no GPUs the scheduler collapses to a single CPU lane.
Task output is appended to per-task ``.log`` files. All status writes go
through the CAS state machine in ``task_state.transition``.
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
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil
from config import TASK_LOGS_DIR
from database import SessionLocal
from models import Task

from services.gpu_monitor import GpuMonitor
from services.python_env import PythonEnvManager
from services.schema_extractor import SchemaExtractor
from services.task_state import transition

logger = logging.getLogger(__name__)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)


class ProcessManager:
    """Manages experiment task subprocesses with a multi-GPU execution queue.

    Args:
        env_manager: PythonEnvManager used to resolve task commands.
        gpu_monitor: GpuMonitor used to detect the number of GPU lanes.
    """

    def __init__(
        self,
        env_manager: PythonEnvManager,
        gpu_monitor: GpuMonitor,
        schema_extractor: SchemaExtractor,
    ):
        """Initialize the manager and start the background scheduler thread."""
        self._env_manager = env_manager
        self._gpu_monitor = gpu_monitor
        self._schema_extractor = schema_extractor
        self._queue: deque[tuple[int, int | None]] = deque()
        self._running: dict[int, subprocess.Popen] = {}
        self._master_fds: dict[int, int] = {}
        self._readers: dict[int, threading.Thread] = {}
        self._recover_monitors: dict[int, threading.Thread] = {}
        self._gpu_slots_capacity = 1
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def gpu_slots(self) -> int:
        """Concurrent task slots available on each GPU (or the CPU lane)."""
        return self._gpu_slots_capacity

    @gpu_slots.setter
    def gpu_slots(self, value: int) -> None:
        with self._lock:
            self._gpu_slots_capacity = max(1, value)
        self._wake.set()

    def get_queue(self) -> list[int]:
        """Return a copy of the ordered task ID queue."""
        with self._lock:
            return [tid for tid, _ in self._queue]

    def reorder_queue(self, task_ids: list[int]) -> None:
        """Move the given task IDs to the front of the queue in order."""
        with self._lock:
            by_id = dict(self._queue)
            front = set(task_ids)
            preserved = [(tid, req) for tid, req in self._queue if tid not in front]
            valid = [(tid, by_id[tid]) for tid in task_ids if tid in by_id]
            self._queue = deque(valid + preserved)

    def remove_from_queue(self, task_id: int) -> bool:
        """Remove a task from the queue; return True if it was present."""
        with self._lock:
            for idx, (tid, _) in enumerate(self._queue):
                if tid == task_id:
                    del self._queue[idx]
                    return True
            return False

    def launch_task(
        self,
        task_id: int,
        model_name: str,
        params: dict,
        env_id: str,
        custom_python_path: str | None = None,
    ) -> None:
        """Stamp the task row with resolved command/env and enqueue it."""
        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if not task:
                return

            base_cmd = self._env_manager.resolve_command(env_id, custom_python_path)
            cmd = base_cmd + self._build_cli_args(model_name, params)
            env_type, env_name = env_id.split(":", 1)

            task.command = " ".join(cmd)
            task.model_name = model_name
            task.dataset_name = params.get("dataset", "")
            task.env_type = env_type
            task.env_name = env_name
            task.extra_params = json.dumps(params)
            session.commit()
            gpu_request = task.gpu_request

        with self._lock:
            self._queue.append((task_id, gpu_request))
        self._wake.set()

    def _loop(self) -> None:
        while not self._stopping:
            try:
                self._reap()
                self._launch_pending()
            except Exception:
                logger.exception("scheduler loop error")
            self._wake.wait(timeout=0.2)
            self._wake.clear()

    def _launch_pending(self) -> None:
        lanes = self._lanes()
        cap = self._gpu_slots_capacity
        usage = self._slot_usage()
        while True:
            tid, assigned = self._pop_dispatchable(lanes, cap, usage)
            if tid is None:
                return
            self._do_launch(tid, assigned)

    def _gpu_count(self) -> int:
        """Return the stable GPU device count (0 if NVML is unavailable)."""
        return self._gpu_monitor.device_count

    def _lanes(self) -> list[int | None]:
        """Return the scheduling lanes: one per GPU, or a single CPU lane."""
        count = self._gpu_count()
        return list(range(count)) if count > 0 else [None]

    def _slot_usage(self) -> dict[int | None, int]:
        """Count tasks occupying each lane (running/stopping/interrupted)."""
        with SessionLocal() as session:
            rows = (
                session.query(Task.gpu_assigned)
                .filter(Task.status.in_(["running", "stopping", "interrupted"]))
                .all()
            )
        usage: dict[int | None, int] = {}
        for (gpu,) in rows:
            usage[gpu] = usage.get(gpu, 0) + 1
        return usage

    def _pick_lane(
        self,
        request: int | None,
        lanes: list[int | None],
        cap: int,
        usage: dict[int | None, int],
    ) -> tuple[bool, int | None]:
        """Resolve a task to a lane with a free slot.

        Returns ``(True, lane)`` when a dispatchable lane exists (``lane`` may be
        ``None`` for the CPU lane), or ``(False, None)`` when the task is
        blocked. Pinned requests target their GPU only; auto requests pick the
        least-loaded lane that still has capacity (ties favor the lower index,
        preserved by lane order).
        """
        if request is not None and request in lanes:
            return (usage.get(request, 0) < cap, request)
        best: int | None = None
        best_load = cap
        found = False
        for lane in lanes:
            load = usage.get(lane, 0)
            if load < cap and load < best_load:
                best = lane
                best_load = load
                found = True
        return (found, best)

    def _pop_dispatchable(
        self,
        lanes: list[int | None],
        cap: int,
        usage: dict[int | None, int],
    ) -> tuple[int | None, int | None]:
        """Pop the first queue task that can be dispatched now.

        ``lanes``/``cap``/``usage`` are computed once per scheduler pass by the
        caller; this only takes the lock for the in-memory queue scan (no I/O
        under the lock) and mutates ``usage`` in place so multiple pops in one
        pass account for just-dispatched tasks without re-querying. Scans in
        order, skipping tasks whose target lane is full so a blocked pinned task
        does not stall later dispatchable tasks.
        """
        with self._lock:
            for idx, (tid, request) in enumerate(self._queue):
                fallback = request is not None and request not in lanes
                ok, target = self._pick_lane(
                    None if fallback else request, lanes, cap, usage
                )
                if not ok:
                    continue
                if fallback:
                    logger.warning(
                        "task %s requested GPU %s but only %d lane(s) exist; "
                        "auto-assigned to lane %s",
                        tid,
                        request,
                        len(lanes),
                        target,
                    )
                del self._queue[idx]
                usage[target] = usage.get(target, 0) + 1
                return tid, target
            return None, None

    def _do_launch(self, task_id: int, assigned_gpu: int | None) -> bool:
        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if not task:
                return False

            env_id = f"{task.env_type}:{task.env_name}"
            custom_python_path = task.python_path or None
            base_cmd = self._env_manager.resolve_command(env_id, custom_python_path)
            params = json.loads(task.extra_params or "{}")
            cmd = base_cmd + self._build_cli_args(task.model_name, params)
            env_type = task.env_type

            try:
                master_fd, slave_fd = pty.openpty()
            except OSError:
                transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "failed",
                    finished_at=datetime.now(),
                )
                return False

            try:
                winsize = struct.pack("HHHH", 24, 80, 0, 0)
                fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
                env = os.environ.copy()
                env["TERM"] = "xterm-256color"
                env["FORCE_COLOR"] = "1"
                if assigned_gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
                proc = subprocess.Popen(
                    cmd,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    cwd=PROJECT_ROOT,
                    start_new_session=True,
                    env=env,
                )
                os.close(slave_fd)
            except Exception:
                os.close(slave_fd)
                os.close(master_fd)
                transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "failed",
                    finished_at=datetime.now(),
                )
                return False

            extra: dict = {}
            if env_type == "custom" and custom_python_path:
                extra["python_path"] = custom_python_path
            if not transition(
                session,
                Task,
                task_id,
                "pending",
                "running",
                pid=proc.pid,
                started_at=datetime.now(),
                gpu_assigned=assigned_gpu,
                **extra,
            ):
                logger.warning(
                    "Launch of task %s aborted: row no longer pending", task_id
                )
                with contextlib.suppress(Exception):
                    self._kill_process_group(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3)
                os.close(master_fd)
                return False

        with self._lock:
            self._running[task_id] = proc
            self._master_fds[task_id] = master_fd

        reader = threading.Thread(
            target=self._read_pty, args=(task_id, master_fd), daemon=True
        )
        reader.start()
        with self._lock:
            self._readers[task_id] = reader
        return True

    def _read_pty(self, task_id: int, master_fd: int) -> None:
        path = TASK_LOGS_DIR / f"{task_id}.log"
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
                        logger.warning("log write failed for task %s", task_id)
                        break
        except Exception:
            logger.exception("reader thread fatal error for task %s", task_id)
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "running",
                    "failed",
                    finished_at=datetime.now(),
                )
        with contextlib.suppress(OSError):
            os.close(master_fd)
        with self._lock:
            self._master_fds.pop(task_id, None)

    def _build_cli_args(
        self,
        model_name: str,
        params: dict,
    ) -> list[str]:
        """Build a ``train.py`` invocation directly from frontend form values.

        Routes the flat ``params`` dict into dotted ``--node.field=value`` flags
        via the cached schema route map. Uses ``-m`` / ``-d`` short flags for
        model and dataset.  Parameters matching their schema default are omitted.
        """
        routes = self._schema_extractor.get_field_routes(model_name)
        defaults = self._schema_extractor.get_field_defaults(model_name)
        args = ["train.py", "-m", model_name]

        dataset = params.get("dataset")
        if dataset:
            args.extend(["-d", str(dataset)])

        for field, value in params.items():
            if field == "dataset":
                continue
            node = routes.get(field)
            if node is None:
                if value is not None:
                    logger.warning(
                        "dropping param '%s' — not in schema for model '%s'",
                        field,
                        model_name,
                    )
                continue
            if value is None:
                continue
            if self._is_default(value, defaults.get(field)):
                continue
            if isinstance(value, bool):
                args.append(f"--{node}.{field}={str(value).lower()}")
            elif isinstance(value, list):
                args.append(f"--{node}.{field}=[{','.join(str(v) for v in value)}]")
            else:
                args.append(f"--{node}.{field}={value}")
        return args

    @staticmethod
    def _is_default(value: object, default: object) -> bool:
        """Return True when *value* equals the schema default.

        Treats ``False`` as equivalent to a ``None`` default to compensate for
        the frontend's ``el-switch`` coercing ``null`` to ``false`` on optional
        boolean fields like ``compile_dynamic``.
        """
        if isinstance(value, list) and isinstance(default, list):
            return value == default
        if default is None and value is False:
            return True
        return value == default

    def preview_command(self, model_name: str, params: dict) -> str:
        """Return the CLI invocation that would be executed for these params."""
        return " ".join(self._build_cli_args(model_name, params))

    def resize_pty(self, task_id: int, cols: int, rows: int) -> bool:
        """Resize a running task's PTY; return True on success."""
        master_fd = self._master_fds.get(task_id)
        if master_fd is None:
            return False
        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
            return True
        except OSError:
            return False

    def _kill_process_group(self, pid: int, sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(pid, sig)

    def _terminate(self, task_id: int) -> bool:
        with self._lock:
            proc = self._running.get(task_id)
        if proc is None:
            return False

        pid = proc.pid
        with contextlib.suppress(Exception):
            self._kill_process_group(pid, signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                self._kill_process_group(pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=3)

        self._cleanup(task_id)
        return True

    def _cleanup(self, task_id: int) -> None:
        master_fd = self._master_fds.pop(task_id, None)
        if master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(master_fd)
        reader = self._readers.pop(task_id, None)
        if reader and reader.is_alive():
            reader.join(timeout=5)
        with self._lock:
            self._running.pop(task_id, None)

    def _reap(self) -> None:
        with self._lock:
            snapshot = list(self._running.items())
        for task_id, proc in snapshot:
            rc = proc.poll()
            if rc is None:
                continue
            self._cleanup(task_id)
            with SessionLocal() as session:
                to = "completed" if rc == 0 else "failed"
                transition(
                    session,
                    Task,
                    task_id,
                    "running",
                    to,
                    exit_code=rc,
                    finished_at=datetime.now(),
                    pid=None,
                )

    def stop_task(self, task_id: int) -> bool:
        """Gracefully stop a task (queue removal or SIGINT)."""
        if self.remove_from_queue(task_id):
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "stopped",
                    finished_at=datetime.now(),
                )
            self._wake.set()
            return True

        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            status = task.status if task else None
            pid = task.pid if task else None

        if status == "interrupted":
            if pid:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.kill(pid, signal.SIGKILL)
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "interrupted",
                    "stopped",
                    finished_at=datetime.now(),
                    pid=None,
                )
            with self._lock:
                self._recover_monitors.pop(task_id, None)
            return True

        if status == "running":
            with SessionLocal() as session:
                claimed = transition(session, Task, task_id, "running", "stopping")
            if not claimed:
                return True
            self._terminate(task_id)
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "stopping",
                    "stopped",
                    finished_at=datetime.now(),
                    pid=None,
                )
            self._wake.set()
            return True

        if status == "pending":
            with SessionLocal() as session:
                return transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "stopped",
                    finished_at=datetime.now(),
                )

        return False

    def kill_task(self, task_id: int) -> bool:
        """Force-kill a task via SIGKILL to its process group."""
        if self.remove_from_queue(task_id):
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "stopped",
                    exit_code=-9,
                    finished_at=datetime.now(),
                )
            self._wake.set()
            return True

        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            status = task.status if task else None
            pid = task.pid if task else None

        if status == "interrupted":
            if pid:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.kill(pid, signal.SIGKILL)
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "interrupted",
                    "stopped",
                    finished_at=datetime.now(),
                    pid=None,
                )
            with self._lock:
                self._recover_monitors.pop(task_id, None)
            return True

        if status == "running":
            with SessionLocal() as session:
                claimed = transition(session, Task, task_id, "running", "stopping")
            if not claimed:
                return True

            with self._lock:
                proc = self._running.get(task_id)
            if proc is not None:
                with contextlib.suppress(Exception):
                    self._kill_process_group(proc.pid, signal.SIGKILL)
            self._cleanup(task_id)

            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "stopping",
                    "stopped",
                    exit_code=-9,
                    finished_at=datetime.now(),
                    pid=None,
                )
            self._wake.set()
            return True

        if status == "pending":
            with SessionLocal() as session:
                return transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "stopped",
                    exit_code=-9,
                    finished_at=datetime.now(),
                )

        return False

    def recover_tasks(self) -> None:
        """Reattach live orphans, re-queue dead in-flight tasks, then queue pending.

        In-flight tasks whose process is gone (interrupted by a prior shutdown,
        or crashed) are put back to ``pending`` so they re-run instead of being
        lost; live orphans are re-attached. Pending tasks are then queued in
        ``id`` order, which matches creation (fold) order and keeps the queue
        stable across restarts.
        """
        with SessionLocal() as session:
            inflight = (
                session.query(Task.id, Task.pid, Task.status)
                .filter(Task.status.in_(["running", "stopping", "interrupted"]))
                .all()
            )
            for task_id, pid, prior_status in inflight:
                if pid and psutil.pid_exists(pid):
                    try:
                        if psutil.Process(pid).is_running():
                            if prior_status != "interrupted":
                                transition(
                                    session, Task, task_id, prior_status, "interrupted"
                                )
                            t = threading.Thread(
                                target=self._recover_monitor,
                                args=(task_id, pid),
                                daemon=True,
                            )
                            t.start()
                            with self._lock:
                                self._recover_monitors[task_id] = t
                            continue
                    except psutil.NoSuchProcess:
                        pass
                transition(
                    session,
                    Task,
                    task_id,
                    prior_status,
                    "pending",
                    pid=None,
                    gpu_assigned=None,
                    started_at=None,
                    finished_at=None,
                    exit_code=None,
                )

            pending = (
                session.query(Task.id, Task.gpu_request)
                .filter(Task.status == "pending")
                .order_by(Task.id)
                .all()
            )
            with self._lock:
                for task_id, gpu_request in pending:
                    self._queue.append((task_id, gpu_request))
        self._wake.set()

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
            task = session.query(Task).get(task_id)
            if task and task.status in ("running", "interrupted"):
                to = "completed" if exit_code == 0 else "failed"
                transition(
                    session,
                    Task,
                    task_id,
                    task.status,
                    to,
                    exit_code=exit_code,
                    finished_at=datetime.now(),
                    pid=None,
                )
        with self._lock:
            self._recover_monitors.pop(task_id, None)

    def shutdown(self) -> None:
        """Stop the scheduler, gracefully stop running tasks, mark them interrupted.

        Each running task gets the same graceful stop the UI uses (SIGINT to its
        process group, then SIGKILL only if it does not exit in time), so the
        trainer can clean up and no orphans survive the backend exit. ``pid`` is
        cleared; on the next start ``recover_tasks`` re-queues these interrupted
        rows as ``pending`` and re-dispatches them against the current GPU set.
        """
        self._stopping = True
        self._wake.set()
        with contextlib.suppress(Exception):
            self._thread.join(timeout=10)

        for task_id in list(self._running):
            self._terminate(task_id)

        with SessionLocal() as session:
            session.query(Task).filter(Task.status.in_(["running", "stopping"])).update(
                {
                    "status": "interrupted",
                    "pid": None,
                }
            )
            session.commit()
