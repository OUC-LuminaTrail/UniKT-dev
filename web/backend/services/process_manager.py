import contextlib
import fcntl
import json
import os
import pty
import signal
import struct
import subprocess
import termios
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from database import SessionLocal
from models import LogChunk, Task

from services.python_env import PythonEnvManager


class ProcessManager:
    def __init__(self, env_manager: PythonEnvManager):
        self._env_manager = env_manager
        self._monitors: dict[int, threading.Thread] = {}
        self._procs: dict[int, subprocess.Popen] = {}
        self._master_fds: dict[int, int] = {}
        self._readers: dict[int, threading.Thread] = {}
        self._queue: list[int] = []
        self._max_concurrent = 1
        self._available_slots = 1
        self._lock = threading.Lock()

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @max_concurrent.setter
    def max_concurrent(self, value: int) -> None:
        with self._lock:
            diff = max(1, value) - self._max_concurrent
            self._max_concurrent = max(1, value)
            self._available_slots += diff
        self._dequeue_next()

    @property
    def running_count(self) -> int:
        return len(self._procs)

    def get_queue(self) -> list[int]:
        return list(self._queue)

    def reorder_queue(self, task_ids: list[int]) -> None:
        with self._lock:
            task_set = set(task_ids)
            preserved = [tid for tid in self._queue if tid not in task_set]
            valid = [tid for tid in task_ids if tid in self._queue]
            self._queue = valid + preserved

    def remove_from_queue(self, task_id: int) -> bool:
        with self._lock:
            if task_id in self._queue:
                self._queue.remove(task_id)
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
        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if not task:
                return

            base_cmd = self._env_manager.resolve_command(env_id, custom_python_path)
            cli_args = self._build_cli_args(model_name, params)
            cmd = base_cmd + cli_args

            env_type, env_name = env_id.split(":", 1)

            task.command = " ".join(cmd)
            task.model_name = model_name
            task.dataset_name = params.get("dataset", "")
            task.env_type = env_type
            task.env_name = env_name
            task.extra_params = json.dumps(params)
            task.status = "pending"
            session.commit()

        self._enqueue(task_id)

    def _enqueue(
        self,
        task_id: int,
    ) -> None:
        with self._lock:
            self._queue.append(task_id)
        self._dequeue_next()

    def _dequeue_next(self) -> None:
        while True:
            with self._lock:
                if not self._queue or self._available_slots <= 0:
                    return
                task_id = self._queue.pop(0)
                self._available_slots -= 1

            try:
                with SessionLocal() as session:
                    task = session.query(Task).get(task_id)
                    if not task or task.status != "pending":
                        if not task:
                            with self._lock:
                                self._queue = [
                                    tid for tid in self._queue if tid != task_id
                                ]
                        with self._lock:
                            self._available_slots += 1
                        continue
                    extra = task.extra_params or "{}"
                    params = json.loads(extra)
                    model_name = task.model_name
                    env_id = f"{task.env_type}:{task.env_name}"
                    custom_python_path = task.python_path or None

                launched = self._do_launch(
                    task_id, model_name, params, env_id, custom_python_path
                )
            except Exception:
                launched = False

            if not launched:
                with self._lock:
                    self._available_slots += 1

    def _do_launch(
        self,
        task_id: int,
        model_name: str,
        params: dict,
        env_id: str,
        custom_python_path: str | None = None,
    ) -> bool:
        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if not task:
                return False

            base_cmd = self._env_manager.resolve_command(env_id, custom_python_path)
            cli_args = self._build_cli_args(model_name, params)
            cmd = base_cmd + cli_args

            env_type, env_name = env_id.split(":", 1)

            task.command = " ".join(cmd)
            task.model_name = model_name
            task.dataset_name = params.get("dataset", "")
            task.env_type = env_type
            task.env_name = env_name
            task.extra_params = json.dumps(params)

            project_root = str(Path(__file__).resolve().parent.parent.parent.parent)

            try:
                master_fd, slave_fd = pty.openpty()
            except OSError:
                task.status = "failed"
                task.finished_at = datetime.now()
                session.commit()
                return False

            try:
                winsize = struct.pack("HHHH", 24, 80, 0, 0)
                fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

                env = os.environ.copy()
                env["TERM"] = "xterm-256color"
                env["FORCE_COLOR"] = "1"

                proc = subprocess.Popen(
                    cmd,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    cwd=project_root,
                    start_new_session=True,
                    env=env,
                )
                os.close(slave_fd)
            except Exception:
                os.close(slave_fd)
                os.close(master_fd)
                task.status = "failed"
                task.finished_at = datetime.now()
                session.commit()
                return False

            task.pid = proc.pid
            task.status = "running"
            task.started_at = datetime.now()
            if env_type == "custom" and custom_python_path:
                task.python_path = custom_python_path

            self._procs[task_id] = proc
            self._master_fds[task_id] = master_fd

            session.commit()

        try:
            reader = threading.Thread(
                target=self._read_pty,
                args=(task_id, master_fd),
                daemon=True,
            )
            reader.start()
            self._readers[task_id] = reader

            t = threading.Thread(
                target=self._monitor_process,
                args=(task_id,),
                daemon=True,
            )
            t.start()
            self._monitors[task_id] = t
        except Exception:
            with contextlib.suppress(Exception):
                self._kill_process_group(proc.pid, signal.SIGKILL)
                proc.wait(timeout=3)
            self._procs.pop(task_id, None)
            os.close(master_fd)
            self._master_fds.pop(task_id, None)
            self._readers.pop(task_id, None)
            self._monitors.pop(task_id, None)
            with SessionLocal() as session:
                task = session.query(Task).get(task_id)
                if task and task.status == "running":
                    task.status = "failed"
                    task.finished_at = datetime.now()
                    session.commit()
            return False

        return True

    def _read_pty(self, task_id: int, master_fd: int) -> None:
        offset = 0
        with SessionLocal() as session:
            last_chunk = (
                session.query(LogChunk)
                .filter_by(source="task", source_id=task_id)
                .order_by(LogChunk.byte_offset.desc())
                .first()
            )
            if last_chunk:
                offset = last_chunk.byte_offset + len(last_chunk.raw_data)
        while True:
            try:
                data = os.read(master_fd, 65536)
                if not data:
                    break
                with SessionLocal() as session:
                    chunk = LogChunk(
                        source="task",
                        source_id=task_id,
                        byte_offset=offset,
                        raw_data=data,
                        created_at=time.time(),
                    )
                    session.add(chunk)
                    session.commit()
                offset += len(data)
            except OSError:
                break
        with contextlib.suppress(OSError):
            os.close(master_fd)
        self._master_fds.pop(task_id, None)

    def _build_cli_args(self, model_name: str, params: dict) -> list[str]:
        args = ["train.py", "-m", model_name]
        for key, value in params.items():
            if key == "model":
                continue
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            elif isinstance(value, list):
                if not value:
                    continue
                args.append(f"--{key}")
                args.extend(str(v) for v in value)
            else:
                args.append(f"--{key}")
                args.append(str(value))
        return args

    def resize_pty(self, task_id: int, cols: int, rows: int) -> bool:
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
            pgid = os.getpgid(pid)
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(pid, sig)

    def _terminate_task_processes(self, task_id: int) -> bool:
        proc = self._procs.get(task_id)
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

        master_fd = self._master_fds.get(task_id)
        if master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(master_fd)

        reader = self._readers.pop(task_id, None)
        if reader and reader.is_alive():
            reader.join(timeout=5)
        self._monitors.pop(task_id, None)
        self._master_fds.pop(task_id, None)

        with self._lock:
            proc_was_present = self._procs.pop(task_id, None) is not None
        return proc_was_present

    def _monitor_process(self, task_id: int) -> None:
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

        slot_returned = False
        with self._lock:
            if self._procs.pop(task_id, None) is not None:
                self._available_slots += 1
                slot_returned = True

        self._master_fds.pop(task_id, None)

        reader = self._readers.pop(task_id, None)
        if reader and reader.is_alive():
            reader.join(timeout=5)

        self._monitors.pop(task_id, None)

        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if task and task.status == "running":
                task.status = "completed" if exit_code == 0 else "failed"
                task.exit_code = exit_code
                task.finished_at = datetime.now()
                task.pid = None
                session.commit()

        if slot_returned:
            self._dequeue_next()

    def stop_task(self, task_id: int) -> bool:
        if self.remove_from_queue(task_id):
            with SessionLocal() as session:
                task = session.query(Task).get(task_id)
                if task and task.status == "pending":
                    task.status = "stopped"
                    task.finished_at = datetime.now()
                    session.commit()
            self._dequeue_next()
            return True

        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if not task or task.status not in ("running", "interrupted"):
                return False

            if task.status == "interrupted":
                pid = task.pid
                if pid:
                    with contextlib.suppress(
                        ProcessLookupError, PermissionError, OSError
                    ):
                        os.kill(pid, signal.SIGKILL)
                task.status = "stopped"
                task.finished_at = datetime.now()
                task.pid = None
                session.commit()
                self._monitors.pop(task_id, None)
                return True

            task.status = "stopping"
            session.commit()

        proc_was_present = self._terminate_task_processes(task_id)

        if proc_was_present:
            with self._lock:
                self._available_slots += 1

        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if task and task.status == "stopping":
                task.status = "stopped"
                task.finished_at = datetime.now()
                task.pid = None
                session.commit()

        self._dequeue_next()
        return True

    def kill_task(self, task_id: int) -> bool:
        if self.remove_from_queue(task_id):
            with SessionLocal() as session:
                task = session.query(Task).get(task_id)
                if task and task.status == "pending":
                    task.status = "stopped"
                    task.exit_code = -9
                    task.finished_at = datetime.now()
                    session.commit()
            self._dequeue_next()
            return True

        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if not task or task.status not in ("running", "interrupted"):
                return False

            if task.status == "interrupted":
                pid = task.pid
                if pid:
                    with contextlib.suppress(
                        ProcessLookupError, PermissionError, OSError
                    ):
                        os.kill(pid, signal.SIGKILL)
                task.status = "stopped"
                task.finished_at = datetime.now()
                task.pid = None
                session.commit()
                self._monitors.pop(task_id, None)
                return True

            task.status = "stopping"
            session.commit()

        proc = self._procs.get(task_id)
        proc_was_present = False
        if proc:
            with contextlib.suppress(Exception):
                self._kill_process_group(proc.pid, signal.SIGKILL)
            with self._lock:
                proc_was_present = self._procs.pop(task_id, None) is not None

        master_fd = self._master_fds.pop(task_id, None)
        if master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(master_fd)

        self._readers.pop(task_id, None)
        self._monitors.pop(task_id, None)

        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if task and task.status == "stopping":
                task.status = "stopped"
                task.finished_at = datetime.now()
                task.exit_code = -9
                task.pid = None
                session.commit()

        if proc_was_present:
            with self._lock:
                self._available_slots += 1
        self._dequeue_next()
        return True

    def recover_tasks(self) -> None:
        with SessionLocal() as session:
            running_tasks = (
                session.query(Task)
                .filter(Task.status.in_(["running", "stopping"]))
                .all()
            )
            for task in running_tasks:
                if task.pid and psutil.pid_exists(task.pid):
                    try:
                        proc = psutil.Process(task.pid)
                        if proc.is_running():
                            task.status = "interrupted"
                            session.commit()
                            t = threading.Thread(
                                target=self._recover_monitor,
                                args=(task.id, task.pid),
                                daemon=True,
                            )
                            t.start()
                            self._monitors[task.id] = t
                            continue
                    except psutil.NoSuchProcess:
                        pass
                task.status = "failed"
                task.finished_at = datetime.now()
                task.pid = None
                session.commit()

            pending_tasks = (
                session.query(Task)
                .filter(Task.status == "pending")
                .order_by(Task.created_at)
                .all()
            )
            with self._lock:
                for task in pending_tasks:
                    self._queue.append(task.id)

        self._dequeue_next()

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
                task.status = "completed" if exit_code == 0 else "failed"
                task.exit_code = exit_code
                task.finished_at = datetime.now()
                task.pid = None
                session.commit()

        self._monitors.pop(task_id, None)

    def shutdown(self) -> None:
        task_ids = list(self._procs.keys())
        for task_id in task_ids:
            self._terminate_task_processes(task_id)
            reader = self._readers.pop(task_id, None)
            if reader and reader.is_alive():
                reader.join(timeout=5)
            self._monitors.pop(task_id, None)
            self._master_fds.pop(task_id, None)

        with SessionLocal() as session:
            session.query(Task).filter(Task.status.in_(["running", "stopping"])).update(
                {
                    "status": "interrupted",
                    "exit_code": -15,
                    "finished_at": datetime.now(),
                    "pid": None,
                }
            )
            session.commit()
