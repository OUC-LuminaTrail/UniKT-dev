import contextlib
import fcntl
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

from config import PROJECT_ROOT
from database import SessionLocal
from models import LogChunk


class PreprocessTask:
    def __init__(self, task_id: int, command: list[str]):
        self.id = task_id
        self.command = command
        self.status: str = "running"
        self.exit_code: int | None = None
        self.started_at: datetime = datetime.now()
        self.finished_at: datetime | None = None
        self.pid: int | None = None


class PreprocessManager:
    def __init__(self):
        self._tasks: dict[int, PreprocessTask] = {}
        self._procs: dict[int, subprocess.Popen] = {}
        self._master_fds: dict[int, int] = {}
        self._readers: dict[int, threading.Thread] = {}
        self._next_id = 1

    def start(self, action: str, dataset: str, params: dict) -> PreprocessTask:
        task_id = self._next_id
        self._next_id += 1

        command = self._build_command(action, dataset, params)

        task = PreprocessTask(task_id, command)
        self._tasks[task_id] = task

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

        task.pid = proc.pid
        self._procs[task_id] = proc
        self._master_fds[task_id] = master_fd

        reader = threading.Thread(
            target=self._read_pty, args=(task_id, master_fd),
            daemon=True,
        )
        reader.start()
        self._readers[task_id] = reader

        t = threading.Thread(
            target=self._monitor, args=(task_id,), daemon=True
        )
        t.start()

        return task

    def _read_pty(self, task_id: int, master_fd: int) -> None:
        offset = 0
        with SessionLocal() as session:
            last_chunk = (
                session.query(LogChunk)
                .filter_by(source="preprocess", source_id=task_id)
                .order_by(LogChunk.byte_offset.desc())
                .first()
            )
            if last_chunk:
                offset = last_chunk.byte_offset
        while True:
            try:
                data = os.read(master_fd, 65536)
                if not data:
                    break
                with SessionLocal() as session:
                    chunk = LogChunk(
                        source="preprocess",
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

    def get(self, task_id: int) -> PreprocessTask | None:
        return self._tasks.get(task_id)

    def list_all(self) -> list[PreprocessTask]:
        return list(self._tasks.values())

    def stop(self, task_id: int) -> bool:
        task = self._tasks.get(task_id)
        if not task or task.status != "running":
            return False
        task.status = "stopping"
        proc = self._procs.get(task_id)
        if proc:
            with contextlib.suppress(Exception):
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGINT)
        return True

    def _build_command(self, action: str, dataset: str, params: dict) -> list[str]:
        cmd = ["python", "data_process.py", action, "-d", dataset]
        if action == "download":
            if params.get("force"):
                cmd.append("--force")
            if params.get("max_retries") is not None:
                cmd.extend(["--max_retries", str(params["max_retries"])])
            if params.get("num_threads") is not None:
                cmd.extend(["--num_threads", str(params["num_threads"])])
        elif action == "process":
            if params.get("min_seq_len") is not None:
                cmd.extend(["--min_seq_len", str(params["min_seq_len"])])
            if params.get("max_seq_len") is not None:
                cmd.extend(["--max_seq_len", str(params["max_seq_len"])])
            if params.get("kfold") is not None:
                cmd.extend(["--kfold", str(params["kfold"])])
            if params.get("seed") is not None:
                cmd.extend(["--seed", str(params["seed"])])
            if params.get("sample_size") is not None:
                cmd.extend(["--sample_size", str(params["sample_size"])])
            if params.get("sample_ratio") is not None:
                cmd.extend(["--sample_ratio", str(params["sample_ratio"])])
            if params.get("sample_strategy"):
                cmd.extend(["--sample_strategy", params["sample_strategy"]])
            if params.get("sample_attempts_bins") is not None:
                cmd.extend(["--sample_attempts_bins", str(params["sample_attempts_bins"])])
            if params.get("sample_correct_bins") is not None:
                cmd.extend(["--sample_correct_bins", str(params["sample_correct_bins"])])
            extra = params.get("extra")
            if extra:
                cmd.extend(["--extra", str(extra)])
        return cmd

    def _monitor(self, task_id: int) -> None:
        proc = self._procs.get(task_id)
        if proc is None:
            return

        try:
            proc.wait()
            exit_code = proc.returncode
        except Exception:
            exit_code = -1

        self._procs.pop(task_id, None)

        reader = self._readers.pop(task_id, None)
        if reader and reader.is_alive():
            reader.join(timeout=5)

        master_fd = self._master_fds.get(task_id)
        if master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(master_fd)

        task = self._tasks.get(task_id)
        if task and task.status in ("running", "stopping"):
            if task.status == "stopping":
                task.status = "stopped"
            else:
                task.status = "completed" if exit_code == 0 else "failed"
            task.exit_code = exit_code
            task.finished_at = datetime.now()
            task.pid = None

    def shutdown(self) -> None:
        for task_id in list(self._procs.keys()):
            proc = self._procs.pop(task_id, None)
            if proc:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            master_fd = self._master_fds.pop(task_id, None)
            if master_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(master_fd)
