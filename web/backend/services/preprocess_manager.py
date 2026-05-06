import contextlib
import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from config import LOG_DIR, PROJECT_ROOT


class PreprocessTask:
    def __init__(self, task_id: int, command: list[str], log_path: str):
        self.id = task_id
        self.command = command
        self.log_path = log_path
        self.status: str = "running"
        self.exit_code: int | None = None
        self.started_at: datetime = datetime.now()
        self.finished_at: datetime | None = None
        self.pid: int | None = None


class PreprocessManager:
    def __init__(self):
        self._tasks: dict[int, PreprocessTask] = {}
        self._procs: dict[int, subprocess.Popen] = {}
        self._next_id = 1

    def start(self, action: str, dataset: str, params: dict) -> PreprocessTask:
        task_id = self._next_id
        self._next_id += 1

        command = self._build_command(action, dataset, params)

        log_path = LOG_DIR / f"preprocess_{task_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        task = PreprocessTask(task_id, command, str(log_path))
        self._tasks[task_id] = task

        log_file = open(log_path, "wb")

        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
        log_file.close()

        task.pid = proc.pid
        self._procs[task_id] = proc

        t = threading.Thread(
            target=self._monitor, args=(task_id,), daemon=True
        )
        t.start()

        return task

    def get(self, task_id: int) -> PreprocessTask | None:
        return self._tasks.get(task_id)

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
