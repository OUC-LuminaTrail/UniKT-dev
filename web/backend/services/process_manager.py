import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import psutil
from database import SessionLocal
from models import Task
from services.environment_resolver import EnvironmentResolver


class ProcessManager:
    def __init__(self):
        self._resolver = EnvironmentResolver()
        self._monitors: dict[int, threading.Thread] = {}

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

            base_cmd = self._resolver.resolve_command(env_id, custom_python_path)
            cli_args = self._build_cli_args(model_name, params)
            cmd = base_cmd + cli_args

            env_type, env_name = env_id.split(":", 1)

            task.command = " ".join(cmd)
            task.model_name = model_name
            task.dataset_name = params.get("dataset", "")
            task.env_type = env_type
            task.env_name = env_name
            task.extra_params = str(params)

            log_path = Path(task.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            project_root = str(Path(__file__).resolve().parent.parent.parent.parent)

            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=project_root,
                    start_new_session=True,
                )

            task.pid = proc.pid
            task.status = "running"
            task.started_at = datetime.now()
            if env_type == "custom" and custom_python_path:
                task.python_path = custom_python_path
            session.commit()

            t = threading.Thread(
                target=self._monitor_process,
                args=(task_id, proc.pid),
                daemon=True,
            )
            t.start()
            self._monitors[task_id] = t

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
                args.append(f"--{key}")
                args.extend(str(v) for v in value)
            else:
                args.append(f"--{key}")
                args.append(str(value))
        return args

    def _monitor_process(self, task_id: int, pid: int) -> None:
        try:
            proc = psutil.Process(pid)
            proc.wait()
            exit_code = proc.returncode
        except psutil.NoSuchProcess:
            exit_code = -1

        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if task:
                task.status = "completed" if exit_code == 0 else "failed"
                task.exit_code = exit_code
                task.finished_at = datetime.now()
                task.pid = None
                session.commit()

        self._monitors.pop(task_id, None)

    def stop_task(self, task_id: int) -> bool:
        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if not task or task.status != "running":
                return False
            if task.pid:
                try:
                    os.kill(task.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                task.status = "stopped"
                task.finished_at = datetime.now()
                task.pid = None
                session.commit()
            return True

    def kill_task(self, task_id: int) -> bool:
        with SessionLocal() as session:
            task = session.query(Task).get(task_id)
            if not task or task.status != "running":
                return False
            if task.pid:
                try:
                    os.killpg(os.getpgid(task.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                task.status = "stopped"
                task.finished_at = datetime.now()
                task.exit_code = -9
                task.pid = None
                session.commit()
            return True

    def recover_tasks(self) -> None:
        with SessionLocal() as session:
            running_tasks = session.query(Task).filter(Task.status == "running").all()
            for task in running_tasks:
                if task.pid and psutil.pid_exists(task.pid):
                    try:
                        proc = psutil.Process(task.pid)
                        if proc.is_running():
                            t = threading.Thread(
                                target=self._monitor_process,
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

    def shutdown(self) -> None:
        pass
