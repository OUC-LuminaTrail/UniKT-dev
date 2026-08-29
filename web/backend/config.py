"""Application configuration constants.

Defines project root path, database path, runs directory, server host/port,
and GPU cache settings used across the web backend.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "web" / "backend" / "data"
DATABASE_PATH = DATA_DIR / "tasks.db"
TASK_LOGS_DIR = DATA_DIR / "task_logs"
PREPROCESS_LOGS_DIR = DATA_DIR / "preprocess_logs"
# Per-search Optuna YAML configs generated from the launch form.
SEARCH_CONFIGS_DIR = DATA_DIR / "search_configs"
RUNS_DIR = PROJECT_ROOT / "runs"
# Search artifacts (study.db, trials) live under runs/hyperparam_search/.
SEARCH_RUNS_DIR = RUNS_DIR / "hyperparam_search"
# JSON substring identifying optuna search tasks in Task.extra_params; matches
# the default json.dumps output of {"task_kind": "optuna", ...}. Shared by the
# search router (include) and the tasks router (exclude) so the two never drift.
SEARCH_TASK_MARKER = '%"task_kind": "optuna"%'
HOST = "127.0.0.1"
PORT = 8765
GPU_CACHE_SECONDS = 2


def read_env_file_value(key: str) -> str | None:
    """Return a ``KEY=VALUE`` entry from the repo-root ``.env``, if present.

    The vite config reads the same file (loadEnv) for KT_WEB_PORT; the backend
    process env does not include it, so mirror the lookup here.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            value = stripped.split("=", 1)[1].strip().strip("'\"")
            return value or None
    return None
