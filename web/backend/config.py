"""Application configuration constants.

Defines project root path, database path, runs directory, server host/port,
and GPU cache settings used across the web backend.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_PATH = PROJECT_ROOT / "web" / "backend" / "data" / "tasks.db"
RUNS_DIR = PROJECT_ROOT / "runs"
HOST = "127.0.0.1"
PORT = 8765
GPU_CACHE_SECONDS = 2
