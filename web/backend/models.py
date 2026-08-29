"""SQLAlchemy ORM models for the task and log database.

Defines the ``Task`` and ``PreprocessTask`` tables used throughout the web backend
for tracking experiment runs and their output logs.
"""

from datetime import datetime

from database import Base
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column


class Task(Base):
    """An experiment task tracked by the process manager.

    Stores metadata, status, command, and timing information for each
    experiment run.
    """

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256))
    command: Mapped[str] = mapped_column(Text)
    model_name: Mapped[str] = mapped_column(String(64))
    dataset_name: Mapped[str] = mapped_column(String(64), default="")
    env_type: Mapped[str] = mapped_column(String(16))
    env_name: Mapped[str] = mapped_column(String(64), default="")
    python_path: Mapped[str] = mapped_column(String(512), default="")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    tags: Mapped[str] = mapped_column(Text, default="[]")
    extra_params: Mapped[str] = mapped_column(Text, default="{}")
    gpu_request: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gpu_assigned: Mapped[int | None] = mapped_column(Integer, nullable=True)


class PreprocessTask(Base):
    """A data download/processing task tracked by the preprocess manager.

    Persists across restarts so interrupted preprocess runs can be observed
    after the backend restarts.
    """

    __tablename__ = "preprocess_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    command: Mapped[str] = mapped_column(Text, nullable=False)
    env_type: Mapped[str] = mapped_column(String(16), default="")
    env_name: Mapped[str] = mapped_column(String(64), default="")
    python_path: Mapped[str] = mapped_column(String(512), default="")
    status: Mapped[str] = mapped_column(String(16), default="running", index=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    params: Mapped[str] = mapped_column(Text, default="{}")
