from datetime import datetime

from sqlalchemy import BLOB, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class Task(Base):
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
    exp_dir: Mapped[str] = mapped_column(String(512), default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    tags: Mapped[str] = mapped_column(Text, default="[]")
    extra_params: Mapped[str] = mapped_column(Text, default="{}")


class LogChunk(Base):
    __tablename__ = "log_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    byte_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_data: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    created_at: Mapped[float] = mapped_column(nullable=False)
