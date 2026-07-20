"""Database engine, session, and declarative base setup.

Creates the SQLAlchemy engine bound to the SQLite database at DATABASE_PATH,
configures WAL mode and NORMAL synchronous pragma on connection, and provides
the Base declarative model class along with an init_db helper.
"""

from config import DATABASE_PATH, PREPROCESS_LOGS_DIR, TASK_LOGS_DIR
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


# check_same_thread=False: the default QueuePool reuses connections across the
# HTTP threadpool and the scheduler/reader/recover worker threads, which the
# sqlite3 default (check_same_thread=True) rejects. WAL + busy_timeout make
# concurrent writes wait rather than fail.
engine = create_engine(
    f"sqlite:///{DATABASE_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine)


def _on_connect(dbapi_connection, _connection_record):
    """Configure WAL, NORMAL synchronous, and a busy timeout on connection."""
    dbapi_connection.execute("PRAGMA journal_mode=WAL")
    dbapi_connection.execute("PRAGMA synchronous=NORMAL")
    dbapi_connection.execute("PRAGMA busy_timeout=5000")


event.listen(engine, "connect", _on_connect)


def init_db():
    """Create the database directory, log directories, and tables if absent."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TASK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
