"""Database engine, session, and declarative base setup.

Creates the SQLAlchemy engine bound to the SQLite database at DATABASE_PATH,
configures WAL mode and NORMAL synchronous pragma on connection, and provides
the Base declarative model class along with an init_db helper.
"""

from config import DATABASE_PATH
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


def _on_connect(dbapi_connection, _connection_record):
    """Configure WAL journal mode and NORMAL synchronous on connection."""
    dbapi_connection.execute("PRAGMA journal_mode=WAL")
    dbapi_connection.execute("PRAGMA synchronous=NORMAL")


event.listen(engine, "connect", _on_connect)


def init_db():
    """Create the database directory and all tables if they do not exist."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
