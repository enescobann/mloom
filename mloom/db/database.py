import os
import threading
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mloom.core.config import config
from mloom.db.classes import Base

# Initialize these as None at the module level.
# They will be populated when the library is explicitly initialized.
engine = None
SessionLocal = None
db_url: Optional[str] = None

_lock = threading.Lock()


def local_db_url() -> str:
    """SQLite file used by the SDK in local mode (~/.mloom/local.db)."""
    return config.local_db_url


def server_db_url() -> str:
    """DATABASE_URL for the server, or a development SQLite file under ~/.mloom."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{config.CONFIG_DIR / 'server.db'}"


SQLITE_BUSY_TIMEOUT_MS = 5000


def _set_sqlite_pragmas(dbapi_connection, _connection_record):
    # WAL lets the dashboard read while the SDK writes; busy_timeout makes
    # concurrent writers (several scripts, server + SDK) wait instead of failing.
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    finally:
        cursor.close()


def _create_engine(url: str):
    if make_url(url).get_backend_name() == "sqlite":
        kwargs = {"connect_args": {"check_same_thread": False}}
        in_memory = make_url(url).database in (None, "", ":memory:")
        if in_memory:
            # A single shared connection, otherwise every session sees an empty DB.
            kwargs["poolclass"] = StaticPool
        new_engine = create_engine(url, **kwargs)
        if not in_memory:
            event.listen(new_engine, "connect", _set_sqlite_pragmas)
        return new_engine

    return create_engine(
        url,
        pool_pre_ping=True,      # Prevents stale connection crashes
        pool_size=20,            # Higher baseline for API throughput
        max_overflow=30          # Allows bursting up to 50 connections
    )


def init_db(url: Optional[str] = None):
    """
    Create the engine and tables for `url` (defaults to the local SQLite file).

    Calling it again with the same URL is a no-op; a different URL replaces
    the current engine.
    """
    global engine, SessionLocal, db_url

    url = url or local_db_url()
    with _lock:
        if engine is not None and db_url == url:
            return engine
        if engine is not None:
            engine.dispose()

        new_engine = _create_engine(url)
        Base.metadata.create_all(bind=new_engine)

        engine = new_engine
        db_url = url
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=new_engine)
        return engine


def reset_db():
    """Dispose of the current engine so the next init_db() starts fresh."""
    global engine, SessionLocal, db_url
    with _lock:
        if engine is not None:
            engine.dispose()
        engine = None
        SessionLocal = None
        db_url = None


def get_session():
    """Return a new Session bound to the current engine."""
    if SessionLocal is None:
        raise RuntimeError("[Mloom] Database not initialized. Call mloom.init() first.")
    return SessionLocal()


def get_db():
    """
    Dependency generator for FastAPI routes or context managers.
    Ensures the DB is initialized before yielding a session.
    """
    db = get_session()
    try:
        yield db
    finally:
        db.close()
