import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from mloom.core.config import config
from mloom.db.classes import Base

# Initialize these as None at the module level.
# They will be populated when the library is explicitly initialized.
engine = None
SessionLocal = None

def init_db():
    """
    Called internally by mloom.init() to establish the DB connection 
    ONLY AFTER the config has been fully populated.
    """
    global engine, SessionLocal
    
    # Prevent double-initialization
    if engine is not None:
        return

    if config.mode == "local":
        db_url = f"sqlite:///{config.local_db_path}"
        engine = create_engine(
            db_url, 
            connect_args={"check_same_thread": False}
        )
        # Create tables safely now that we know the path is correct
        Base.metadata.create_all(bind=engine)
        
    else:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError(
                "[Mloom] DATABASE_URL environment variable is not set. "
                "In remote mode, you must provide a valid PostgreSQL connection string."
            )
        engine = create_engine(
            db_url,
            pool_pre_ping=True,      # Prevents stale connection crashes
            pool_size=20,            # Higher baseline for API throughput
            max_overflow=30          # Allows bursting up to 50 connections
        )

    # Bind the sessionmaker to the newly created engine
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency generator for FastAPI routes or context managers.
    Ensures the DB is initialized before yielding a session.
    """
    if SessionLocal is None:
        raise RuntimeError("[Mloom] Database not initialized. Call mloom.init() first.")
        
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()