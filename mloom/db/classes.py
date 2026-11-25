from sqlalchemy import Column, String, Integer, Float, ForeignKey, JSON, DateTime, Enum, func
from sqlalchemy.orm import relationship, DeclarativeBase

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC timestamp; all DateTime columns store naive UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"
    id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=utcnow, server_default=func.now())

    #relation
    runs = relationship("Run", back_populates="project", cascade="all, delete-orphan")

class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    run_name = Column(String)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False, index=True)
    run_type = Column(Enum("LLM", "CLASSIC", name="run_type"))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    timestamp = Column(DateTime, default=utcnow, server_default=func.now())
    latency = Column(Integer)

    tags = Column(JSON)
    error_trace = Column(JSON, nullable=True)
    call_site = Column(JSON, nullable=True)

    #relation
    project = relationship("Project", back_populates="runs")
    metrics = relationship("LLMMetrics", back_populates="run", cascade="all, delete-orphan")

class LLMMetrics(Base):
    __tablename__ = "llm_metrics"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False, index=True)

    model_name = Column(String)
    prompt = Column(String)
    response = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_cost = Column(Float)
    latency = Column(Integer)

    timestamp = Column(DateTime, default=utcnow, server_default=func.now())
    start_time = Column(DateTime)
    end_time = Column(DateTime)

    #relation
    run = relationship("Run", back_populates="metrics")
