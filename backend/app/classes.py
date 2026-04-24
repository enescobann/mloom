from sqlalchemy import Column, String, Integer, Float, ForeignKey, JSON, DateTime, Enum, func
from sqlalchemy.orm import relationship, DeclarativeBase

import datetime

class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    #relation
    runs = relationship("Run", back_populates="project", cascade="all, delete-orphan")

class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    run_name = Column(String)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    run_type = Column(Enum("LLM", "CLASSIC", name="run_type"))
    timestamp = Column(DateTime, server_default=func.now())
    latency = Column(Integer)

    tags = Column(JSON)

    #relation
    project = relationship("Project", back_populates="runs")
    metrics = relationship("LLMMetrics", back_populates="run", cascade="all, delete-orphan")

class LLMMetrics(Base):
    __tablename__ = "llm_metrics"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)

    model_name = Column(String)
    prompt = Column(String)
    response = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_cost = Column(Float)
    latency = Column(Integer)
    
    #relation
    run = relationship("Run", back_populates="llm_data")
