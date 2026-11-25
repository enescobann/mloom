import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from mloom.db.database import init_db, server_db_url, get_db
from mloom.db.crud import create_run_record

from mloom.db.classes import Project, Run, LLMMetrics
from mloom.server.schemas import (
    ProjectResponse, ProjectCreate, ProjectUpdate, ProjectSummary,
    RunResponse, RunCreate,
    LLMMetricsResponse,
)
from mloom.server.trace_logger import log_run_trace


@asynccontextmanager
async def lifespan(app: FastAPI):
    # DATABASE_URL, or a SQLite file under ~/.mloom for development.
    init_db(server_db_url())
    yield


app = FastAPI(title="mloom", lifespan=lifespan)

# Routes are plain `def` on purpose: SQLAlchemy calls are blocking, and FastAPI
# runs sync routes in a threadpool instead of stalling the event loop.


def _derive_project_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or uuid.uuid4().hex[:12]


@app.post("/projects/", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    project_id = project.id or _derive_project_id(project.name)
    if db.get(Project, project_id) is not None:
        raise HTTPException(status_code=409, detail=f"Project {project_id!r} already exists")

    new_project = Project(id=project_id, name=project.name)
    db.add(new_project)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="A project with this id or name already exists")
    db.refresh(new_project)
    return new_project

@app.get("/projects/", response_model=list[ProjectSummary])
def list_projects(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Project, func.count(Run.id))
        .outerjoin(Run, Run.project_id == Project.id)
        .group_by(Project.id)
        .order_by(Project.created_at.desc(), Project.id)
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        ProjectSummary(id=p.id, name=p.name, created_at=p.created_at, run_count=count)
        for p, count in rows
    ]

@app.get("/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)):
    db_project = db.get(Project, project_id)

    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return db_project

@app.get("/projects/{project_id}/runs", response_model=list[RunResponse])
def list_project_runs(
    project_id: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return (
        db.query(Run)
        .options(selectinload(Run.metrics))
        .filter(Run.project_id == project_id)
        .order_by(Run.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

@app.put("/projects/{project_id}", response_model=ProjectResponse)
def update_project(project_id: str, project_data: ProjectUpdate, db: Session = Depends(get_db)):
    db_project = db.get(Project, project_id)

    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")

    db_project.name = project_data.name
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="A project with this name already exists")
    db.refresh(db_project)

    return db_project

@app.delete("/projects/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_db)):
    db_project = db.get(Project, project_id)
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")

    db.delete(db_project)
    db.commit()

    return {"message": "Project deleted successfully"}

#BULK POST
@app.post("/runs/", response_model=RunResponse)
def create_run(run: RunCreate, db: Session = Depends(get_db)):
    # Missing projects are created on the fly, same as local mode.
    new_run = create_run_record(db, run.model_dump(mode="json"))

    # Live trace output to server terminal
    log_run_trace(new_run)

    return new_run

@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: int, db: Session = Depends(get_db)):
    db_run = db.get(Run, run_id)

    if db_run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return db_run

@app.delete("/runs/{run_id}")
def delete_run(run_id: int, db: Session = Depends(get_db)):
    db_run = db.get(Run, run_id)

    if db_run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    db.delete(db_run)
    db.commit()

    return {"message": "Run deleted successfully"}

@app.get("/runs/{run_id}/metrics/{metric_id}", response_model=LLMMetricsResponse)
def get_llm_metric(run_id: int, metric_id: int, db: Session = Depends(get_db)):
    db_metric = db.query(LLMMetrics).filter(
        LLMMetrics.id == metric_id,
        LLMMetrics.run_id == run_id
    ).first()

    if db_metric is None:
        raise HTTPException(status_code=404, detail="Metric not found")

    return db_metric

@app.delete("/runs/{run_id}/metrics/{metric_id}")
def delete_llm_metric(run_id: int, metric_id: int, db: Session = Depends(get_db)):
    db_metric = db.query(LLMMetrics).filter(
        LLMMetrics.id == metric_id,
        LLMMetrics.run_id == run_id
    ).first()

    if db_metric is None:
        raise HTTPException(status_code=404, detail="Metric not found")

    db.delete(db_metric)
    db.commit()
    return {"message": "Metric deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mloom.server.main:app", host="127.0.0.1", port=8000, reload=True)
