from fastapi import FastAPI, Depends, HTTPException

from sqlalchemy.orm import Session
from database import engine, get_db

from classes import Base, Project, Run, LLMMetrics
from schemas import (
    ProjectResponse, ProjectCreate,
    RunResponse, RunCreate,
    LLMMetricsResponse, LLMMetricsCreate
)

Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.post("/projects/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    new_project = Project(name=project.name)

    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return new_project

@app.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()

    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return db_project

@app.put("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: int, project_data: ProjectCreate, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()

    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db_project.name = project_data.name
    db.commit()
    db.refresh(db_project)
    
    return db_project

@app.delete("/projects/{project_id}")
async def delete_project(project_id: int, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(db_project)
    db.commit()
    
    return {"message": "Project deleted successfully"}

#BULK POST
@app.post("/runs/", response_model=RunResponse)
async def create_run(run: RunCreate, db: Session = Depends(get_db)):
    db_project = db.query(Project).filter(Project.id == run.project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")

    new_run = Run(
        project_id=run.project_id,
        run_type=run.run_type.value,
        run_name=run.run_name,
        tags=run.tags,
        latency=run.latency
    )

    for metric_data in run.metrics:
        input_tokens = metric_data.input_tokens if metric_data.input_tokens is not None else (len(metric_data.prompt.split()) if metric_data.prompt else 0)
        output_tokens = metric_data.output_tokens if metric_data.output_tokens is not None else (len(metric_data.response.split()) if metric_data.response else 0)
        total_cost = metric_data.total_cost if metric_data.total_cost is not None else ((input_tokens * 0.00001) + (output_tokens * 0.00003))

        new_metric = LLMMetrics(
            run_id=run.id,
            model_name=metric_data.model_name,
            prompt=metric_data.prompt,
            response=metric_data.response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
            latency=metric_data.latency
        )

        new_run.metrics.append(new_metric)

    db.add(new_run)
    db.commit()
    db.refresh(new_run)

    return new_run

@app.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: int, db: Session = Depends(get_db)):
    db_run = db.query(Run).filter(Run.id == run_id).first()

    if db_run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return db_run

@app.delete("/runs/{run_id}")
async def delete_run(run_id: int, db: Session = Depends(get_db)):
    db_run = db.query(Run).filter(Run.id == run_id).first()

    if db_run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    db.delete(db_run)
    db.commit()

    return {"message": "Run deleted successfully"}

@app.get("/runs/{run_id}/metrics/{metric_id}", response_model=LLMMetricsResponse)
async def get_llm_metric(run_id: int, metric_id: int, db: Session = Depends(get_db)):
    db_metric = db.query(LLMMetrics).filter(
        LLMMetrics.id == metric_id,
        LLMMetrics.run_id == run_id
    ).first()

    if db_metric is None:
        raise HTTPException(status_code=404, detail="Metric not found")

    return db_metric

@app.delete("/runs/{run_id}/metrics/{metric_id}")
async def delete_llm_metric(run_id: int, metric_id: int, db: Session = Depends(get_db)):
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)