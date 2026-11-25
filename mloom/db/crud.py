"""
Persistence helpers shared by the local-mode emitter and the server, so both
modes store runs exactly the same way.
"""

from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mloom.db.classes import Project, Run, LLMMetrics


def to_naive_utc(value):
    """Normalise an ISO string / epoch / datetime into a naive UTC datetime."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str):
        try:
            # Python 3.10's fromisoformat does not accept a trailing "Z".
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def get_or_create_project(db: Session, project_id: str) -> Project:
    """Fetch a project, creating it on first use. Safe against concurrent creators."""
    project = db.get(Project, project_id)
    if project is not None:
        return project

    db.add(Project(id=project_id, name=f"Project_{project_id}"))
    try:
        db.commit()
    except IntegrityError:
        # Another writer created it between our SELECT and INSERT.
        db.rollback()
    project = db.get(Project, project_id)
    if project is None:
        raise RuntimeError(f"[Mloom] Could not create project {project_id!r}.")
    return project


def create_run_record(db: Session, data: dict) -> Run:
    """Insert a run and its metrics in a single transaction and return the Run."""
    project_id = str(data["project_id"])
    get_or_create_project(db, project_id)

    run = Run(
        run_name=data.get("run_name") or "Unnamed Run",
        project_id=project_id,
        run_type=data.get("run_type") or "LLM",
        latency=data.get("latency", data.get("latency_ms")),
        tags=data.get("tags") or {},
        error_trace=data.get("error_trace"),
        call_site=data.get("call_site"),
        start_time=to_naive_utc(data.get("start_time")),
        end_time=to_naive_utc(data.get("end_time")),
    )

    for metric_data in data.get("metrics") or []:
        run.metrics.append(LLMMetrics(
            model_name=metric_data.get("model_name"),
            prompt=metric_data.get("prompt"),
            response=metric_data.get("response"),
            input_tokens=metric_data.get("input_tokens"),
            output_tokens=metric_data.get("output_tokens"),
            total_cost=metric_data.get("total_cost"),
            latency=metric_data.get("latency"),
            start_time=to_naive_utc(metric_data.get("start_time")),
            end_time=to_naive_utc(metric_data.get("end_time")),
        ))

    db.add(run)
    db.commit()
    db.refresh(run)
    return run
