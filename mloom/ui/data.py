"""
Read-only queries behind the dashboard.

Everything returns plain pandas DataFrames / dicts so the Streamlit layer can
cache results and never holds on to ORM objects or open sessions.
"""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import select

from mloom.db import database
from mloom.db.classes import Run, LLMMetrics

STATUS_OK = "OK"
STATUS_FAILED = "Failed"

RUN_COLUMNS = [
    "id", "project_id", "run_name", "run_type", "latency", "status", "error_type",
    "llm_calls", "tokens", "cost", "start",
]
METRIC_COLUMNS = [
    "id", "run_id", "model_name", "input_tokens", "output_tokens", "tokens",
    "total_cost", "latency", "is_llm",
]


def to_local(value: Optional[datetime]) -> Optional[datetime]:
    """Naive UTC (as stored) -> naive local time for display."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    return value.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


def _session(db_url: str):
    database.init_db(db_url)
    return database.get_session()


def load_metrics(db_url: str) -> pd.DataFrame:
    """One row per metric, without the (potentially large) prompt/response text."""
    with _session(db_url) as db:
        rows = db.execute(select(
            LLMMetrics.id, LLMMetrics.run_id, LLMMetrics.model_name,
            LLMMetrics.input_tokens, LLMMetrics.output_tokens,
            LLMMetrics.total_cost, LLMMetrics.latency,
        )).all()

    df = pd.DataFrame(rows, columns=["id", "run_id", "model_name", "input_tokens",
                                     "output_tokens", "total_cost", "latency"])
    if df.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    # Metrics with token counts come from LLM calls; @track_metric steps have none.
    df["is_llm"] = df["input_tokens"].notna() | df["output_tokens"].notna()
    df["input_tokens"] = df["input_tokens"].fillna(0).astype("int64")
    df["output_tokens"] = df["output_tokens"].fillna(0).astype("int64")
    df["tokens"] = df["input_tokens"] + df["output_tokens"]
    df["total_cost"] = df["total_cost"].fillna(0.0).astype("float64")
    df["model_name"] = df["model_name"].fillna("unknown")
    return df[METRIC_COLUMNS]


def load_runs(db_url: str, metrics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per run with per-run LLM call, token and cost totals."""
    error_type = Run.error_trace["error_type"].as_string()
    with _session(db_url) as db:
        rows = db.execute(select(
            Run.id, Run.project_id, Run.run_name, Run.run_type, Run.latency,
            Run.start_time, Run.timestamp, error_type.label("error_type"),
        )).all()

    if not rows:
        return pd.DataFrame(columns=RUN_COLUMNS)

    df = pd.DataFrame(rows, columns=["id", "project_id", "run_name", "run_type", "latency",
                                     "start_time", "timestamp", "error_type"])
    df["run_name"] = df["run_name"].fillna("")
    df["status"] = df["error_type"].map(lambda e: STATUS_OK if e is None or pd.isna(e) else STATUS_FAILED)
    df["start"] = [
        to_local(start if start is not None and not pd.isna(start) else ts)
        for start, ts in zip(df["start_time"], df["timestamp"])
    ]
    df["start"] = pd.to_datetime(df["start"])

    if metrics is None:
        metrics = load_metrics(db_url)
    if metrics.empty:
        df["llm_calls"], df["tokens"], df["cost"] = 0, 0, 0.0
    else:
        totals = metrics.groupby("run_id").agg(
            llm_calls=("is_llm", "sum"), tokens=("tokens", "sum"), cost=("total_cost", "sum"),
        )
        df = df.merge(totals, how="left", left_on="id", right_index=True)
        df["llm_calls"] = df["llm_calls"].fillna(0).astype("int64")
        df["tokens"] = df["tokens"].fillna(0).astype("int64")
        df["cost"] = df["cost"].fillna(0.0).astype("float64")

    return df[RUN_COLUMNS].sort_values("id", ascending=False).reset_index(drop=True)


def load_dataset(db_url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(runs, metrics) for the whole database."""
    metrics = load_metrics(db_url)
    return load_runs(db_url, metrics), metrics


def load_run_detail(db_url: str, run_id: int) -> Optional[dict]:
    """Everything about one run, including prompts, responses and its error trace."""
    with _session(db_url) as db:
        run = db.get(Run, run_id)
        if run is None:
            return None
        metrics = sorted(run.metrics, key=lambda m: m.id)
        return {
            "id": run.id,
            "project_id": run.project_id,
            "run_name": run.run_name or "",
            "run_type": run.run_type,
            "latency": run.latency,
            "start": to_local(run.start_time or run.timestamp),
            "end": to_local(run.end_time),
            "tags": run.tags or {},
            "error_trace": run.error_trace or None,
            "call_site": run.call_site or None,
            "metrics": [
                {
                    "id": m.id,
                    "model_name": m.model_name or "unknown",
                    "prompt": m.prompt or "",
                    "response": m.response or "",
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "total_cost": m.total_cost,
                    "latency": m.latency,
                    "start": to_local(m.start_time),
                    "is_llm": m.input_tokens is not None or m.output_tokens is not None,
                }
                for m in metrics
            ],
        }
