import asyncio
import json
from datetime import datetime

import pytest

import mloom
from mloom import track_metric, track_run
from mloom.core.config import config
from mloom.db import database


# ─── init() / config ─────────────────────────────────────────────────

def test_public_api_exports():
    for name in ["track_run", "track_llm", "track_metric", "init", "autotrack", "flush", "config"]:
        assert hasattr(mloom, name)
    assert mloom.track_llm is mloom.track_run


def test_init_defaults_to_local_and_creates_db(mloom_home):
    mloom.init(project_id="p1")

    assert config.mode == "local"
    assert database.engine is not None
    assert (mloom_home / "local.db").exists()


def test_init_remote_never_needs_a_database(mloom_home):
    mloom.init(project_id="p1", mode="remote", backend_url="http://example.invalid:8000/")

    assert config.mode == "remote"
    assert config.backend_url == "http://example.invalid:8000"
    assert database.engine is None
    assert not (mloom_home / "local.db").exists()


def test_init_uses_mode_saved_by_server_connect(mloom_home):
    mloom_home.mkdir(parents=True)
    (mloom_home / "config.json").write_text(json.dumps(
        {"mode": "remote", "backend_url": "http://10.0.0.5:8000", "project_id": None}
    ))

    mloom.init(project_id="p1")

    assert config.mode == "remote"
    assert config.backend_url == "http://10.0.0.5:8000"


def test_init_rejects_unknown_mode():
    with pytest.raises(ValueError):
        mloom.init(project_id="p1", mode="cloud")


def test_init_coerces_project_id_to_str():
    mloom.init(project_id=42)
    assert config.project_id == "42"


def test_config_dir_falls_back_to_home(monkeypatch, tmp_path):
    monkeypatch.delenv("MLOOM_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config.CONFIG_DIR == tmp_path / ".mloom"


def test_init_db_reuses_engine_for_same_url_and_swaps_for_new_one(tmp_path):
    first = database.init_db(f"sqlite:///{tmp_path / 'a.db'}")
    assert database.init_db(f"sqlite:///{tmp_path / 'a.db'}") is first

    second = database.init_db(f"sqlite:///{tmp_path / 'b.db'}")
    assert second is not first
    assert database.db_url.endswith("b.db")

    database.reset_db()
    assert database.engine is None and database.SessionLocal is None


# ─── track_run ───────────────────────────────────────────────────────

def test_sync_track_run_records_run(local_project, fetch_runs):
    @track_run(name="sync-pipeline")
    def pipeline(x):
        return x * 2

    assert pipeline(21) == 42

    [run] = fetch_runs()
    assert run["project_id"] == local_project
    assert run["run_name"] == "sync-pipeline"
    assert run["run_type"] == "LLM"
    assert run["error_trace"] is None
    assert run["latency"] >= 0
    assert isinstance(run["start_time"], datetime)
    assert isinstance(run["end_time"], datetime)
    assert run["start_time"] <= run["end_time"]
    assert run["call_site"]["caller_function"] == "test_sync_track_run_records_run"
    assert run["call_site"]["caller_file"] == __file__


def test_bare_decorator_uses_function_name(local_project, fetch_runs):
    @track_run
    def my_job():
        return "done"

    assert my_job() == "done"
    [run] = fetch_runs()
    assert run["run_name"] == "my_job"


def test_async_track_run_records_run(local_project, fetch_runs):
    @track_run(name="async-pipeline", run_type="CLASSIC")
    async def pipeline(x):
        await asyncio.sleep(0.01)
        return x + 1

    assert asyncio.run(pipeline(1)) == 2

    [run] = fetch_runs()
    assert run["run_name"] == "async-pipeline"
    assert run["run_type"] == "CLASSIC"
    assert run["latency"] >= 10
    assert run["start_time"] is not None and run["end_time"] is not None


def test_nested_metrics_are_collected(local_project, fetch_runs):
    @track_metric("retrieval")
    def retrieve(query):
        return ["doc-a", "doc-b"]

    @track_metric("async-rerank")
    async def rerank(docs):
        return list(reversed(docs))

    @track_metric
    def postprocess(text):
        return text.upper()

    @track_run(name="rag")
    def rag(query):
        docs = retrieve(query)
        docs = asyncio.run(rerank(docs))
        return postprocess(" ".join(docs))

    assert rag("hello") == "DOC-B DOC-A"

    [run] = fetch_runs()
    names = [m["model_name"] for m in run["metrics"]]
    assert names == ["retrieval", "async-rerank", "custom_metric"]

    retrieval = run["metrics"][0]
    assert "hello" in retrieval["prompt"]
    assert "doc-a" in retrieval["response"]
    assert retrieval["start_time"] is not None and retrieval["end_time"] is not None


def test_track_metric_outside_run_is_ignored(local_project, fetch_runs):
    @track_metric("lonely")
    def lonely():
        return 1

    assert lonely() == 1
    assert fetch_runs() == []


def test_failed_run_stores_error_trace_and_redacts_secrets(local_project, fetch_runs):
    @track_run(name="failing")
    def failing():
        openai_api_key = "sk-very-secret"
        db_password = "hunter2"
        user_count = 3
        raise RuntimeError("model exploded")

    with pytest.raises(RuntimeError, match="model exploded"):
        failing()

    [run] = fetch_runs()
    trace = run["error_trace"]
    assert trace["error_type"] == "RuntimeError"
    assert trace["error_message"] == "model exploded"
    assert "RuntimeError" in trace["traceback_formatted"]

    innermost = trace["frames"][-1]
    assert innermost["function"] == "failing"
    assert innermost["locals"]["openai_api_key"] == "[REDACTED]"
    assert innermost["locals"]["db_password"] == "[REDACTED]"
    assert innermost["locals"]["user_count"] == "3"
    # Source lines are kept verbatim; only captured variable values are redacted.
    assert "sk-very-secret" not in json.dumps(trace["frames"])
    assert "hunter2" not in json.dumps(trace["frames"])

    assert trace["source_context"]["error_line"] == innermost["lineno"]
    assert run["call_site"]["caller_function"] == "test_failed_run_stores_error_trace_and_redacts_secrets"


def test_failed_metric_is_recorded_and_reraised(local_project, fetch_runs):
    @track_metric("flaky-tool")
    def flaky():
        raise ValueError("tool failed")

    @track_run(name="agent")
    def agent():
        try:
            flaky()
        except ValueError:
            return "recovered"

    assert agent() == "recovered"

    [run] = fetch_runs()
    assert run["error_trace"] is None
    [metric] = run["metrics"]
    assert metric["response"] == "ERROR: tool failed"


def test_async_failure_is_recorded(local_project, fetch_runs):
    @track_run(name="async-fail")
    async def boom():
        token = "abc123"
        raise KeyError("missing")

    with pytest.raises(KeyError):
        asyncio.run(boom())

    [run] = fetch_runs()
    assert run["error_trace"]["error_type"] == "KeyError"
    assert run["error_trace"]["frames"][-1]["locals"]["token"] == "[REDACTED]"


# ─── emitter ─────────────────────────────────────────────────────────

def test_flush_can_be_called_repeatedly(local_project, fetch_runs):
    @track_run
    def job():
        return 1

    job()
    assert mloom.flush(timeout=10)
    job()
    assert mloom.flush(timeout=10)
    job()

    assert len(fetch_runs()) == 3


def test_runs_for_multiple_projects_are_all_stored(fetch_runs):
    @track_run
    def job():
        return 1

    mloom.init(project_id="alpha")
    job()
    mloom.flush(timeout=10)
    mloom.init(project_id="beta")
    job()

    runs = fetch_runs()
    assert [r["project_id"] for r in runs] == ["alpha", "beta"]

    from mloom.db.classes import Project
    with database.get_session() as db:
        assert {p.id for p in db.query(Project)} == {"alpha", "beta"}


def test_missing_project_id_drops_event(capsys):
    from mloom.sdk.emitter import emit_event

    emit_event({"run_name": "orphan", "metrics": []})
    assert mloom.flush(timeout=10)
    assert "project_id" in capsys.readouterr().err


# The emitter passes a requests-style timeout, which TestClient warns about.
@pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")
def test_remote_mode_posts_runs_to_server(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from mloom.sdk import emitter
    from mloom.server.main import app

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'server.db'}")
    with TestClient(app) as client:
        monkeypatch.setattr(emitter, "session", client)
        mloom.init(project_id="remote-proj", mode="remote", backend_url="http://testserver")

        @track_metric("step")
        def step():
            return "ok"

        @track_run(name="remote-run")
        def job():
            return step()

        job()
        assert mloom.flush(timeout=10)

        runs = client.get("/projects/remote-proj/runs").json()

    [run] = runs
    assert run["run_name"] == "remote-run"
    assert run["start_time"] is not None and run["end_time"] is not None
    assert run["call_site"]["caller_function"] == "test_remote_mode_posts_runs_to_server"
    assert [m["model_name"] for m in run["metrics"]] == ["step"]
    assert run["metrics"][0]["start_time"] is not None


def test_sqlite_file_databases_use_wal(tmp_path):
    from sqlalchemy import text

    engine = database.init_db(f"sqlite:///{tmp_path / 'wal.db'}")
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == database.SQLITE_BUSY_TIMEOUT_MS
