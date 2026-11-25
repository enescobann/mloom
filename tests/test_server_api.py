import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from mloom.db import database


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'server.db'}")
    from mloom.server.main import app

    with TestClient(app) as test_client:
        yield test_client


def _run_payload(project_id="proj-1", name="run", **extra):
    payload = {
        "project_id": project_id,
        "run_name": name,
        "run_type": "LLM",
        "latency": 250,
        "tags": {"env": "test"},
        "start_time": "2026-03-01T12:00:00+00:00",
        "end_time": "2026-03-01T12:00:00.250000+00:00",
        "call_site": {"caller_file": "app.py", "caller_line": 10, "caller_function": "main"},
        "metrics": [
            {"model_name": "gpt-4o", "prompt": "hi", "response": "hello",
             "input_tokens": 10, "output_tokens": 5, "total_cost": 0.000125, "latency": 200,
             "start_time": "2026-03-01T12:00:00.010000+00:00",
             "end_time": "2026-03-01T12:00:00.210000+00:00"},
            {"model_name": "retrieval", "prompt": "q", "response": "docs", "latency": 30},
        ],
    }
    payload.update(extra)
    return payload


def test_import_does_not_need_a_database(mloom_home):
    # Fresh interpreter: importing the app must not touch any database.
    code = (
        "import mloom.server.main\n"
        "from mloom.db import database\n"
        "assert database.engine is None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
    assert not mloom_home.exists() or not any(mloom_home.glob("*.db"))


def test_lifespan_falls_back_to_sqlite_under_mloom_home(mloom_home):
    from mloom.server.main import app

    with TestClient(app) as test_client:
        assert test_client.get("/projects/").json() == []
    assert database.db_url == f"sqlite:///{mloom_home / 'server.db'}"


def test_create_run_with_metrics_auto_creates_project(client):
    response = client.post("/runs/", json=_run_payload())

    assert response.status_code == 200, response.text
    run = response.json()
    assert run["project_id"] == "proj-1"
    assert run["run_name"] == "run"
    assert run["tags"] == {"env": "test"}
    assert run["call_site"]["caller_function"] == "main"
    assert run["start_time"] == "2026-03-01T12:00:00"
    assert run["end_time"] == "2026-03-01T12:00:00.250000"
    assert run["timestamp"] is not None

    gpt, retrieval = run["metrics"]
    assert gpt["input_tokens"] == 10 and gpt["output_tokens"] == 5
    assert gpt["total_cost"] == pytest.approx(0.000125)
    assert gpt["start_time"] == "2026-03-01T12:00:00.010000"
    # Custom metrics without token counts get no invented tokens or cost.
    assert retrieval["input_tokens"] is None
    assert retrieval["total_cost"] is None

    project = client.get("/projects/proj-1")
    assert project.status_code == 200
    assert project.json()["id"] == "proj-1"


def test_create_run_accepts_numeric_project_id(client):
    response = client.post("/runs/", json=_run_payload(project_id=7))

    assert response.status_code == 200, response.text
    assert response.json()["project_id"] == "7"


def test_create_run_stores_error_trace(client):
    error_trace = {"error_type": "ValueError", "error_message": "nope", "frames": []}
    response = client.post("/runs/", json=_run_payload(error_trace=error_trace))

    assert response.status_code == 200
    run_id = response.json()["id"]
    assert client.get(f"/runs/{run_id}").json()["error_trace"] == error_trace


def test_create_project_derives_id_from_name(client):
    response = client.post("/projects/", json={"name": "My Chat Bot!"})

    assert response.status_code == 200, response.text
    assert response.json()["id"] == "my-chat-bot"
    assert response.json()["name"] == "My Chat Bot!"


def test_create_project_with_explicit_id_and_conflicts(client):
    response = client.post("/projects/", json={"id": "bot", "name": "Bot"})
    assert response.status_code == 200
    assert response.json()["id"] == "bot"

    assert client.post("/projects/", json={"id": "bot", "name": "Other"}).status_code == 409
    assert client.post("/projects/", json={"id": "bot-2", "name": "Bot"}).status_code == 409


def test_update_and_delete_project(client):
    client.post("/runs/", json=_run_payload(project_id="tmp"))

    response = client.put("/projects/tmp", json={"name": "Renamed"})
    assert response.status_code == 200
    assert response.json()["name"] == "Renamed"

    assert client.delete("/projects/tmp").status_code == 200
    assert client.get("/projects/tmp").status_code == 404
    assert client.get("/runs/1").status_code == 404


def test_list_projects_with_run_counts(client):
    client.post("/runs/", json=_run_payload(project_id="a"))
    client.post("/runs/", json=_run_payload(project_id="a"))
    client.post("/runs/", json=_run_payload(project_id="b"))
    client.post("/projects/", json={"id": "empty", "name": "Empty"})

    response = client.get("/projects/")

    assert response.status_code == 200
    counts = {p["id"]: p["run_count"] for p in response.json()}
    assert counts == {"a": 2, "b": 1, "empty": 0}

    assert len(client.get("/projects/", params={"limit": 2}).json()) == 2
    assert len(client.get("/projects/", params={"offset": 2}).json()) == 1


def test_list_project_runs_newest_first_with_pagination(client):
    for i in range(5):
        client.post("/runs/", json=_run_payload(project_id="p", name=f"run-{i}"))
    client.post("/runs/", json=_run_payload(project_id="other", name="other-run"))

    response = client.get("/projects/p/runs")
    assert response.status_code == 200
    names = [r["run_name"] for r in response.json()]
    assert names == ["run-4", "run-3", "run-2", "run-1", "run-0"]
    assert len(response.json()[0]["metrics"]) == 2

    page = client.get("/projects/p/runs", params={"limit": 2, "offset": 1}).json()
    assert [r["run_name"] for r in page] == ["run-3", "run-2"]


def test_list_runs_validation_and_missing_project(client):
    assert client.get("/projects/missing/runs").status_code == 404
    client.post("/runs/", json=_run_payload(project_id="p"))
    assert client.get("/projects/p/runs", params={"limit": 0}).status_code == 422
    assert client.get("/projects/p/runs", params={"offset": -1}).status_code == 422


def test_get_and_delete_metric(client):
    run = client.post("/runs/", json=_run_payload()).json()
    metric_id = run["metrics"][0]["id"]

    response = client.get(f"/runs/{run['id']}/metrics/{metric_id}")
    assert response.status_code == 200
    assert response.json()["model_name"] == "gpt-4o"

    assert client.delete(f"/runs/{run['id']}/metrics/{metric_id}").status_code == 200
    assert client.get(f"/runs/{run['id']}/metrics/{metric_id}").status_code == 404


def test_delete_run(client):
    run = client.post("/runs/", json=_run_payload()).json()

    assert client.delete(f"/runs/{run['id']}").status_code == 200
    assert client.get(f"/runs/{run['id']}").status_code == 404
    assert client.delete(f"/runs/{run['id']}").status_code == 404
