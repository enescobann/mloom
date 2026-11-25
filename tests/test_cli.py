import json
import sys

import pytest
from click.testing import CliRunner

import mloom
from mloom import cli as cli_module
from mloom.cli import cli
from mloom.sdk.cost import calculate_cost
from mloom.sdk.emitter import emit_event


@pytest.fixture
def runner():
    return CliRunner()


def _record_runs():
    """Two runs in local mode: one OK with LLM metrics, one failed."""
    emit_event({
        "project_id": "chatbot",
        "run_name": "answer-question",
        "run_type": "LLM",
        "latency": 120,
        "start_time": "2026-01-01T10:00:00+00:00",
        "end_time": "2026-01-01T10:00:00.120000+00:00",
        "metrics": [
            {"model_name": "gpt-4o", "input_tokens": 100, "output_tokens": 50,
             "total_cost": calculate_cost("gpt-4o", 100, 50), "latency": 80},
            {"model_name": "retrieval", "latency": 20},
        ],
    })

    @mloom.track_run(name="broken-run")
    def broken():
        secret_token = "tok-123"
        attempts = 2
        raise ValueError("bad input")

    mloom.init(project_id="other")
    with pytest.raises(ValueError):
        broken()
    assert mloom.flush(timeout=10)


# ─── server connect ──────────────────────────────────────────────────

def test_server_connect_remote_writes_config(runner, mloom_home):
    result = runner.invoke(cli, ["server", "connect", "http://192.168.1.5:8000/"])

    assert result.exit_code == 0, result.output
    assert "Remote Mode" in result.output
    saved = json.loads((mloom_home / "config.json").read_text())
    assert saved["mode"] == "remote"
    assert saved["backend_url"] == "http://192.168.1.5:8000"


def test_server_connect_adds_scheme(runner, mloom_home):
    result = runner.invoke(cli, ["server", "connect", "10.0.0.7:9000"])

    assert result.exit_code == 0, result.output
    saved = json.loads((mloom_home / "config.json").read_text())
    assert saved["backend_url"] == "http://10.0.0.7:9000"


def test_server_connect_local_writes_config(runner, mloom_home):
    runner.invoke(cli, ["server", "connect", "http://example.com"])
    result = runner.invoke(cli, ["server", "connect", "local"])

    assert result.exit_code == 0, result.output
    assert "Local SQLite" in result.output
    saved = json.loads((mloom_home / "config.json").read_text())
    assert saved["mode"] == "local"
    # The remote URL is remembered for a later switch back.
    assert saved["backend_url"] == "http://example.com"


# ─── runs ────────────────────────────────────────────────────────────

def test_runs_empty(runner):
    result = runner.invoke(cli, ["runs"])

    assert result.exit_code == 0, result.output
    assert "No runs found." in result.output


def test_runs_lists_recent_runs(runner, local_project):
    _record_runs()

    result = runner.invoke(cli, ["runs"])

    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0].split() == ["ID", "NAME", "STATUS", "LATENCY", "TOKENS", "COST", "TIME"]
    # Newest first
    assert "broken-run" in lines[1] and "FAILED" in lines[1]
    ok_row = lines[2].split()
    assert ok_row[:6] == ["1", "answer-question", "OK", "120ms", "150", f"${calculate_cost('gpt-4o', 100, 50):.6f}"]


def test_runs_filters_by_project_and_limit(runner, local_project):
    _record_runs()

    result = runner.invoke(cli, ["runs", "--project", "chatbot"])
    assert "answer-question" in result.output
    assert "broken-run" not in result.output

    result = runner.invoke(cli, ["runs", "--limit", "1"])
    assert "broken-run" in result.output
    assert "answer-question" not in result.output


# The CLI uses requests' timeout argument, which TestClient warns about.
@pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")
def test_runs_in_remote_mode_reads_from_server(runner, monkeypatch, tmp_path):
    import requests
    from fastapi.testclient import TestClient
    from mloom.server.main import app

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'server.db'}")
    with TestClient(app) as client:
        client.post("/runs/", json={
            "project_id": "svc", "run_name": "remote-run", "latency": 7,
            "metrics": [{"model_name": "gpt-4o", "input_tokens": 3, "output_tokens": 4, "total_cost": 0.5}],
        })
        monkeypatch.setattr(requests, "get", client.get)
        runner.invoke(cli, ["server", "connect", "http://testserver"])

        result = runner.invoke(cli, ["runs"])
        filtered = runner.invoke(cli, ["runs", "-p", "svc"])
        missing = runner.invoke(cli, ["runs", "-p", "nope"])
        trace = runner.invoke(cli, ["trace", "1"])

    assert result.exit_code == 0, result.output
    row = result.output.strip().splitlines()[1].split()
    assert row[:6] == ["1", "remote-run", "OK", "7ms", "7", "$0.500000"]
    assert "remote-run" in filtered.output
    assert missing.exit_code != 0 and "not found" in missing.output
    assert trace.exit_code == 0 and '"remote-run"' in trace.output


# ─── trace ───────────────────────────────────────────────────────────

def test_trace_shows_failure_details(runner, local_project):
    _record_runs()

    result = runner.invoke(cli, ["trace", "2"])

    assert result.exit_code == 0, result.output
    out = result.output
    assert '"broken-run"' in out and "FAILED" in out
    assert "ValueError" in out and "bad input" in out
    assert "Call Site:" in out and "_record_runs" in out
    assert "Stack Trace:" in out
    stack = out.split("Stack Trace:")[1].split("Source Context:")[0]
    assert "in broken" in stack
    assert "sync_wrapper" not in stack and "decorators.py" not in stack
    assert "(1 internal mloom frame hidden)" in stack
    assert "secret_token = [REDACTED]" in out
    assert "attempts = 2" in out
    assert "tok-123" not in out


def test_trace_successful_run(runner, local_project):
    _record_runs()

    result = runner.invoke(cli, ["trace", "1"])

    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "No errors recorded for this run." in result.output


def test_trace_missing_run(runner, local_project):
    result = runner.invoke(cli, ["trace", "999"])

    assert result.exit_code == 1
    assert "not found" in result.output


# ─── ui / server start ───────────────────────────────────────────────

def test_ui_launches_streamlit(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(cli_module.subprocess, "call", lambda cmd: calls.append(cmd) or 0)

    result = runner.invoke(cli, ["ui", "--port", "8600", "--no-open"])

    assert result.exit_code == 0, result.output
    [cmd] = calls
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert cmd[4].endswith("ui/app.py")
    assert cmd[5:] == [
        "--server.port", "8600",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--client.toolbarMode", "minimal",
    ]


def test_server_start_runs_uvicorn(runner, monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))

    result = runner.invoke(cli, ["server", "start", "--host", "0.0.0.0", "--port", "9001"])

    assert result.exit_code == 0, result.output
    assert calls == [("mloom.server.main:app", {"host": "0.0.0.0", "port": 9001})]
