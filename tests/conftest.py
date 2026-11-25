import pytest

# Manual end-to-end scripts that need live servers (LM Studio, a running API).
collect_ignore = [
    "real_world_test.py",
    "test_e2e.py",
    "test_mloom_sdk.py",
    "test_real_scenario.py",
    "test_trace.py",
]


def _reset_state():
    from mloom.core.config import config
    from mloom.db import database
    from mloom.sdk.emitter import flush

    assert flush(timeout=10), "emitter queue did not drain"
    config.reset()
    database.reset_db()


@pytest.fixture(autouse=True)
def mloom_home(tmp_path, monkeypatch):
    """Point HOME / MLOOM_HOME at a temp dir so tests never touch ~/.mloom."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MLOOM_HOME", str(home / ".mloom"))
    monkeypatch.delenv("DATABASE_URL", raising=False)

    _reset_state()
    yield home / ".mloom"
    _reset_state()


@pytest.fixture
def local_project():
    """Initialise the SDK in local mode and return the project id."""
    import mloom

    mloom.init(project_id="test-project", mode="local")
    return "test-project"


@pytest.fixture
def fetch_runs():
    """Flush pending events and return all runs (newest last) as plain dicts."""
    import mloom
    from mloom.db import database
    from mloom.db.classes import Run

    def _fetch():
        assert mloom.flush(timeout=10)
        with database.get_session() as db:
            runs = db.query(Run).order_by(Run.id).all()
            return [
                {
                    "id": r.id,
                    "project_id": r.project_id,
                    "run_name": r.run_name,
                    "run_type": r.run_type,
                    "latency": r.latency,
                    "tags": r.tags,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "error_trace": r.error_trace,
                    "call_site": r.call_site,
                    "metrics": [
                        {
                            "model_name": m.model_name,
                            "prompt": m.prompt,
                            "response": m.response,
                            "input_tokens": m.input_tokens,
                            "output_tokens": m.output_tokens,
                            "total_cost": m.total_cost,
                            "latency": m.latency,
                            "start_time": m.start_time,
                            "end_time": m.end_time,
                        }
                        for m in sorted(r.metrics, key=lambda m: m.id)
                    ],
                }
                for r in runs
            ]

    return _fetch
