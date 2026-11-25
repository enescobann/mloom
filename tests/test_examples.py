import os
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mloom.db import database
from mloom.db.classes import Run

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run_example(name: str, home: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "MLOOM_HOME": str(home)}
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "[Mloom Error]" not in result.stderr
    assert "[Mloom Warning]" not in result.stderr
    return result


def stored_runs(home: Path) -> list:
    database.init_db(f"sqlite:///{home / 'local.db'}")
    with database.get_session() as db:
        return [
            {
                "project_id": r.project_id,
                "run_name": r.run_name,
                "failed": bool(r.error_trace),
                "error_trace": r.error_trace,
                "start_time": r.start_time,
                "tokens": sum((m.input_tokens or 0) + (m.output_tokens or 0) for m in r.metrics),
                "cost": sum(m.total_cost or 0 for m in r.metrics),
            }
            for r in db.query(Run).order_by(Run.id)
        ]


def test_quickstart_runs_offline(mloom_home):
    result = run_example("quickstart.py", mloom_home)

    assert "mloom ui" in result.stdout
    runs = stored_runs(mloom_home)
    assert [r["run_name"] for r in runs] == ["answer_question"] * 3 + ["sync_knowledge_base"]
    assert [r["failed"] for r in runs] == [False, False, False, True]
    locals_ = runs[-1]["error_trace"]["frames"][-1]["locals"]
    assert locals_["api_key"] == "[REDACTED]"
    assert locals_["fetched_pages"] == "0"


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_demo_data_populates_dashboard(mloom_home):
    result = run_example("demo_data.py", mloom_home)

    assert "Recorded 40 runs" in result.stdout
    runs = stored_runs(mloom_home)
    assert len(runs) == 40
    assert {r["project_id"] for r in runs} == {"support-bot", "doc-search", "eval-harness"}
    assert 2 <= sum(r["failed"] for r in runs) <= 12
    assert sum(r["tokens"] for r in runs) > 0 and sum(r["cost"] for r in runs) > 0
    starts = [r["start_time"] for r in runs]
    assert starts == sorted(starts)
    assert (starts[-1] - starts[0]).total_seconds() > 86400

    # The generated data renders without errors in every section of the dashboard.
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "mloom" / "ui" / "app.py"),
                           default_timeout=60).run()
    assert not at.exception
    assert len(at.get("vega_lite_chart")) == 4
    for run_id in at.dataframe[0].value["id"].tolist():
        at.selectbox(key="detail_run_id").set_value(run_id).run()
        assert not at.exception, run_id
        assert not at.warning, run_id
