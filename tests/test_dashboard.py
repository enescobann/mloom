import json
import warnings
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mloom
from mloom.sdk.cost import calculate_cost
from mloom.sdk.emitter import emit_event

APP_PATH = str(Path(mloom.__file__).parent / "ui" / "app.py")


def charts(at: AppTest) -> list:
    # The element type was renamed in newer Streamlit releases.
    return at.get("vega_lite_chart") + at.get("arrow_vega_lite_chart")


def run_app() -> AppTest:
    # Each test has its own MLOOM_HOME, so cached queries never leak between tests.
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    return rerun(at)


def rerun(at: AppTest) -> AppTest:
    """Run (or rerun) the app and fail on any exception, warning element or Python warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    relevant = [w for w in caught if not issubclass(w.category, ResourceWarning)]
    assert not relevant, [f"{w.category.__name__}: {w.message} ({w.filename})" for w in relevant]
    return at


def all_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def all_code(at: AppTest) -> str:
    return "\n".join(c.value for c in at.code)


@pytest.fixture
def populated():
    """An OK LLM run, a failed run with a secret local, and a run in a second project."""
    mloom.init(project_id="shop", verbose=False)

    emit_event({
        "project_id": "shop",
        "run_name": "answer_question",
        "latency": 1200,
        "start_time": "2026-01-01T10:00:00+00:00",
        "end_time": "2026-01-01T10:00:01.200000+00:00",
        "call_site": {"caller_file": "/app/bot.py", "caller_line": 42, "caller_function": "handle"},
        "metrics": [
            {"model_name": "retrieve", "prompt": "{'args': ('refunds',)}", "response": "['doc-1']",
             "latency": 30, "start_time": "2026-01-01T10:00:00.010000+00:00"},
            {"model_name": "gpt-4o", "prompt": "[{'role': 'user', 'content': 'How do refunds work?'}]",
             "response": "Refunds take 5 days.", "input_tokens": 1000, "output_tokens": 500,
             "total_cost": calculate_cost("gpt-4o", 1000, 500), "latency": 1100,
             "start_time": "2026-01-01T10:00:00.050000+00:00"},
        ],
    })

    @mloom.track_run(name="sync_inventory")
    def sync_inventory(warehouse):
        inventory_api_key = "inv-secret-123"
        retries = 3
        raise TimeoutError(f"warehouse {warehouse} did not answer")

    with pytest.raises(TimeoutError):
        sync_inventory("berlin")

    mloom.init(project_id="analytics", verbose=False)
    emit_event({
        "project_id": "analytics",
        "run_name": "weekly_report",
        "latency": 300,
        "start_time": "2026-01-03T08:00:00+00:00",
        "metrics": [
            {"model_name": "gpt-4o-mini", "prompt": "p", "response": "r", "input_tokens": 200,
             "output_tokens": 100, "total_cost": calculate_cost("gpt-4o-mini", 200, 100), "latency": 280},
        ],
    })
    assert mloom.flush(timeout=10)


# ─── Special states ────────────────────────────────────────────────────

def test_empty_state_without_database(mloom_home):
    at = run_app()

    assert "No runs recorded yet." in [i.value for i in at.info]
    assert "mloom.init" in all_code(at) and "@mloom.track_run" in all_code(at)
    assert not at.tabs
    assert not (mloom_home / "local.db").exists()


def test_empty_state_with_empty_database():
    mloom.init(project_id="nothing-yet")

    at = run_app()

    assert "No runs recorded yet." in [i.value for i in at.info]


def test_remote_mode_shows_notice(mloom_home):
    mloom_home.mkdir(parents=True)
    (mloom_home / "config.json").write_text(json.dumps(
        {"mode": "remote", "backend_url": "http://10.0.0.5:8000", "project_id": None}
    ))

    at = run_app()

    [notice] = at.info
    assert "http://10.0.0.5:8000" in notice.value and "v0.2" in notice.value
    assert "mloom runs" in all_code(at)
    assert not at.tabs


# ─── Overview ──────────────────────────────────────────────────────────

def test_overview_kpis_and_charts(populated):
    at = run_app()

    assert [t.label for t in at.tabs] == ["Overview", "Runs", "Run detail"]
    kpis = {m.label: m.value for m in at.metric[:6]}
    total_cost = calculate_cost("gpt-4o", 1000, 500) + calculate_cost("gpt-4o-mini", 200, 100)
    assert kpis == {
        "Runs": "3",
        "Failure rate": "33.3%",
        "Total tokens": "1,800",
        "Total cost": f"${total_cost:,.4f}",
        "p50 latency": kpis["p50 latency"],
        "p95 latency": kpis["p95 latency"],
    }
    assert kpis["p50 latency"].endswith(" ms") and kpis["p95 latency"].endswith(" ms")
    # Runs over time, latency distribution, tokens by model, cost by model.
    assert len(charts(at)) == 4


def test_sidebar_filters(populated):
    at = run_app()

    at.radio(key="filter_status").set_value("Failed")
    rerun(at)
    assert at.metric[0].value == "1"
    assert at.metric[1].value == "100.0%"

    at.radio(key="filter_status").set_value("All")
    at.selectbox(key="filter_project").set_value("analytics")
    rerun(at)
    assert at.metric[0].value == "1"
    assert at.selectbox(key="detail_run_id").options == ["#3 · weekly_report · OK"]

    at.selectbox(key="filter_project").set_value("All projects")
    at.text_input(key="filter_search").set_value("ANSWER")
    rerun(at)
    assert at.metric[0].value == "1"

    at.text_input(key="filter_search").set_value("no-such-run")
    rerun(at)
    assert "No runs match the current filters." in [i.value for i in at.info]


def test_refresh_button_reloads(populated):
    at = run_app()
    at.sidebar.button[0].click()
    rerun(at)
    assert at.metric[0].value == "3"


# ─── Runs table ────────────────────────────────────────────────────────

def test_runs_table_newest_first(populated):
    at = run_app()

    table = at.dataframe[0].value
    assert list(table.columns) == ["id", "run_name", "project_id", "status", "latency",
                                   "llm_calls", "tokens", "cost", "start"]
    assert table["id"].tolist() == [3, 2, 1]
    first = table.iloc[2]
    assert first["status"] == "OK"
    assert first["llm_calls"] == 1
    assert first["tokens"] == 1500
    assert table.iloc[1]["status"] == "Failed"


# ─── Run detail ────────────────────────────────────────────────────────

def test_run_detail_ok_run(populated):
    at = run_app()
    at.selectbox(key="detail_run_id").set_value(1)
    rerun(at)

    markdown = all_markdown(at)
    assert "Run #1" in markdown and ":green[**OK**]" in markdown
    assert not at.error
    assert 'File "/app/bot.py", line 42, in handle()' in all_code(at)

    detail_kpis = {m.label: m.value for m in at.metric[6:]}
    assert detail_kpis["LLM calls"] == "1"
    assert detail_kpis["Steps"] == "1"
    assert detail_kpis["Tokens"] == "1,500"

    assert [e.label for e in at.expander] == ["1. retrieve · 30 ms", "2. gpt\\-4o · 1,100 ms"]
    assert "How do refunds work?" in all_code(at)
    assert "Refunds take 5 days." in all_code(at)


def test_run_detail_failed_run(populated):
    at = run_app()
    at.selectbox(key="detail_run_id").set_value(2)
    rerun(at)

    assert ":red[**FAILED**]" in all_markdown(at)
    [error] = at.error
    assert "TimeoutError" in error.value and "berlin" in error.value
    assert "Traceback (most recent call last)" in all_code(at)

    source = next(m.value for m in at.markdown if '<td class="mloom-ln">' in m.value)
    error_row = source.split('class="mloom-error"')[1].split("</tr>")[0]
    assert "raise TimeoutError" in error_row and "&lt;- TimeoutError" in error_row

    local_vars = next(m.value for m in at.markdown if '<td class="mloom-name">' in m.value)
    assert '<span class="mloom-redacted">[REDACTED]</span>' in local_vars
    assert "inventory_api_key" in local_vars
    assert "inv-secret-123" not in local_vars
    assert "&#x27;berlin&#x27;" in local_vars and ">3<" in local_vars
    assert any("hidden because the variable name looks like a secret" in c.value for c in at.caption)


def test_every_run_detail_renders(populated):
    at = run_app()
    run_ids = at.dataframe[0].value["id"].tolist()
    assert len(run_ids) == 3
    for run_id in run_ids:
        at.selectbox(key="detail_run_id").set_value(run_id)
        rerun(at)
        assert f"Run #{run_id}" in all_markdown(at)
