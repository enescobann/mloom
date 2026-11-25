"""
mloom dashboard for local mode.

Launch with `mloom ui` (or `streamlit run mloom/ui/app.py`). It reads the
local SQLite store (~/.mloom/local.db, or $MLOOM_HOME/local.db).
"""

import html
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from mloom.core.config import config
from mloom.ui import data

CACHE_TTL_SECONDS = 10
ALL_PROJECTS = "All projects"
STATUS_FILTERS = ["All", data.STATUS_OK, data.STATUS_FAILED]
CHART_HEIGHT = 260
PRIMARY_COLOR = "#4E79A7"
FAILED_COLOR = "#E15759"
STATUS_SCALE = alt.Scale(domain=[data.STATUS_OK, data.STATUS_FAILED], range=[PRIMARY_COLOR, FAILED_COLOR])

QUICKSTART_SNIPPET = '''import mloom

mloom.init(project_id="my-app")

@mloom.track_run
def answer(question):
    return f"You asked: {question}"

answer("What does mloom record?")
mloom.flush()'''

CSS = f"""
<style>
.mloom-code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.85rem;
  border: 1px solid rgba(128, 128, 128, 0.25);
  border-radius: 0.5rem;
  overflow-x: auto;
  margin-bottom: 1rem;
}}
.mloom-code table {{ border-collapse: collapse; width: 100%; margin: 0; }}
.mloom-code tr, .mloom-code td {{ border: none; background: transparent; }}
.mloom-code td {{ padding: 0.1rem 0.75rem; white-space: pre; vertical-align: top; }}
.mloom-code td.mloom-ln {{ color: rgba(128, 128, 128, 0.9); text-align: right; width: 1%; user-select: none; }}
.mloom-code td.mloom-name {{ width: 1%; font-weight: 600; }}
.mloom-code tr.mloom-error td {{ background: rgba(225, 87, 89, 0.16); }}
.mloom-code tr.mloom-error td.mloom-ln {{ color: {FAILED_COLOR}; font-weight: 700; }}
.mloom-redacted {{
  color: {FAILED_COLOR};
  background: rgba(225, 87, 89, 0.12);
  border-radius: 0.25rem;
  padding: 0 0.35rem;
  font-weight: 600;
}}
</style>
"""


# ─── Formatting ────────────────────────────────────────────────────────

def fmt_int(value) -> str:
    return f"{int(value or 0):,}"


def fmt_ms(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f} ms"


def fmt_cost(value) -> str:
    value = float(value or 0.0)
    if value == 0 or value >= 0.01:
        return f"${value:,.4f}"
    return f"${value:,.6f}"


def fmt_time(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return value.strftime("%Y-%m-%d %H:%M:%S")


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def md_escape(text) -> str:
    """Escape user-provided text before it goes into a markdown string."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


TABLE_COLUMNS = {
    "latency": st.column_config.NumberColumn("Latency", format="%d ms"),
    "tokens": st.column_config.NumberColumn("Tokens", format="localized"),
    "cost": st.column_config.NumberColumn("Cost", format="$%.6f"),
    "start": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm:ss"),
}


# ─── Data (cached) ─────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_dataset(db_url: str):
    return data.load_dataset(db_url)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_run_detail(db_url: str, run_id: int):
    return data.load_run_detail(db_url, run_id)


# ─── Special states ────────────────────────────────────────────────────

def render_remote_notice():
    st.title("mloom")
    st.info(
        f"mloom is configured to send runs to a remote server ({config.backend_url}). "
        "This dashboard reads the local SQLite store only; a dashboard for remote "
        "servers is planned for v0.2."
    )
    st.markdown("Until then, inspect runs from the terminal:")
    st.code("mloom runs --limit 20\nmloom trace <run-id>", language="bash")
    st.caption("Switch back to local mode with `mloom server connect local`.")


def render_empty_state(db_path: Path):
    st.title("mloom")
    st.info("No runs recorded yet.")
    st.markdown("Track a function, run your script, then press **Refresh data**:")
    st.code(QUICKSTART_SNIPPET, language="python")
    st.caption(f"Runs are stored in {db_path}.")


# ─── Overview ──────────────────────────────────────────────────────────

def render_kpis(view: pd.DataFrame):
    failed = int((view["status"] == data.STATUS_FAILED).sum())
    latency = pd.to_numeric(view["latency"], errors="coerce").dropna()
    p50 = latency.quantile(0.5) if not latency.empty else None
    p95 = latency.quantile(0.95) if not latency.empty else None

    cols = st.columns(6)
    cols[0].metric("Runs", fmt_int(len(view)), border=True)
    cols[1].metric("Failure rate", f"{failed / len(view):.1%}", border=True,
                   help=f"{failed:,} of {len(view):,} runs failed")
    cols[2].metric("Total tokens", fmt_int(view["tokens"].sum()), border=True)
    cols[3].metric("Total cost", fmt_cost(view["cost"].sum()), border=True)
    cols[4].metric("p50 latency", fmt_ms(p50), border=True)
    cols[5].metric("p95 latency", fmt_ms(p95), border=True)


def runs_over_time_chart(view: pd.DataFrame):
    starts = view["start"].dropna()
    if starts.empty:
        return None

    daily = (starts.max() - starts.min()) > pd.Timedelta(days=2)
    freq, label_format = ("D", "%b %d") if daily else ("h", "%b %d %H:00")
    buckets = pd.date_range(starts.min().floor(freq), starts.max().floor(freq), freq=freq)

    counts = (
        view.dropna(subset=["start"])
        .assign(bucket=lambda df: df["start"].dt.floor(freq))
        .groupby(["bucket", "status"]).size()
        .unstack(fill_value=0)
        .reindex(index=buckets, columns=[data.STATUS_OK, data.STATUS_FAILED], fill_value=0)
        .rename_axis(index="bucket", columns=None)
        .reset_index()
        .melt(id_vars="bucket", var_name="status", value_name="runs")
    )
    counts["label"] = counts["bucket"].dt.strftime(label_format)
    order = [b.strftime(label_format) for b in buckets]

    return alt.Chart(counts).mark_bar().encode(
        x=alt.X("label:O", sort=order, title=None, axis=alt.Axis(labelAngle=0 if daily else -45)),
        y=alt.Y("runs:Q", title="Runs", stack=True),
        color=alt.Color("status:N", scale=STATUS_SCALE, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("label:O", title="Period"), alt.Tooltip("status:N", title="Status"),
                 alt.Tooltip("runs:Q", title="Runs")],
    ).properties(height=CHART_HEIGHT)


def latency_chart(view: pd.DataFrame):
    latency = view.assign(latency=pd.to_numeric(view["latency"], errors="coerce")).dropna(subset=["latency"])
    if latency.empty:
        return None
    return alt.Chart(latency[["latency", "status"]]).mark_bar().encode(
        x=alt.X("latency:Q", bin=alt.Bin(maxbins=30), title="Latency (ms)"),
        y=alt.Y("count():Q", title="Runs", stack=True),
        color=alt.Color("status:N", scale=STATUS_SCALE, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("latency:Q", bin=alt.Bin(maxbins=30), title="Latency (ms)"),
                 alt.Tooltip("count():Q", title="Runs")],
    ).properties(height=CHART_HEIGHT)


def by_model_chart(by_model: pd.DataFrame, field: str, title: str, value_format: str):
    return alt.Chart(by_model).mark_bar(color=PRIMARY_COLOR).encode(
        x=alt.X(f"{field}:Q", title=title, axis=alt.Axis(format=value_format)),
        y=alt.Y("model_name:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("model_name:N", title="Model"),
                 alt.Tooltip("calls:Q", title="LLM calls", format=","),
                 alt.Tooltip("tokens:Q", title="Tokens", format=","),
                 alt.Tooltip("cost:Q", title="Cost", format="$,.6f")],
    ).properties(height=CHART_HEIGHT)


def render_overview(view: pd.DataFrame, view_metrics: pd.DataFrame):
    render_kpis(view)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Runs over time")
        chart = runs_over_time_chart(view)
        if chart is None:
            st.caption("No start times recorded.")
        else:
            st.altair_chart(chart)
    with right:
        st.markdown("##### Latency distribution")
        chart = latency_chart(view)
        if chart is None:
            st.caption("No latencies recorded.")
        else:
            st.altair_chart(chart)

    llm = view_metrics[view_metrics["is_llm"].astype(bool)]
    left, right = st.columns(2)
    if llm.empty:
        left.markdown("##### Tokens by model")
        left.caption("No LLM calls recorded for the selected runs.")
        right.markdown("##### Cost by model")
        right.caption("No LLM calls recorded for the selected runs.")
        return

    by_model = llm.groupby("model_name").agg(
        calls=("id", "count"), tokens=("tokens", "sum"), cost=("total_cost", "sum"),
    ).reset_index()
    with left:
        st.markdown("##### Tokens by model")
        st.altair_chart(by_model_chart(by_model, "tokens", "Tokens", ",d"))
    with right:
        st.markdown("##### Cost by model")
        st.altair_chart(by_model_chart(by_model, "cost", "Cost (USD)", "$,.4f"))


# ─── Runs table ────────────────────────────────────────────────────────

def render_runs_table(view: pd.DataFrame):
    table = view[["id", "run_name", "project_id", "status", "latency", "llm_calls",
                  "tokens", "cost", "start"]]
    event = st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="runs_table",
        column_config={
            "id": st.column_config.NumberColumn("ID", format="%d"),
            "run_name": st.column_config.TextColumn("Name"),
            "project_id": st.column_config.TextColumn("Project"),
            "status": st.column_config.TextColumn("Status"),
            "llm_calls": st.column_config.NumberColumn("LLM calls", format="%d"),
            **TABLE_COLUMNS,
        },
    )
    st.caption("Select a row to open it in the Run detail tab. Times are local.")

    rows = getattr(getattr(event, "selection", None), "rows", None) or []
    if rows:
        picked = int(table.iloc[rows[0]]["id"])
        # Only follow the table when its selection changes, so the selectbox
        # in the detail tab stays usable on its own.
        if st.session_state.get("_table_pick") != picked:
            st.session_state["_table_pick"] = picked
            st.session_state["detail_run_id"] = picked


# ─── Run detail ────────────────────────────────────────────────────────

def _code_table(rows_html: list) -> str:
    return f'<div class="mloom-code"><table>{"".join(rows_html)}</table></div>'


def render_source_context(source: dict, error_type: str) -> str:
    start = source.get("start_line", 1)
    error_line = source.get("error_line")
    rows = []
    for offset, line in enumerate(source.get("lines", [])):
        lineno = start + offset
        is_error = lineno == error_line
        marker = f"   <- {error_type}" if is_error else ""
        rows.append(
            f'<tr class="{"mloom-error" if is_error else ""}">'
            f'<td class="mloom-ln">{lineno}</td>'
            f"<td>{html.escape(line + marker)}</td></tr>"
        )
    return _code_table(rows)


def render_locals(local_vars: dict) -> str:
    rows = []
    for name, value in local_vars.items():
        if value == "[REDACTED]":
            shown = '<span class="mloom-redacted">[REDACTED]</span>'
        else:
            shown = html.escape(str(value))
        rows.append(f'<tr><td class="mloom-name">{html.escape(name)}</td><td>{shown}</td></tr>')
    return _code_table(rows)


def render_error(error_trace: dict):
    error_type = error_trace.get("error_type", "Error")
    message = error_trace.get("error_message", "")
    st.markdown("#### Error")
    st.error(f"**{md_escape(error_type)}**" + (f": {md_escape(message)}" if message else ""))

    frames = error_trace.get("frames") or []
    traceback_text = error_trace.get("traceback_formatted")
    if not traceback_text and frames:
        traceback_text = "\n".join(
            f'File "{f.get("filename", "?")}", line {f.get("lineno", "?")}, in {f.get("function", "?")}\n'
            f'    {f.get("code_line", "").strip()}'
            for f in frames
        )
    if traceback_text:
        st.markdown("**Stack trace**")
        st.code(traceback_text.rstrip(), language="python")

    source = error_trace.get("source_context")
    if source and source.get("lines"):
        filename = Path(source.get("filename", "?")).name
        st.markdown(f"**Source context** · `{filename}`, line {source.get('error_line', '?')}")
        st.markdown(render_source_context(source, error_type), unsafe_allow_html=True)

    if frames:
        innermost = frames[-1]
        local_vars = innermost.get("locals") or {}
        st.markdown(f"**Local variables** in `{innermost.get('function', '?')}()`")
        if local_vars:
            st.markdown(render_locals(local_vars), unsafe_allow_html=True)
            redacted = sum(1 for v in local_vars.values() if v == "[REDACTED]")
            if redacted:
                st.caption(f"{redacted} value(s) hidden because the variable name looks like a secret.")
        else:
            st.caption("No local variables were captured.")


def render_metrics(metrics: list):
    st.markdown("#### LLM calls and steps")
    if not metrics:
        st.caption("No LLM calls or steps were recorded in this run.")
        return

    table = pd.DataFrame([
        {
            "n": i,
            "kind": "LLM call" if m["is_llm"] else "Step",
            "model_name": m["model_name"],
            "latency": m["latency"],
            "input_tokens": m["input_tokens"],
            "output_tokens": m["output_tokens"],
            "cost": m["total_cost"],
            "start": m["start"],
        }
        for i, m in enumerate(metrics, 1)
    ])
    table["start"] = pd.to_datetime(table["start"])
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "n": st.column_config.NumberColumn("#", format="%d"),
            "kind": st.column_config.TextColumn("Kind"),
            "model_name": st.column_config.TextColumn("Model / step"),
            "input_tokens": st.column_config.NumberColumn("Input tokens", format="localized"),
            "output_tokens": st.column_config.NumberColumn("Output tokens", format="localized"),
            **TABLE_COLUMNS,
        },
    )

    for i, m in enumerate(metrics, 1):
        title = f"{i}. {md_escape(m['model_name'])} · {fmt_ms(m['latency'])}"
        if m["response"].startswith("ERROR:"):
            title += " · error"
        with st.expander(title):
            st.markdown("**Prompt**")
            st.code(m["prompt"] or "(empty)", language=None, wrap_lines=True)
            st.markdown("**Response**")
            st.code(m["response"] or "(empty)", language=None, wrap_lines=True)


def render_run_detail(detail: dict):
    failed = bool(detail["error_trace"])
    metrics = detail["metrics"]
    llm_calls = [m for m in metrics if m["is_llm"]]

    st.markdown(f"### Run #{detail['id']} · {md_escape(detail['run_name'])}")
    status = ":red[**FAILED**]" if failed else ":green[**OK**]"
    st.markdown(
        f"{status} · Project `{detail['project_id']}` · Type {detail['run_type'] or '-'}"
    )

    cols = st.columns(5)
    cols[0].metric("Latency", fmt_ms(detail["latency"]), border=True)
    cols[1].metric("LLM calls", fmt_int(len(llm_calls)), border=True)
    cols[2].metric("Tokens", fmt_int(sum((m["input_tokens"] or 0) + (m["output_tokens"] or 0)
                                         for m in llm_calls)), border=True)
    cols[3].metric("Cost", fmt_cost(sum(m["total_cost"] or 0.0 for m in llm_calls)), border=True)
    cols[4].metric("Steps", fmt_int(len(metrics) - len(llm_calls)), border=True)

    timing = f"Started {fmt_time(detail['start'])} · Ended {fmt_time(detail['end'])} (local time)"
    if detail["tags"]:
        timing += " · Tags " + md_escape(json.dumps(detail["tags"]))
    st.caption(timing)

    call_site = detail["call_site"]
    if call_site:
        st.markdown("**Call site**")
        st.code(
            f'File "{call_site.get("caller_file", "?")}", line {call_site.get("caller_line", "?")}, '
            f'in {call_site.get("caller_function", "?")}()',
            language=None,
        )

    render_metrics(metrics)

    if failed:
        render_error(detail["error_trace"])


def render_detail_tab(view: pd.DataFrame, db_url: str):
    ids = [int(i) for i in view["id"]]
    labels = {
        int(row.id): f"#{row.id} · {row.run_name or '(unnamed)'} · {row.status}"
        for row in view.itertuples()
    }
    if st.session_state.get("detail_run_id") not in labels:
        st.session_state["detail_run_id"] = ids[0]

    run_id = st.selectbox("Run", ids, format_func=lambda i: labels.get(i, f"#{i}"), key="detail_run_id")
    detail = cached_run_detail(db_url, run_id)
    if detail is None:
        st.info(f"Run #{run_id} no longer exists. Press Refresh data.")
        return
    render_run_detail(detail)


# ─── Page ──────────────────────────────────────────────────────────────

def apply_filters(runs: pd.DataFrame, project: str, status: str, search: str) -> pd.DataFrame:
    view = runs
    if project != ALL_PROJECTS:
        view = view[view["project_id"] == project]
    if status != "All":
        view = view[view["status"] == status]
    if search:
        view = view[view["run_name"].str.contains(search, case=False, regex=False)]
    return view


def main():
    st.set_page_config(page_title="mloom", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # Re-read on every rerun so `mloom server connect` takes effect without a restart.
    config.load(force=True)
    if config.mode != "local":
        render_remote_notice()
        return

    db_path = config.local_db_path
    db_url = config.local_db_url

    with st.sidebar:
        st.title("mloom")
        st.caption("Local LLM run tracking")
        if st.button("Refresh data"):
            st.cache_data.clear()

    if not db_path.exists():
        render_empty_state(db_path)
        return

    runs, metrics = cached_dataset(db_url)
    if runs.empty:
        render_empty_state(db_path)
        return

    with st.sidebar:
        projects = sorted(runs["project_id"].dropna().unique().tolist())
        project = st.selectbox("Project", [ALL_PROJECTS, *projects], key="filter_project")
        status = st.radio("Status", STATUS_FILTERS, horizontal=True, key="filter_status")
        search = st.text_input("Run name", placeholder="Search by name", key="filter_search").strip()
        st.divider()
        st.caption(f"Database: {db_path}")
        st.caption(f"Data is cached for {CACHE_TTL_SECONDS} s. Times are shown in local time.")

    view = apply_filters(runs, project, status, search)
    view_metrics = metrics[metrics["run_id"].isin(view["id"])]

    st.title("Runs")
    scope = "all projects" if project == ALL_PROJECTS else f"project {project}"
    st.caption(
        f"{len(view):,} of {len(runs):,} runs · {scope} · "
        f"latest run {fmt_time(runs['start'].max())}"
    )
    if view.empty:
        st.info("No runs match the current filters.")
        return

    overview_tab, runs_tab, detail_tab = st.tabs(["Overview", "Runs", "Run detail"])
    with overview_tab:
        render_overview(view, view_metrics)
    with runs_tab:
        render_runs_table(view)
    with detail_tab:
        render_detail_tab(view, db_url)


main()
