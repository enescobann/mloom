import subprocess
import sys
from datetime import timezone
from pathlib import Path

import click
from mloom.core.config import config

UI_APP_PATH = Path(__file__).parent / "ui" / "app.py"
_REMOTE_TIMEOUT = 5.0


@click.group()
def cli():
    """Mloom CLI: The Local-First LLMOps Platform"""
    pass


# ─── Data access (local SQLite or remote server) ──────────────────────

def _remote_get(path: str, **params):
    import requests

    url = f"{config.backend_url.rstrip('/')}{path}"
    response = requests.get(url, params=params or None, timeout=_REMOTE_TIMEOUT)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _summarize_run(run: dict) -> dict:
    """Reduce a run (with its metrics) to the columns shown by `mloom runs`."""
    from mloom.db.crud import to_naive_utc

    metrics = run.get("metrics") or []
    return {
        "id": run["id"],
        "name": run.get("run_name") or "",
        "ok": not run.get("error_trace"),
        "latency": run.get("latency"),
        "tokens": sum((m.get("input_tokens") or 0) + (m.get("output_tokens") or 0) for m in metrics),
        "cost": sum(m.get("total_cost") or 0.0 for m in metrics),
        "time": to_naive_utc(run.get("start_time") or run.get("timestamp")),
    }


def _fetch_runs_local(project, limit):
    from sqlalchemy import func
    from mloom.db import database
    from mloom.db.classes import Run, LLMMetrics

    database.init_db(config.local_db_url)
    with database.get_session() as db:
        query = db.query(Run)
        if project:
            query = query.filter(Run.project_id == project)
        runs = query.order_by(Run.id.desc()).limit(limit).all()

        # Aggregate tokens/cost only for the runs being shown.
        totals = {}
        run_ids = [r.id for r in runs]
        if run_ids:
            totals = {
                run_id: (tokens or 0, cost or 0.0)
                for run_id, tokens, cost in db.query(
                    LLMMetrics.run_id,
                    func.sum(func.coalesce(LLMMetrics.input_tokens, 0)
                             + func.coalesce(LLMMetrics.output_tokens, 0)),
                    func.sum(LLMMetrics.total_cost),
                ).filter(LLMMetrics.run_id.in_(run_ids)).group_by(LLMMetrics.run_id)
            }

        rows = []
        for r in runs:
            tokens, cost = totals.get(r.id, (0, 0.0))
            rows.append({
                "id": r.id,
                "name": r.run_name or "",
                "ok": not r.error_trace,
                "latency": r.latency,
                "tokens": int(tokens),
                "cost": float(cost),
                "time": r.start_time or r.timestamp,
            })
        return rows


def _fetch_runs_remote(project, limit):
    if project:
        runs = _remote_get(f"/projects/{project}/runs", limit=limit)
        if runs is None:
            raise click.ClickException(f"Project {project!r} not found on {config.backend_url}.")
    else:
        runs = []
        for proj in _remote_get("/projects/", limit=1000) or []:
            runs.extend(_remote_get(f"/projects/{proj['id']}/runs", limit=limit) or [])
        runs.sort(key=lambda r: r["id"], reverse=True)
    return [_summarize_run(r) for r in runs[:limit]]


def _fetch_run(run_id: int):
    """Return a run as a plain dict, or None if it does not exist."""
    if config.mode != "local":
        return _remote_get(f"/runs/{run_id}")

    from mloom.db import database
    from mloom.db.classes import Run

    database.init_db(config.local_db_url)
    with database.get_session() as db:
        run = db.get(Run, run_id)
        if run is None:
            return None
        return {
            "id": run.id,
            "run_name": run.run_name,
            "error_trace": run.error_trace,
            "call_site": run.call_site,
        }


def _format_time(value) -> str:
    if value is None:
        return "-"
    # Stored as naive UTC; show it in the user's local time zone.
    return value.replace(tzinfo=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 3] + "..."


# ─── Commands ──────────────────────────────────────────────────────────

@cli.command("ui")
@click.option("--port", type=int, default=8501, show_default=True, help="Port for the Streamlit server.")
@click.option("--open/--no-open", "open_browser", default=True, help="Open the dashboard in a browser.")
def ui(port, open_browser):
    """Launch the local dashboard (Streamlit)."""
    # Headless skips Streamlit's first-run email prompt; usage statistics stay off for a local-first tool.
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(UI_APP_PATH),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--client.toolbarMode", "minimal",
    ]
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, webbrowser.open, args=(f"http://localhost:{port}",)).start()
    try:
        sys.exit(subprocess.call(cmd))
    except KeyboardInterrupt:
        pass


@cli.group()
def server():
    """Run a server or choose where runs go (Local SQLite vs Remote Server)"""
    pass

@server.command("start")
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to listen on.")
def start(host: str, port: int):
    """
    Start the tracking server (uses DATABASE_URL, or SQLite under ~/.mloom).

    Example:
      mloom server start --host 0.0.0.0 --port 8000
    """
    try:
        import uvicorn
        import fastapi  # noqa: F401
    except ImportError:
        raise click.ClickException('Server dependencies missing. Install with: pip install "mloom[server]"')

    click.secho(f"[Mloom] Starting server on http://{host}:{port}", fg="green")
    uvicorn.run("mloom.server.main:app", host=host, port=port)

@server.command("connect")
@click.argument("target")
def connect(target: str):
    """
    Connect to a server.
    Example:
      mloom server connect local
      mloom server connect http://192.168.1.5:8000
    """
    config.load()
    if target.lower() == "local":
        config.mode = "local"
        config.save()
        click.secho("[Mloom] Switched to Local SQLite mode.", fg="green")
        click.secho(f"[Mloom] Database path: {config.local_db_path}", fg="blue")
    else:
        if not target.startswith(("http://", "https://")):
            target = f"http://{target}"
        target = target.rstrip("/")

        config.mode = "remote"
        config.backend_url = target
        config.save()
        click.secho(f"[Mloom] Switched to Remote Mode. Target: {target}", fg="green")


@cli.command("runs")
@click.option("--project", "-p", default=None, help="Only show runs of this project.")
@click.option("--limit", "-n", default=20, show_default=True, type=click.IntRange(min=1),
              help="Number of runs to show.")
def runs(project, limit):
    """
    List the most recent runs.

    Example:
      mloom runs --project my-chatbot --limit 10
    """
    config.load()
    try:
        if config.mode == "local":
            rows = _fetch_runs_local(project, limit)
        else:
            rows = _fetch_runs_remote(project, limit)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Could not load runs: {e}")

    if not rows:
        click.echo("No runs found.")
        return

    click.secho(
        f"{'ID':>6}  {'NAME':<28}  {'STATUS':<6}  {'LATENCY':>9}  {'TOKENS':>8}  {'COST':>11}  TIME",
        bold=True,
    )
    for row in rows:
        status = "OK" if row["ok"] else "FAILED"
        latency = f"{row['latency']}ms" if row["latency"] is not None else "-"
        cost = f"${row['cost']:.6f}"
        click.echo(
            f"{row['id']:>6}  {_truncate(row['name'], 28):<28}  "
            + click.style(f"{status:<6}", fg="green" if row["ok"] else "red")
            + f"  {latency:>9}  {row['tokens']:>8}  {cost:>11}  {_format_time(row['time'])}"
        )


@cli.command("trace")
@click.argument("run_id", type=int)
def trace(run_id: int):
    """
    Inspect the trace of a specific run (post-mortem debugging).

    Example:
      mloom trace 42
    """
    config.load()
    try:
        run = _fetch_run(run_id)
    except Exception as e:
        raise click.ClickException(f"Could not load run #{run_id}: {e}")

    if not run:
        click.secho(f"[Mloom] Run #{run_id} not found.", fg="red")
        sys.exit(1)

    # ─── Header ────────────────────────────────────────────────
    run_name = run.get("run_name") or ""
    error_trace = run.get("error_trace")
    status = "FAILED" if error_trace else "OK"
    status_color = "red" if error_trace else "green"

    click.echo()
    click.secho("╔" + "═" * 58 + "╗", fg=status_color)
    title = f'  Run #{run_id} — "{run_name}" — {status}'
    click.secho(f"║{title:<58}║", fg=status_color)

    if error_trace:
        error_type = error_trace.get("error_type", "Unknown")
        error_message = error_trace.get("error_message", "")
        error_line = f"  Error: {error_type} — {error_message}"
        if len(error_line) > 58:
            error_line = error_line[:55] + "..."
        click.secho("╠" + "═" * 58 + "╣", fg="red")
        click.secho(f"║{error_line:<58}║", fg="red")

    click.secho("╚" + "═" * 58 + "╝", fg=status_color)

    # ─── Call Site ─────────────────────────────────────────────
    call_site = run.get("call_site")
    if call_site:
        click.echo()
        click.secho("Call Site:", bold=True)
        caller_file = call_site.get("caller_file", "?")
        caller_line = call_site.get("caller_line", "?")
        caller_func = call_site.get("caller_function", "?")
        click.echo(f'  File "{caller_file}", line {caller_line}, in {caller_func}()')

    # ─── Error Details ─────────────────────────────────────────
    if not error_trace:
        click.echo()
        click.secho("No errors recorded for this run.", fg="green")
        return

    # Stack Trace
    frames = error_trace.get("frames", [])
    if frames:
        click.echo()
        click.secho("Stack Trace:", bold=True)
        for frame in frames:
            filename = frame.get("filename", "?")
            lineno = frame.get("lineno", "?")
            function = frame.get("function", "?")
            code_line = frame.get("code_line", "").strip()
            click.echo(f'  File "{filename}", line {lineno}, in {function}')
            if code_line:
                click.echo(f"    {code_line}")

    # Source Context
    source_ctx = error_trace.get("source_context")
    if source_ctx:
        click.echo()
        click.secho("Source Context:", bold=True)
        ctx_lines = source_ctx.get("lines", [])
        start = source_ctx.get("start_line", 1)
        error_lineno = source_ctx.get("error_line", -1)
        error_type = error_trace.get("error_type", "Error")

        for i, line_text in enumerate(ctx_lines):
            current_lineno = start + i
            if current_lineno == error_lineno:
                click.secho(
                    f"→ {current_lineno:>4} │ {line_text}   ← {error_type}",
                    fg="red", bold=True
                )
            else:
                click.secho(f"  {current_lineno:>4} │ {line_text}", dim=True)

    # Local Variables
    if frames:
        innermost = frames[-1]
        local_vars = innermost.get("locals", {})
        if local_vars:
            click.echo()
            click.secho("Local Variables:", bold=True)
            for var_name, var_value in local_vars.items():
                if var_value == "[REDACTED]":
                    click.echo(f"  {click.style(var_name, fg='yellow')}"
                               f" = {click.style('[REDACTED]', fg='red')}")
                else:
                    click.echo(f"  {click.style(var_name, fg='cyan')}"
                               f" = {var_value}")

    click.echo()

if __name__ == "__main__":
    cli()
