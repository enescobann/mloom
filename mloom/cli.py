import click
from mloom.core.config import config

@click.group()
def cli():
    """Mloom CLI: The Local-First LLMOps Platform"""
    pass

@cli.group()
def server():
    """Manage server connections (Local SQLite vs Remote Server)"""
    pass

@server.command("connect")
@click.argument("target")
def connect(target: str):
    """
    Connect to a server.
    Example: 
      mloom server connect local
      mloom server connect http://192.168.1.5:8000
    """
    if target.lower() == "local":
        config.mode = "local"
        config._save_config()
        click.secho("[Mloom] Switched to Local SQLite mode.", fg="green")
        click.secho(f"[Mloom] Database path: {config.local_db_path}", fg="blue")
    else:
        if not target.startswith("http"):
            target = f"http://{target}"
            
        config.mode = "remote"
        config.backend_url = target
        config._save_config()
        click.secho(f"[Mloom] Switched to Remote Mode. Target: {target}", fg="green")


@cli.command("trace")
@click.argument("run_id", type=int)
def trace(run_id: int):
    """
    Inspect the trace of a specific run (post-mortem debugging).

    Example:
      mloom trace 42
    """
    # Initialize local DB connection
    from mloom.db.database import init_db, SessionLocal
    config.load()
    if config.mode != "local":
        click.secho("[Mloom] Trace inspection is only available in local mode.", fg="red")
        return
    
    init_db()
    from mloom.db.classes import Run

    with SessionLocal() as db:
        run = db.query(Run).filter(Run.id == run_id).first()

    if not run:
        click.secho(f"[Mloom] Run #{run_id} not found.", fg="red")
        return

    # ─── Header ────────────────────────────────────────────────
    error_trace = run.error_trace
    status = "FAILED" if error_trace else "OK"
    status_color = "red" if error_trace else "green"

    click.echo()
    click.secho("╔" + "═" * 58 + "╗", fg=status_color)
    title = f'  Run #{run_id} — "{run.run_name}" — {status}'
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
    call_site = run.call_site
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