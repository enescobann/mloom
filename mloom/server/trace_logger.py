"""
Server-side trace logger for Mloom.

Outputs structured, human-readable trace information to stdout
in real-time. Used by both the FastAPI server process and the
local-mode SDK emitter so developers can see what's happening
by watching the terminal.
"""

import logging

logger = logging.getLogger("mloom.trace")

# Only add a handler if one doesn't exist already (avoid duplicates)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)


def _format_success(run_name: str, run_id, latency, call_site: dict | None) -> str:
    """Format a one-line summary for a successful run."""
    site_str = ""
    if call_site:
        filename = call_site.get("caller_file", "?")
        # Use just the basename for brevity
        basename = filename.rsplit("/", 1)[-1] if "/" in filename else filename
        line = call_site.get("caller_line", "?")
        func = call_site.get("caller_function", "?")
        site_str = f" — from {basename}:{line} in {func}()"

    latency_str = f"{latency}ms" if latency is not None else "?ms"
    return f'[Mloom] ✓ Run "{run_name}" (Run #{run_id}) — {latency_str}{site_str}'


def _format_failure(run_name: str, run_id, error_trace: dict) -> str:
    """Format a full trace block for a failed run."""
    error_type = error_trace.get("error_type", "Unknown")
    error_message = error_trace.get("error_message", "")
    frames = error_trace.get("frames", [])
    source_ctx = error_trace.get("source_context")

    lines = []
    lines.append(f'[Mloom] ✗ Run "{run_name}" (Run #{run_id}) — FAILED')
    lines.append(f"  ┌─ Error: {error_type} — {error_message}")

    # Show the innermost frame (most useful)
    if frames:
        innermost = frames[-1]
        filename = innermost.get("filename", "?")
        lineno = innermost.get("lineno", "?")
        function = innermost.get("function", "?")
        code_line = innermost.get("code_line", "").strip()
        lines.append(f'  ├─ File "{filename}", line {lineno}, in {function}')
        if code_line:
            lines.append(f"  │    {code_line}")

    # Source context
    if source_ctx:
        lines.append("  ├─ Source Context:")
        ctx_lines = source_ctx.get("lines", [])
        start = source_ctx.get("start_line", 1)
        error_line = source_ctx.get("error_line", -1)

        for i, line_text in enumerate(ctx_lines):
            current_lineno = start + i
            if current_lineno == error_line:
                lines.append(f"  │  → {current_lineno:>4} │ {line_text}")
            else:
                lines.append(f"  │    {current_lineno:>4} │ {line_text}")

    # Locals from innermost frame
    if frames:
        innermost_locals = frames[-1].get("locals", {})
        # Filter out noise — show only meaningful variables
        display_locals = {
            k: v for k, v in innermost_locals.items()
            if not k.startswith("_") and v != "[REDACTED]"
        }
        redacted_count = sum(1 for v in innermost_locals.values() if v == "[REDACTED]")

        if display_locals:
            parts = [f"{k}={v}" for k, v in list(display_locals.items())[:8]]
            locals_str = ", ".join(parts)
            if len(display_locals) > 8:
                locals_str += ", ..."
            lines.append(f"  └─ Locals: {locals_str}")
            if redacted_count:
                lines.append(f"     ({redacted_count} variable(s) redacted)")
        elif redacted_count:
            lines.append(f"  └─ Locals: ({redacted_count} variable(s) redacted)")
        else:
            lines.append("  └─ (no local variables captured)")

    return "\n".join(lines)


def log_run_trace(run) -> None:
    """
    Log trace information for a run to stdout.

    Accepts either a SQLAlchemy Run ORM object or a plain dict with
    the same keys (for flexibility in both local and remote modes).

    Args:
        run: A Run ORM instance or a dict with run_name, id, latency,
             error_trace, and call_site keys.
    """
    # Support both ORM objects and dicts
    if isinstance(run, dict):
        run_name = run.get("run_name", "Unnamed Run")
        run_id = run.get("id", "?")
        latency = run.get("latency")
        error_trace = run.get("error_trace")
        call_site = run.get("call_site")
    else:
        run_name = getattr(run, "run_name", "Unnamed Run")
        run_id = getattr(run, "id", "?")
        latency = getattr(run, "latency", None)
        error_trace = getattr(run, "error_trace", None)
        call_site = getattr(run, "call_site", None)

    if error_trace:
        message = _format_failure(run_name, run_id, error_trace)
        logger.error(message)
    else:
        message = _format_success(run_name, run_id, latency, call_site)
        logger.info(message)
