"""
Trace capture module for Mloom SDK.

Provides structured exception context and call-site provenance
using Python's `traceback` and `inspect` modules.
"""

import sys
import traceback
import inspect
import linecache
from fnmatch import fnmatch

# ─── Redaction Patterns ───────────────────────────────────────────────
# Variable names matching ANY of these (case-insensitive fnmatch) get redacted.
# Intentionally broad — false positives (over-redacting) are safer than leaks.

REDACT_PATTERNS = [
    # Authentication & API Keys
    "*api_key*",  "*apikey*",  "*api_secret*",
    "*secret*",   "*secret_key*",
    "*key*",
    "*token*",    "*access_token*", "*refresh_token*", "*bearer*",
    "*password*", "*passwd*",  "*pwd*",
    "*auth*",     "*authorization*",
    "*credential*", "*credentials*",

    # Connection Strings & DSNs
    "*connection_string*", "*conn_str*",
    "*database_url*",      "*db_url*",    "*db_uri*",
    "*dsn*",               "*jdbc*",
    "*redis_url*",         "*mongo_uri*",

    # Certificates & Cryptographic Material
    "*private_key*", "*privkey*", "*pem*",
    "*cert*",        "*certificate*",
    "*signing_key*", "*encryption_key*",

    # Session & Cookie Data
    "*session_id*",  "*session_key*", "*session_token*",
    "*cookie*",      "*csrf*",        "*xsrf*",

    # OAuth & SSO
    "*oauth*",       "*client_secret*", "*client_id*",
    "*id_token*",    "*jwt*",

    # Hashing & Signing
    "*hash*",   "*salt*",   "*nonce*",
    "*hmac*",   "*signature*", "*digest*",

    # Cloud Provider Credentials
    "*aws_access*",     "*aws_secret*",
    "*azure_key*",      "*azure_secret*",   "*azure_token*",
    "*gcp_key*",        "*gcp_credentials*",
    "*service_account*",

    # Webhook & Integration Secrets
    "*webhook_secret*", "*signing_secret*",
    "*slack_token*",    "*discord_token*",
    "*stripe_key*",     "*sendgrid_key*",

    # SSH
    "*ssh_key*",   "*ssh_pass*",   "*passphrase*",

    # Encryption Passphrases
    "*master_key*", "*root_key*", "*encryption_password*",
]

# Context window: 5 lines before + 5 lines after the failure line
_CONTEXT_LINES = 5


def _should_redact(var_name: str) -> bool:
    """Check if a variable name matches any redaction pattern."""
    name_lower = var_name.lower()
    return any(fnmatch(name_lower, pattern) for pattern in REDACT_PATTERNS)


def _safe_repr(value, max_length: int = 200) -> str:
    """Safely convert a value to a string representation, truncating if needed."""
    try:
        s = repr(value)
        if len(s) > max_length:
            return s[:max_length] + "..."
        return s
    except Exception:
        return "[Unrepresentable Object]"


def _redact_locals(local_vars: dict) -> dict:
    """
    Redact sensitive variables from a locals() dict.
    Returns a new dict with values replaced by '[REDACTED]' where needed.
    """
    redacted = {}
    for name, value in local_vars.items():
        # Skip dunder/private variables — they're almost never useful for debugging
        if name.startswith("__"):
            continue
        if _should_redact(name):
            redacted[name] = "[REDACTED]"
        else:
            redacted[name] = _safe_repr(value)
    return redacted


def _get_source_context(filename: str, error_lineno: int) -> dict | None:
    """
    Extract source code lines around the failure point.
    Returns None if the source can't be read (e.g. <stdin>, compiled code).
    """
    try:
        # Try to read the source file via linecache (handles encoding, caching)
        all_lines = linecache.getlines(filename)
        if not all_lines:
            return None

        total_lines = len(all_lines)
        start_line = max(1, error_lineno - _CONTEXT_LINES)
        end_line = min(total_lines, error_lineno + _CONTEXT_LINES)

        context_lines = []
        for i in range(start_line, end_line + 1):
            context_lines.append(all_lines[i - 1].rstrip("\n"))

        return {
            "filename": filename,
            "start_line": start_line,
            "end_line": end_line,
            "lines": context_lines,
            "error_line": error_lineno,
        }
    except Exception:
        return None


def capture_exception_context(exc: BaseException, tb) -> dict:
    """
    Capture a structured error context from an exception and its traceback.

    Uses:
        - traceback.format_exception() for the full formatted traceback string
        - traceback.extract_tb() for structured frame data
        - inspect/linecache for source context around the failure
        - Frame f_locals for local variable snapshots (with redaction)

    Args:
        exc: The caught exception instance.
        tb: The traceback object (from sys.exc_info()[2]).

    Returns:
        A JSON-serializable dict with error type, message, frames, 
        formatted traceback, and source context.
    """
    # 1. Formatted traceback string (the familiar Python output)
    formatted_lines = traceback.format_exception(type(exc), exc, tb)
    traceback_formatted = "".join(formatted_lines)

    # 2. Structured frames from traceback.extract_tb
    extracted = traceback.extract_tb(tb)

    # 3. Walk the actual traceback chain to get f_locals from each frame
    frames = []
    current_tb = tb
    for frame_summary in extracted:
        frame_locals = {}
        if current_tb is not None:
            try:
                frame_locals = _redact_locals(current_tb.tb_frame.f_locals)
            except Exception:
                frame_locals = {"_error": "[Could not capture locals]"}
            current_tb = current_tb.tb_next

        frames.append({
            "filename": frame_summary.filename,
            "lineno": frame_summary.lineno,
            "function": frame_summary.name,
            "code_line": frame_summary.line or "",
            "locals": frame_locals,
        })

    # 4. Source context from the innermost (last) frame
    source_context = None
    if frames:
        innermost = frames[-1]
        source_context = _get_source_context(
            innermost["filename"], innermost["lineno"]
        )

    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback_formatted": traceback_formatted,
        "frames": frames,
        "source_context": source_context,
    }


def capture_call_site(skip_frames: int = 2) -> dict | None:
    """
    Capture the call site of the decorated function.

    Uses inspect.stack() to find the caller's file, line, and function name,
    skipping internal decorator/wrapper frames.

    Args:
        skip_frames: Number of internal frames to skip to reach the user's
                     actual call site. Default is 2 (this function + the wrapper).

    Returns:
        A minimal dict with caller_file, caller_line, caller_function,
        or None if the stack can't be inspected.
    """
    try:
        stack = inspect.stack()
        # Guard against unexpectedly shallow stacks
        if len(stack) <= skip_frames:
            return None

        frame_info = stack[skip_frames]
        return {
            "caller_file": frame_info.filename,
            "caller_line": frame_info.lineno,
            "caller_function": frame_info.function,
        }
    except Exception:
        # inspect.stack() can fail in some edge cases (frozen modules, etc.)
        return None
    finally:
        # Avoid reference cycles from holding frame objects
        del stack
