from datetime import datetime, timezone
import sys
import time
import functools
from .emitter import emit_event
from mloom.core.config import config
from mloom.sdk.cost import calculate_cost
from .contextvar import active_run_metrics

_PATCH_MARKER = "_mloom_patched"


def _extract_text(response):
    """Safely extracts actual text rather than stringifying a whole Pydantic object."""
    try:
        return response.choices[0].message.content
    except Exception:
        return str(response)


def _record_telemetry(start_time, kwargs, response=None, error=None):
    """Centralized logic to format and send the metric."""
    end_time_utc = datetime.now(timezone.utc)
    start_time_utc = datetime.fromtimestamp(start_time, tz=timezone.utc)
    latency_ms = int((end_time_utc - start_time_utc).total_seconds() * 1000)

    input_tokens, output_tokens = 0, 0

    # Safely extract tokens if they exist (they don't in standard streaming)
    usage = getattr(response, "usage", None) if response is not None else None
    if usage:
        input_tokens = getattr(usage, "prompt_tokens", None) or 0
        output_tokens = getattr(usage, "completion_tokens", None) or 0

    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])
    is_streaming = kwargs.get("stream", False)

    # Handle formatting based on success, failure, or stream
    if error:
        response_text = f"ERROR: {str(error)}"
    elif is_streaming:
        response_text = "[Streaming Response - Content aggregation not yet supported]"
    else:
        response_text = _extract_text(response)

    metric_data = {
        "model_name": model,
        "prompt": str(messages), # Consider truncating this if it gets too large
        "response": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "start_time": start_time_utc.isoformat(),
        "end_time": end_time_utc.isoformat(),
        "latency": latency_ms,
        "total_cost": calculate_cost(model, input_tokens, output_tokens)
    }

    # Safely check ContextVar (Must use .get(None) to avoid LookupError)
    current_metrics = active_run_metrics.get(None)

    if current_metrics is not None:
        # We are inside a @track_run pipeline
        current_metrics.append(metric_data)
    else:
        # Standalone API call
        payload = {
            "project_id": config.project_id,
            "tags": {"provider": "openai"},
            "run_name": "OpenAI API Call",
            "run_type": "LLM",
            "latency": latency_ms, # Standardized key name to match earlier config
            "start_time": metric_data["start_time"],
            "end_time": metric_data["end_time"],
            "metrics": [metric_data]
        }
        emit_event(payload)


def _safe_record(start_time, kwargs, response=None, error=None):
    """Telemetry must never break or alter the user's OpenAI call."""
    try:
        _record_telemetry(start_time, kwargs, response=response, error=error)
    except Exception as e:
        print(f"[Mloom Error]: Failed to record OpenAI call. Reason: {e}", file=sys.stderr)


def patch_openai():
    try:
        # Must import BOTH sync and async classes
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except ImportError:
        return  # User doesn't have openai installed, exit silently

    # Patching twice would record every call twice.
    if getattr(Completions.create, _PATCH_MARKER, False):
        return

    original_sync_create = Completions.create
    original_async_create = AsyncCompletions.create

    # --- 1. SYNC WRAPPER ---
    @functools.wraps(original_sync_create)
    def sync_wrapped_create(*args, **kwargs):
        start_time = time.time()
        try:
            response = original_sync_create(*args, **kwargs)
        except Exception as e:
            _safe_record(start_time, kwargs, error=e)
            raise  # Always re-raise so user code behaves normally!
        _safe_record(start_time, kwargs, response=response)
        return response

    # --- 2. ASYNC WRAPPER ---
    @functools.wraps(original_async_create)
    async def async_wrapped_create(*args, **kwargs):
        start_time = time.time()
        try:
            response = await original_async_create(*args, **kwargs)
        except Exception as e:
            _safe_record(start_time, kwargs, error=e)
            raise
        _safe_record(start_time, kwargs, response=response)
        return response

    setattr(sync_wrapped_create, _PATCH_MARKER, True)
    setattr(async_wrapped_create, _PATCH_MARKER, True)

    # Apply the patches
    Completions.create = sync_wrapped_create
    AsyncCompletions.create = async_wrapped_create
