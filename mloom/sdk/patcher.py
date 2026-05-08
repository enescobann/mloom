import time
import functools
from .emitter import emit_event
from mloom.core.config import config
from .contextvar import active_run_metrics

def patch_openai():
    try:
        import openai
        # Must import BOTH sync and async classes
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except ImportError:
        return  # User doesn't have openai installed, exit silently

    original_sync_create = Completions.create
    original_async_create = AsyncCompletions.create

    def _extract_text(response):
        """Safely extracts actual text rather than stringifying a whole Pydantic object."""
        try:
            return response.choices[0].message.content
        except Exception:
            return str(response)

    def _record_telemetry(start_time, kwargs, response=None, error=None):
        """Centralized logic to format and send the metric."""
        latency_ms = int((time.time() - start_time) * 1000)
        
        input_tokens, output_tokens = 0, 0
        
        # Safely extract tokens if they exist (they don't in standard streaming)
        if response and hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)

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
            "latency": latency_ms
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
                "latency": latency_ms, # Standardized key name to match earlier config
                "metrics": [metric_data]
            }
            emit_event(payload)

    # --- 1. SYNC WRAPPER ---
    @functools.wraps(original_sync_create)
    def sync_wrapped_create(*args, **kwargs):
        start_time = time.time()
        try:
            response = original_sync_create(*args, **kwargs)
            _record_telemetry(start_time, kwargs, response=response)
            return response
        except Exception as e:
            _record_telemetry(start_time, kwargs, error=e)
            raise  # Always re-raise so user code behaves normally!

    # --- 2. ASYNC WRAPPER ---
    @functools.wraps(original_async_create)
    async def async_wrapped_create(*args, **kwargs):
        start_time = time.time()
        try:
            response = await original_async_create(*args, **kwargs)
            _record_telemetry(start_time, kwargs, response=response)
            return response
        except Exception as e:
            _record_telemetry(start_time, kwargs, error=e)
            raise

    # Apply the patches
    Completions.create = sync_wrapped_create
    AsyncCompletions.create = async_wrapped_create