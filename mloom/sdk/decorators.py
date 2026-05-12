from datetime import datetime, timezone
import time
import functools
import asyncio
from .emitter import emit_event
from .contextvar import active_run_metrics
from mloom.core.config import config

def _safe_stringify(obj, max_length=2000):
    """Utility to safely stringify inputs/outputs without memory blowouts."""
    try:
        s = str(obj)
        return s if len(s) <= max_length else s[:max_length] + "... [TRUNCATED]"
    except Exception:
        return "[Unserializable Object]"

def track_run(name: str = None, run_type: str = None):
    """Parent Decorator. Initializes tracking elements and emits."""
    
    def decorator(func):
        run_name = name or func.__name__

        # ASYNC WRAPPER
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                contextvar_token = active_run_metrics.set([])
                try:
                    return await func(*args, **kwargs)
                finally:
                    _finalize_run(start_time, run_name, run_type, contextvar_token)
            return async_wrapper

        # SYNC WRAPPER
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                contextvar_token = active_run_metrics.set([])
                try:
                    return func(*args, **kwargs)
                finally:
                    _finalize_run(start_time, run_name, run_type, contextvar_token)
            return sync_wrapper

    return decorator

def _finalize_run(start_time, run_name, run_type, contextvar_token):
    """Helper to keep track_run DRY."""
    end_time_utc = datetime.now(timezone.utc)
    start_time_utc = datetime.fromtimestamp(start_time, tz=timezone.utc)
    latency_ms = int((end_time_utc - start_time_utc).total_seconds() * 1000)
    collected_metrics = active_run_metrics.get()
    
    payload = {
        "project_id": config.project_id,
        "run_name": run_name,
        "run_type": run_type or "LLM",
        "latency": latency_ms,
        "metrics": collected_metrics,
        "start_time": start_time_utc.isoformat(),
        "end_time": end_time_utc.isoformat()

    }
    emit_event(payload)
    active_run_metrics.reset(contextvar_token)


def track_metric(metric_type: str = "custom_metric"):
    """Child Decorator. Measures execution time and captures I/O safely."""

    def decorator(func):
        
        def _record_metric(start_time, args, kwargs, result=None, error=None):
            """Helper to format and append the metric data safely."""
            current_metrics = active_run_metrics.get(None) # Safe fallback if parent isn't active
            if current_metrics is not None:
                end_time_utc = datetime.now(timezone.utc)
                start_time_utc = datetime.fromtimestamp(start_time, tz=timezone.utc)
                latency_ms = int((end_time_utc - start_time_utc).total_seconds() * 1000)
                
                # Combine args and kwargs safely
                inputs = {"args": args, "kwargs": kwargs}
                
                response_str = f"ERROR: {str(error)}" if error else _safe_stringify(result)

                metric_data = {
                    "model_name": metric_type,
                    "prompt": _safe_stringify(inputs),
                    "response": response_str,
                    "latency": latency_ms,
                    "start_time": start_time_utc.isoformat(),
                    "end_time": end_time_utc.isoformat()
                }
                current_metrics.append(metric_data)

        # ASYNC WRAPPER
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    _record_metric(start_time, args, kwargs, result=result)
                    return result
                except Exception as e:
                    _record_metric(start_time, args, kwargs, error=e)
                    raise # Re-raise so we don't break the user's app!
            return async_wrapper

        # SYNC WRAPPER
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    _record_metric(start_time, args, kwargs, result=result)
                    return result
                except Exception as e:
                    _record_metric(start_time, args, kwargs, error=e)
                    raise
            return sync_wrapper

    return decorator