#I am aiming to provide tools for custom pipelines for developers.
#Currentyl only will calculate latency and func output.

import time
import functools
from emitter import emit_event

def track_llm(name: str = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            result = func(*args, **kwargs)

            latency_ms= (time.time() - start_time) * 1000

            payload = {
                "run_name": name or func.__name__,
                "latency_ms": latency_ms,
            }

            emit_event(payload)

            return result
        return wrapper
    return decorator