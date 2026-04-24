#I am aiming to provide tools for custom pipelines for developers.
#Currentyl only will gets run time of the funcs, inputs and outputs.

import time
import functools
from .emitter import emit_event
from .contextvar import active_run_metrics
from .config import config

def track_run(name: str = None, run_type: str = None):
    """Parent Decorator.
    Initalizes the trackinge lements and emits.
    track_metric decorators needed to capture metrics.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            #needs to be initialized before func call to let child decorater store data in it.
            contextvar_token = active_run_metrics.set([])
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency_ms = (time.time() - start_time) * 1000


                collected_metrics = active_run_metrics.get()
                payload = {
                    "project_id": config.project_id,
                    "run_name": name or func.__name__,
                    "run_type": run_type,
                    "latency": latency_ms,
                    "metrics": collected_metrics
                }

                emit_event(payload)

                active_run_metrics.reset(contextvar_token)

        return wrapper
    return decorator

def track_metric(metric_type: str = "custom_metric"):
    """Child Decorator.
    Measures execution time of the function, catches prompt and 
    outpu if there is any and stores it on the active_run_metrics.

    Dont needed on the calls of supported SDKs.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            
            latency_ms = (time.time() - start_time) * 1000

            current_metrics = active_run_metrics.get()

            if current_metrics is not None:
                metric_data = {
                    "model_name": metric_type,
                    "prompt": str(kwargs),
                    "response": str(result)
                    #...
                }

                current_metrics.append(metric_data)
            
            return result
        return wrapper
    return decorator