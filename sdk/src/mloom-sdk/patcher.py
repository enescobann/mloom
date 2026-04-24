#auto pather for populer sdks. For now only OpenAI.

import time
from .emitter import emit_event
from .config import config
from .contextvar import active_run_metrics

def patch_openai():
    try:
        import openai
        from openai.resources.chat.completions import Completions
    except ImportError:
        return

    original_create = Completions.create

    def wrapped_create(*args, **kwargs):
        start_time = time.time()

        response = original_create(*args, **kwargs)

        latency_ms = (time.time() - start_time) * 1000

        if response and hasattr(response, "usage"):
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        else:
            input_tokens = 0
            output_tokens = 0

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        metric_data = {
            "model_name": model,
            "prompt": str(messages), #keep it simple for now or JSON serialize
            "response": str(response),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": int(latency_ms)
            #maybe costs here later
        }

        #if we are inside a @track_run pipeline
        current_metrics = active_run_metrics.get()

        if current_metrics is not None:
            #we are inside a pipeline so append
            current_metrics.append(metric_data)
        else:
            #NOT in a pipeline so pass the metric straight
            payload = {
                "project_id": config.project_id,
                "tags": {"provider": "openai",},
                "run_name": "OpenAI API Call",
                "latency_ms": latency_ms,
                "metrics": [metric_data]
                }
            emit_event(payload)

        return response
    
    Completions.create = wrapped_create