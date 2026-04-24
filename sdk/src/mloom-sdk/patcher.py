#auto pather for populer sdks. For now only OpenAI.

import time
from .emitter import emit_event
from .config import config

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

        payload = {
            "project_id": config.project_id,
            "provider": "openai",
            "run_name": "OpenAI API Call",
            "model_name": model,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt": str(messages), #keep it simple for now or JSON serialize
        }

        emit_event(payload)

        return response
    
    Completions.create = wrapped_create