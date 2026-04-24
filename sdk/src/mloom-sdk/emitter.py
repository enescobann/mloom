import requests
import threading
from config import config

def _emit_event_sync(payload: dict):
    """Actual emitter and error handler."""
    if not config.backend_url or not config.project_id:
        return

    try:
        run_url = f"{config.backend_url}/runs/"
        provider = payload.get("provider")
        run_payload = {
            "run_type": "LLM",
            "run_name": payload.get("run_name", "Unnamed Run"),
            "project_id": config.project_id,
            "latency": payload.get("latency_ms"),
            "tags": {"provider": provider} if provider else {}
        }
        run_response = requests.post(run_url, json=run_payload, timeout=2.0)
        run_response.raise_for_status()
        run_data = run_response.json()
        run_id = run_data["id"]

        metrics_url = f"{config.backend_url}/runs/{run_id}/metrics/"
        metrics_payload = {
            "model_name": payload.get("model_name"),
            "prompt": payload.get("prompt"),
            "response": payload.get("response"),
            "input_tokens": payload.get("input_tokens"),
            "output_tokens": payload.get("output_tokens")
        }
        metrics_response = requests.post(metrics_url, json=metrics_payload, timeout=2.0)
        metrics_response.raise_for_status()

    except Exception as e:
        pass

def emit_event(payload: dict):
    """Fires the HTTP request in a background thread."""
    thread = threading.Thread(target=_emit_event_sync, args=(payload,))
    thread.daemon = True
    thread.start()