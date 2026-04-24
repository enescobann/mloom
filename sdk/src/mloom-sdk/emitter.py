from time import time
from requests import request, Session
import requests
import threading
from .config import config

session = Session()
session.headers.update({
    "Content-Type": "application/json"
})

def _emit_event_sync(_payload: dict):
    """Actual emitter and error handler."""
    if not config.backend_url or not config.project_id:
        return

    try:
        url = f"{config.backend_url}/runs/"
        provider = _payload.get("provider")

        payload = {
            "run_type": "LLM",
            "run_name": _payload.get("run_name", "Unnamed Run"),
            "project_id": config.project_id,
            "latency": _payload.get("latency", _payload.get("latency_ms")),
            "tags": {"provider": provider} if provider else {},
            "metrics": _payload.get("metrics", [])
        }
        response = session.post(url, json=payload, timeout=2.0)
        response.raise_for_status()

    except Exception as e:
        pass

def emit_event(payload: dict):
    """Fires the HTTP request in a background thread."""
    thread = threading.Thread(target=_emit_event_sync, args=(payload,))
    thread.daemon = True
    thread.start()