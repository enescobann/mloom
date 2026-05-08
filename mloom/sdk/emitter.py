import threading
import queue
import atexit
from requests import Session
from mloom.core.config import config

# 1. Setup a global session
session = Session()
session.headers.update({"Content-Type": "application/json"})

# 2. Create a thread-safe Queue (maxsize prevents memory leaks if backend is down)
_event_queue = queue.Queue(maxsize=10000)

# 3. Cache to avoid repeated DB calls in local mode
_project_verified = False 

def _emit_event_sync(_payload: dict):
    """Actual emitter and error handler."""
    global _project_verified
    
    if not config.project_id:
        print("[Mloom Warning]: `project_id` is not set. Event not recorded.")
        return

    try:
        if config.mode == "local":
            # Lazy imports
            from mloom.db.database import SessionLocal
            from mloom.db.classes import Run, LLMMetrics, Project
            
            with SessionLocal() as db:
                # Cache project check so we don't query it on every single run
                if not _project_verified:
                    proj = db.query(Project).filter_by(id=config.project_id).first()
                    if not proj:
                        proj = Project(id=config.project_id, name=f"Project_{config.project_id}")
                        db.add(proj)
                        db.commit()
                    _project_verified = True

                new_run = Run(
                    run_name=_payload.get("run_name", "Unnamed Run"),
                    project_id=config.project_id,
                    run_type=_payload.get("run_type", "LLM"),
                    latency=_payload.get("latency", _payload.get("latency_ms")),
                    tags=_payload.get("tags", {})
                )
                db.add(new_run)
                db.commit()
                db.refresh(new_run)

                metrics_to_add = []
                for metric_data in _payload.get("metrics", []):
                    metrics_to_add.append(LLMMetrics(
                        run_id=new_run.id,
                        model_name=metric_data.get("model_name"),
                        prompt=metric_data.get("prompt"),
                        response=metric_data.get("response"),
                        input_tokens=metric_data.get("input_tokens"),
                        output_tokens=metric_data.get("output_tokens"),
                        latency=metric_data.get("latency"),
                        total_cost=metric_data.get("total_cost")
                    ))
                
                # Bulk save metrics instead of committing one by one
                if metrics_to_add:
                    db.add_all(metrics_to_add)
                    db.commit()

        else:
            if not config.backend_url:
                print("[Mloom Warning]: `backend_url` is not set in remote mode.")
                return
            
            url = f"{config.backend_url}/runs/"
            payload = {
                "run_type": _payload.get("run_type", "LLM"),
                "run_name": _payload.get("run_name", "Unnamed Run"),
                "project_id": config.project_id,
                "latency": _payload.get("latency", _payload.get("latency_ms")),
                "tags": _payload.get("tags", {}),
                "metrics": _payload.get("metrics", [])
            }
            # The session handles connection pooling automatically
            response = session.post(url, json=payload, timeout=2.0)
            response.raise_for_status()

    except Exception as e:
        print(f"[Mloom Error]: Failed to emit event. Reason: {e}")

def _worker_loop():
    """Background thread loop that processes events from the queue."""
    while True:
        payload = _event_queue.get()
        if payload is None:  # Sentinel value indicating shutdown
            _event_queue.task_done()
            break
            
        _emit_event_sync(payload)
        _event_queue.task_done()

# 4. Start the single background worker thread
_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()

def emit_event(payload: dict):
    """Puts the event in the queue. Almost instantaneous."""
    try:
        # put_nowait ensures the main application thread NEVER blocks
        _event_queue.put_nowait(payload)
    except queue.Full:
        print("[Mloom Error]: Telemetry queue is full. Event dropped.")

# 5. Graceful shutdown to prevent data loss
def flush_events():
    """Ensures all pending events are sent before the program exits."""
    if _worker_thread.is_alive():
        _event_queue.put(None)  # Send shutdown signal
        _worker_thread.join(timeout=3.0)  # Wait up to 3 seconds for queue to drain

atexit.register(flush_events)