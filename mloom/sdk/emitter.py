import atexit
import os
import queue
import sys
import threading
import time
from typing import Optional

from requests import Session
from mloom.core.config import config

_QUEUE_MAXSIZE = 10000
_REMOTE_TIMEOUT = 2.0
_STOP = object()


def _new_session() -> Session:
    s = Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


# 1. Setup a global session (only ever used from the worker thread)
session = _new_session()

# 2. Create a thread-safe Queue (maxsize prevents memory leaks if backend is down)
_event_queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)

# 3. The worker thread is started lazily on the first event.
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_shutting_down = False


def _warn(message: str):
    print(message, file=sys.stderr)


def _emit_local(payload: dict, project_id: str):
    # Lazy imports: SQLAlchemy is only needed once something is recorded locally.
    from mloom.db import database
    from mloom.db.crud import create_run_record
    from mloom.server.trace_logger import log_run_trace

    if database.engine is None:
        database.init_db(config.local_db_url)

    with database.get_session() as db:
        new_run = create_run_record(db, {**payload, "project_id": project_id})
        # Live trace output to terminal
        log_run_trace(new_run)


def _emit_remote(payload: dict, project_id: str):
    if not config.backend_url:
        _warn("[Mloom Warning]: `backend_url` is not set in remote mode.")
        return

    url = f"{config.backend_url.rstrip('/')}/runs/"
    body = {
        "run_type": payload.get("run_type") or "LLM",
        "run_name": payload.get("run_name") or "Unnamed Run",
        "project_id": project_id,
        "latency": payload.get("latency", payload.get("latency_ms")),
        "tags": payload.get("tags") or {},
        "metrics": payload.get("metrics") or [],
        "error_trace": payload.get("error_trace"),
        "call_site": payload.get("call_site"),
        "start_time": payload.get("start_time"),
        "end_time": payload.get("end_time"),
    }
    # The session handles connection pooling automatically
    response = session.post(url, json=body, timeout=_REMOTE_TIMEOUT)
    response.raise_for_status()


def _emit_event_sync(_payload: dict):
    """Actual emitter and error handler."""
    project_id = _payload.get("project_id") or config.project_id
    if not project_id:
        _warn("[Mloom Warning]: `project_id` is not set. Event not recorded.")
        return

    try:
        if config.mode == "local":
            _emit_local(_payload, str(project_id))
        else:
            _emit_remote(_payload, str(project_id))
    except Exception as e:
        _warn(f"[Mloom Error]: Failed to emit event. Reason: {e}")


def _worker_loop(event_queue: queue.Queue):
    """Background thread loop that processes events from the queue."""
    while True:
        payload = event_queue.get()
        try:
            if payload is _STOP:
                return
            _emit_event_sync(payload)
        finally:
            event_queue.task_done()


def _ensure_worker():
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(
                target=_worker_loop, args=(_event_queue,), name="mloom-emitter", daemon=True
            )
            _worker_thread.start()


def emit_event(payload: dict):
    """Puts the event in the queue. Almost instantaneous."""
    if _shutting_down:
        # The worker is gone once the interpreter is exiting; write inline instead.
        _emit_event_sync(payload)
        return

    _ensure_worker()
    try:
        # put_nowait ensures the main application thread NEVER blocks
        _event_queue.put_nowait(payload)
    except queue.Full:
        _warn("[Mloom Error]: Telemetry queue is full. Event dropped.")


def flush(timeout: Optional[float] = None) -> bool:
    """
    Block until every queued event has been written.

    Returns False if `timeout` (seconds) expired first. The worker keeps
    running afterwards, so flush() can be called any number of times.
    """
    if threading.current_thread() is _worker_thread:
        return False

    event_queue = _event_queue
    deadline = None if timeout is None else time.monotonic() + timeout
    with event_queue.all_tasks_done:
        if event_queue.unfinished_tasks and not _shutting_down:
            _ensure_worker()
        while event_queue.unfinished_tasks:
            if deadline is None:
                event_queue.all_tasks_done.wait()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                event_queue.all_tasks_done.wait(remaining)
    return True


def flush_events(timeout: Optional[float] = None) -> bool:
    """Backwards-compatible alias for flush()."""
    return flush(timeout)


# 4. Graceful shutdown to prevent data loss
def _shutdown():
    global _shutting_down
    flush(timeout=5.0)
    _shutting_down = True
    worker = _worker_thread
    if worker is not None and worker.is_alive():
        try:
            _event_queue.put_nowait(_STOP)
        except queue.Full:
            return
        worker.join(timeout=1.0)


def _reset_after_fork():
    # Threads, locks and pooled sockets do not survive fork(); start clean in the child.
    global _event_queue, _worker_thread, _worker_lock, session
    _event_queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    _worker_thread = None
    _worker_lock = threading.Lock()
    session = _new_session()


atexit.register(_shutdown)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
