from contextvars import ContextVar

#big global var that you can pass around

active_run_metrics: ContextVar[list] = ContextVar("active_run_metrics", default=None)