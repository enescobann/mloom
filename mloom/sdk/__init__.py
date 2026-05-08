from mloom.core.config import init
from .decorators import track_run as track_llm
from .decorators import track_metric
from .patcher import patch_openai

def autotrack():
    """Call this to automatically patch supported libraries."""
    patch_openai()

__all__ = ["init", "track_llm", "track_metric", "autotrack"]
