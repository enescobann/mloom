# sdk/src/mlflow_copy/__init__.py
from .config import init
from .decorators import track_llm
from .patcher import patch_openai

def autotrack():
    """Call this to automatically patch supported libraries."""
    patch_openai()

__all__ = ["init", "track_llm", "autotrack"]
