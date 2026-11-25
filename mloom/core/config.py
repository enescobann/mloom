import json
import os
import warnings
from pathlib import Path
from typing import Optional

VALID_MODES = ("local", "remote")


class SDKConfig:
    def __init__(self):
        # Set defaults in memory immediately. NO disk I/O here.
        self.reset()

    def reset(self):
        """Restore in-memory defaults and forget anything loaded from disk."""
        self.backend_url: str = "http://localhost:8000"
        self.api_key: Optional[str] = None
        self.project_id: Optional[str] = None
        self.mode: str = "local"
        # Print a live trace line for every recorded run (see trace_output_enabled()).
        self.verbose: bool = True

        self._is_loaded = False

    @property
    def CONFIG_DIR(self) -> Path:
        # Resolved on every access so MLOOM_HOME / HOME changes are respected.
        override = os.environ.get("MLOOM_HOME")
        return Path(override) if override else Path.home() / ".mloom"

    @property
    def CONFIG_FILE(self) -> Path:
        return self.CONFIG_DIR / "config.json"

    @property
    def local_db_path(self) -> Path:
        """Dynamically generates the path and ensures the dir exists."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return self.CONFIG_DIR / "local.db"

    @property
    def local_db_url(self) -> str:
        return f"sqlite:///{self.local_db_path}"

    def load(self, force: bool = False):
        """Explicitly load config from disk. Called during init()."""
        if (self._is_loaded and not force) or not self.CONFIG_FILE.exists():
            return

        try:
            with open(self.CONFIG_FILE, "r") as f:
                data = json.load(f)
                self.mode = data.get("mode", self.mode)
                self.backend_url = data.get("backend_url", self.backend_url)
                self.project_id = data.get("project_id", self.project_id)
            self._is_loaded = True

        except json.JSONDecodeError:
            warnings.warn(f"[Mloom] Config file at {self.CONFIG_FILE} is corrupted. Using default settings.")
        except IOError as e:
            warnings.warn(f"[Mloom] Could not read config file: {e}")

    def save(self):
        """Persists the current in-memory state to disk."""
        try:
            self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, "w") as f:
                json.dump({
                    "mode": self.mode,
                    "backend_url": self.backend_url,
                    "project_id": self.project_id
                }, f, indent=4)
        except IOError as e:
            warnings.warn(f"[Mloom] Could not save config to disk: {e}")

# Python modules are naturally singletons.
# This instance will be shared across the entire application.
config = SDKConfig()

def trace_output_enabled() -> bool:
    """False when init(verbose=False) was used or MLOOM_QUIET is set."""
    quiet = os.environ.get("MLOOM_QUIET", "").strip().lower() in ("1", "true", "yes", "on")
    return config.verbose and not quiet


def init(
    project_id: str,
    backend_url: Optional[str] = None,
    api_key: Optional[str] = None,
    mode: Optional[str] = None,
    persist: bool = False,
    verbose: bool = True,
):
    """
    Entry point for the developer to initialize the SDK.

    `mode` is "local" (SQLite under ~/.mloom) or "remote" (POST to a mloom
    server). When omitted, the mode saved by `mloom server connect` is used,
    falling back to "local".

    `verbose=False` (or the MLOOM_QUIET=1 environment variable) silences the
    live trace printed to the terminal for every recorded run.
    """
    if mode is not None and mode not in VALID_MODES:
        raise ValueError(f"[Mloom] Invalid mode {mode!r}. Expected one of {VALID_MODES}.")

    # 1. Load any existing settings from disk first
    config.load()

    # 2. Override with explicit arguments passed to init()
    config.project_id = str(project_id) if project_id is not None else None
    if mode is not None:
        config.mode = mode
    config.api_key = api_key
    config.verbose = verbose

    if backend_url:
        config.backend_url = backend_url.rstrip("/")

    # 3. Only write to disk if the developer explicitly wants to save this profile
    if persist:
        config.save()

    # 4. Only local mode talks to a database from the client side.
    if config.mode == "local":
        from mloom.db.database import init_db
        init_db(config.local_db_url)
