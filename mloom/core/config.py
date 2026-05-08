import json
import warnings
from pathlib import Path
from typing import Optional

class SDKConfig:
    def __init__(self):
        # Set defaults in memory immediately. NO disk I/O here.
        self.CONFIG_DIR = Path.home() / ".mloom"
        self.CONFIG_FILE = self.CONFIG_DIR / "config.json"
        
        self.backend_url: str = "http://localhost:8000"
        self.api_key: Optional[str] = None
        self.project_id: Optional[str] = None
        self.mode: str = "local"
        
        self._is_loaded = False

    @property
    def local_db_path(self) -> Path:
        """Dynamically generates the path and ensures the dir exists."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return self.CONFIG_DIR / "local.db"

    def load(self):
        """Explicitly load config from disk. Called during init()."""
        if self._is_loaded or not self.CONFIG_FILE.exists():
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

def init(
    project_id: str, 
    backend_url: Optional[str] = None, 
    api_key: Optional[str] = None,
    mode: str = "remote",
    persist: bool = False
):
    """
    Entry point for the developer to initialize the SDK.
    """
    # 1. Load any existing settings from disk first
    config.load()
    
    # 2. Override with explicit arguments passed to init()
    config.project_id = project_id
    config.mode = mode
    config.api_key = api_key
    
    if backend_url:
        config.backend_url = backend_url
        
    # 3. Only write to disk if the developer explicitly wants to save this profile
    if persist:
        config.save()
        
    # 4. Initialize the database connection (SQLite or Postgres) based on config
    from mloom.db.database import init_db
    init_db()