class SDKConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SDKConfig, cls).__new__(cls)
            cls._instance.backend_url = "http://localhost:8000"
            cls._instance.api_key = None
            cls._instance.project_id = None
        return cls._instance

config = SDKConfig()

def init(backend_url: str, project_id: str, api_key: str | None = None):
    config.backend_url = backend_url
    config.project_id = project_id
    config.api_key = api_key