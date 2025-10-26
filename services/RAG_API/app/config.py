import os
from dataclasses import dataclass

@dataclass
class Settings:
    # Search
    search_endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    search_api_key: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    search_index: str   = os.getenv("AZURE_SEARCH_INDEX", "")

    # AOAI
    aoai_endpoint: str  = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    aoai_api_key: str   = os.getenv("AZURE_OPENAI_API_KEY", "")
    aoai_deployment: str= os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    aoai_api_version: str=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    # App
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))

settings = Settings()
