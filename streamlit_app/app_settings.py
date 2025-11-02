# streamlit_app/config.py
import os
from dotenv import load_dotenv

# Always load root .env (repo root), then local streamlit_app/.env if present
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(REPO_ROOT, ".env"), override=False)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return default

# --- What your Streamlit code can import ---
AOAI_ENDPOINT   = _env("AZURE_OPENAI_ENDPOINT", "AZ_OPENAI_ENDPOINT")
AOAI_KEY        = _env("AZURE_OPENAI_API_KEY", "AZ_OPENAI_API_KEY", "AZ_OPENAI_KEY")
AOAI_API_VER    = _env("AZURE_OPENAI_API_VERSION", "AZ_OPENAI_API_VERSION") or "2024-12-01-preview"
AOAI_CHAT_DEPLOY  = _env("AZ_OPENAI_CHAT_DEPLOY") or "gpt-4o-mini"
AOAI_EMBED_DEPLOY = _env("AZ_OPENAI_EMBED_DEPLOY") or "text-embedding-3-large"

SEARCH_ENDPOINT = _env("AZURE_SEARCH_ENDPOINT", "AZ_SEARCH_ENDPOINT")
SEARCH_KEY      = _env("AZURE_SEARCH_API_KEY", "AZ_SEARCH_KEY")
SEARCH_INDEX    = _env("AZURE_SEARCH_INDEX", "AZ_SEARCH_INDEX") or "atm-knowledge"

#  THIS WAS MISSING - ADD THIS LINE:
RAG_API_URL     = _env("RAG_API_URL") or "http://localhost:8000"