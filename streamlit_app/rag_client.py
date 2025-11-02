"""
RAG API Client for Streamlit
FILE: streamlit_app/rag_client.py

Handles communication between Streamlit and FastAPI RAG service.
"""

import httpx
import pandas as pd
from typing import Dict, Any, Optional, List
import app_settings as config

# --- JSON-safe conversion for pandas/NumPy/datetimes ---
import pandas as pd
import numpy as np
from datetime import datetime, date
import requests  


def df_to_json_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-serializable list[dict].
    - Datetime/Timestamp -> 'YYYY-MM-DD' strings
    - NumPy scalars -> native Python types
    - NaN -> None
    """
    df = df.copy()

    # 1) Normalize any datetime-like columns to ISO strings
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            # If the column is 'object' and some cells are datetime/date, format them
            df[col] = df[col].apply(
                lambda x: x.isoformat() if isinstance(x, (datetime, date)) else x
            )

    # 2) Convert numpy scalars and NaN to native types
    def to_native(x):
        if isinstance(x, (np.integer,)):   return int(x)
        if isinstance(x, (np.floating,)):  return float(x)
        if pd.isna(x):                     return None
        return x

    records = df.to_dict(orient="records")
    records = [{k: to_native(v) for k, v in row.items()} for row in records]
    return records


class RAGClient:
    """Client for interacting with the RAG API."""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or config.RAG_API_URL
        self.timeout = 30.0
    
    def health_check(self) -> Dict[str, Any]:
        """Check if RAG API is running and configured."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return {"status": "ok", "data": response.json()}
        except httpx.ConnectError:
            return {"status": "error", "message": "Cannot connect to RAG API. Is it running?"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    

    def upload_data(self, df: pd.DataFrame, timeout_sec: int = 900) -> dict:
        """POST /data/upload with a long read-timeout for large datasets."""
        payload = {"data": df_to_json_records(df)}
        url = f"{self.base_url}/data/upload"
        # timeout=(connect_timeout, read_timeout)
        resp = requests.post(url, json=payload, timeout=(10, timeout_sec))
        resp.raise_for_status()
        return resp.json()
    
    def query(self, question: str, atm_id: Optional[str] = None, use_tools: bool = True) -> Dict[str, Any]:
        """Send a natural language query to the RAG system."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/query",
                    json={
                        "query": question,
                        "atm_id": atm_id,
                        "use_tools": use_tools
                    }
                )
                response.raise_for_status()
                return {"status": "success", "data": response.json()}
        except httpx.TimeoutException:
            return {"status": "error", "message": "Request timed out. Query may be too complex."}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_cashout_risk(self, threshold: float = 0.2) -> Dict[str, Any]:
        """Get ATMs at risk of cash-out."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.base_url}/tools/cashout-risk",
                    params={"threshold": threshold}
                )
                response.raise_for_status()
                return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_refill_suggestions(self, buffer_days: int = 2) -> Dict[str, Any]:
        """Get refill recommendations."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.base_url}/tools/refill-suggestions",
                    params={"buffer_days": buffer_days}
                )
                response.raise_for_status()
                return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "message": str(e)}