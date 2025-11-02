"""
FastAPI RAG Service for ATM Cash Management - FINAL CORRECTED VERSION
FILE: services/RAG_API/app.py
"""

from __future__ import annotations
import os, re, hashlib
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from pydantic import BaseModel
import traceback

# Load environment variables
load_dotenv()
load_dotenv("../../.env")

# ============= JSON Serialization Helper =============
def safe_json_value(val: Any) -> Any:
    """Convert pandas/numpy types to JSON-safe Python types."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, np.bool_):
        return bool(val)
    return val

# ============= App Setup =============
app = FastAPI(title="ATM RAG API", version="1.2.0")

# create the API service and allow CORS from your Streamlit front-end (default http://localhost:8501).
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# search by 
def getenv(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default

# ============= Configuration =============
# azure openai config

#URL of your Azure OpenAI service.
AOAI_ENDPOINT = getenv("AZURE_OPENAI_ENDPOINT", "AOAI_ENDPOINT")
#API key to talk to your Azure OpenAI.
AOAI_KEY = getenv("AZURE_OPENAI_API_KEY", "AOAI_KEY") 

#API versions and model to use for Azure OpenAI.
AOAI_API_VERSION = getenv("AZURE_OPENAI_API_VERSION", "AZ_OPENAI_API_VERSION", default="2024-08-01-preview")
AOAI_CHAT_DEPLOY = getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "AZ_OPENAI_CHAT_DEPLOY", default="gpt-4o-mini")
AOAI_EMBED_DEPLOY = getenv("AZURE_OPENAI_EMBED_MODEL", "AZ_OPENAI_EMBED_DEPLOY", default="text-embedding-3-large")

# azure search config
SEARCH_ENDPOINT = getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = getenv("AZURE_SEARCH_API_KEY")

# knowledge data (sop,logic)
SEARCH_INDEX = getenv("AZURE_SEARCH_INDEX", default="atm-knowledge")
KB_SEM_CONFIG = getenv("KB_SEMANTIC_CONFIG", default="kb-semantic")

#actual data 
DATA_SEM_CONFIG = getenv("DATA_SEMANTIC_CONFIG", default="atm-semantic")
DATA_INDEX_V2 = getenv("ATM_DATA_INDEX_V2", default="atm-data-v2")

# both are using content field for searching 
DATA_VECTOR_FIELD = getenv("DATA_VECTOR_FIELD", default="content_vector")
KB_VECTOR_FIELD = getenv("KB_VECTOR_FIELD", default="content_vector")

# embedding support semantic search 
INDEX_WITH_EMBEDDINGS = getenv("INDEX_WITH_EMBEDDINGS", default="false").lower() == "true"

# ============= Heavy Dependencies =============
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import tools

# ============= Globals =============
atm_dataframe: Optional[pd.DataFrame] = None
_aoai_client: Optional[AzureOpenAI] = None
_kb_client: Optional[SearchClient] = None
_data_client: Optional[SearchClient] = None

def aoai_client() -> AzureOpenAI:
    assert _aoai_client is not None, "AzureOpenAI client not initialized"
    return _aoai_client

def kb_client() -> SearchClient:
    assert _kb_client is not None, "Knowledge SearchClient not initialized"
    return _kb_client

def data_client() -> SearchClient:
    assert _data_client is not None, "Data SearchClient not initialized"
    return _data_client

# ============= Startup =============
@app.on_event("startup")
async def _init_clients() -> None:
    global _aoai_client, _kb_client, _data_client

    missing = []
    if not AOAI_ENDPOINT: missing.append("AZURE_OPENAI_ENDPOINT")
    if not AOAI_KEY: missing.append("AZURE_OPENAI_API_KEY")
    if not SEARCH_ENDPOINT: missing.append("AZURE_SEARCH_ENDPOINT")
    if not SEARCH_KEY: missing.append("AZURE_SEARCH_API_KEY")

    if missing:
        print(f"[WARN] Missing env: {', '.join(missing)}")

    if AOAI_ENDPOINT and AOAI_KEY:
        try:
            _aoai_client = AzureOpenAI(
                api_key=AOAI_KEY,
                api_version=AOAI_API_VERSION,
                azure_endpoint=AOAI_ENDPOINT,
            )
            print("[OK] AzureOpenAI client initialized")
        except Exception as e:
            print(f"[ERROR] AzureOpenAI init failed: {e}")

    if SEARCH_ENDPOINT and SEARCH_KEY:
        cred = AzureKeyCredential(SEARCH_KEY)
        try:
            _kb_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX, credential=cred)
            print(f"[OK] Knowledge SearchClient: {SEARCH_INDEX}")
        except Exception as e:
            print(f"[ERROR] KB SearchClient init failed: {e}")

        try:
            _data_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=DATA_INDEX_V2, credential=cred)
            print(f"[OK] Data SearchClient: {DATA_INDEX_V2}")
        except Exception as e:
            print(f"[ERROR] Data SearchClient init failed: {e}")

@app.on_event("startup")
async def _show_routes() -> None:
    print("=== ROUTES ===")
    for r in app.routes:
        if isinstance(r, APIRoute):
            print(f"{','.join(sorted(r.methods)):8s} {r.path}")
    print("==============")

# ============= Models =============
class QueryRequest(BaseModel):
    query: str
    atm_id: Optional[str] = None
    use_tools: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    tool_results: Optional[Dict[str, Any]] = None

class IndexDocumentRequest(BaseModel):
    content: str
    title: str
    category: str
    metadata: Optional[Dict[str, Any]] = None

class UploadDataRequest(BaseModel):
    data: List[Dict[str, Any]]

# ============= Search Functions =============
def get_embedding(text: str) -> List[float]:
    resp = aoai_client().embeddings.create(input=text, model=AOAI_EMBED_DEPLOY)
    return resp.data[0].embedding

def hybrid_search_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    try:
        query_vector = get_embedding(query)
        print(f'this is vector embedding{query_vector[:5]}')
        vq = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields=KB_VECTOR_FIELD)
        kwargs: Dict[str, Any] = dict(
            search_text=query,
            vector_queries=[vq],
            select=["id","content","title","category","@search.score"],
            top=top_k,
        )
        if KB_SEM_CONFIG:
            kwargs.update(query_type="semantic", semantic_configuration_name=KB_SEM_CONFIG)
        results = kb_client().search(**kwargs)
        return [{
            "id": r.get("id"),
            "content": r.get("content"),
            "title": r.get("title"),
            "category": r.get("category"),
            "score": r.get("@search.score"),
            "source": "knowledge_base",
        } for r in results]
    except Exception as e:
        print(f"[KB search error] {e}")
        return []

def search_transactional_data(query: str, top_k: int = 50) -> List[Dict[str, Any]]:
    try:
        new_data_client = SearchClient(
            endpoint=SEARCH_ENDPOINT,
            index_name=DATA_INDEX_V2,
            credential=AzureKeyCredential(SEARCH_KEY)
        )
        docs: List[Dict[str, Any]] = []

        # Try semantic search first
        try:
            results = new_data_client.search(
                search_text=query,
                select=["ATM_ID","City","Location","Date","Withdrawals","Deposits","@search.score"],
                top=top_k, query_type="semantic", semantic_configuration_name=DATA_SEM_CONFIG,
            )
            for r in results:
                docs.append({
                    "ATM_ID": r.get("ATM_ID"), "City": r.get("City"), "Location": r.get("Location"),
                    "Date": r.get("Date"), "Withdrawals": r.get("Withdrawals"), "Deposits": r.get("Deposits"),
                    "score": r.get("@search.score"), "source": "transaction_data",
                })
        except Exception:
            pass

        # Try vector hybrid
        if not docs:
            try:
                qvec = get_embedding(query)
                vq = VectorizedQuery(vector=qvec, k_nearest_neighbors=top_k, fields=DATA_VECTOR_FIELD)
                results = new_data_client.search(
                    search_text=query, vector_queries=[vq],
                    select=["ATM_ID","City","Location","Date","Withdrawals","Deposits","@search.score"], top=top_k,
                )
                for r in results:
                    docs.append({
                        "ATM_ID": r.get("ATM_ID"), "City": r.get("City"), "Location": r.get("Location"),
                        "Date": r.get("Date"), "Withdrawals": r.get("Withdrawals"), "Deposits": r.get("Deposits"),
                        "score": r.get("@search.score"), "source": "transaction_data",
                    })
            except Exception:
                pass

        # Keyword fallback
        if not docs:
            results = new_data_client.search(
                search_text=query,
                select=["ATM_ID","City","Location","Date","Withdrawals","Deposits","@search.score"],
                top=top_k,
            )
            for r in results:
                docs.append({
                    "ATM_ID": r.get("ATM_ID"), "City": r.get("City"), "Location": r.get("Location"),
                    "Date": r.get("Date"), "Withdrawals": r.get("Withdrawals"), "Deposits": r.get("Deposits"),
                    "score": r.get("@search.score", 0), "source": "transaction_data",
                })
        return docs
    except Exception as e:
        print(f"[Data search ERROR] {e}")
        return []

# ============= Intent Routing =============
def extract_atm_id_from_query(query: str) -> Optional[str]:
    """Extract ATM_ID from query text."""
    m = re.search(r'ATM[_-\s]?(\d+)', query, re.IGNORECASE)
    if not m: return None
    return f"ATM_{m.group(1).zfill(3)}"

def route_query_to_tool(query: str, atm_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Route query to appropriate tool based on intent."""
    try:
        q = query.lower()

        if not atm_id:
            atm_id = extract_atm_id_from_query(query)

        # Critical/Immediate attention
        if any(k in q for k in ["critical", "immediate attention", "urgent atms"]):
            results = tools.compute_cashout_risk(threshold=0.1)
            critical = [r for r in results if r.get("risk_category") in ("CRITICAL", "HIGH")]
            return {"tool": "compute_cashout_risk", "results": critical}

        # Risk distribution overview
        if "risk distribution" in q or ("overall" in q and "risk" in q):
            return {"tool": "risk_distribution_snapshot", "results": tools.risk_distribution_snapshot()}

        # Cash out risk
        if any(k in q for k in ["risk", "cash out", "cashout", "running out", "shortage"]):
            return {"tool": "compute_cashout_risk", "results": tools.compute_cashout_risk(threshold=0.2)}

        # Weekend preparation
        if any(k in q for k in ["weekend", "this weekend", "weekend preparation"]):
            return {"tool": "weekend_preparation_report", "results": tools.weekend_preparation_report()}

        # Urgent refills (next 24 hours)
        if any(k in q for k in ["urgent refill", "next 24 hours", "immediate refill"]):
            return {"tool": "refill_plan_next_24h", "results": tools.refill_plan_next_24h()}

        # Refill suggestions
        if any(k in q for k in ["refill amounts", "refill plan", "replenish", "how much cash"]):
            return {"tool": "refill_suggestion", "results": tools.refill_suggestion(buffer_days=2)}

        # Top performers
        if any(k in q for k in ["top 10", "top performing", "highest withdrawal", "best atm"]):
            return {"tool": "atm_performance_ranking", "results": tools.atm_performance_ranking(metric="withdrawals", top_n=10)}

        # City comparison
        if "compare" in q and ("cit" in q or "cities" in q):
            return {"tool": "city_comparison_report", "results": tools.city_comparison_report()}

        # Demand pattern
        if ("pattern" in q or "demand for" in q) and atm_id:
            return {"tool": "demand_pattern_analysis", "results": tools.demand_pattern_analysis(atm_id=atm_id)}

        # Location optimization
        if any(k in q for k in ["where to place", "new atm", "expansion", "best location", "maximum impact"]):
            return {"tool": "location_optimization_analysis", "results": tools.location_optimization_analysis()}

        # Operational efficiency
        if any(k in q for k in ["efficiency", "operational", "metrics", "kpi"]):
            return {"tool": "operational_efficiency_metrics", "results": tools.operational_efficiency_metrics()}

        # Predictions
        if any(k in q for k in ["predict", "prediction", "forecast", "tomorrow", "next day"]):
            return {"tool": "predict_atm_demand", "results": tools.predict_atm_demand(atm_id=atm_id)}

        return None
    except Exception as e:
        print(f"[Tool routing error] {e}")
        traceback.print_exc()
        return None

# ============= Answer Generation =============
def generate_answer(query: str, tool_data: Optional[Dict[str, Any]] = None) -> str:
    """Generate answer using KB, transaction data, and tool results."""
    
    # Search both indexes
    knowledge_docs = hybrid_search_knowledge(query, top_k=2)
    data_docs = search_transactional_data(query, top_k=30)

    # Build knowledge context
    kb_context = ""
    if knowledge_docs:
        kb_parts = []
        for doc in knowledge_docs:
            title = doc.get('title', 'Untitled')
            content = (doc.get('content') or '')[:500]
            kb_parts.append(f"**{title}**\n{content}")
        kb_context = "\n\n".join(kb_parts)

    # Build transaction data context
    data_context = ""
    if data_docs:
        df = pd.DataFrame(data_docs)
        for c in ("Withdrawals", "Deposits"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        unique_atms = df['ATM_ID'].nunique() if 'ATM_ID' in df else 0
        unique_cities = df['City'].nunique() if 'City' in df else 0
        total_withdrawals = df['Withdrawals'].sum() if 'Withdrawals' in df else 0
        avg_withdrawals = df['Withdrawals'].mean() if 'Withdrawals' in df else 0

        data_summary = f"""DATA SUMMARY ({len(df)} transactions):
- Unique ATMs: {unique_atms}
- Cities: {unique_cities}
- Total Withdrawals: {total_withdrawals:,.0f} KD
- Avg Withdrawal: {avg_withdrawals:,.0f} KD"""

        if len(df) > 15 and {'ATM_ID','Withdrawals'}.issubset(df.columns):
            atm_summary = df.groupby('ATM_ID').agg({
                'City': 'first',
                'Location': 'first',
                'Withdrawals': 'sum',
            }).reset_index().sort_values('Withdrawals', ascending=False).head(15)
            data_context = f"{data_summary}\n\nTOP ATMs:\n{atm_summary.to_string(index=False)}"
        else:
            cols = [c for c in ['ATM_ID','City','Location','Date','Withdrawals'] if c in df.columns]
            data_context = f"{data_summary}\n\nRECORDS:\n{df[cols].head(15).to_string(index=False)}"

    # Build tool context
    tool_context = ""
    if tool_data and "results" in tool_data:
        tool_name = tool_data.get("tool", "")
        results = tool_data["results"]

        # Format predictions specially
        if tool_name == "predict_atm_demand":
            if isinstance(results, dict):
                if "prediction" in results:
                    p = results["prediction"]
                    tool_context = f"""🔮 ML PREDICTION:
- ATM: {p.get('ATM_ID')}
- Predicted Demand: {p.get('Predicted_Demand_KD', 0):,} KD
- Date: {p.get('Prediction_Date')}
- Model: {p.get('Model_Used')}
- Recommendation: {p.get('Recommendation')}"""
                elif "predictions" in results:
                    preds = results["predictions"][:5]
                    lines = [f"- {p['ATM_ID']} ({p['City']}): {p['Predicted_Demand_KD']:,} KD" for p in preds]
                    tool_context = f"""🔮 ML PREDICTIONS (Top 5):
{chr(10).join(lines)}
Total Demand: {results.get('total_predicted_demand_kd', 0):,} KD"""

        # Format risk distribution
        elif tool_name == "risk_distribution_snapshot":
            counts = results.get("risk_counts", {})
            tool_context = f"""🚨 RISK DISTRIBUTION:
- CRITICAL: {counts.get('CRITICAL', 0)} ATMs
- HIGH: {counts.get('HIGH', 0)} ATMs
- MEDIUM: {counts.get('MEDIUM', 0)} ATMs
- LOW: {counts.get('LOW', 0)} ATMs
Total: {results.get('total_atms', 0)} ATMs"""

        # Format other tool results
        elif isinstance(results, list) and len(results) > 0:
            df_tool = pd.DataFrame(results).head(10)
            tool_context = f"""🔧 {tool_name.upper()}:\n{df_tool.to_string(index=False)}"""
        elif isinstance(results, dict):
            tool_context = f"🔧 {tool_name.upper()}: {str(results)[:600]}"

    # System prompt
    system_prompt = """You are an ATM cash management expert.

CRITICAL RULES:
1. For "how many ATMs" → Count UNIQUE ATM_IDs (use nunique()), NOT transaction count
2. For predictions → Use ML prediction results as PRIMARY source, cite specific KD amounts and ATM_IDs
3. For risk → Identify ATMs by ID with their risk levels (CRITICAL/HIGH/MEDIUM/LOW)
4. For refills → Provide specific amounts in KD per ATM_ID
5. Always cite ATM_IDs, cities, and amounts with numbers

RESPONSE FORMAT:
**ANSWER:** Direct answer with specific numbers
**EVIDENCE:** Bullet points with ATM_IDs and values
**ACTION:** What to do next with timing"""

    # User prompt
    user_prompt = f"""Question: {query}

ML PREDICTIONS & ANALYTICS:
{tool_context if tool_context else "No analytics available"}

TRANSACTION DATA:
{data_context if data_context else "No historical data"}

POLICIES & PROCEDURES:
{kb_context if kb_context else "No policies"}

Answer using the format above. Be specific with ATM_IDs and KD amounts."""

    # Generate answer
    try:
        resp = aoai_client().chat.completions.create(
            model=AOAI_CHAT_DEPLOY,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        traceback.print_exc()
        return f"Error generating answer: {str(e)}"

# ============= Endpoints =============
@app.get("/health")
def health_check() -> Dict[str, Any]:
    data_count = None
    try:
        if _data_client:
            data_count = _data_client.get_document_count()
    except Exception:
        pass

    return {
        "status": "healthy",
        "aoai_configured": bool(AOAI_ENDPOINT and AOAI_KEY and _aoai_client),
        "search_configured": bool(SEARCH_ENDPOINT and SEARCH_KEY and _data_client and _kb_client),
        "data_loaded": atm_dataframe is not None,
        "chat_deploy": AOAI_CHAT_DEPLOY,
        "embed_deploy": AOAI_EMBED_DEPLOY,
        "kb_index": SEARCH_INDEX,
        "data_index": DATA_INDEX_V2,
        "data_count": data_count,
    }

@app.post("/data/upload")
async def upload_atm_data(request: UploadDataRequest) -> Dict[str, Any]:
    """Upload ATM CSV rows with proper JSON serialization."""
    global atm_dataframe
    try:
        print(f"[Upload] Received {len(request.data)} records")
        
        atm_dataframe = pd.DataFrame(request.data)
        
        # Column normalization
        colmap = {"Location_Type": "Location", "CityName": "City", "ATMId": "ATM_ID", "TxnDate": "Date"}
        for src, dst in colmap.items():
            if src in atm_dataframe.columns and dst not in atm_dataframe.columns:
                atm_dataframe[dst] = atm_dataframe[src]

        # Sanitize ALL numeric columns
        for col in atm_dataframe.columns:
            if atm_dataframe[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                atm_dataframe[col] = atm_dataframe[col].apply(safe_json_value)

        tools.atm_dataframe = atm_dataframe

        # Build search documents
        search_docs: List[Dict[str, Any]] = []
        for _, row in atm_dataframe.iterrows():
            atm_id = str(row.get("ATM_ID", ""))
            city = str(row.get("City", ""))
            location = str(row.get("Location", row.get("Location_Type", "")))
            date_str = str(row.get("Date", ""))
            
            # Use safe_json_value for all numeric fields
            withdrawals = safe_json_value(row.get("Withdrawals", 0))
            deposits = safe_json_value(row.get("Deposits", 0))
            day_of_week = safe_json_value(row.get("DayOfWeek", 0))
            is_holiday = safe_json_value(row.get("IsHoliday", 0))
            weather = str(row.get("Weather", "clear"))

            searchable_content = (
                f"ATM {atm_id} in {city} at {location}. "
                f"Date: {date_str}. Withdrawals: {withdrawals} KD. "
                f"Deposits: {deposits} KD. Day: {day_of_week}. "
                f"Holiday: {'Yes' if is_holiday == 1 else 'No'}. Weather: {weather}."
            )

            doc = {
                "id": f"{atm_id}_{date_str}".replace("/", "-").replace(" ", "_"),
                "ATM_ID": atm_id,
                "City": city,
                "Location": location,
                "Date": date_str,
                "Withdrawals": withdrawals,
                "Deposits": deposits,
                "DayOfWeek": day_of_week,
                "IsHoliday": is_holiday,
                "Weather": weather,
                "searchable_content": searchable_content,
            }

            search_docs.append(doc)

        if not (_data_client and SEARCH_ENDPOINT and SEARCH_KEY):
            return {"status": "loaded_in_memory_only", "rows": len(atm_dataframe)}

        # Upload to Azure AI Search
        client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=DATA_INDEX_V2, credential=AzureKeyCredential(SEARCH_KEY))
        batch_size = 2000
        for i in range(0, len(search_docs), batch_size):
            batch = search_docs[i:i + batch_size]
            result = client.upload_documents(documents=batch)
            failed = [r for r in result if not getattr(r, "succeeded", False)]
            print(f"[{DATA_INDEX_V2}] Indexed {i + len(batch)} / {len(search_docs)}; failed: {len(failed)}")

        return {
            "status": "success",
            "rows": len(atm_dataframe),
            "indexed_to_search": len(search_docs),
            "index_name": DATA_INDEX_V2
        }
    except ValueError as ve:
        print(f"[Upload ValueError] {ve}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(ve)}")
    except Exception as e:
        print(f"[Upload Exception] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Main RAG query endpoint."""
    try:
        tool_results: Optional[Dict[str, Any]] = None
        
        if request.use_tools:
            try:
                tool_results = route_query_to_tool(request.query, request.atm_id)
                if tool_results and tool_results.get("tool") == "predict_atm_demand":
                    try:
                        tool_results["summary"] = tools.summarize_predictions(tool_results.get("results", {}))
                    except Exception as e:
                        print(f"[Summary error] {e}")
            except Exception as e:
                print(f"[Tool error] {e}")
                traceback.print_exc()

        answer = generate_answer(request.query, tool_results)

        kb_sources = hybrid_search_knowledge(request.query, top_k=3)
        data_sources = search_transactional_data(request.query, top_k=5)
        sources = kb_sources + data_sources

        return QueryResponse(answer=answer, sources=sources, tool_results=tool_results)
    except Exception as e:
        print(f"[QUERY ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/index/document")
async def index_document(doc: IndexDocumentRequest) -> Dict[str, Any]:
    """Index a single document in atm-knowledge."""
    try:
        if _aoai_client is None:
            raise RuntimeError("Azure OpenAI not configured")
        doc_id = f"{doc.category}_{hashlib.sha1(doc.title.encode('utf-8')).hexdigest()[:16]}"
        content_vector = get_embedding(doc.content)
        payload = {
            "id": doc_id,
            "content": doc.content,
            "title": doc.title,
            "category": doc.category,
            "content_vector": content_vector,
            "metadata": doc.metadata or {},
        }
        result = kb_client().upload_documents(documents=[payload])
        return {"status": "indexed", "doc_id": doc_id, "result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

# ============= Local Runner =============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("APP_PORT", 8000)))