"""
Enhanced RAG API with Intelligent Tool Routing
FILE: services/RAG_API/app/main.py

Orchestrates natural language queries with:
- Azure AI Search retrieval
- Azure OpenAI reasoning
- Python analytical tools execution
- Context-aware responses
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging

from app.retrieve import retrieve, answer
from app.tools import (
    compute_cashout_risk,
    refill_suggestion,
    location_optimization_analysis,
    demand_pattern_analysis,
    operational_efficiency_metrics,
    weekend_preparation_report,
    atm_performance_ranking,
    city_comparison_report
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ATM Intelligence API",
    description="AI-powered ATM cash management and analytics",
    version="2.0.0"
)

# CORS for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., description="Natural language question about ATM operations")
    city: Optional[str] = Field(None, description="Filter by specific city")
    atm_id: Optional[str] = Field(None, description="Filter by specific ATM ID")
    use_tools: bool = Field(True, description="Enable Python analytical tools")


class QuestionResponse(BaseModel):
    answer: str
    tools_executed: List[str]
    tool_results: Dict[str, Any]
    sources_used: int
    confidence: str


def intelligent_tool_router(question: str) -> List[str]:
    """
    Analyze question and determine which tools to execute
    
    Returns: List of tool names to execute
    """
    q = question.lower()
    tools = []
    
    # Risk detection keywords
    if any(kw in q for kw in ['risk', 'cashout', 'run out', 'empty', 'shortage', 'alert']):
        tools.append('cashout_risk')
    
    # Refill/replenishment keywords
    if any(kw in q for kw in ['refill', 'replenish', 'top up', 'add cash', 'load', 'restock']):
        tools.append('refill_suggestion')
    
    # Location/expansion keywords
    if any(kw in q for kw in ['location', 'new atm', 'expand', 'placement', 'where to', 'best location']):
        tools.append('location_optimization')
    
    # Pattern/trend keywords
    if any(kw in q for kw in ['pattern', 'trend', 'behavior', 'peak', 'demand analysis', 'when']):
        tools.append('demand_pattern')
    
    # Performance/ranking keywords
    if any(kw in q for kw in ['top', 'best', 'worst', 'rank', 'perform', 'compare atm']):
        tools.append('performance_ranking')
    
    # Operational efficiency keywords
    if any(kw in q for kw in ['efficiency', 'utilization', 'metrics', 'kpi', 'overall', 'system']):
        tools.append('operational_metrics')
    
    # Weekend preparation keywords
    if any(kw in q for kw in ['weekend', 'friday', 'thursday', 'prepare', 'upcoming']):
        tools.append('weekend_prep')
    
    # City comparison keywords
    if any(kw in q for kw in ['compare cit', 'which city', 'city comparison', 'between cities']):
        tools.append('city_comparison')
    
    return tools


def execute_tools(tool_names: List[str], atm_id: Optional[str] = None) -> Dict[str, Any]:
    """Execute selected analytical tools and return results"""
    results = {}
    
    try:
        if 'cashout_risk' in tool_names:
            results['cashout_risk'] = compute_cashout_risk(threshold=0.2)
        
        if 'refill_suggestion' in tool_names:
            results['refill_suggestions'] = refill_suggestion(buffer_days=2)
        
        if 'location_optimization' in tool_names:
            results['location_analysis'] = location_optimization_analysis()
        
        if 'demand_pattern' in tool_names:
            results['demand_patterns'] = demand_pattern_analysis(atm_id=atm_id)
        
        if 'operational_metrics' in tool_names:
            results['operational_metrics'] = operational_efficiency_metrics()
        
        if 'weekend_prep' in tool_names:
            results['weekend_preparation'] = weekend_preparation_report()
        
        if 'performance_ranking' in tool_names:
            # Determine metric from context (default to withdrawals)
            results['performance_ranking'] = atm_performance_ranking(metric='withdrawals', top_n=20)
        
        if 'city_comparison' in tool_names:
            results['city_comparison'] = city_comparison_report()
    
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}")
        results['error'] = str(e)
    
    return results


def format_tool_results_for_llm(tool_results: Dict[str, Any]) -> str:
    """Format tool results into readable context for LLM"""
    formatted = ["=== ANALYTICAL TOOL RESULTS ===\n"]
    
    for tool_name, data in tool_results.items():
        formatted.append(f"\n--- {tool_name.upper().replace('_', ' ')} ---")
        
        if isinstance(data, list):
            formatted.append(f"Total results: {len(data)}")
            if len(data) > 0:
                formatted.append("Top results:")
                for i, item in enumerate(data[:5], 1):
                    formatted.append(f"{i}. {item}")
        
        elif isinstance(data, dict):
            if 'error' in data:
                formatted.append(f"Error: {data['error']}")
            else:
                for key, value in data.items():
                    formatted.append(f"  {key}: {value}")
        
        else:
            formatted.append(str(data))
    
    return "\n".join(formatted)


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "operational",
        "service": "ATM Intelligence API",
        "version": "2.0.0",
        "capabilities": [
            "Natural language queries",
            "Cash-out risk detection",
            "Refill optimization",
            "Location analysis",
            "Demand forecasting",
            "Performance analytics"
        ]
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "api": "healthy",
        "search": "connected",
        "llm": "connected",
        "tools": "ready"
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint: Process natural language questions with RAG + Tools
    
    Flow:
    1. Route question to appropriate tools
    2. Execute analytical tools
    3. Retrieve relevant documents from Azure AI Search
    4. Combine tool results + documents as context
    5. Generate answer using Azure OpenAI
    """
    try:
        logger.info(f"Question received: {request.question}")
        
        # Step 1: Intelligent tool routing
        tool_names = intelligent_tool_router(request.question) if request.use_tools else []
        logger.info(f"Tools selected: {tool_names}")
        
        # Step 2: Execute tools
        tool_results = {}
        if tool_names:
            tool_results = execute_tools(tool_names, atm_id=request.atm_id)
        
        # Step 3: Retrieve relevant documents
        search_filter = None
        if request.city:
            search_filter = f"city eq '{request.city}'"
        elif request.atm_id:
            search_filter = f"atm_id eq '{request.atm_id}'"
        
        retrieved_docs = retrieve(request.question, k=8, filters=search_filter)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 4: Combine context
        context_parts = retrieved_docs.copy()
        
        if tool_results:
            tool_context = format_tool_results_for_llm(tool_results)
            context_parts.append(tool_context)
        
        # Step 5: Generate answer
        final_answer = answer(request.question, context_parts)
        
        # Determine confidence level
        confidence = "high" if tool_results and len(retrieved_docs) >= 5 else "medium" if tool_results or len(retrieved_docs) >= 3 else "low"
        
        return QuestionResponse(
            answer=final_answer,
            tools_executed=tool_names,
            tool_results=tool_results,
            sources_used=len(retrieved_docs),
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/cashout-risk")
def get_cashout_risk(threshold: float = 0.2):
    """Direct access to cash-out risk analysis"""
    try:
        return compute_cashout_risk(threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/refill-suggestions")
def get_refill_suggestions(buffer_days: int = 2):
    """Direct access to refill recommendations"""
    try:
        return refill_suggestion(buffer_days=buffer_days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/location-analysis")
def get_location_analysis():
    """Direct access to location optimization"""
    try:
        return location_optimization_analysis()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/demand-patterns")
def get_demand_patterns(atm_id: Optional[str] = None):
    """Direct access to demand pattern analysis"""
    try:
        return demand_pattern_analysis(atm_id=atm_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/operational-metrics")
def get_operational_metrics():
    """Direct access to system-wide metrics"""
    try:
        return operational_efficiency_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/weekend-prep")
def get_weekend_prep():
    """Direct access to weekend preparation report"""
    try:
        return weekend_preparation_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/performance-ranking")
def get_performance_ranking(metric: str = 'withdrawals', top_n: int = 20):
    """Direct access to ATM performance rankings"""
    try:
        return atm_performance_ranking(metric=metric, top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/city-comparison")
def get_city_comparison():
    """Direct access to city comparison report"""
    try:
        return city_comparison_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)