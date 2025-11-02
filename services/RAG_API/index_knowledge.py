"""
Knowledge Base Indexing Script
FILE: services/RAG_API/index_knowledge.py

Populates the Azure AI Search index with ATM operational knowledge.
Run this after setup_index.py to add sample documents.
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

load_dotenv("../../.env")

AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_EMBED_DEPLOY = os.getenv("AZ_OPENAI_EMBED_DEPLOY", "text-embedding-3-large")
AOAI_API_VERSION = os.getenv("AZ_OPENAI_API_VERSION", "2024-08-01-preview")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "atm-knowledge")

# Initialize clients
aoai_client = AzureOpenAI(
    api_key=AOAI_KEY,
    api_version=AOAI_API_VERSION,
    azure_endpoint=AOAI_ENDPOINT
)

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_KEY)
)

# Knowledge base documents
KNOWLEDGE_DOCS = [
    {
        "title": "ATM Cash Management Best Practices",
        "category": "SOP",
        "content": """
ATM Cash Management Standard Operating Procedures:

1. Daily Monitoring:
- Check cash levels every morning before 9 AM
- Review predicted demand from ML model
- Identify ATMs below 2-day buffer threshold

2. Refill Scheduling:
- Schedule refills 24 hours in advance
- Prioritize ATMs with risk_ratio > 0.8
- Weekend refills must be completed by Thursday 6 PM

3. Risk Categories:
- CRITICAL: risk_ratio > 1.0 (immediate action required)
- HIGH: risk_ratio 0.9-1.0 (refill within 24 hours)
- MEDIUM: risk_ratio 0.7-0.9 (refill within 48 hours)
- LOW: risk_ratio < 0.7 (monitor only)

4. Cash Allocation:
- Maintain 2-day buffer for normal ATMs
- Maintain 3-day buffer for high-traffic locations
- Maximum capacity: 50,000 KD per ATM

5. Emergency Procedures:
- If ATM shows CRITICAL status, dispatch within 2 hours
- Contact regional manager for ATMs in remote areas
- Keep emergency cash reserves at main branch
        """
    },
    {
        "title": "Weekend Cash Planning Guidelines",
        "category": "SOP",
        "content": """
Weekend Preparation Protocol:

Thursday Operations:
- Run weekend_preparation_report() by 2 PM
- Identify all ATMs needing refills
- Calculate total cash requirements
- Reserve armored car services

Friday Morning:
- Complete all critical refills by 11 AM
- Verify high-traffic ATMs are at full capacity
- Double-check shopping mall and tourist area ATMs

Weekend Multipliers:
- Friday demand: 1.3x weekday average
- Saturday demand: 1.5x weekday average
- Holiday weekends: 2.0x weekday average

Target Buffer:
- Standard ATMs: 2.5 days
- High-traffic ATMs: 3.5 days
- Tourist locations: 4.0 days
        """
    },
    {
        "title": "Location Optimization Strategy",
        "category": "Strategy",
        "content": """
ATM Placement and Expansion Guidelines:

High-Demand Indicators:
- demand_per_atm > 3000 KD/day
- utilization_rate > 70%
- ATM_Count < 3 in area
- Population density > 5000/km²

Optimal Location Types (by priority):
1. Shopping malls (highest withdrawal frequency)
2. Bank branches (trusted locations)
3. Gas stations (24/7 accessibility)
4. Airports (tourist demand)
5. Universities (student population)

Expansion Score Formula:
expansion_score = (demand_per_atm × 0.6) + (100/ATM_count × 0.4)

Cities to prioritize:
- Kuwait City (highest total demand)
- Salmiya (high tourist traffic)
- Hawalli (dense population)

Avoid:
- Locations with > 5 ATMs within 500m
- Areas with declining transaction trends
- Industrial zones with low foot traffic
        """
    },
    {
        "title": "Demand Forecasting Model Guide",
        "category": "Technical",
        "content": """
ATM Cash Demand Prediction System:

Model Details:
- Algorithm: Random Forest (primary), XGBoost, CatBoost
- R² Score: ~0.85
- Features: 13 numeric + 5 categorical
- Update frequency: Daily

Key Features:
- Lag_1, Lag_7: Previous day and week withdrawals
- RollingMean_3, RollingMean_7: Moving averages
- Month_Sin, Month_Cos: Seasonal patterns
- Holiday_Flag: 1 for holidays, 0 otherwise
- Day_of_Week: Day name (Monday-Sunday)
- Location_Type: bank_branch, shopping_mall, etc.
- Weather_Condition: clear, rainy, snowy

Interpretation:
- Predictions are in Kuwaiti Dinars (KD)
- Auto-rounded to nearest 5/10/20 KD denominations
- Include 15% safety margin for volatility
- Holiday predictions typically 30-50% higher

When to Override Model:
- Major national events (National Day, Eid)
- Bank promotions or special campaigns
- Construction/road closures near ATM
- Unusual weather events
        """
    },
    {
        "title": "Performance Metrics and KPIs",
        "category": "Analytics",
        "content": """
ATM Performance Measurement Framework:

Primary KPIs:
1. Utilization Rate = (Daily Demand / Current Balance) × 100
- Target: 50-70%
- Critical if > 90%

2. Cash Buffer Days = Current Balance / Daily Demand
- Target: 2-3 days
- Critical if < 1 day

3. Volatility = (Std Dev / Mean) × 100
- Low: < 15% (stable demand)
- Medium: 15-30% (moderate variability)
- High: > 30% (unpredictable)

4. Peak Multiplier = Peak Day Avg / Off-Peak Day Avg
- Typical: 1.2-1.5x
- High: > 2.0x (requires special attention)

Ranking Metrics:
- By Withdrawals: Total KD dispensed
- By Utilization: Efficiency of cash deployment
- By Efficiency: Withdrawals per KD balance

City-Level Metrics:
- Total ATM count per city
- Demand per ATM (indicates saturation)
- Buffer days (indicates cash availability)
- Volatility (indicates market stability)

Red Flags:
- Utilization > 90% (cash-out risk)
- Buffer days < 1 (urgent refill needed)
- Volatility > 40% (demand unpredictability)
        """
    },
    {
        "title": "Holiday and Special Event Planning",
        "category": "Operations",
        "content": """
Special Events Cash Management:

National Holidays (Kuwait):
- National Day (Feb 25): 2.0x normal demand
- Liberation Day (Feb 26): 1.8x normal demand
- Eid Al-Fitr: 2.5x normal demand (3-day period)
- Eid Al-Adha: 2.2x normal demand (4-day period)
- New Year's Day: 1.5x normal demand

Preparation Timeline:
- 7 days before: Run demand forecasts with holiday flag
- 5 days before: Order additional cash from central bank
- 3 days before: Schedule armored car services
- 2 days before: Begin refills for high-priority ATMs
- 1 day before: Complete all refills, verify all ATMs operational

Shopping Mall Events:
- Major sales (Black Friday, etc.): 1.6x normal
- Back-to-school season: 1.3x normal
- Wedding season (summer): 1.4x normal in mall ATMs

Tourist Season Adjustments:
- Summer (June-August): 0.8x (many locals travel)
- Winter (Dec-Feb): 1.2x (tourist arrivals)
- Airport ATMs: Maintain 5-day buffer during holidays

Post-Event Review:
- Compare actual vs predicted demand
- Update model if deviation > 20%
- Document lessons learned for next event
        """
    },
    {
        "title": "Troubleshooting Common Issues",
        "category": "Technical",
        "content": """
ATM Cash Management Issue Resolution:

Issue: Model Prediction Too Low
Possible Causes:
- Recent promotional campaign not captured in data
- Nearby competitor ATM out of service
- New business opened nearby
Solution: Apply 20-30% buffer, update model with recent data

Issue: Frequent Cash-Outs Despite Predictions
Possible Causes:
- Denominations mismatch (too many large bills)
- Fraud or theft suspected
- Data quality issues (incorrect balance reporting)
Solution: Verify physical count, check security footage, audit data pipeline

Issue: Low Utilization (Cash Sitting Idle)
Possible Causes:
- Over-refilling based on worst-case scenarios
- Competition from nearby ATMs
- Location traffic decreased
Solution: Reduce buffer to 1.5 days, consider relocating ATM

Issue: High Volatility in Predictions
Possible Causes:
- Seasonal business nearby (stadium, event venue)
- New construction affecting access
- Irregular holiday patterns
Solution: Use higher buffer (3-4 days), segment analysis by day-of-week

Issue: RAG System Not Finding Relevant Info
Possible Causes:
- Query too vague or ambiguous
- Knowledge base incomplete
- Embedding mismatch
Solution: Rephrase query with specific keywords, add more documents to index

Data Quality Checks:
- Verify Date column has no gaps
- Check for negative withdrawal values
- Ensure ATM_ID consistency
- Validate Location and City fields
        """
    }
]

def get_embedding(text: str) -> list:
    """Generate embedding using Azure OpenAI."""
    response = aoai_client.embeddings.create(
        input=text,
        model=AOAI_EMBED_DEPLOY
    )
    return response.data[0].embedding

def index_documents():
    """Index all knowledge documents into Azure AI Search."""
    print(f"🔄 Indexing {len(KNOWLEDGE_DOCS)} documents...\n")
    
    indexed_docs = []
    for i, doc in enumerate(KNOWLEDGE_DOCS, 1):
        print(f"[{i}/{len(KNOWLEDGE_DOCS)}] Processing: {doc['title']}")
        
        # Generate embedding
        embedding = get_embedding(doc["content"])
        
        # Prepare document
        search_doc = {
            "id": f"{doc['category']}_{abs(hash(doc['title']))}".replace("-", "_"),
            "content": doc["content"],
            "title": doc["title"],
            "category": doc["category"],
            "content_vector": embedding,
            "metadata": '{"source": "internal", "version": "1.0"}'
        }
        
        indexed_docs.append(search_doc)
        print(f"   ✅ Embedded ({len(embedding)} dimensions)")
    
    # Upload to search
    print(f"\n📤 Uploading to Azure AI Search...")
    try:
        result = search_client.upload_documents(documents=indexed_docs)
        success_count = sum(1 for r in result if r.succeeded)
        print(f"✅ Successfully indexed {success_count}/{len(indexed_docs)} documents!")
        return True
    except Exception as e:
        print(f"❌ Error uploading documents: {e}")
        return False

if __name__ == "__main__":
    print("📚 ATM Knowledge Base Indexing\n")
    print(f"   Endpoint: {SEARCH_ENDPOINT}")
    print(f"   Index: {SEARCH_INDEX}")
    print(f"   Embedding Model: {AOAI_EMBED_DEPLOY}\n")
    
    if not all([AOAI_ENDPOINT, AOAI_KEY, SEARCH_ENDPOINT, SEARCH_KEY]):
        print("❌ Missing required environment variables!")
        exit(1)
    
    index_documents()
    print("\n🎉 Indexing complete!")