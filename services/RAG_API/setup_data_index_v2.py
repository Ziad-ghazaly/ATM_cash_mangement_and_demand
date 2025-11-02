"""
Create ENHANCED atm-data-v2 index - ONLY searchable_content is searchable
FILE: services/RAG_API/setup_data_index_v2.py
"""

import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from azure.core.credentials import AzureKeyCredential

load_dotenv("../../.env")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
DATA_INDEX = "atm-data-v2"

def create_enhanced_data_index():
    """
    Create index where:
    - searchable_content = SEARCHABLE (for queries)
    - All other fields = FILTERABLE/RETRIEVABLE only (not searchable)
    """
    
    print("🔧 Creating Enhanced ATM Data Index (v2)...")
    print(f"   Endpoint: {SEARCH_ENDPOINT}")
    print(f"   Index: {DATA_INDEX}\n")
    
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_KEY)
    )
    
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
        ),
        #  NOT searchable - only filterable (can use in filters, not full-text search)
        SimpleField(
            name="ATM_ID",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="City",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="Location",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="Date",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="Withdrawals",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="Deposits",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="DayOfWeek",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        SimpleField(
            name="IsHoliday",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        SimpleField(
            name="Weather",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        # ⭐ ONLY THIS FIELD IS SEARCHABLE - all text search happens here
        SearchableField(
            name="searchable_content",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft",
        ),
    ]
    
    # Semantic configuration for better relevance
    semantic_config = SemanticConfiguration(
        name="atm-semantic",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=None,  # No title field
            content_fields=[
                SemanticField(field_name="searchable_content"),  # Primary content
            ],
            keywords_fields=[],  # No keyword fields needed
        ),
    )
    
    semantic_search = SemanticSearch(configurations=[semantic_config])
    
    index = SearchIndex(
        name=DATA_INDEX,
        fields=fields,
        semantic_search=semantic_search,
    )
    
    try:
        result = index_client.create_or_update_index(index)
        print(f"✅ Index '{result.name}' created successfully!")
        print(f"   - Total fields: {len(result.fields)}")
        print(f"   - Searchable field: searchable_content ONLY")
        print(f"   - Semantic config: atm-semantic")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    create_enhanced_data_index()