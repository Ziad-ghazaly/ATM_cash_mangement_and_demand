"""
Azure AI Search Index Setup Script
FILE: services/RAG_API/setup_index.py
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
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential

# Load environment
load_dotenv("../../.env")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "atm-knowledge")

def create_search_index():
    """Create the Azure AI Search index with vector capabilities."""
    
    print("🔧 Setting up Azure AI Search Index...")
    print(f"   Endpoint: {SEARCH_ENDPOINT}")
    print(f"   Index: {SEARCH_INDEX}\n")
    
    if not SEARCH_ENDPOINT or not SEARCH_KEY:
        print("❌ Missing AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_API_KEY in .env")
        return False
    
    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_KEY)
    )
    
    # Define fields
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
        ),
        SimpleField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="vector-profile",
        ),
        SimpleField(
            name="metadata",
            type=SearchFieldDataType.String,
        ),
    ]
    
    # Configure vector search
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine",
                }
            )
        ],
    )
    
    # Create index
    index = SearchIndex(
        name=SEARCH_INDEX,
        fields=fields,
        vector_search=vector_search,
    )
    
    try:
        result = index_client.create_or_update_index(index)
        print(f"✅ Index '{result.name}' created successfully!")
        print(f"   - Fields: {len(result.fields)}")
        print(f"   - Vector search enabled: {result.vector_search is not None}")
        return True
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return False

if __name__ == "__main__":
    create_search_index()