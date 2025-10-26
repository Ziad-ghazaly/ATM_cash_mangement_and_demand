import os
from dotenv import load_dotenv; load_dotenv()
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, SearchableField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticField, SemanticSettings
)

endpoint, key, index_name = os.getenv("AZ_SEARCH_ENDPOINT"), os.getenv("AZ_SEARCH_KEY"), os.getenv("AZ_SEARCH_INDEX")
client = SearchIndexClient(endpoint, AzureKeyCredential(key))

index = SearchIndex(
    name=index_name,
    fields=[
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
        SimpleField(name="city", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="atm_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="date_range", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=3072, vector_search_profile_name="vec-prof")
    ],
    vector_search=VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="vec-prof", algorithm_configuration_name="hnsw")]
    ),
    semantic_settings=SemanticSettings(
        configurations=[SemanticConfiguration(
            name="default",
            prioritized_fields={"content_fields":[SemanticField(field_name="content")]}
        )]
    )
)
try: client.delete_index(index_name)
except: pass
client.create_index(index)
print("Index created:", index_name)
