from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from .config import settings

def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=settings.search_endpoint,
        index_name=settings.search_index,
        credential=AzureKeyCredential(settings.search_api_key),
    )

def get_aoai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=settings.aoai_api_key,
        api_version=settings.aoai_api_version,
        azure_endpoint=settings.aoai_endpoint,
    )
