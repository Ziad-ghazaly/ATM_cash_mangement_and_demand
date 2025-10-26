import os
from dotenv import load_dotenv; load_dotenv()
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from app.llm import embed_texts, chat

search = SearchClient(os.getenv("AZ_SEARCH_ENDPOINT"), os.getenv("AZ_SEARCH_INDEX"),
                      AzureKeyCredential(os.getenv("AZ_SEARCH_KEY")))

def retrieve(query:str, k:int=8, filters:str|None=None):
    vec = embed_texts([query])[0]
    kwargs = dict(search_text=query, vectors=[Vector(value=vec, k=k, fields="contentVector")],
                  query_type="semantic", semantic_configuration_name="default", top=k)
    if filters: kwargs["filter"] = filters
    results = search.search(**kwargs)
    return [r["content"] for r in results]

def answer(question:str, ctx_docs:list[str]):
    ctx = "\n\n---\n\n".join(ctx_docs[:6])
    sys = "You are a banking analytics assistant. Use only the provided context or Python tool outputs."
    msgs = [{"role":"system","content":sys},
            {"role":"user","content":f"Question: {question}\n\nContext:\n{ctx}"}]
    return chat(msgs).content
