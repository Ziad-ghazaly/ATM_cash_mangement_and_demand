import os, uuid, glob, tiktoken
from dotenv import load_dotenv; load_dotenv()
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from app.llm import embed_texts

enc = tiktoken.get_encoding("cl100k_base")
def chunk(text, max_tokens=700, overlap=120):
    toks = enc.encode(text); i=0; out=[]
    while i < len(toks):
        out.append(enc.decode(toks[i:i+max_tokens]))
        i += (max_tokens - overlap)
    return out

search = SearchClient(os.getenv("AZ_SEARCH_ENDPOINT"), os.getenv("AZ_SEARCH_INDEX"),
                      AzureKeyCredential(os.getenv("AZ_SEARCH_KEY")))

def ingest_path(pattern):
    for fp in glob.glob(pattern):
        raw = open(fp, "r", encoding="utf-8", errors="ignore").read()
        chunks = chunk(raw)
        vecs = embed_texts(chunks)
        docs = [{"id": f"{uuid.uuid4()}-{i}", "content": c, "contentVector": v}
                for i,(c,v) in enumerate(zip(chunks, vecs))]
        search.upload_documents(docs)
        print("Ingested", fp, "chunks:", len(docs))

if __name__ == "__main__":
    ingest_path("docs/*.md"); ingest_path("docs/*.txt")
