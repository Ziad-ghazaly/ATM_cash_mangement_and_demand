from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .clients import get_search_client, get_aoai_client
from .config import settings

app = FastAPI(title="RAG API")

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/ask")
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    # 1) Retrieve context from Azure AI Search
    search = get_search_client()
    results = search.search(search_text=q, top=req.top_k)
    passages = []
    for r in results:
        passages.append(r.get("content") or r.get("chunk") or "")

    context = "\n\n".join(p for p in passages if p) or "No relevant context."

    # 2) Generate with Azure OpenAI (Chat Completions)
    aoai = get_aoai_client()
    resp = aoai.chat.completions.create(
        model=settings.aoai_deployment,
        messages=[
            {"role": "system", "content": "Be concise. If unsure, say you don't know."},
            {"role": "user", "content": f"Question: {q}\n\nContext:\n{context}"}
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    return {"answer": answer, "used_context": passages[:req.top_k]}
