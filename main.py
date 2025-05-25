from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("my-index")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    # Step 1: Embed the query
    embedding_response = client.embeddings.create(
        input=request.query,
        model="text-embedding-3-small"
    )
    vector = embedding_response.data[0].embedding

    # Step 2: Search Pinecone
    pinecone_results = index.query(
        vector=vector,
        top_k=5,
        include_metadata=True,
        namespace="__default__"
    )

    matches = [
        {
            "score": m.get("score"),
            "text": m["metadata"].get("text", ""),
            "source": m["metadata"].get("source", "Unknown"),
            "page": m["metadata"].get("page", None)
        }
        for m in pinecone_results["matches"]
        if "metadata" in m
    ]

    # Step 3: Ask GPT to re-rank with reasoning
    context = "\n\n".join(
        [f"[{i+1}] Source: {m['source']}, Page {m['page']}\n{m['text']}" for i, m in enumerate(matches)]
    )

    rerank_prompt = f"""
You are a technical assistant for Eermafirst Ballast Water Treatment Systems.
Given the question: "{request.query}"

And the following extracted document segments:
{context}

Rank the top 3 segments by relevance to the question. For each, return:
- A short explanation of why it is relevant
- Source filename and page
- Original score (if available)
"""

    chat_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful technical assistant."},
            {"role": "user", "content": rerank_prompt}
        ]
    )

    return {
        "query": request.query,
        "summary": chat_response.choices[0].message.content,
        "raw_matches": matches  # for debugging if needed
    }
