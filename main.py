from fastapi import FastAPI
from pydantic import BaseModel
import pinecone
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
index = pinecone.Index("my-index")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    embed = openai.Embedding.create(input=request.query, model="text-embedding-3-small")
    vector = embed["data"][0]["embedding"]
    result = index.query(vector=vector, top_k=5, include_metadata=True, namespace="__default__")
    matches = [m["metadata"]["text"] for m in result["matches"] if "metadata" in m]
    return {"results": matches}
