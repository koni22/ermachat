from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from pinecone import Pinecone

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("my-index")  # Replace with your actual index name

# FastAPI app
app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    # Get embedding
    embed = openai.Embedding.create(input=request.query, model="text-embedding-3-small")
    vector = embed["data"][0]["embedding"]

    # Query Pinecone
    result = index.query(vector=vector, top_k=5, include_metadata=True, namespace="__default__")
    matches = [m["metadata"]["text"] for m in result["matches"] if "metadata" in m]
    return {"results": matches}
