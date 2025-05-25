from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI
from pinecone import Pinecone

# Initialize OpenAI with project + key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project="proj_3VRKy2L0xtFsBDtizPISqqiK"  # Force correct project
)

# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("my-index")

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    embedding_response = client.embeddings.create(
        input=request.query,
        model="text-embedding-3-small"
    )
    vector = embedding_response.data[0].embedding
    print("DEBUG: Vector length is", len(vector))  # Should print 384

    result = index.query(vector=vector, top_k=5, include_metadata=True, namespace="__default__")
    return {"results": [m["metadata"]["text"] for m in result["matches"] if "metadata" in m]}
