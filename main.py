from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI
from pinecone import Pinecone

# Initialize OpenAI and Pinecone clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("my-index")  # Replace with your actual index name

# Set up FastAPI
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    # Get embedding using OpenAI v1 client
    response = openai_client.embeddings.create(
        input=request.query,
        model="text-embedding-3-small"
    )
    vector = response.data[0].embedding

    # Query Pinecone for relevant documents
    result = index.query(vector=vector, top_k=5, include_metadata=True, namespace="__default__")
    matches = [m["metadata"]["text"] for m in result["matches"] if "metadata" in m]
    return {"results": matches}
