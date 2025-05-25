from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

# Print SDK version for debug
import openai
print("OpenAI SDK version:", openai.__version__)

# Init OpenAI client with enforced project
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project="proj_3VRKy2L0xtFsBDtizPISqqiK"  # Replace with yours if needed
)

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
    # Generate embedding
    response = client.embeddings.create(
        input=request.query,
        model="text-embedding-3-small"
    )
    vector = response.data[0].embedding
    print("DEBUG: Vector length is", len(vector))  # Should be 384

    # Query Pinecone
    result = index.query(
        vector=vector,
        top_k=5,
        include_metadata=True,
        namespace="__default__"
    )

    # Return matched text
    return {
    "results": [
        {
            "score": match.get("score"),
            "metadata": match.get("metadata", {})
        }
        for match in result["matches"]
    ]
}
