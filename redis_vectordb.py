import os
import time
from typing import List, Dict, Any
import numpy as np
import redis
from redis.commands.search.query import Query
from dotenv import load_dotenv
from embed import NomicEmbedder
import base64

# Load environment variables
load_dotenv()

# Get Redis configuration from environment variables
INDEX_NAME = "ds4300"



# Initialize with the embedding dimension nomic embed text model uses
def initialize_redis_index(embedding_dimension: int = 768):
    """Initialize Redis vector index."""
    
    # Initialize Redis client
    redis_client = redis.Redis(host="localhost", port="6379", decode_responses=True)
    print(redis_client.ping())

    try:
        redis_client.ft(INDEX_NAME).info()
        print(f"Index {INDEX_NAME} already exists")
    except:
        # Create index if it doesn't exist
        redis_client.ft(INDEX_NAME).create_index(
            fields=[
                redis.commands.search.field.VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": embedding_dimension,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                redis.commands.search.field.TextField("text"),
            ]
        )
        print(f"Index {INDEX_NAME} created successfully.")
    return redis_client

def upload_embeddings_to_redis(
    client,
    embeddings: List[np.ndarray],
    documents: List[str]
):
    """Upload embeddings to Redis vector database."""
    total_vectors = len(embeddings)
    print(f"Uploading {total_vectors} vectors to Redis...")
    
    
    # Use a pipeline for faster insertion
    pipeline = client.pipeline()
    for i in range(total_vectors):
        vector_id = f"doc_{i}"
        embedding = np.array(embeddings[i], dtype=np.float32)
        
        # Store the embedding as bytes and text associated with the embedding
        pipeline.hset(vector_id, "text", documents[i]['text'])
        pipeline.hset(vector_id, "embedding", embedding.tobytes())
    
    pipeline.execute()
    print(f"Successfully uploaded {total_vectors} vectors to Redis.")

def query(client, query_text: str, top_k: int = 1):
    """Query the Redis index and return the most relevant context."""
    # Embed the user query
    embedder = NomicEmbedder()
    query_embedding = embedder.embed_chunks([query_text])[0]
    
    # Ensure the query embedding is a numpy array of float32
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding, dtype=np.float32)
    else:
        query_embedding = query_embedding.astype(np.float32)
    
    # Convert the query embedding to bytes
    query_embedding_bytes = query_embedding.tobytes()
    
    # Prepare the query
    q = Query(
        f"*=>[KNN {top_k} @embedding $query_vector AS score]"
    ).sort_by("score").dialect(2)
    
    params = {"query_vector": query_embedding_bytes}
    
    # Execute the query
    results = client.ft(INDEX_NAME).search(q, query_params=params)
    
    # Extract context from search results
    contexts = []
    for doc in results.docs:
        if hasattr(doc, 'text'):
            contexts.append(doc.text)
    
    # Join contexts
    context_str = "\n\n".join(contexts)
    return context_str

def delete_index(redis_client):
    """Delete the Redis index."""
    redis_client.ft(INDEX_NAME).dropindex()
    print(f"Index {INDEX_NAME} deleted.")

def main():
    """Main function to upload embeddings to Redis (for testing)."""
    # Initialize Redis index
    embedder = NomicEmbedder()
    
    client = initialize_redis_index(embedding_dimension=embedder.get_embedding_dimension())
    
    # Sample data
    sample_texts = [
        "redis was found in 2009",
        "Sample document 2",
        "Sample document 3",
        "Sample document 4",
        "Sample document 5"
    ]
    
    # Generate embeddings
    sample_embeddings = embedder.embed_chunks(sample_texts)
    
    
    # Upload embeddings to Redis
    upload_embeddings_to_redis(client, sample_embeddings, sample_texts)
    
    # Query example
    result = query(client, "when was redis found?", top_k=1)
    print("\nQuery result:", result)

if __name__ == "__main__":
    main()
