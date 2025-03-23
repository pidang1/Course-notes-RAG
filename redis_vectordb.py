import os
import time
import numpy as np
import struct
from typing import List, Dict, Any
from dotenv import load_dotenv
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# load environment variables
load_dotenv()

# get Redis connection details from environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
INDEX_NAME = os.getenv("REDIS_INDEX_NAME", "ds4300_index")

# initialize Redis client
client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False
)

def initialize_redis():
    """Initialize Redis index."""
    # Check if index already exists
    try:
        client.ft(INDEX_NAME).info()
        print(f"Index {INDEX_NAME} already exists")
    except:
        print(f"Creating index {INDEX_NAME}...")
        
        # define schema with text field and vector field
        schema = [
            TextField("text"),
            VectorField("embedding", 
                      "FLAT", {
                          "TYPE": "FLOAT32",
                          "DIM": 768,  # Adjust based on your embedding model
                          "DISTANCE_METRIC": "COSINE"
                      })
        ]
        
        # creates the index
        client.ft(INDEX_NAME).create_index(
            schema,
            definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
        )
        print(f"Created index {INDEX_NAME}")
    
    return client

def upload_embeddings_to_redis(
    client,
    embeddings: List[np.ndarray], 
    documents: List[Dict[str, Any]]
):
    """upload embeddings to Redis."""
    total_vectors = len(embeddings)
    print(f"Uploading {total_vectors} vectors to Redis...")
    
    # use pipeline for batch insertion
    pipe = client.pipeline()
    
    for i in range(total_vectors):
        vector_id = f"doc:{i}"
        vector_embedding = embeddings[i]
        vector_metadata = documents[i]
        
        # convert numpy array to bytes
        embedding_bytes = struct.pack(f'{len(vector_embedding)}f', *vector_embedding)
        
        # creates hash data
        hash_data = {
            "text": vector_metadata.get("text", ""),
            "embedding": embedding_bytes
        }
        
        # add other metadata fields
        for key, value in vector_metadata.items():
            if key != "text":
                hash_data[f"meta_{key}"] = str(value)
        
        # Set hash in Redis
        pipe.hset(vector_id, mapping=hash_data)

    pipe.execute()
    
    print(f"Successfully uploaded {total_vectors} vectors to Redis.")

def main():
    """Main function to upload embeddings to Redis."""
    # Initialize Redis
    client = initialize_redis()
    
    # Example: Load your embeddings and documents here
    # In a real scenario, you would load these from files or generate them
    sample_embeddings = [np.random.rand(768) for _ in range(5)]  # Replace with actual embeddings
    sample_documents = [
        {"text": "Sample document 1", "source": "source1", "category": "category1"},
        {"text": "Sample document 2", "source": "source1", "category": "category2"},
        {"text": "Sample document 3", "source": "source2", "category": "category1"},
        {"text": "Sample document 4", "source": "source2", "category": "category2"},
        {"text": "Sample document 5", "source": "source3", "category": "category3"},
    ]
    
    # Upload embeddings to Redis
    upload_embeddings_to_redis(client, sample_embeddings, sample_documents)
    
    # Query example
    import struct
    query_embedding = np.random.rand(768)  # Replace with actual query embedding
    query_embedding_bytes = struct.pack(f'{len(query_embedding)}f', *query_embedding)
    
    # Create vector query
    q = f"*=>[KNN 3 @embedding $embedding AS score]"
    
    # Execute search
    query_results = client.ft(INDEX_NAME).search(
        q,
        query_params={"embedding": query_embedding_bytes}
    )
    
    print("\nQuery results:", query_results)

if __name__ == "__main__":
    main()