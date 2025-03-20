import os
import time
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
import pinecone

# Load environment variables
load_dotenv()

# Get Pinecone API key and environment from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

def initialize_pinecone():
    """Initialize Pinecone client and connect to index."""
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    
    # Check if index exists, if not create it
    if INDEX_NAME not in pinecone.list_indexes():
        print(f"Creating index {INDEX_NAME}...")
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Adjust dimension based on your embedding model
            metric="cosine"
        )
        # Wait for index to be initialized
        time.sleep(1)
    
    # Connect to the index
    return pinecone.Index(INDEX_NAME)

def upload_embeddings_to_pinecone(
    index,
    embeddings: List[np.ndarray], 
    documents: List[Dict[str, Any]]
):
    
    #Upload embeddings to Pinecone index.
    total_vectors = len(embeddings)
    print(f"Uploading {total_vectors} vectors to Pinecone...")
    
    # Prepare vectors in Pinecone format
    vectors = []
    for i in range(total_vectors):
        vector_id = f"doc_{i}"
        vector_embedding = embeddings[i].tolist()
        vector_metadata = documents[i]
        vectors.append((vector_id, vector_embedding, vector_metadata))
    
    # Upsert to Pinecone
    index.upsert(vectors=vectors)
    
    print(f"Successfully uploaded {total_vectors} vectors to Pinecone.")
