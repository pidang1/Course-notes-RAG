import os
import time
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Get Pinecone API key and environment from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ds4300")

# Initialize Pinecone client
client = Pinecone(api_key=PINECONE_API_KEY)

def initialize_pinecone():
    """Initialize Pinecone index."""
    existing_indexes = [index_info['name'] for index_info in client.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating index {INDEX_NAME}...")
        client.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Adjust this based on your embedding model
            metric="cosine"
        )
        time.sleep(2)  # Wait for index to be initialized

    return client.Index('ds4300')

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

def main():
    """Main function to upload embeddings to Pinecone."""
    # Initialize Pinecone
    index = initialize_pinecone()
    
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
    
    # Upload embeddings to Pinecone
    upload_embeddings_to_pinecone(index, sample_embeddings, sample_documents)
    
    # Query example
    query_results = index.query(
        vector=np.random.rand(768).tolist(),  # Replace with actual query embedding
        top_k=3,
        include_metadata=True
    )
    print("\nQuery results:", query_results)

if __name__ == "__main__":
    main()