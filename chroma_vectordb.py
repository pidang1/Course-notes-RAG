import os
import time
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

# Get Chroma settings from environment variables
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "ds4300")

# Initialize Chroma client
client = chromadb.Client(chromadb.Settings(persist_directory=CHROMA_PERSIST_DIR))

def initialize_chroma():
    """Initialize Chroma collection."""
    # Get or create collection
    try:
        # Try to get existing collection
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")
    except:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"Created new collection: {COLLECTION_NAME}")

    return collection

def upload_embeddings_to_chroma(
    collection,
    embeddings: List[np.ndarray], 
    documents: List[Dict[str, Any]]
):
    """Upload embeddings to Chroma collection."""
    total_vectors = len(embeddings)
    print(f"Uploading {total_vectors} vectors to Chroma...")
    
    # Prepare data for Chroma
    ids = [f"doc_{i}" for i in range(total_vectors)]
    texts = [doc.get("text", "") for doc in documents]
    metadatas = [{k: v for k, v in doc.items() if k != "text"} for doc in documents]
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=[emb.tolist() for emb in embeddings],
        documents=texts,
        metadatas=metadatas
    )
    
    print(f"Successfully uploaded {total_vectors} vectors to Chroma.")

def main():
    """Main function to upload embeddings to Chroma."""
    # Initialize Chroma
    collection = initialize_chroma()
    
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
    
    # Upload embeddings to Chroma
    upload_embeddings_to_chroma(collection, sample_embeddings, sample_documents)
    
    # Query example
    query_embedding = np.random.rand(768).tolist()  # Replace with actual query embedding
    
    # Execute search
    query_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    print("\nQuery results:", query_results)

if __name__ == "__main__":
    main()