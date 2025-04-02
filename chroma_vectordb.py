import os
# import time
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
import chromadb
from embed import MxbaiEmbedder
from uuid import uuid4


# Load environment variables
load_dotenv()

# Global variables
CHROMA_PERSIST_DIR = "./chroma"
COLLECTION_NAME = "ds4300"



def initialize_chroma():
    """Initialize Chroma collection."""
    # Get or create collection
    # Initialize Chroma client
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        # Try to get existing collection
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")
    except:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name=COLLECTION_NAME,
            # Use cosine similarity
            metadata={"hnsw:space": "cosine"}  
        )
        print(f"Created new collection: {COLLECTION_NAME}")

    return collection

def upload_embeddings_to_chroma(
    collection,
    embeddings: List[np.ndarray], 
    documents: List[str]
):
    """Upload embeddings to Chroma collection."""
    total_vectors = len(embeddings)
    print(f"Uploading {total_vectors} vectors to Chroma...")
    
    # Prepare data for Chroma
    ids = [f"doc_{uuid4}" for i in range(total_vectors)]
    emb_lists = [emb.tolist() for emb in embeddings]
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=emb_lists,
        documents=documents
    )
    print("Added to collection successfully")
    
    print(f"Successfully uploaded {total_vectors} vectors to Chroma.")
    
def query_chroma(collection, query_text: str, top_k: int = 1):
    """Query the Chroma collection and return the most relevant context."""
    # Embed the user query
    embedder = MxbaiEmbedder()
    query_embedding = embedder.embed_chunks([query_text])[0]
    
    # Execute search
    query_results = collection.query(
        query_embeddings=[query_embedding.tolist()],  # Convert numpy array to list
        n_results=top_k,
        include=["documents", "distances"]
    )
    
    # Extract relevant documents
    if "documents" in query_results and len(query_results["documents"]) > 0:
        contexts = query_results["documents"][0]  # First query result
        context_str = "\n\n".join(contexts)
        return context_str
    
    return "No relevant documents found."

def main():
    """Main function to upload embeddings to Chroma."""
    # Initialize Chroma
    collection = initialize_chroma()
    
    embedder = MxbaiEmbedder()
    sample_embeddings = embedder.embed_chunks(["Sample text 1", "Sample text 2", "Sample text 3"])
    sample_documents = ["Sample document 1", "Sample document 2", "Sample document 3"]
    
    # Upload embeddings to Chroma
    upload_embeddings_to_chroma(collection, sample_embeddings, sample_documents)
    
    
    # Execute search
    result = query_chroma(collection, "When was Redis created?", top_k=1)
    
    print("\nQuery results:", result)

if __name__ == "__main__":
    # start_time = time.time()
    main()
    # print(f"Execution completed in {time.time() - start_time:.2f} seconds")