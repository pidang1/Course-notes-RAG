import os
import time
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from embed import SentenceTransformerEmbedder
from pinecone import Pinecone

## Embedding pipeline to emebd all documents using MPNetEmbedder and Pinecone's Vector DB

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
    
def clear_pinecone_index(index):
    """Clear the Pinecone index."""
    print(f"Clearing index {INDEX_NAME}...")
    index.delete(delete_all=True)
    print(f"Index {INDEX_NAME} cleared successfully.")

# Query the Pinecone index and return the most relevant context
def query_pinecone(index, query: str, top_k=1):
        # Embed the user query
        embedder = SentenceTransformerEmbedder()
        query_embedding = embedder.embed_chunks([query])[0].tolist()
        print(f"Query embedding: {query_embedding}")
        
        # Search Pinecone for relevant context
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract and format the context from search results
        contexts = []
        for match in search_results['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                contexts.append(match['metadata']['text'])
        
        # Create a prompt that includes both the context and the original query
        context_str = "\n\n".join(contexts)
        
        return context_str
    
def main():
    """Main function to upload embeddings to Pinecone. (testing and is not actually used in indexing script)"""
    # Initialize Pinecone
    index = initialize_pinecone()
    
    
    sample_embeddings = [np.random.rand(768) for _ in range(5)]  
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