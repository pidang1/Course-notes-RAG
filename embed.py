from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import ollama

class SentenceTransformerEmbedder:
    """A simple class that embeds text using the sentence-transformers/all-mpnet-base-v2 model"""
    
    def __init__(self):
        """Initialize the embedder with the all-mpnet-base-v2 model"""
        print("Loading all-mpnet-base-v2 embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded successfully with embedding dimension: {self.embedding_dim}")
        
    # Embed a list of text chunks
    def embed_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        
        # Empty case
        if not chunks:
            return []
        # Use the model to encode the chunks
        embeddings = self.model.encode(chunks)
        return embeddings
    
    # Function to get the embedding dimension
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

# Simple test function
def test_embedder():
    # Create sample text chunks
    chunks = [
        "Sample embedding"
    ]
    
    # Create embedder
    embedder = SentenceTransformerEmbedder()
    
    # Get embeddings for list of chunks
    embeddings = embedder.embed_chunks(chunks)
    
    # Print results
    print(f"Created {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings[0].shape}")
    
    # Show sample values from first embedding
    print(f"Sample values from first embedding: {embeddings[0][:5]}")
    
    return embeddings

if __name__ == "__main__":
    test_embedder()


class NomicEmbedder:
    """A simple class that embeds text using the nomic-embed-text model"""
    
    def __init__(self):
        """Initialize the embedder with the nomic-embed-text model"""
        print("Loading nomic-embed-text embedding model...")
        self.model = ollama
        # Get embedding dimension by testing with a sample text
        sample_response = self.model.embeddings(model='nomic-embed-text', prompt="Sample text")
        sample_embedding = np.array(sample_response['embedding'])
        self.embedding_dim = len(sample_embedding)
        print(f"Model loaded successfully with embedding dimension: {self.embedding_dim}")
    # Embed a list of text chunks
    def embed_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        # Empty case
        if not chunks:
            return []
        # Use the model to encode the chunks chunk by chunk
        embeddings = []
        for chunk in chunks:
            response = self.model.embeddings(model='nomic-embed-text', prompt=chunk)
            embedding = np.array(response['embedding'])
            embeddings.append(embedding)
        return embeddings
    
    # Function to get the embedding dimension
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim


class MxbaiEmbedder:
    """A simple class that embeds text using the mxbai-embed-large model"""
    
    def __init__(self):
        """Initialize the embedder with the mxbai-embed-large model"""
        print("Loading mxbai-embed-large embedding model...")
        self.model = ollama
        # Get embedding dimension by testing with a sample text
        sample_response = self.model.embeddings(model='mxbai-embed-large', prompt="Sample text")
        sample_embedding = np.array(sample_response['embedding'])
        self.embedding_dim = len(sample_embedding)
        print(f"Model loaded successfully with embedding dimension: {self.embedding_dim}")
    # Embed a list of text chunks
    def embed_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        # Empty case
        if not chunks:
            return []
        # Use the model to encode the chunks chunk by chunk
        embeddings = []
        for chunk in chunks:
            response = self.model.embeddings(model='mxbai-embed-large', prompt=chunk)
            embedding = np.array(response['embedding'])
            embeddings.append(embedding)
        return embeddings
    
    # Function to get the embedding dimension
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim


