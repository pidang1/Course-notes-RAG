# from embed import NomicEmbedder
from embed import MxbaiEmbedder

def testEmbedder():
    # Create sample text chunks
    chunks = [
        "Sample embedding", "sample embedding 2"
    ]
    
    # Create embedder
    embedder = MxbaiEmbedder()
    
    # Get embeddings for list of chunks
    embeddings = embedder.embed_chunks(chunks)
    
    # Print results
    print(f"Created {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    
    # Show sample values from first embedding
    print(f"Sample values from first embedding: {embeddings[0][:5]}")
    
    return embeddings

testEmbedder()