from embed import NomicEmbedder

def testEmbedder():
    # Create sample text chunks
    chunks = [
        "Sample embedding"
    ]
    
    # Create embedder
    embedder = NomicEmbedder()
    
    # Get embeddings for list of chunks
    embeddings = embedder.embed_chunks(chunks)
    
    # Print results
    print(f"Created {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings[0].shape}")
    
    # Show sample values from first embedding
    print(f"Sample values from first embedding: {embeddings[0][:5]}")
    
    return embeddings

testEmbedder()