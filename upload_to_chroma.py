import os
import sys
from typing import List, Dict, Any
import numpy as np
from document_loader import load_documents
from embed import MxbaiEmbedder
from chroma_vectordb import initialize_chroma, upload_embeddings_to_chroma
from chunking import chunk_text
import time

def process_documents(directory_path: str, chunk_size: int = 500, overlap: int = 100) -> tuple:
    """Process documents from a directory, create chunks, and embed them."""
    # Load documents
    documents = load_documents(directory_path)
    
    if not documents:
        print("No documents found.")
        return [], []
    
    # Initialize embedder
    embedder = MxbaiEmbedder()
    
    all_chunks = []
    
    # Process each document
    for filename, pages in documents.items():
        print(f"Processing {filename}...")
        
        # Combine all pages into a single text
        complete_text = ""
        for page in pages:
            complete_text += page.page_content + " "
        
        # Create chunks from the document
        chunks = chunk_text(complete_text, chunk_size, overlap)
        print(f"Created {len(chunks)} chunks from {filename}")
        
        
        
        # Add to master lists
        all_chunks.extend(chunks)
    
    # Embed all chunks
    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embedder.embed_chunks(all_chunks)
    print(f"Successfully embedded {len(embeddings)} chunks")
    
    return embeddings, all_chunks

def perform_upload_chroma(path: str, chunk_size: int, overlap: int):
    print(f"Processing documents from {path}")
    
    # Process documents (load, chunk, embed)
    embeddings, metadata, embed_time = process_documents(path, chunk_size, overlap)
    
    if len(embeddings) == 0:
        print("No embeddings were generated. Exiting...")
        return
    
    # Upload to Redis
    upload_start_time = time.time()
    index = initialize_chroma()
    upload_embeddings_to_chroma(index, embeddings, metadata)
    upload_time = time.time() - upload_start_time
    print("Process completed successfully!")
    
    statistics = {
        "embed_time": embed_time,
        "upload_time": upload_time,
        "chunk_count": len(embeddings),
     }
    return index, statistics

def main():
    if len(sys.argv) < 2:
        print("Please provide a directory path containing PDF documents")
        print("Usage: python upload_to_chroma.py <directory_path>")
        return
    
    directory_path = sys.argv[1]
    
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return
    
    print(f"Processing documents from {directory_path}")
    
    # Process documents (load, chunk, embed)
    embeddings, metadata = process_documents(directory_path)
    
    if len(embeddings) == 0:
        print("No embeddings were generated. Exiting...")
        return
    
    # Upload to Redis
    index = initialize_chroma()
    print("metadata[0]:", metadata[0])
    upload_embeddings_to_chroma(index, embeddings, metadata)
    print("Process completed successfully!")
    return index

if __name__ == "__main__":
    main()
