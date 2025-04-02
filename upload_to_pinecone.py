import os
import sys
from typing import List, Dict, Any
import numpy as np
from document_loader import load_documents
from embed import SentenceTransformerEmbedder
from pinecone_vectordb import initialize_pinecone, upload_embeddings_to_pinecone
from chunking import chunk_text
import time

# filepath: c:\Users\daoho\Desktop\Progamming\Course-notes-RAG\main.py

def process_documents(directory_path: str, chunk_size: int = 500, overlap: int = 100) -> tuple:
    """Process documents from a directory, create chunks, and embed them."""
    # Load documents
    documents = load_documents(directory_path)
    
    if not documents:
        print("No documents found.")
        return [], []
    
    # Initialize embedder
    
    embedder = SentenceTransformerEmbedder()
    
    all_chunks = []
    all_metadata = []
    
    # Process each document
    embed_start_time = time.time()
    for filename, pages in documents.items():
        print(f"Processing {filename}...")
        
        # Combine all pages into a single text
        complete_text = ""
        for page in pages:
            complete_text += page.page_content + " "
        
        # Create chunks from the document
        chunks = chunk_text(complete_text, chunk_size, overlap)
        print(f"Created {len(chunks)} chunks from {filename}")
        
        # Create metadata for each chunk
        metadata_list = [
            {
                "source": filename,
                "text": chunk 
            }
            for chunk in chunks
        ]
        
        # Add to master lists
        all_chunks.extend(chunks)
        all_metadata.extend(metadata_list)
    
    # Embed all chunks
    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embedder.embed_chunks(all_chunks)
    embed_time = time.time() - embed_start_time
    print(f"Successfully embedded {len(embeddings)} chunks")
    
    return embeddings, all_metadata, embed_time

def perform_upload_pinecone(path: str, chunk_size: int, overlap: int):
    print(f"Processing documents from {path}")
    
    # Process documents (load, chunk, embed)
    embeddings, metadata, embed_time = process_documents(path, chunk_size, overlap)
    
    if len(embeddings) == 0:
        print("No embeddings were generated. Exiting...")
        return
    
    # Upload to Redis
    upload_start_time = time.time()
    index = initialize_pinecone()
    upload_embeddings_to_pinecone(index, embeddings, metadata)
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
        print("Usage: python upload_to_pinecone.py <directory_path>")
        return
    
    directory_path = sys.argv[1]
    
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return
    
    print(f"Processing documents from {directory_path}")
    
    # Process documents (load, chunk, embed)
    embeddings, metadata, embed_time = process_documents(directory_path)
    
    if len(embeddings) == 0:
        print("No embeddings were generated. Exiting...")
        return
    
    # Upload to Pinecone
    index = initialize_pinecone()
    upload_embeddings_to_pinecone(index, embeddings, metadata)
    print("Process completed successfully!")
    return index

if __name__ == "__main__":
    main()
