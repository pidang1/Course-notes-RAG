from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pinecone_vectordb import clear_pinecone_index, initialize_pinecone
from upload_to_chroma import perform_upload_chroma
from upload_to_redis import perform_upload_redis
from upload_to_pinecone import perform_upload_pinecone
from query_question import query_question
from llm_models.llama import LLM
import time
import sys
import os
import csv
from datetime import datetime
import psutil

questions = [
    "How many databases can Redis have?",
    "What is the purpose of logical replication (row based) in databases, and how is it different from statement-based replication?",
]

db_embedding_map = {
    "chroma": "mxbai-embed-large",
    "redis": "nomic-embed-large",
    "pinecone": "sentence-transformer"
}
    
db_upload_map = {
    "chroma": perform_upload_chroma,
    "redis": perform_upload_redis,
    "pinecone": perform_upload_pinecone
}


@dataclass
# Various statistics and results from the pipeline run
class PipelineRun:
    # Configuration
    embedding_model: str
    database: str
    llm_model: str
    chunk_size: int
    overlap: int
    question: str
    
    # Timing metrics
    embedding_time: float = 0.0
    upload_time: float = 0.0
    query_time: float = 0.0
    
    # Results
    num_chunks: int = 0
    answer: str = ""
    memory_usage: float = 0.0  # Memory usage in MB
    score: Optional[float] = None  # For manual qualitative evaluation later
    

def run_pipeline_variant(
    path: str,
    question: str,
    llm_model: LLM,
    database: str,
    chunk_size: int,
    overlap: int,
    prompt: str,
    ) -> PipelineRun:
    """Run a specific variant of the pipeline and collect statistics"""
    
    
    result = PipelineRun(
        embedding_model=db_embedding_map[database],
        database=database,
        llm_model=llm_model,
        chunk_size=chunk_size,
        overlap=overlap,
        question=question,
    )
    
    # Retrieve the appropriate upload function based on the database
    db_upload_func = db_upload_map[database]
    
    # Measure memory usage before execution
    process = psutil.Process(os.getpid()) 
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    # Embed the dataset and retrieve statistics
    index, statistics = db_upload_func(path, chunk_size, overlap)
    
    # Measure memory usage after embedding
    mem_after_embedding = process.memory_info().rss / (1024 * 1024)
    
    
    # Store the statistics in the result object
    result.embedding_time = statistics["embed_time"]
    result.upload_time = statistics["upload_time"]
    result.num_chunks = statistics["chunk_count"]
    
    # Measure memory usage before query
    mem_before_query = process.memory_info().rss / (1024 * 1024)
    
    query_start_time = time.time()
    answer = query_question(database, index, question, llm_model, prompt)
    query_time = time.time() - query_start_time
    
    # Measure memory usage after query
    mem_after_query = process.memory_info().rss / (1024 * 1024)
    
    result.query_time = query_time
    result.answer = answer
    
    # Calculate peak memory usage during the run
    result.memory_usage = max(mem_after_embedding, mem_after_query) - mem_before
    
    print(result)
    
    return result

# Example of running experiments
results: List[PipelineRun] = []


if len(sys.argv) < 2:
    print("Please provide a directory path containing PDF documents")
    print("Usage: python experiment.py <directory_path>")
    quit()

directory_path = sys.argv[1]

if not os.path.isdir(directory_path):
    print(f"Error: {directory_path} is not a valid directory")
    quit()

# Configuration variants to test
llama = LLM("llama3.2")
mistral = LLM("mistral")
databases = ["chroma", "redis", "pinecone"]
llm_models = [llama, mistral]
prompts = [
    """Synthesize the information across these documents to provide a comprehensive answer. 
    
    Review the following retrieved passages and determine which are relevant to answering: \"{user_query}\"\n\n
    {retrieved_passage}\n""",
    """
    You have retrieved multiple documents related to: "{user_query}"

    {retrieved_passage}

    Return the answer to the question if only the retrieved passages are relevant, else return "I don't know".
    """]
chunk_sizes = [200, 500]
overlaps = [0, 100]

# Write results out into CSV
def write_results_to_csv(results, csv_path, append=False):
    """Write results to CSV file, either creating a new file or appending to existing one"""
    # Define CSV headers based on PipelineRun fields
    headers = [
        "embedding_model", "database", "llm_model", "chunk_size", "overlap", 
        "question", "embedding_time", "upload_time", "query_time", 
        "num_chunks", "answer", "memory_usage", "score", 
    ]
    
    # Open file in append mode if append=True and file exists, otherwise write mode
    mode = 'a' if append and os.path.exists(csv_path) else 'w'
    
    # Write results to CSV
    with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        # Write header only if we're creating a new file
        if mode == 'w':
            writer.writeheader()
        
        for result in results:
            # Convert the result to a dictionary
            result_dict = {
                "embedding_model": result.embedding_model,
                "database": result.database,
                "llm_model": result.llm_model.model_name if hasattr(result.llm_model, 'model_name') else str(result.llm_model),
                "chunk_size": result.chunk_size,
                "overlap": result.overlap,
                "question": result.question,
                "embedding_time": result.embedding_time,
                "upload_time": result.upload_time,
                "query_time": result.query_time,
                "num_chunks": result.num_chunks,
                "answer": result.answer,
                "memory_usage": result.memory_usage if result.memory_usage is not None else "",
                "score": result.score if result.score is not None else ""
            }
            writer.writerow(result_dict)
    
    print(f"Results written to {csv_path}")

csv_path = "./experiment_results.csv"

# Run experiments with different configurations
for db in databases:
    db_results = []
    for llm in llm_models:
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                for prompt in prompts:
                    for question in questions:
                        try:
                            # Record memory usage before the run
                            
                            run = run_pipeline_variant(
                                path=directory_path,
                                question=question,
                                llm_model=llm,
                                database=db,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                prompt=prompt
                            )
                            
                            db_results.append(run)
                        except Exception as e:
                            print(f"Error running experiment with {db}, {llm.model_name}, chunk_size={chunk_size}, overlap={overlap}: {e}")
                            continue
                    # Delete pinecone index after each overlap switch
                    if db == "pinecone":
                        index = initialize_pinecone()
                        clear_pinecone_index(index)
                # Delete pinecone index after each chunk size switch
                if db == "pinecone":
                    index = initialize_pinecone()
                    clear_pinecone_index(index)
    # Write results for this database to CSV
    print(f"Writing results for database: {db}")
    # For the first database, create a new file; for others, append
    append_mode = db != databases[0]
    write_results_to_csv(db_results, csv_path, append=append_mode)
    
    print(f"Completed experiments for database: {db}")
    
    



