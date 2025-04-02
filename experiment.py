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


questions = [
    "How many databases can Redis have?",
    "What is the purpose of logical replication (row based) in databases, and how is it different from statement-based replication?",
    "Describe the CAP Theorem and how many can it simultaneously provide?",
    "Compare the use cases for Redis lists vs. Redis sets.",
    "What is the purpose of logical replication (row based) in databases, and how is it different from statement-based replication?"
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
    score: Optional[float] = None  # For manual qualitative evaluation later
    

def run_pipeline_variant(
    path: str,
    question: str,
    llm_model: LLM,
    database: str,
    chunk_size: int,
    overlap: int,
    prompt: str
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
    
    # Embed the dataset and retrieve statistics
    index, statistics = db_upload_func(path, chunk_size, overlap)
    
    # Store the statistics in the result object
    result.embedding_time = statistics["embed_time"]
    result.upload_time = statistics["upload_time"]
    result.num_chunks = statistics["chunk_count"]
    
    query_start_time = time.time()
    answer = query_question(database, index, question, llm_model, prompt)
    query_time = time.time() - query_start_time
    
    result.query_time = query_time
    result.answer = answer
    
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
embedding_models = ["sentence_transformer", "nomic", "mxbai"]
databases = ["chroma", "redis", "pinecone"]
llm_models = [llama, mistral]
prompts = [
    """Synthesize the information across these documents to provide a comprehensive answer. 
    Highlight areas where sources agree or disagree, and explain the significance of these patterns 
    given the following context:
    
    Review the following retrieved passages and determine which are relevant to answering: \"{user_query}\"\n\n
    {retrieved_passage}\n\n
    For each relevant passage, explain why it contains useful information. 
    Then provide a comprehensive answer using only the relevant information.""",
    """
    You have retrieved multiple documents related to: "{user_query}"

    {retrieved_passage}

    Synthesize the information across these documents to provide a comprehensive answer. Highlight areas where sources agree or disagree, and explain the significance of these patterns.
    """]
chunk_sizes = [100, 500]
overlaps = [0, 100]

# Run experiments with different configurations

for llm in llm_models:
    for db in databases:
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                for prompt in prompts:
                    for question in questions:
                        run = run_pipeline_variant(
                            path=directory_path,
                            question=question,
                            llm_model=llm,
                            database=db,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            prompt=prompt
                        )
                        results.append(run)
                    # Delete pinecone index after each overlap switch
                    index = initialize_pinecone()
                    clear_pinecone_index(index)
                # Delete pinecone index after each chunk size switch
                index = initialize_pinecone()
                clear_pinecone_index(index)
# Write results out into CSV

csv_path = f"./experiment_results.csv"

# Define CSV headers based on PipelineRun fields
headers = [
    "embedding_model", "database", "llm_model", "chunk_size", "overlap", 
    "question", "embedding_time", "upload_time", "query_time", 
    "num_chunks", "answer", "score"
]

# Write results to CSV
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
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
            "score": result.score if result.score is not None else ""
        }
        writer.writerow(result_dict)

print(f"Results written to {csv_path}")


