import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

questions = [
    "How many databases can Redis have?",
    "What is the purpose of logical replication (row based) in databases, and how is it different from statement-based replication?",
    "Describe the CAP Theorem and how many can it simultaneously provide?",
    "Compare the use cases for Redis lists vs. Redis sets.",
    "What is the purpose of logical replication (row based) in databases, and how is it different from statement-based replication?"
]

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
    score: Optional[float] = None  # For manual evaluation later
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

def run_pipeline_variant(
    question: str,
    llm_model: str,
    database: str,
    chunk_size: int,
    overlap: int
) -> PipelineRun:
    """Run a specific variant of the pipeline and collect statistics"""
    db_embedding_map = {
        "chroma": "mxbai-embed-large",
        "redis": "nomic-embed-large",
        "pinecone": "sentence-transformer"
    }
    
    
    result = PipelineRun(
        embedding_model=embedding_model,
        database=database,
        llm_model=llm_model,
        chunk_size=chunk_size,
        overlap=overlap,
        question=question
    )
    
    
    
    return result

# Example of running experiments
results: List[PipelineRun] = []

# Configuration variants to test
embedding_models = ["sentence_transformer", "nomic", "mxbai"]
databases = ["chroma", "redis", "pinecone"]
llm_models = ["llama 3.2", "mixtral"]
chunk_sizes = [100, 500, 1000]
overlaps = [0, 100]

# Run experiments with different configurations
for question in questions:
    for llm in llm_models:
        for db in databases:
            for chunk_size in chunk_sizes:
                for overlap in overlaps:
                    run = run_pipeline_variant(
                        question=question,
                        llm_model=llm,
                        database=db,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                    results.append(run)

# Save results for analysis
# TODO: Implement saving results to CSV/JSON
