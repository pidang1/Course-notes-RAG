import os
import time
import json
from typing import List, Dict
from langchain_text_splitters import TokenTextSplitter

def chunk_by_tokens(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

def test_chunking_strategies