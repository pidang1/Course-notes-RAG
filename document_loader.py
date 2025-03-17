import os
import re
import time
from typing import Dict, List
from langchain_community.document_loaders import PyPDFLoader
#pip install langchain-community

# loads PDF documents from specified directory
def load_documents(directory_path: str) -> Dict[str, List]:
    documents = {}
    print(f"Loading documents from {directory_path}...")

    # processes all PDF files in directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            try:
                print(f"Processing {filename}...")
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                # store the pages
                documents[filename] = pages
                print(f"Successfully loaded {filename} ({len(pages)} pages)")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Loaded {len(documents)} PDF documents")
    return documents
        
# process text to one word per line and clean unwanted characters
def preprocess_text(text: str) -> str:
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    words = cleaned_text.split()
    one_word_per_line = "\n".join(words)
    return one_word_per_line




