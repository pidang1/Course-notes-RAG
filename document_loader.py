import os
import re
import time
from typing import Dict, List
from langchain_community.document_loaders import PyPDFLoader
#pip install langchain-community

# loads PDF documents from specified directory
def load_documents(directory_path: str) -> Dict[str, List[str]]:
    documents = {}
    print(f"Loading documents from {directory_path}...")

    # process all PDF files in directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if filename.endswith('.pdf'):
            try:
                print(f"Processing {filename}...")
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                #combines all pages into one text
                text = "\n\n".join([page.page_content for page in pages])

                documents[filename] = pages
                print(f"Successfully Loaded {filename} ({len(pages)} pages)")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print (f"Loaded {len(documents)} PDF documents")
    return documents
        
