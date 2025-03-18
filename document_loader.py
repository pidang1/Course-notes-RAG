import os
import re
import time
from typing import Dict, List
from langchain_community.document_loaders import PyPDFLoader

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
    # removes all punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # replaces all whitespace with single spaces
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    # splits words
    words = cleaned_text.split()
    # join with lines
    one_word_per_line = "\n".join(words)
    return one_word_per_line

path = "C:\\Users\\pierr\\Documents\\ds4300notes"

def test_document_loader(pdf_directory):
    #test and load documents from a specified directory
    documents = load_documents(pdf_directory)
    
    if documents:
        # only gets first documents
        filename = list(documents.keys())[0]
        pages = documents[filename]
        
        # processes all pages
        all_text = ""
        for i, page in enumerate(pages):
            all_text += page.page_content + " "
            print(f"Added page {i+1}/{len(pages)}")
        
        print(f"\nTotal characters in document: {len(all_text)}")
        
        # process entire text
        start_time = time.time()
        processed_text = preprocess_text(all_text)
        processing_time = time.time() - start_time
        
        # count words in  processed text
        lines = processed_text.split("\n")
        word_count = len(lines)

        print(f"Document: {filename}")
        print(f"Pages processed: {len(pages)}")
        print(f"# Words in document: {word_count}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # prints all words
        print("\n=== PROCESSED TEXT ===")
        print(processed_text)
        print("\nPreprocessing successful!")

if __name__ == "__main__":
    test_document_loader(path)