
from pinecone_vectordb import clear_pinecone_index, initialize_pinecone

index = initialize_pinecone()
clear_pinecone_index(index)