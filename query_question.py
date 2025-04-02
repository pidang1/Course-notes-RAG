from llm_models.llama import LLM
from chroma_vectordb import query_chroma
from pinecone_vectordb import query_pinecone
from redis_vectordb import query_redis

def query_question(indexName, index, query, llm, prompt):
    
    query_function_map = {
        "chroma": query_chroma,
        "redis": query_redis,
        "pinecone": query_pinecone,
    }
    
    # Get the chunk based on the index name
    chunk = query_function_map[indexName](index, query)
    
    # Format the prompt with the user query and retrieved chunks
    context = prompt.format(user_query=query, retrieved_passage=chunk)
    
    response = llm.generate_response(context)
    return response
    
    
    
    
    
    
    
    