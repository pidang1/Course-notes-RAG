# from embed import NomicEmbedder
from pinecone_vectordb import query, initialize_pinecone

def testQuery():
    # Initialize Pinecone
    initialize_pinecone()
    
    query_text = "When was Redis found?"
    
    # Query the index
    results = query(query_text, initialize_pinecone)
    
    # Print results
    print("Query Results:")
    for result in results:
        print(result)
    

testQuery()