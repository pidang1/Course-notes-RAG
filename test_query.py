# from embed import NomicEmbedder
from pinecone_vectordb import query, initialize_pinecone

def testQuery():
    # Initialize Pinecone
    db = initialize_pinecone()
    
    query_text = "explain the 3 principles of the CAP theorem"
    
    # Query the index
    results = query(db, query_text)
    
    # Print results
    print("Query Results:", results)
    

testQuery()