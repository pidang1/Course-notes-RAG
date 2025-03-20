from pinecone import Pinecone
import os
from typing import List, Dict, Any
from llm_models.llama import LLM
from embed import MPNetEmbedder
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Get Pinecone API key and environment from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class RAGSearch:
    #Initialize the RAG search with Pinecone and LLM components
    def __init__(self, index_name: str, llm_model_name: str = "llama3.2"):
        # Initialize the Pinecone client
        client = Pinecone(api_key=PINECONE_API_KEY)
        
        # Connect to the specified index
        self.index = client.Index(index_name)
        
        # Initialize the embedder
        self.embedder = MPNetEmbedder()
        
        # Initialize the LLM model
        self.llm = LLM(llm_model_name)
        
    def search_and_respond(self, user_query: str, top_k: int = 5) -> str:
        """
        Process a user query by embedding it, retrieving relevant context from Pinecone,
        and generating a response using the LLM model
        
        Args:
            user_query: The user's question or prompt
            top_k: Number of most relevant documents to retrieve
            
        Returns:
            The LLM's response with context-enhanced knowledge
        """
        # 1. Embed the user query
        query_embedding = self.embedder.embed_chunks([user_query])[0].tolist()
        
        # 2. Search Pinecone for relevant context
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # 3. Extract and format the context from search results
        contexts = []
        for match in search_results['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                contexts.append(match['metadata']['text'])
        
        # 4. Create a prompt that includes both the context and the original query
        context_str = "\n\n".join(contexts)
        enhanced_prompt = f"""Use the following information to answer the question.

Context information:
{context_str}

User question: {user_query}"""

        # 5. Send the enhanced prompt to the LLM and get the response
        response = self.llm.generate_response(enhanced_prompt)
        
        return response

def main():
    index_name = os.getenv("PINECONE_INDEX_NAME", "ds4300")
    model_name = "llama3.2"  
    
    rag_search = RAGSearch(index_name, model_name)
    
    print("Welcome to RAG Search! Type 'exit' to quit.")
    
    while True:
        user_query = input("\nEnter your question: ")
        
        if user_query.lower() == 'exit':
            print("Thank you for using RAG Search. Goodbye!")
            break
        response = rag_search.search_and_respond(user_query)
        
        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main()