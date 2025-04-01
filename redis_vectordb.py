# import os
# from typing import List
# from embed import NomicEmbedder  # Changed import here
# import redis
# from redis.commands.search.query import Query
# import numpy as np


# class RedisVectorDB:
#     def __init__(self, index_name: str, embedding_model: NomicEmbedder, redis_host: str = "localhost", redis_port: int = 6379):
#         self.index_name = index_name
#         self.embedding_model = embedding_model
#         self.redis_host = redis_host
#         self.redis_port = redis_port
#         self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port)

#     def create_index(self, embedding_dimension: int):
#         """
#         Creates a Redis index with the specified name and embedding dimension.
#         """
#         try:
#             self.redis_client.ft(self.index_name).info()
#             print("Index already exists")
#             return
#         except:
#             pass

#         self.redis_client.ft(self.index_name).create_index(
#             fields=[
#                 redis.commands.search.field.VectorField(
#                     "embedding",
#                     "FLAT",
#                     {
#                         "TYPE": "FLOAT32",
#                         "DIM": embedding_dimension,
#                         "DISTANCE_METRIC": "COSINE",
#                     },
#                 )
#             ]
#         )
#         print(f"Index {self.index_name} created successfully.")

#     def add_vectors(self, texts: List[str]):
#         """
#         Adds text embeddings to the Redis vector database.
#         """
#         embeddings = self.embedding_model.embed_chunks(texts)
        
#         # Use a pipeline for faster insertion
#         pipeline = self.redis_client.pipeline()
#         for i, embedding in enumerate(embeddings):
#             # Generate a unique key for each vector
#             key = f"vector:{i}"  # Changed key naming
#             # Store the embedding as a numpy array
#             pipeline.hset(key, mapping={"embedding": embedding.astype(np.float32).tobytes()})  # Convert embedding to bytes
#         pipeline.execute()
#         print(f"Added {len(embeddings)} vectors to Redis.")

#     def search(self, query: str, top_k: int = 5) -> List[str]:
#         """
#         Searches the Redis vector database for the most similar vectors to the query.
#         """
#         query_embedding = self.embedding_model.embed_chunks([query])[0]
        
#         # Ensure the query embedding is a numpy array of float32
#         if not isinstance(query_embedding, np.ndarray):
#             query_embedding = np.array(query_embedding, dtype=np.float32)
#         else:
#             query_embedding = query_embedding.astype(np.float32)
            
#         # Convert the query embedding to bytes
#         query_embedding_bytes = query_embedding.tobytes()
        
#         # Prepare the query
#         q = Query(
#             f"*=>[KNN {top_k} @embedding $query_vector AS score]"
#         ).sort_by("score").dialect(2)
        
#         params = {"query_vector": query_embedding_bytes}
        
#         # Execute the query
#         results = self.redis_client.ft(self.index_name).search(q, query_params=params)
        
#         # Extract and return the results
#         return [(result.id, result.score) for result in results.docs]

#     def delete_index(self):
#         """
#         Deletes the Redis index.
#         """
#         self.redis_client.ft(self.index_name).dropindex()
#         print(f"Index {self.index_name} deleted.")

# def test_redis_vector_db():
#     # Initialize the NomicEmbedder
#     nomic_embedder = NomicEmbedder()
#     embedding_dimension = nomic_embedder.get_embedding_dimension()
    
#     # Configure Redis connection
#     redis_host = "localhost"
#     redis_port = 6379
#     index_name = "my_index"
    
#     # Initialize the RedisVectorDB
#     redis_db = RedisVectorDB(index_name, nomic_embedder, redis_host, redis_port)
    
#     # Create the index
#     redis_db.create_index(embedding_dimension)
    
#     # Sample texts to add
#     texts = [
#         "This is the first sample text.",
#         "Here is the second text for testing.",
#         "The third text is a bit longer."
#     ]
    
#     # Add the vectors to the database
#     redis_db.add_vectors(texts)
    
#     # Perform a search
#     query = "testing the search functionality"
#     results = redis_db.search(query)
#     print(f"Search results for query '{query}': {results}")
    
#     # Clean up: Delete the index
#     redis_db.delete_index()

# if __name__ == "__main__":
#     test_redis_vector_db()
