# import time
# import psutil
# from typing import List, Dict
# from sentence_transformers import SentenceTransformer
# from nomic.embed import EmbeddingModel
# from InstructorEmbedding import INSTRUCTOR



# # generates embeddings using sentence transformers
# def embed_with_sentence_transformer(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> Dict:
#     print(f"Generating embeddings using model: {model_name}...")
    
#     # measures speed
#     start_time = time.time()

#     # measures memory usage before embedding
#     memory_before = psutil.Process().memory_info().rss / 1024 / 1024

#     # load and generate embeddings using SenctenceTransformer
#     model = SentenceTransformer(model_name)
#     embeddings = model.encode(chunks)

#     # measures memory usage after embedding
#     memory_after = psutil.Process().memory_info().rss / 1024 / 1024

#     # calculate total time
#     total_time = time.time() - start_time
    
#     return {
#         "model": f"sentence-transformers/{model_name}",
#         "embeddings": embeddings,
#         "time_seconds": total_time,
#         "memory_usage_mb": memory_after - memory_before,
#         "dimension": embeddings.shape[1] if len(embeddings) > 0 else 0
#     }