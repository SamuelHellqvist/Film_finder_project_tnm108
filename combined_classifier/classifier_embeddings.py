
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def classify(user_input, movie_embeddings):
    # Get user embedding
    user_embedding = np.array(embedder.embed_query(user_input)).reshape(1, -1)

    # Compute similarity
    similarities = cosine_similarity(user_embedding, movie_embeddings)[0]

    # Return top 10 as (movie_id, score)
    top_indices = np.argsort(similarities)[-10:][::-1]
    return [(int(idx + 1), float(similarities[idx])) for idx in top_indices]  # +1 if CSV IDs start at 1
