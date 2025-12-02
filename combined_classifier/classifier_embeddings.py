from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def classify(user_input: str, movie_embeddings: np.ndarray):
    """
    Returns: list of (movie_index, score)
    movie_index is the row index in movies_df and movie_embeddings
    """
    # 1. Embed user query
    user_embedding = np.array(
        embedder.embed_query(user_input), dtype=np.float32
    ).reshape(1, -1)

    # 2. Compute cosine similarity
    similarities = cosine_similarity(user_embedding, movie_embeddings)[0]  # (num_movies,)

    # 3. Top 10 indices
    top_indices = np.argsort(similarities)[-10:][::-1]

    # 4. Return (index, score)
    return [(int(idx), float(similarities[idx])) for idx in top_indices]
