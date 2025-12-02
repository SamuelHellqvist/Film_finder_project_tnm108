from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def classify(user_input, movie_embeddings: np.ndarray, movies_df):
    """
    user_input: text from user
    movie_embeddings: np.array of shape (num_movies, emb_dim)
    movies_df: DataFrame aligned with movie_embeddings (row i ↔ movie i)
    Returns: list of (movie_index, score)
    """

    # 1. Get user embedding
    user_embedding = np.array(embedder.embed_query(user_input), dtype=np.float32).reshape(1, -1)

    # 2. Compute cosine similarity: user vs all movies
    similarities = cosine_similarity(user_embedding, movie_embeddings)[0]  # shape: (num_movies,)

    # 3. Get top 10 most similar movies (indices into movie_embeddings / movies_df)
    top_indices = np.argsort(similarities)[-10:][::-1]

    # 4. Use the DataFrame index as movie_id
    results = []
    for idx in top_indices:
        movie_id = int(idx)  # index-based ID
        results.append((movie_id, float(similarities[idx])))

    return results
