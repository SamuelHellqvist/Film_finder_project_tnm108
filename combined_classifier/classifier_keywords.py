from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
print("TF-IDF vectorizer initialized.")

def classify(user_input: str, movies_df: pd.DataFrame, usecols=None):
    """
    TF-IDF keyword-based similarity.
    Returns: list of (movie_index, score)
    """
    if usecols is None:
        usecols = ["Title", "Description"]

    film_data = movies_df[usecols].copy()
    film_data["Description"] = film_data["Description"].fillna("")

    keyword_matrix = tfidf_vectorizer.fit_transform(film_data["Description"])
    input_tfidf = tfidf_vectorizer.transform([user_input])

    similarities = cosine_similarity(input_tfidf, keyword_matrix)[0]
    # Normalize similarities to [0, 1]
    min_s = similarities.min()
    max_s = similarities.max()
    if max_s > min_s:
        similarities = (similarities - min_s) / (max_s - min_s)
    else:
        similarities = np.ones_like(similarities)
        
    top_indices = similarities.argsort()[-10:][::-1]

    # IMPORTANT: use DataFrame index as ID
    return [(int(idx), float(similarities[idx])) for idx in top_indices]
