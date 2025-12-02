
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Define the TF-IDF model globally (optional for efficiency in repeated calls)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

def classify(user_input, movie_df_path, usecols=['Description']):
    """
    Compute TF-IDF similarity between the user's input and movie descriptions.
    Returns a list of (row_index, similarity_score) tuples sorted by similarity (desc).
    
    Parameters:
    - user_input (str): The user's text description.
    - movie_df_path (str): Path to the CSV file containing movies.
    - usecols (list[str]): Columns to read. Must include 'Description'.
    
    Example return: [(42, 0.713), (7, 0.689), ...]  # indices into the CSV
    """
    # Load movie data from CSV (only necessary columns)
    film_df = pd.read_csv(movie_df_path, usecols=usecols)

    # Ensure 'Description' column exists
    if 'Description' not in film_df.columns:
        raise ValueError("CSV must contain a 'Description' column.")

    # Clean descriptions
    film_df['Description'] = film_df['Description'].fillna('')

    # Fit TF-IDF on all movie descriptions
    keyword_matrix = tfidf_vectorizer.fit_transform(film_df['Description'])

    # Transform the user input
    input_tfidf = tfidf_vectorizer.transform([user_input])

    # Compute cosine similarity between user input and each movie description
    similarities = cosine_similarity(input_tfidf, keyword_matrix)[0]  # shape: (num_movies,)

    # Top 10 most similar rows by index
    top_indices = similarities.argsort()[-10:][::-1]

    # Return (row_index, score) tuples; row_index is the DataFrame positional index (0..N-1)
    results = [(int(idx), float(similarities[idx])) for idx in top_indices]
    return results
