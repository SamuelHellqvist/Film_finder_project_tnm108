
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Define the TF-IDF model globally (optional for efficiency)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
print("TF-IDF vectorizer initialized.")
def classify(user_input, movie_df_path, usecols):
    # Load movie data from CSV
    film_data = pd.read_csv(movie_df_path, usecols=usecols)

    # Ensure descriptions are clean
    film_data['Description'] = film_data['Description'].fillna('')

    # Create TF-IDF matrix for all movie descriptions
    keyword_matrix = tfidf_vectorizer.fit_transform(film_data['Description'])

    # Transform user input into TF-IDF
    input_tfidf = tfidf_vectorizer.transform([user_input])

    # Compute cosine similarity
    similarities = cosine_similarity(input_tfidf, keyword_matrix)[0]

    # Get top 10 most similar movies
    top_indices = similarities.argsort()[-10:][::-1]

    # Return list of (movie_title, score)
    results = [(film_data.iloc[idx]['Title'], float(similarities[idx])) for idx in top_indices]
    return results
