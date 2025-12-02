
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Define the TF-IDF model globally (optional for efficiency)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
print("TF-IDF vectorizer initialized.")

def classify(user_input, movie_df, usecols):
    film_data = movie_df[usecols].copy()
    film_data['Description'] = film_data['Description'].fillna('')

    keyword_matrix = tfidf_vectorizer.fit_transform(film_data['Description'])
    input_tfidf = tfidf_vectorizer.transform([user_input])

    similarities = cosine_similarity(input_tfidf, keyword_matrix)[0]
    top_indices = similarities.argsort()[-10:][::-1]

    results = [(int(film_data.index[idx]), float(similarities[idx])) for idx in top_indices]
    return results
