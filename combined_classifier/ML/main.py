import pandas as pd
import numpy as np
from .classifier_embeddings import classify as classify_embeddings
from .classifier_sentiment import classify as classify_sentiment
from .classifier_keywords import classify as classify_keywords

# Load movie data from CSV
movies_df = pd.read_csv("test_data/16k_Movies.csv")
#movies_df = pd.read_csv("test_data/16k_Movies.csv")
movie_embeddings = np.load("test_data/movie_embeddings.npy")
emotion_vectors = np.load("test_data/emotion_vectors.npy")  # must be (num_movies, 7)

#user_input = input("Describe the movie you want: ")
#GET IT FROM FLASK INSTEAD...

def run_classifier(user_input):
    """Return the top 10 movie recommendations for a given text."""

    # Run each model
    results_embeddings = classify_embeddings(user_input, movie_embeddings)
    results_sentiment = classify_sentiment(user_input, emotion_vectors)
    results_keywords = classify_keywords(
        user_input, movies_df, usecols=["Title", "Description"]
    )

    # Combine scores
    combined_scores = {}
    for res in [results_embeddings, results_sentiment, results_keywords]:
        for movie_id, score in res:
            combined_scores[movie_id] = combined_scores.get(movie_id, 0) + score

    # Sort and pick top 10
    top_movies = sorted(
        combined_scores.items(), key=lambda x: x[1], reverse=True
    )[:10]

     # Sort ALL movies by combined score (best first)
    sorted_movies = sorted(
        combined_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Now pick up to 10 movies with UNIQUE titles
    formatted = []
    seen_titles = set()

    for movie_id, score in sorted_movies:
        movie = movies_df.iloc[movie_id]
        title = movie["Title"]

        if title in seen_titles:
            continue  # skip duplicate title

        seen_titles.add(title)
        formatted.append({
            "title": title,
            "description": movie["Description"],
            "score": float(score),
        })

        if len(formatted) == 10:
            break  # we have our top 10 unique movies

    return formatted