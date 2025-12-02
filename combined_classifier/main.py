import pandas as pd
import numpy as np
from combined_classifier.classifier_embeddings import classify as classify_embeddings
from combined_classifier.classifier_sentiment import classify as classify_sentiment
from combined_classifier.classifier_keywords import classify as classify_keywords

# Load movie data from CSV
movies_df = pd.read_csv("data/movies.csv")

# Load embeddings
movie_embeddings = np.load("data/movie_embeddings.npy")

# Get user input
user_input = input("Describe the movie you want: ")

# Run classifiers
results_embeddings = classify_embeddings(user_input, movie_embeddings)
results_sentiment = classify_sentiment(user_input, movies_df)
results_keywords = classify_keywords(user_input, movies_df)

# Combine results
combined_scores = {}
for res in [results_embeddings, results_sentiment, results_keywords]:
    for movie_id, score in res:
        combined_scores[movie_id] = combined_scores.get(movie_id, 0) + score

# Sort and show top 5
top_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop recommendations:")
for movie_id, score in top_movies:
    movie = movies_df[movies_df["id"] == movie_id].iloc[0]
    print(f"{movie['title']} - {movie['description']} (Score: {score:.2f})")
