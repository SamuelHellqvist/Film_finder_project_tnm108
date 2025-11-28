import numpy as np
import pandas as pd
from transformers import pipeline

# 1. Load data (from your local CSV)
df = pd.read_csv("test_data/16k_Movies.csv")  # adapt path
# Make sure you have a column with descriptions
# e.g., df["description"] = df["plot"] or similar

# 2. Emotion classifier
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# 3. Process all descriptions
emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

vectors = []

for text in df['Description'][0:20]:
    if isinstance(text, float) and pd.isna(text):
        text = ""
    scores = classifier(text)[0]  # list of dicts
    # Ensure fixed order of labels
    scores_by_label = {item["label"]: item["score"] for item in scores}
    vec = [scores_by_label[label] for label in emotion_labels]
    vectors.append(vec)

vectors = np.array(vectors, dtype="float32")  # shape (N, 7)

# 4. Save everything
#df[["movie_id", "title", "description"]].to_parquet("movies.parquet", index=False)
np.save("emotion_vectors.npy", vectors)

