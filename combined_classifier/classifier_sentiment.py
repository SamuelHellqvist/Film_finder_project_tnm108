import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load emotion model once
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

def classify(user_input, movie_emotion_vectors, movies_df):
    """
    user_input: user description string
    movie_emotion_vectors: numpy array of shape (num_movies, 7)
    movies_df: DataFrame with movie metadata including 'id'
    """

    # 1. Encode user input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]   # shape: (7,)
    user_emotion = probs.numpy().reshape(1, -1)         # shape: (1, 7)

    # 2. Compute cosine similarity between user emotion and movie emotions
    similarities = cosine_similarity(user_emotion, movie_emotion_vectors)[0]

    # 3. Select top 10 movies
    top_idx = np.argsort(similarities)[-10:][::-1]

    # 4. Return list of (movie_id, score)
    results = []
    for idx in top_idx:
        movie_id = int(idx)  # index as ID
        results.append((movie_id, float(similarities[idx])))

    return results
