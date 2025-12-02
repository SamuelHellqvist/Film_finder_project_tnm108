import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

def classify(user_input: str, movie_emotion_vectors: np.ndarray):
    """
    user_input: user text
    movie_emotion_vectors: shape (num_movies, 7) – one emotion prob vector per movie
    Returns: list of (movie_index, score)
    """
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]  # (7,)

    user_emotion = probs.numpy().reshape(1, -1)  # (1, 7)

    similarities = cosine_similarity(user_emotion, movie_emotion_vectors)[0]  # (num_movies,)
    top_indices = np.argsort(similarities)[-10:][::-1]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]
