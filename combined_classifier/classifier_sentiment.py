import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load the model and request full score distribution

tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

def classify(user_input, movie_embeddings):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]  # shape: (7,)

    # Convert to numpy for cosine similarity
    probs_np = probs.numpy()  # shape (7,)

    # Compute similarity
    similarities = cosine_similarity(probs_np.reshape(1, -1), movie_embeddings)[0]

    # Return top 10 as (movie_id, score)
    top_indices = np.argsort(similarities)[-10:][::-1]
    return [(int(idx + 1), float(similarities[idx])) for idx in top_indices]  # +1 if CSV IDs start at 1
