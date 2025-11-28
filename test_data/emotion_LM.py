#from transformers import pipeline
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


# Load the model and request full score distribution

tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# skrivas ut på bokstavlig ordning. 

# read the descriptions from the data file
usecols = ["Title", "Description"]

# Read the CSV file
df = pd.read_csv("test_data/16k_Movies.csv", usecols=usecols)

title_texts = df['Title'].tolist()

desc_texts = df['Description'].tolist()
# for Description in desc_texts[0:20]: 
#     print(Description)

#print(title_texts[275] )
#print(desc_texts[275])


#text = ("Ed and Lorraine Warren, world-renowned investigators of supernatural events, are called in to help a family terrorized by dark forces. In the family’s house, deep in the countryside, the Warrens are forced to confront a powerful demonic presence.")
#text = desc_texts[5]
text = "this is a scary movie. very scary and frightening with a lot of suspense and horror elements. but alos, i am angry and feeling bad"
#text = "A hilarious space adventure where the hero travels between planets and fights monsters, filled with comedy and fun moments."
inputs = tokenizer(text, return_tensors="pt")


#vectors = np.load("emotion_vectors.npy")
#print(vectors)


inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0]  # shape: (7,)

# Convert to numpy for cosine similarity
probs_np = probs.numpy()  # shape (7,)

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# 4. Print all emotions sorted by score
scores_list = list(zip(EMOTION_LABELS, probs_np))
scores_list.sort(key=lambda x: x[1], reverse=True)

print("All emotions sorted:")
for label, score in scores_list:
    print(f"{label}: {score:.4f}")

# Top two emotions
top_two = scores_list[:2]
print("\nTop two emotions:")
for label, score in top_two:
    print(f"{label}: {score:.4f}")

# 5. Load your precomputed emotion vectors for all movies
# This file should have shape (num_movies, 7)
emotion_data = np.load("test_data/emotion_vectors.npy")

# 6. Cosine similarity between this movie and all movies
# probs_np has shape (7,) so wrap it in a list to make shape (1,7)
similarities = cosine_similarity([probs_np], emotion_data)[0]  # shape (num_movies,)

best_idx = similarities.argmax()
print("\nMost similar movie index:", best_idx)
print("Similarity score:", similarities[best_idx])

print("\nMost similar movie title:", title_texts[best_idx])
print("Most similar movie description:", desc_texts[best_idx])