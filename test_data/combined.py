# File made for combining sentiment analysis and the embeddings to make a "combined" classifyer

# imports
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from torch import cosine_similarity

# Step 1: Get the users input
userInput = input("Write a review: ")
print(userInput)

# Step 2: Embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embeddings from the dataset
review_embeddings = np.load("test_data/movie_embeddings.npy")

# Embeddings from the user input
user_embedding = embedder.embed_query(userInput)


# calculate cosine similarity with the embeddings
# similarities = cosine_similarity([user_embedding], review_embeddings)
similarities = cosine_similarity([user_embedding], review_embeddings)[0]
top_10_indices = np.argsort(similarities)[-10:][::-1]
top_10_similarities = similarities[top_10_indices]
print(top_10_similarities)