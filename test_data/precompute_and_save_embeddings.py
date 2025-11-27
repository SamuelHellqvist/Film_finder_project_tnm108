import pandas as pd
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load data
df = pd.read_csv("test_data/16k_Movies.csv", usecols=["Description"])
texts = df["Description"].fillna("").tolist()

# Initialize embedder
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Compute embeddings
embeddings = embedder.embed_documents(texts)

# Save embeddings to disk
np.save("movie_embeddings.npy", embeddings)

print("done!")