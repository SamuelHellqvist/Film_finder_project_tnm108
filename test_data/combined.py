
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Get user input
userInput = input("Write a review: ")
print("Your review:", userInput)

# STEP 2: TF-IDF

# Defined the model
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# get the keybword matrix to use as data to compare the user input to
# Read the CSV file
usecols = ['Title', 'Description', 'Genres']

# get whats important
film_data = pd.read_csv("test_data/16k_Movies.csv", usecols=usecols)

# use the descriptions to get a matrix of keyowrds
film_data['combined_keywords'] = film_data['Description'].fillna('')

# use tf-idf on the data
keyword_matrix = tfidf_vectorizer.fit_transform(film_data['combined_keywords'])

# Get the tf-idf from the user input
userString = [userInput]
input_tfidf = tfidf_vectorizer.transform(userString)

# use cosine similarity between the tf-idf from user input and from keywords
cos_similarity = cosine_similarity(input_tfidf, keyword_matrix)
csims = cos_similarity[0]

film_data['Percentage Match'] = csims * 100
film_data = film_data.sort_values(by=['Percentage Match'], ascending=False)
film_data = film_data[(film_data['Percentage Match'] > 0.0)]
film_data = film_data.set_index('Percentage Match')

# creating a variable for the 10 best matches
top_recommendations = film_data.head(10)



# STEP 3: Embeddings

# load embeddings model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load dataset embeddings
review_embeddings = np.load("test_data/movie_embeddings.npy")

# Get user embedding and reshape
user_embedding = np.array(embedder.embed_query(userInput)).reshape(1, -1)

# Calculate cosine similarity
similarities = cosine_similarity(user_embedding, review_embeddings)[0]

# Get top 10 most similar reviews
top_10_indices = np.argsort(similarities)[-10:][::-1]
top_10_similarities = similarities[top_10_indices]

print("Top 10 similarities:", top_10_similarities)
print("Top 10 indices:", top_10_indices)



# STEP 4: Emotions / sentiment analysis

# get the model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
