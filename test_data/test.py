import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings


# == functions ==
# loader function to let the user know that the system is actaully running
import threading
import time
import sys

def show_loading(message="Loading..."):
    stop_flag = threading.Event()

    def loader():
        while not stop_flag.is_set():
            for char in "|/-\\":
                sys.stdout.write(f"\r{message} {char}")
                sys.stdout.flush()
                time.sleep(0.2)

    t = threading.Thread(target=loader)
    t.start()
    return stop_flag

#using the latest version of the embedder
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# trying to only save some of the columns
usecols = ["Title", "Description", "Genres"]

# Read the CSV file
film_data = pd.read_csv("test_data/16k_Movies.csv", usecols=usecols)
film_data['combined_keywords'] = film_data['Description'].fillna('')

def discriptors(idx):
    try:
        val = film_data.iloc[idx, 1]
        return None if pd.isna(val) else val
    except Exception:
        return None

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
keyword_matrix = tfidf_vectorizer.fit_transform(film_data['combined_keywords'])

# User input
userInput = input("Write a review: ")

while userInput == "":
    userInput = input("Please write a review: ")


# test text: I want to see a funny movie in space where the hero travels between 
# planets and fights monsters

# Turn the string into an array (Needed for tfidf)
userString = [userInput]

input_tfidf = tfidf_vectorizer.transform(userString)

# control variable if we use embeddings or the method from the previuous project
control = 1
if control == 0:
    # Cosine similarity
    cos_similarity = cosine_similarity(input_tfidf, keyword_matrix)
    csims = cos_similarity[0]

    # Output
    film_data['Percentage Match'] = csims * 100
    film_data = film_data.sort_values(by=['Percentage Match'], ascending=False)
    film_data = film_data[(film_data['Percentage Match'] > 0.0)]
    film_data = film_data.set_index('Percentage Match')

    # Print the top 10 similar movies with additional context
    top_recommendations = film_data.head(10)
    print("\n\n############### Your Review #######################")
    print(userInput)
    print("############### YOU SHOULD WATCH ##################")
    if len(film_data) != 0:
        for idx, row in top_recommendations.iterrows():
            print(f"{row['Title']} | Genres: {row['Genres']} | Percentage Match: {idx:.2f}%")
    else:
        print("No movies found.")

    print("\n\n")

elif control == 1:
    # embeddings used
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    import pandas as pd
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from sklearn.metrics.pairwise import cosine_similarity

    # we only use the descriptions right now. using title so that we actually 
    # can write what movie we recomend
    usecols = ["Title", "Description"]

    # Read the CSV file
    df = pd.read_csv("test_data/16k_Movies.csv", usecols=usecols)

    # 2. Initialize embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Compute embeddings for reviews
    # using the flag to get the "loading"
    # stop_flag = show_loading("Computing embeddings")

    review_texts = df['Description'].tolist()
    titles = df['Title'].tolist()
    # compute embeddings here (really slow)
    # review_embeddings = embedder.embed_documents(review_texts)

    # using embeddings from npy file
    review_embeddings = np.load("test_data/movie_embeddings.npy")

    # 4. Embedd the user input
    user_embedding = embedder.embed_query(userInput)

    # 5. Compute similarity
    similarities = cosine_similarity([user_embedding], review_embeddings)[0]

    # stop the flag to since the program has loaded everything
    # stop_flag.set()

    # 6. Find best match
    best_idx = similarities.argmax()
    print(best_idx)
    print("Best match:", titles[best_idx])
    print("Similarity score:", similarities[best_idx])

