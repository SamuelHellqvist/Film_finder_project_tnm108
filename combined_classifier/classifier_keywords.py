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