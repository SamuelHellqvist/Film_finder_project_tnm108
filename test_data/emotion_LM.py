from transformers import pipeline
import pandas as pd

# Load the model and request full score distribution
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# read the descriptions from the data file
usecols = ["Title", "Description"]

# Read the CSV file
df = pd.read_csv("test_data/16k_Movies.csv", usecols=usecols)

desc_texts = df['Description'].tolist()

print(desc_texts[1])


text = (
    "Ed and Lorraine Warren, world-renowned investigators of supernatural events, are called in to help a family terrorized by dark forces. In the family’s house, deep in the countryside, the Warrens are forced to confront a powerful demonic presence."
)

# Run classifier
scores = classifier(text)[0]

# Sort emotions by score (descending)
sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

# Get top two
top_two = sorted_scores[:2]

# Print them
print("Top two emotions:")
for item in top_two:
    print(f"{item['label']}: {item['score']:.4f}")
