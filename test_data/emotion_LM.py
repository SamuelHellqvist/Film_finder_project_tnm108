from transformers import pipeline

# Load the model and request full score distribution
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

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
