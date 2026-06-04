import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Sample sentences
sent1 = "dog bites man"
sent2 = "man bites dog"

# Bag-of-words representation
bow1 = set(word_tokenize(sent1))
bow2 = set(word_tokenize(sent2))

print("Sentence 1 BoW:", bow1)
print("Sentence 2 BoW:", bow2)

print("\nAre they same in Bag-of-Words?", bow1 == bow2)