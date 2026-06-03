import nltk
from nltk.corpus import treebank
from nltk.tag import UnigramTagger

tagged_sents = treebank.tagged_sents()

train_data = tagged_sents[:3000]
test_data = tagged_sents[3000:]

tagger = UnigramTagger(train_data)

accuracy = tagger.accuracy(test_data)

print("Accuracy:", accuracy)