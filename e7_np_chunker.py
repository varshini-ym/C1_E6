import nltk
from nltk import RegexpParser

# Function to build NP Chunker
def build_np_chunker():
    grammar = r"""
        NP: {<DT>?<JJ>*<CD>?<NN|NNS>+}
    """
    return RegexpParser(grammar)

# Function to extract NP phrases
def chunk_nps(tagged_sent):
    chunker = build_np_chunker()
    tree = chunker.parse(tagged_sent)

    np_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            phrase = " ".join(word for word, tag in subtree.leaves())
            np_phrases.append(phrase)

    return np_phrases

# Demonstration
tagged_sent = [
    ("the", "DT"),
    ("two", "CD"),
    ("old", "JJ"),
    ("wooden", "JJ"),
    ("bridges", "NNS")
]

nps = chunk_nps(tagged_sent)

print("Noun Phrases Found:")
for np in nps:
    print(np)