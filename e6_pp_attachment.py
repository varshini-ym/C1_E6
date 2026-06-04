import nltk
from nltk import CFG
from nltk.parse import ChartParser

# Define the grammar
grammar = CFG.fromstring("""
S -> NP VP
VP -> V NP | V NP PP
NP -> Pronoun | Det N | Det N PP
PP -> P NP

Pronoun -> 'I'
V -> 'shot'
Det -> 'an' | 'my'
N -> 'elephant' | 'pyjamas'
P -> 'in'
""")

# Input sentence
sentence = "I shot an elephant in my pyjamas".split()

print("Sentence:", " ".join(sentence))

# Create parser
parser = ChartParser(grammar)

# Generate parse trees
trees = list(parser.parse(sentence))

print("Number of parse trees found:", len(trees))

# Display parse trees
for i, tree in enumerate(trees, 1):
    print(f"\nParse Tree {i}:")
    print(tree)

# Optional: Draw parse trees graphically
for tree in trees:
    tree.draw()