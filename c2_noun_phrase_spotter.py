import nltk

def noun_phrase_spotter(sentence):
    words = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(words)

    noun_phrases = []
    current_np = []

    for word, tag in tagged:

        if tag.startswith('JJ'):
            current_np.append(word)

        elif tag.startswith('NN'):
            current_np.append(word)

        else:
            if current_np:
                noun_phrases.append(" ".join(current_np))
                current_np = []

    if current_np:
        noun_phrases.append(" ".join(current_np))

    return noun_phrases


sentence = "The beautiful garden has red flowers and a large wooden table."

result = noun_phrase_spotter(sentence)

print("Noun Phrases:")
print(result)