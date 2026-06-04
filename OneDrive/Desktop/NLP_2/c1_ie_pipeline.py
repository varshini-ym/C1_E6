import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')


def ie_pipeline(text):

    relations = []

    # Step 1: Sentence Tokenization
    sentences = sent_tokenize(text)

    for sentence in sentences:

        print("\nSentence:", sentence)

        # Step 2: Word Tokenization
        words = word_tokenize(sentence)

        # Step 3: POS Tagging
        tagged_words = pos_tag(words)

        # Step 4: Named Entity Recognition
        ne_tree = ne_chunk(tagged_words)

        print("NE Tree:")
        print(ne_tree)

        entities = []

        # Step 5: Extract Named Entities
        for subtree in ne_tree:

            if hasattr(subtree, 'label'):

                entity_name = " ".join(
                    word for word, tag in subtree.leaves()
                )

                entity_type = subtree.label()

                entities.append((entity_type, entity_name))

        print("Entities Found:", entities)

        # Step 6: Find ORG and GPE entities
        orgs = []
        gpes = []

        for entity_type, entity_name in entities:

            if entity_type == "ORGANIZATION":
                orgs.append(entity_name)

            elif entity_type == "GPE":
                gpes.append(entity_name)

        # Step 7: Create relations
        for org in orgs:
            for gpe in gpes:
                relations.append((org, gpe))

    return relations


# Sample Text
text = """
Google opened a new office in India.
Microsoft expanded its operations in Hyderabad.
Infosys is headquartered in Bangalore.
"""

print("\nExtracted Relations:")
print(ie_pipeline(text))