#!/usr/bin/env python3

import re
import nltk
from nltk import word_tokenize, pos_tag

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Returns all nouns found in a sentence
def find_nouns(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Get the POS tags for each token
    pos_tags = pos_tag(tokens)

    # Filter and return the nouns (NN: singular, NNS: plural, NNP: proper singular, NNPS: proper plural)
    nouns = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return nouns


# Example usage
sentence = "turn on the living room lights in 10 hours"
nouns = find_nouns(sentence)
print(nouns)

# TODO : Implement regex entity matching