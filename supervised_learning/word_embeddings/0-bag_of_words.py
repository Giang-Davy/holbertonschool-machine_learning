#!/usr/bin/env python3
"""0-bag_of_words.py"""


import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    - sentences (list of str): Sentences to analyze
    - vocab (list of str or None): Vocabulary to use for analysis.
                                    If None, it's built from sentences.

    Returns:
    - embeddings (np.ndarray): Matrix of shape (s, f) with word counts
    - features (list of str): The vocabulary used
    """
    tokenized_sentences = []
    for sentence in sentences:
        # Simple word tokenization (lowercased words)
        words = re.findall(r'\b\w+\b', sentence.lower())
        tokenized_sentences.append(words)

    if vocab is None:
        # Build vocab from all unique words
        vocab_set = set()
        for words in tokenized_sentences:
            vocab_set.update(words)
        vocab = sorted(vocab_set)

    # Mapping word to index
    word_index = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, words in enumerate(tokenized_sentences):
        for word in words:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, vocab

