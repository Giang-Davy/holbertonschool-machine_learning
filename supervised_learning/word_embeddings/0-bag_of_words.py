#!/usr/bin/env python3
"""0-bag_of_words.py"""


import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """baguage de mots"""

    cleaned = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
        words = sentence.split()
        cleaned.append(words)

    if vocab is None:
        vocab_set = set()
        for sentence in cleaned:
            for word in sentence:
                vocab_set.add(word)
        features = sorted(vocab_set)

    embeddings = []
    s = len(sentences)
    f = len(features)
    for sentence in cleaned:
        liste = np.zeros(f, dtype=int)
        embeddings.append(liste)
        for word in sentence:
            if word in features:
                index = features.index(word)
                liste[index] += 1
    embeddings = np.array(embeddings)

    return embeddings, features
