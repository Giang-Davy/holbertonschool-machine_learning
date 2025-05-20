#!/usr/bin/env python3
"""0-bag_of_words.py"""


import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """baguage de mots"""

    cleaned = []
    for sentence in sentences:
        sentence = sentence.lower()
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sentence)
        cleaned.append(words)

    if vocab is None:
        vocab_set = set()
        for sentence in cleaned:
            for word in sentence:
                vocab_set.add(word)
        features = sorted(vocab_set)
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sentence in enumerate(cleaned):
        for word in sentence:
            if word in features:
                index = features.index(word)
                embeddings[i, index] += 1

    embeddings = np.array(embeddings)
    features = np.array(features)

    return embeddings, features
