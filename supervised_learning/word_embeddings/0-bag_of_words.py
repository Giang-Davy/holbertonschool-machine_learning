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
        features = sorted(vocab_set)  # Trie pour garantir l'ordre
    else:
        features = list(vocab)  # S'assurer que c'est une liste, pas un np.array

    # Build word to index mapping for fast lookup
    word2idx = {word: idx for idx, word in enumerate(features)}

    embeddings = []
    f = len(features)
    for sentence in cleaned:
        liste = np.zeros(f, dtype=int)
        for word in sentence:
            if word in word2idx:
                liste[word2idx[word]] += 1
        embeddings.append(liste)
    embeddings = np.array(embeddings)
    features = np.array(features)

    return embeddings, features
