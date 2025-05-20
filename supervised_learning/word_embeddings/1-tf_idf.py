#!/usr/bin/env python3
"""1-tf_idf.py"""

import numpy as np
import re


def tf_idf(sentences, vocab=None):
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
    embeddings = np.zeros((s, f), dtype=float)
    for i, sentence in enumerate(cleaned):
        for word in sentence:
            tf = sentence.count(word) / len(sentence)
            if word in features:
                index = features.index(word)
                embeddings[i, index] = tf

    idfs = np.zeros(f, dtype=float)
    for j, word in enumerate(features):
        d_w = sum(1 for sentence in cleaned if word in sentence)
        idfs[j] = np.log((1 + s) / (1 + d_w)) + 1

    embeddings *= idfs

    return embeddings, features
