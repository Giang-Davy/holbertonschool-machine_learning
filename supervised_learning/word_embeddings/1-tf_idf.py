#!/usr/bin/env python3
"""1-tf_idf_sklearn.py"""


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """idf"""
    if vocab is not None:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
    else:
        vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(sentences)
    embeddings = tfidf_matrix.toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
