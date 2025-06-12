#!/usr/bin/env python3
"""2-qa.py"""


import os
import numpy as np
from sentence_transformers import SentenceTransformer


def semantic_search(corpus_path, sentence):
    """recherche sementique"""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []

    for filename in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, filename),
                  "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            documents.append(text)
    embeddings = model.encode(documents)
    sentence_emb = model.encode(sentence)
    dot = np.dot(embeddings, sentence_emb)
    similaire = np.argmax(dot)

    return documents[similaire]
