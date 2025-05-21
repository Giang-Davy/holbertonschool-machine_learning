#!/usr/bin/env python3
"""2-word2vec.py"""


import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5,
                   cbow=True, epochs=5, seed=0, workers=1):
    """Train a Word2Vec model."""
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,  # mettre ici aussi le sentences
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )
    return model
