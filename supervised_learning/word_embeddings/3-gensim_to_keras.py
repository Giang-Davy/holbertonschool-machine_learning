#!/usr/bin/env python3
"""3-gensim_to_keras.py"""


import tensorflow as tf


def gensim_to_keras(model):
    """d'un modèle gensim vers un modele keras"""

    vec = model.wv
    vocab_size, vector_size = vec.vectors.shape
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                          output_dim=vector_size,
                                          weights=[vec.vectors],
                                          trainable=True)
    return embedding
