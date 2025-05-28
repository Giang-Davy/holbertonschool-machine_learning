#!/usr/bin/env python3
"""5-sdp_attention.py"""


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """sdp attention calcule"""
    #  obtenir dk sous float
    dk = tf.cast(tf.shape(Q)[-1], tf.float32)
    #  transposition de K
    K_transposed = tf.transpose(K, perm=[0, 2, 1])
    score = tf.matmul(Q, K_transposed) / tf.math.sqrt(dk)
    if mask is not None:
        score += (mask * -1e9)
    weight = tf.nn.softmax(score)
    output = tf.matmul(weight, V)

    return output, weight
