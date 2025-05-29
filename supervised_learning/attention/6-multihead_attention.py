#!/usr/bin/env python3
"""6-multihead_attention.py"""


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """classe"""
    def __init__(self, dm, h):
        """initialisation"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.depth = dm // h
        self.dm = dm
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """rappelle"""
        batch_size = tf.shape(Q)[0]
        seq_len = tf.shape(Q)[1]

        # Projeter Q, K, V
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # Reshape et transpose pour multi-head
        def split_heads(x):
            x = tf.reshape(x, (batch_size, seq_len, self.h, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Attention
        output, weights = sdp_attention(q, k, v, mask)

        # Concaténer les têtes
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, seq_len, self.dm))
        output = self.linear(output)

        return output, weights
