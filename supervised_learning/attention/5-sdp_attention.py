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
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        #  obtenir le tuple nécéssaire pour le reshape
        batchq = tf.shape(Q)[0]
        batch2q = tf.shape(Q)[1]
        reshapeq = tf.reshape(q, [batchq, batch2q, self.h ,self.depth])
        #  mettre self.h comme deuxième dimension.
        q = tf.transpose(reshapeq, perm=[0, 2, 1, 3])
        #  refaire pareil pour k et v
        batchk = tf.shape(K)[0]
        batch2k = tf.shape(K)[1]
        reshapek = tf.reshape(k, [batchk, batch2k, self.h ,self.depth])
        k = tf.transpose(reshapek, perm=[0, 2, 1, 3])
        #  v
        batchv = tf.shape(V)[0]
        batch2v = tf.shape(V)[1]
        reshapev = tf.reshape(v, [batchv, batch2v, self.h ,self.depth])
        v = tf.transpose(reshapev, perm=[0, 2, 1, 3])
        #  appelle de la fonction 5-sdp attention
        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, [batchq, batch2q, self.dm])
        output = self.linear(output)

        return output, weights
