#!/usr/bin/env python3
"""7-transformer_encoder_block.py"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """block encoder"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """initialisation"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """rappelle"""
        Q = x
        K = x
        V = x
        output, weights = self.mha(Q, K, V, mask)
        result = self.dropout1(output, training)
        add1 = result + x
        layernorm = self.layernorm1(add1)
        hidd1 = self.dense_hidden(layernorm)
        dense1 = self.dense_output(hidd1)
        result2 = self.dropout2(dense1, training)
        add2 = result2 + layernorm
        layernorm2 = self.layernorm2(add2)

        return layernorm2
