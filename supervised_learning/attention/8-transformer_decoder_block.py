#!/usr/bin/env python3
"""8-transformer_decoder_block.py"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Classe decoderblock"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """initialisation"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """rappelle"""
        Q = x
        K = x
        V = x
        output1, weights = self.mha1(Q, K, V, look_ahead_mask)
        result = self.dropout1(output1, training)
        add1 = result + x
        layernorm1 = self.layernorm1(add1)
        Q = layernorm1
        K = encoder_output
        V = encoder_output
        output2, weights = self.mha2(Q, K, V, padding_mask)
        result2 = self.dropout2(output2, training)
        add2 = layernorm1 + result2
        layernorm2 = self.layernorm2(add2)
        hidd = self.dense_hidden(layernorm2)
        final_dense = self.dense_output(hidd)
        result3 = self.dropout3(final_dense, training)
        add3 = layernorm2 + result3
        layernorm3 = self.layernorm3(add3)

        return layernorm3
