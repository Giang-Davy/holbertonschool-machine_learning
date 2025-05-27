#!/usr/bin/env python3
"""2-rnn_decoder.py"""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder"""
    def __init__(self, vocab, embedding, units, batch):
        """initialisation"""
        super(RNNDecoder, self).__init__()
        self.attention = SelfAttention(units)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """rappelle"""
        context, _ = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        # Concaténation sur le dernier axe (features)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, s = self.gru(x)
        output = tf.squeeze(output, axis=1)
        y = self.F(output)
        return y, s
