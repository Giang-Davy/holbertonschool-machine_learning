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
        embedded = self.embedding(x)
        # S'assurer que embedded a la forme (batch, 1, embedding_dim)
        if len(embedded.shape) == 2:
            embedded = tf.expand_dims(embedded, axis=1)
        # context a déjà la forme (batch, units), on l'expand
        context = tf.expand_dims(context, axis=1)
        # Concaténation sur le dernier axe (features)
        concat_result = tf.concat([embedded, context], axis=-1)
        output, s = self.gru(concat_result, initial_state=s_prev)
        y = self.F(output)
        y = tf.squeeze(y, axis=1)
        return y, s
