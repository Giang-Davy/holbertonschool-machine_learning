#!/usr/bin/env python3
"""0-rnn_encoder.py"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """classe de RNN"""
    def __init__(self, vocab, embedding, units, batch):
        """initialisation"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding)
        self.units = units
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """initialisation des états cachés"""
        hidden = tf.zeros((self.batch, self.units))
        return hidden

    def call(self, x, initial):
        """rappelle"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
