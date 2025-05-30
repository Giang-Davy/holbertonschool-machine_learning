#!/usr/bin/env python3
"""9-transformer_encoder.py"""


import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Classe decoderblock"""
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """initialisation"""
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(
            max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for i in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """rappelle"""
        lenght = tf.shape(x)[1]
        embed = self.embedding(x)
        embed = embed * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        encoding = self.positional_encoding[:lenght, :]
        encoding = tf.cast(encoding, tf.float32)
        encoding = tf.expand_dims(encoding, axis=0)
        somme = embed + encoding
        embed = self.dropout(somme, training)
        for bloc in self.blocks:
            embed = bloc(embed, training, mask)
        return embed
