#!/usr/bin/env python3
"""10-transformer_decoder.py"""


import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder"""
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """initialisation"""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(
            max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for i in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """rappelle"""
        length = tf.shape(x)[1]
        embed = self.embedding(x)
        embed = embed * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        encoding = self.positional_encoding[:length, :]
        encoding = tf.cast(encoding, tf.float32)
        encoding = tf.expand_dims(encoding, axis=0)
        somme = embed + encoding
        embed = self.dropout(somme, training)
        for bloc in self.blocks:
            embed = bloc(embed, encoder_output, training,
                         look_ahead_mask, padding_mask)
        return embed
