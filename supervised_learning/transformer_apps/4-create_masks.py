#!/usr/bin/env python3
"""4-create_masks.py"""


import tensorflow as tf


def create_masks(inputs, target):
    """création de masque"""
    #  -----Encoder Mask------
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = tf.expand_dims(encoder_mask, axis=1)
    encoder_mask = tf.expand_dims(encoder_mask, axis=2)
    # (batch_size, 1, 1, seq_len_in)

    #  ------Combined Mask-------
    seq_len = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len, seq_len)), -1, 0
    )  # (seq_len_out, seq_len_out)

    tgt_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    tgt_padding_mask = tf.expand_dims(tgt_padding_mask, axis=1)
    tgt_padding_mask = tf.expand_dims(tgt_padding_mask, axis=2)
    # (batch_size, 1, 1, seq_len_out)

    # broadcast to (batch_size, 1, seq_len_out, seq_len_out)
    combined_mask = tf.maximum(
        look_ahead_mask,
        tf.squeeze(tgt_padding_mask, axis=2)
    )
    combined_mask = tf.expand_dims(combined_mask, axis=1)

    #  -----Decoder Mask------
    # mask sur l'entrée (inputs), pas sur la target
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = tf.expand_dims(decoder_mask, axis=1)
    decoder_mask = tf.expand_dims(decoder_mask, axis=2)
    # (batch_size, 1, 1, seq_len_in)

    return encoder_mask, combined_mask, decoder_mask
