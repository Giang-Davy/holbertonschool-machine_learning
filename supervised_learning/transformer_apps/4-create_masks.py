#!/usr/bin/env python3
"""4-create_masks.py"""


import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


def create_masks(inputs, target):
    """création de masque"""
    #  -----Encoder Mask------
    #  comaparer l'entrée à zéro
    commparaison = tf.math.equal(inputs, 0)
    float_comparaison = tf.cast(commparaison, tf.float32)
    # (batch_size, seq_len_in) => (batch_size, 1, 1, seq_len_in)
    encoder_mask = tf.expand_dims(float_comparaison, axis=1)
    encoder_mask = tf.expand_dims(encoder_mask, axis=2)

    #  -----Decoder Mask------
    commparaison = tf.math.equal(target, 0)
    float_comparaison2 = tf.cast(commparaison, tf.float32)
    # (batch_size, seq_len_in) => (batch_size, 1, 1, seq_len_in)
    decoder_mask = tf.expand_dims(float_comparaison2, axis=1)
    decoder_mask = tf.expand_dims(decoder_mask, axis=2)

    #  ------Combined Mask-------
    seq_len = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    tgt_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    tgt_padding_mask = tf.expand_dims(tgt_padding_mask, axis=1)
    tgt_padding_mask = tf.expand_dims(tgt_padding_mask, axis=2)

    combined_mask = tf.maximum(look_ahead_mask, tgt_padding_mask)

    return encoder_mask, decoder_mask, combined_mask
