#!/usr/bin/env python3
"""1-self_attention.py"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """classe de attention"""
    def __init__(self, units):
        """initialisation"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """rappelle"""
        x = self.W(s_prev)
        W_s = tf.expand_dims(x, axis=1)
        U = self.U(hidden_states)
        tenseur = W_s + U
        tanh = tf.tanh(tenseur)
        tanh = self.V(tanh)
        weight = tf.nn.softmax(tanh, axis=1)
        hidden_states = hidden_states * weight
        context = tf.reduce_sum(hidden_states, axis=1)

        return context, weight
