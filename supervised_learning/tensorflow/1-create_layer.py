#!/usr/bin/env python3
"""fonction"""


import tensorflow as tf


def create_layer(prev, n, activation):
	"""
	Retourne :
	Le tensor de sortie de la couche créée.
	"""
	initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
	layer = tf.keras.layers.Dense(units=n, activation=activation, kernel_initializer=initializer, name='layer')
	return layer(prev)
