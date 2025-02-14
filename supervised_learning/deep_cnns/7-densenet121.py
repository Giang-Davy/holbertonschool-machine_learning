#!/usr/bin/env python3
"""fonction"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block

def inception_network():
	"""réseau inception"""
	input = K.layers.Input(shape=(224, 224, 3))
	initializer = K.initializers.VarianceScaling(scale=2.0)

	norm1 = K.layers.BatchNormalization()(input)
	activation1 = K.layers.ReLU()(norm1)

	conv1 = K.layers.Conv2D(
		filters=64,
		kernel_size=(7, 7),
		activation=None,
		padding="same",
		strides=2,
		kernel_initializer=initializer)(activation1)

	pool1 = K.layers.MaxPooling2D(
		pool_size=(3, 3),
		strides=(2, 2),
		padding="same")(conv1)
	norm2 = K.layers.BatchNormalization()(pool1)
	activation2 = K.layers.ReLU()(norm2)

	conv2 = K.layers.Conv2D(
		64, (1, 1), padding='same', activation=None)(activation2)
	norm3 = K.layers.BatchNormalization()(conv2)
	activation3 = K.layers.ReLU()(norm3)

	conv3 = K.layers.Conv2D(
		192, (3, 3), padding='same', activation=None)(activation3)
	norm4 = K.layers.BatchNormalization()(conv3)
	activation4 = K.layers.ReLU()(norm4)

	poolbis = K.layers.MaxPooling2D(
		(3, 3), strides=(2, 2), padding='same')(activation4)

	incept3a = inception_block(poolbis, [64, 96, 128, 16, 32, 32])
	incept3b = inception_block(incept3a, [128, 128, 192, 32, 96, 64])
	pool2 = K.layers.MaxPooling2D(
		pool_size=(3, 3),
		strides=(2, 2),
		padding="same")(incept3b)

	incept4a = inception_block(pool2, [192, 96, 208, 16, 48, 64])
	incept4b = inception_block(incept4a, [160, 112, 224, 24, 64, 64])
	incept4c = inception_block(incept4b, [128, 128, 256, 24, 64, 64])
	incept4d = inception_block(incept4c, [112, 144, 288, 32, 64, 64])
	pool3 = K.layers.MaxPooling2D(
		pool_size=(3, 3),
		strides=(2, 2),
		padding="same")(incept4d)

	incept5a = inception_block(pool3, [256, 160, 320, 32, 128, 128])
	incept5b = inception_block(incept5a, [384, 192, 384, 48, 128, 128])

	avg_pool = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(incept5b)
	dropout = K.layers.Dropout(0.4)(avg_pool)
	output_layer = K.layers.Dense(1000, activation='softmax')(dropout)

	model = K.models.Model(inputs=input, outputs=output_layer)
	return model
