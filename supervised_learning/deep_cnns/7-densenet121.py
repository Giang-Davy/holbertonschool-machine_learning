#!/usr/bin/env python3
"""DenseNet-121"""


from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Construit le modèle DenseNet-121"""
    inputs = K.layers.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal(seed=0)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(inputs)
    activation1 = K.layers.Activation('relu')(batch_norm1)

    # Convolution initiale
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=2,
        padding="same",
        kernel_initializer=initializer)(activation1)

    # MaxPooling
    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding="same")(conv1)

    # Dense Block 1
    dense1, nb_filters = dense_block(pool1, 64, growth_rate, 6)

    # Transition Layer 1
    transition1, nb_filters = transition_layer(dense1, nb_filters, compression)

    # Dense Block 2
    dense2, nb_filters = dense_block(transition1, nb_filters, growth_rate, 12)

    # Transition Layer 2
    transition2, nb_filters = transition_layer(dense2, nb_filters, compression)

    # Dense Block 3
    dense3, nb_filters = dense_block(transition2, nb_filters, growth_rate, 24)

    # Transition Layer 3
    transition3, nb_filters = transition_layer(dense3, nb_filters, compression)

    # Dense Block 4
    dense4, nb_filters = dense_block(transition3, nb_filters, growth_rate, 16)

    # Global Average Pooling
    avg_pool = K.layers.GlobalAveragePooling2D()(dense4)

    # Fully Connected Layer
    outputs = K.layers.Dense(1000, activation="softmax")(avg_pool)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model
