#!/usr/bin/env python3
"""0-vanilla.py"""


import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """
    Builds a convolutional autoencoder.
    """
    # Encoder
    X = K.Input(shape=input_dims)
    x = X
    for nodes in filters:
        x = K.layers.Conv2D(
            filters=nodes, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = K.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    encoder = K.models.Model(inputs=X, outputs=x)

    # Decoder
    latent_input = K.Input(shape=latent_dims)
    x = latent_input
    for nodes in reversed(filters[1:]):
        x = K.layers.Conv2D(
            filters=nodes, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = K.layers.UpSampling2D(size=(2, 2))(x)
    x = K.layers.Conv2D(
        filters[0], kernel_size=(3, 3), activation='relu', padding='valid')(x)

    decoder = K.Model(inputs=latent_input, outputs=x)

    # Autoencoder complete
    auto_input = K.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = K.Model(inputs=auto_input, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
