#!/usr/bin/env python3
"""2-convolutional.py"""

import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """autoencoder"""
    input_encoder = K.Input(shape=input_dims)
    x = input_encoder

    for num_filters in filters:
        x = K.layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = K.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder = K.models.Model(inputs=input_encoder, outputs=x)

    input_decoder = K.Input(shape=latent_dims)
    x = input_decoder

    for num_filters in reversed(filters[1:]):
        x = K.layers.Conv2D(
            filters=num_filters, kernel_size=(3, 3),
            padding='same', activation='relu')(x)
        x = K.layers.UpSampling2D(size=(2, 2))(x)

    x = K.layers.Conv2D(
        filters=filters[0], kernel_size=(3, 3),
        padding='valid', activation='relu')(x)
    x = K.layers.UpSampling2D(size=(2, 2))(x)

    output_decoder = K.layers.Conv2D(
        filters=input_dims[2], kernel_size=(3, 3),
        activation='sigmoid', padding='same')(x)

    decoder = K.models.Model(inputs=input_decoder, outputs=output_decoder)

    output_autoencoder = decoder(encoder(input_encoder))
    autoencoder_model = K.models.Model(
        inputs=input_encoder, outputs=output_autoencoder
    )

    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder_model
