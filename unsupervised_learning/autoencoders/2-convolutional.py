#!/usr/bin/env python3
"""2-convolutional.py"""

import tensorflow.keras as K

def autoencoder(input_dims, filters, latent_dims):
    """
    autoencoder
    """
    # Encoder
    X = K.Input(shape=input_dims)
    x = X
    for nodes in filters:
        x = K.layers.Conv2D(filters=nodes, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    encoder = K.Model(inputs=X, outputs=x)

    # Decoder
    latent_input = K.Input(shape=latent_dims)
    x = latent_input
    for idx, nodes in enumerate(reversed(filters)):
        x = K.layers.Conv2D(filters=nodes, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = K.layers.UpSampling2D(size=(2, 2))(x)
    x = K.layers.Conv2D(filters=input_dims[2], kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    decoder = K.Model(inputs=latent_input, outputs=x)

    # Autoencoder complet
    auto_input = K.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = K.Model(inputs=auto_input, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
