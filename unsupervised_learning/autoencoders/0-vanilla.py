#!/usr/bin/env python3
"""0-vanilla.py"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
   autoencoder
    """
    # Encoder
    input_layer = keras.Input(shape=(input_dims,))
    x = input_layer
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent_layer = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=input_layer, outputs=latent_layer)

    # Decoder
    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output_layer = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_input, outputs=output_layer)

    # Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=auto_input, outputs=decoded)

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
