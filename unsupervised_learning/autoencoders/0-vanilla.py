#!/usr/bin/env python3
"""0-vanilla.py"""


import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """autoencoder normal"""

    input_layer = K.Input(shape=(input_dims,))
    encoded = input_layer
    for nodes in hidden_layers:
        encoded = K.layers.Dense(nodes, activation='relu')(encoded)
    output_layer = K.layers.Dense(latent_dims, activation='relu')(encoded)
    decoded = output_layer
    for nodes in hidden_layers[::-1]:
        decoded = K.layers.Dense(nodes, activation='relu')(decoded)
    last_layer = K.layers.Dense(input_dims, activation='sigmoid')(decoded)

    auto = K.Model(inputs=input_layer, outputs=last_layer)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    # Modèle encodeur
    encoder = K.Model(inputs=input_layer, outputs=output_layer)

    # Décodeur (autonome, à partir du latent space)
    latent_inputs = K.Input(shape=(latent_dims,))
    decoder_layer = latent_inputs
    for nodes in hidden_layers[::-1]:
        decoder_layer = K.layers.Dense(nodes, activation='relu')(decoder_layer)
    decoder_output = K.layers.Dense(input_dims, activation='sigmoid')(decoder_layer)
    decoder = K.Model(inputs=latent_inputs, outputs=decoder_output)

    return encoder, decoder, auto
