#!/usr/bin/env python3
"""2-conv"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input.
        filters (list): Number of filters for each convolutional layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        encoder (Model): The encoder model.
        decoder (Model): The decoder model.
        auto (Model): The full autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    latent = keras.layers.Conv2D(latent_dims[0], (3, 3), activation='relu', padding='same')(x)

    encoder = keras.Model(inputs, latent, name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    # Second to last convolution with valid padding
    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu', padding='valid')(x)
    # Last convolution
    outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)

    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # Autoencoder
    auto_inputs = keras.Input(shape=input_dims)
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)
    auto = keras.Model(auto_inputs, decoded, name="autoencoder")

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
