#!/usr/bin/env python3

import tensorflow.keras as K


def sampling(args):
    """
    Reparameterization trick to sample z ~ N(mu, sigma^2).
    """
    z_mean, z_log_var = args
    epsilon = K.backend.random_normal(shape=K.backend.shape(z_mean))
    return z_mean + K.backend.exp(0.5 * z_log_var) * epsilon


def vae_loss(inputs, outputs, z_mean, z_log_var, input_dims):
    """
    Custom loss function for Variational Autoencoder.
    Combines reconstruction loss and KL divergence.
    """
    reconstruction_loss = K.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims  # Scale by input dimensions
    kl_loss = 1 + z_log_var - K.backend.square(z_mean) - K.backend.exp(z_log_var)
    kl_loss = -0.5 * K.backend.sum(kl_loss, axis=-1)
    return K.backend.mean(reconstruction_loss + kl_loss)


def autoencoder(input_dims, hidden_layers, latent_dims):
    # Encoder
    inputs = K.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = K.layers.Dense(nodes, activation='relu')(x)

    # Latent space representation (mean and log variance)
    z_mean = K.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = K.layers.Dense(latent_dims, activation=None)(x)

    # Sampling layer (reparameterization trick)
    z = K.layers.Lambda(sampling)([z_mean, z_log_var])

    # Encoder model outputs mean, log variance, and sampled z
    encoder = K.Model(inputs, [z, z_mean, z_log_var], name="encoder")

    # Decoder (reversed hidden layers)
    latent_inputs = K.Input(shape=(latent_dims,))
    x = latent_inputs
    reversed_layers = list(reversed(hidden_layers))
    for nodes in reversed_layers:
        x = K.layers.Dense(nodes, activation='relu')(x)

    # Output layer of the decoder
    decoded = K.layers.Dense(input_dims, activation='sigmoid')(x)

    # Decoder model
    decoder = K.Model(latent_inputs, decoded, name="decoder")

    # Autoencoder model (input to output through encoder and decoder)
    outputs = decoder(encoder(inputs)[0])
    auto = K.Model(inputs, outputs, name="autoencoder")

    # Add custom loss
    auto.add_loss(vae_loss(inputs, outputs, z_mean, z_log_var, input_dims))
    auto.compile(optimizer=K.optimizers.Adam())

    return encoder, decoder, auto
