#!/usr/bin/env python3


import tensorflow.keras as K


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
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.backend.random_normal(shape=K.backend.shape(z_mean))
        return z_mean + K.backend.exp(0.5 * z_log_var) * epsilon
    
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
    auto = K.Model(inputs, decoder(encoder(inputs)[0]), name="autoencoder")
    
    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
