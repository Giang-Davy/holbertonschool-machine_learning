#!/usr/bin/env python3
"""2-convolutional.py"""

import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """
    Builds a convolutional autoencoder.
    """
    # Partie Encodeur
    entree_encodeur = K.Input(shape=input_dims)
    x = entree_encodeur

    # Construction des couches convolutives
    for n_filtres in filters:
        x = K.layers.Conv2D(
            filters=n_filtres, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = K.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encodeur = K.models.Model(inputs=entree_encodeur, outputs=x)

    # Partie Décodeur
    entree_decodeur = K.Input(shape=latent_dims)
    x = entree_decodeur

    # Reconstruction inverse avec gestion des dimensions
    for n_filtres in reversed(filters[1:]):  # On ignore le premier filtre
        x = K.layers.Conv2D(
            filters=n_filtres, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = K.layers.UpSampling2D(size=(2, 2))(x)

    # Couche spéciale pour ajuster les dimensions
    x = K.layers.Conv2D(
        filters=filters[0], kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = K.layers.UpSampling2D(size=(2, 2))(x)

    # Dernière couche de reconstruction
    sortie_decodeur = K.layers.Conv2D(
        filters=input_dims[2], kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    decodeur = K.models.Model(inputs=entree_decodeur, outputs=sortie_decodeur)

    # Assemblage final corrigé
    sortie_autoencodeur = decodeur(encodeur(entree_encodeur))
    autoencodeur_complet = K.models.Model(
        inputs=entree_encodeur, outputs=sortie_autoencodeur
    )

    # Configuration de l'apprentissage
    autoencodeur_complet.compile(optimizer='adam', loss='binary_crossentropy')

    return encodeur, decodeur, autoencodeur_complet
