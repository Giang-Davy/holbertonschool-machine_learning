#!/usr/bin/env python3
"""fonction"""


from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Ajoute une couche de transition avec compressiDense Block.

    X : sortie du Dense Block.
    nb_filters : nombre initial de filtres.
    compression : facteur de réduction (entre 0 et 1).

    Retourne : la sortie de la couche de transition et le nouveau noe filtres.
    """
    # Calcul du nombre de filtres après compression
    nb_filters_reduced = int(nb_filters * compression)

    # Normalisation et activation
    norm = K.layers.BatchNormalization()(X)
    act = K.layers.ReLU()(norm)

    # Convolution 1x1 pour réduire les filtres
    conv = K.layers.Conv2D(
        filters=nb_filters_reduced,  # Réduction
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(act)

    # Average Pooling pour réduire la taille spatiale
    pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                     strides=(2, 2), padding='same')(conv)

    return pool, nb_filters_reduced
