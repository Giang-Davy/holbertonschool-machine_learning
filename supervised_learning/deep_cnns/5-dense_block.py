#!/usr/bin/env python3
"""fonction"""


from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Construire un Dense Block avec un nombre spécifié de couches.

    X : sortie de la couche précédente.
    nb_filters : nombre de filtres dans X.
    growth_rate : taux de croissance pour le Dense Block.
    layers : nombre de couches dans le Dense Block.

    Retourne : La sortie concaténée de chaque couche au sein de filtres.
    """

    for i in range(layers):
        # Normalisation + Activation avant la première convolution (1x1)
        norm1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.ReLU()(norm1)

        # Bottleneck: Convolution 1x1
        bottleneck = K.layers.Conv2D(
            filters=4 * growth_rate,  # Le nombre de fit 4 fois le growth_rate
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=K.initializers.he_normal(seed=0)
        )(act1)

        # Normalisation + Activation avant la deuxième convolution (3x3)
        norm2 = K.layers.BatchNormalization()(bottleneck)
        act2 = K.layers.ReLU()(norm2)

        # Convolution 3x3
        conv3x3 = K.layers.Conv2D(
            filters=growth_rate,  # Le taux de croissance est ajoutée la couche
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=K.initializers.he_normal(seed=0)
        )(act2)

        # Concaténation de l'entrée X avec la sortie de la couche
        X = K.layers.Concatenate()([X, conv3x3])

    return X, nb_filters + layers * growth_rate
