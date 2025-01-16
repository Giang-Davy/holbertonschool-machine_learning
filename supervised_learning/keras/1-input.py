#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Construire un modèle de réseau neuronal avec Keras"""

    # Créer l'entrée du modèle
    input_layer = K.Input(shape=(nx,))

    # Ajouter la première couche avec input_dim pour la couche d'entrée
    x = K.layers.Dense(
            units=layers[0],
            activation=activations[0],
            kernel_regularizer=K.regularizers.l2(lambtha))(input_layer)
    x = K.layers.Dropout(rate=1 - keep_prob)(x)

    # Ajouter les autres couches avec boucle
    for i in range(1, len(layers)):
        x = K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha))(x)
        x = K.layers.Dropout(rate=1 - keep_prob)(x)
    # Créer le modèle
    model = K.models.Model(inputs=input_layer, outputs=x)

    # Compiler le modèle
    model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    return model
