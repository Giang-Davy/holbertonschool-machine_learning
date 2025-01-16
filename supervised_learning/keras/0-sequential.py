#!/usr/bin/env python3
"""fonction"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Construire un modèle de réseau neuronal avec Keras"""

    # Créer le modèle séquentiel
    model = K.models.Sequential()

    # Ajouter la première couche avec input_dim pour la couche d'entrée
    model.add(K.layers.Dense(units=layers[0], activation=activations[0], input_dim=nx))

    # Ajouter les autres couches avec boucle
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(units=layers[i], activation=activations[i]))

        # Ajouter Dropout après chaque couche cachée
        model.add(K.layers.Dropout(rate=1 - keep_prob))

    # Ajouter la régularisation L2 à chaque couche cachée
    for layer in model.layers:
        if isinstance(layer, K.layers.Dense):
            layer.kernel_regularizer = K.regularizers.l2(lambtha)

    # Compiler le modèle
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
