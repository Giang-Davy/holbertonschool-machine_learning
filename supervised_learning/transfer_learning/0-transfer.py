#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras as K
import numpy as np


def preprocess_data(X, Y):
    """bonne chance"""
    X = X / 255.0
    Y = K.utils.to_categorical(Y, 10)

    return X, Y

def main():
    # Définir l'entrée du modèle
    input_layer = K.layers.Input(shape=(32, 32, 3))  # Entrée avec les dimensions CIFAR-10
    resized = K.layers.Lambda(lambda x: tf.image.resize(x, (224, 224)))(input_layer)

    # Charger le modèle pré-entraîné (ResNet50)
    base_model = K.applications.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # On gèle les poids de ResNet50

    # Appliquer le modèle pré-entraîné aux données redimensionnées
    x = base_model(resized)

    # Ajouter des couches supplémentaires pour la classification
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(224, activation='relu')(x)
    output = K.layers.Dense(10, activation='softmax')(x)

    # Créer le modèle complet
    model = K.models.Model(inputs=input_layer, outputs=output)

    # Compiler le modèle
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    # Charger et prétraiter les données CIFAR-10
    (X_train, Y_train), (X_valid, Y_valid) = tf.keras.datasets.cifar10.load_data()

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_valid_p, Y_valid_p = preprocess_data(X_valid, Y_valid)

    # Entraînement du modèle
    model.fit(X_train_p, Y_train_p, validation_data=(X_valid_p, Y_valid_p), epochs=15)

    # Sauvegarder le modèle
    model.save("cifar10.h5V2")

    # Évaluation du modèle
    score = model.evaluate(X_valid_p, Y_valid_p)
    print("Loss:", score[0])
    print("Accuracy:", score[1])

if __name__ == "__main__":
    main()
