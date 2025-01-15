#!/usr/bin/env python3


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def evaluate(X, Y, save_path):
    # Charger le modèle depuis le chemin donné
    model = tf.keras.models.load_model(save_path)

    # Créer un session pour évaluer le modèle
    with tf.Session() as sess:
        # Restaurer le modèle dans la session
        model._make_predict_function()
        
        # Évaluer la précision et la perte
        loss, accuracy = model.evaluate(X, Y)

        # Prédire les sorties
        predictions = model.predict(X)

    return predictions, accuracy, loss
