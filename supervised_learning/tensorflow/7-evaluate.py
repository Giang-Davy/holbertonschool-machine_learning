#!/usr/bin/env python3
"""fonction"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def evaluate(X, Y, save_path):
    """
    Fonction qui évalue la sortie d'un réseau de neurones.
    
    Args:
        X (ndarray): Données d'entrée à évaluer.
        Y (ndarray): Étiquettes one-hot correspondant à X.
        save_path (str): Le chemin pour charger le modèle.

    Returns:
        predictions: Prédictions du réseau de neurones.
        accuracy_value: Précision du modèle.
        loss_value: Perte du modèle.
    """
    # Charger le modèle à partir du chemin
    x, y = create_placeholders(X.shape[1], Y.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    
    # Créer un Saver
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        # Restaurer le modèle depuis le chemin donné
        saver.restore(sess, save_path)

        # Obtenir les prédictions
        predictions = sess.run(y_pred, feed_dict={x: X})

        # Calculer la perte et l'exactitude
        loss_value = sess.run(loss, feed_dict={x: X, y: Y})
        accuracy_value = sess.run(accuracy, feed_dict={x: X, y: Y})

    return predictions, accuracy_value, loss_value
