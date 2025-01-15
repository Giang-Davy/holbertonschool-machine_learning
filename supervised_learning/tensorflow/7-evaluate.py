#!/usr/bin/env python3
"""
Module pour évaluer la sortie d'un réseau neuronal
"""
import tensorflow.compat.v1 as tf


def evaluate(input_data, one_hot_labels, model_path):
    """
    Évalue la sortie d'un réseau neuronal

    Args:
        input_data: numpy.ndarray contenant les données d'entrée à évaluer
        one_hot_labels: numpy.ndarray contenant les étiquettes one-hot
        model_path: emplacement du modèle à charger

    Returns:
        Prédiction du réseau, précision et perte respectivement
    """
    # Création d'une nouvelle session
    with tf.Session() as sess:
        # Importation du métagraphe sauvegardé
        saver = tf.train.import_meta_graph(model_path + '.meta')
        # Restauration des variables du modèle
        saver.restore(sess, model_path)

        # Récupération des tenseurs nécessaires
        input_tensor = tf.get_collection('x')[0]
        label_tensor = tf.get_collection('y')[0]
        prediction_tensor = tf.get_collection('y_pred')[0]
        accuracy_tensor = tf.get_collection('accuracy')[0]
        loss_tensor = tf.get_collection('loss')[0]

        # Évaluation du modèle
        feed_dict = {input_tensor: input_data, label_tensor: one_hot_labels}
        predictions, accuracy, loss = sess.run(
            [prediction_tensor, accuracy_tensor, loss_tensor],
            feed_dict=feed_dict
        )

        return predictions, accuracy, loss
