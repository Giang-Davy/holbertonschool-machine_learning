#!/usr/bin/env python3


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def evaluate(X, Y, save_path):
    # Charger le modèle sauvegardé
    saver = tf.train.import_meta_graph(save_path + '.meta')
    
    with tf.Session() as sess:
        # Restaurer les poids du modèle
        saver.restore(sess, save_path)
        
        # Récupérer les tenseurs de prédiction, de perte et d'exactitude depuis le graphe
        graph = tf.get_default_graph()
        prediction_op = graph.get_tensor_by_name("prediction:0")  # Nom à adapter selon votre modèle
        loss_op = graph.get_tensor_by_name("loss:0")  # Nom à adapter selon votre modèle
        accuracy_op = graph.get_tensor_by_name("accuracy:0")  # Nom à adapter selon votre modèle
        
        # Calculer la prédiction, la perte et l'exactitude
        predictions, loss, accuracy = sess.run([prediction_op, loss_op, accuracy_op], feed_dict={X: X, Y: Y})
    
    return predictions, accuracy, loss
