#!/usr/bin/env python3
"""LeNet-5 modifié"""


import tensorflow.compat.v1 as tf


tf.set_random_seed(42)  # Fixe le seed pour TensorFlow

def lenet5(x, y):
    """modèle LeNet-5"""
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Première couche convolutionnelle
    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=(5, 5),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)

    # Première couche de pooling
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=2)

    # Deuxième couche convolutionnelle
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=(5, 5),
                             padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)

    # Deuxième couche de pooling
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2)

    # Aplatissement des données pour les couches entièrement connectées
    flat = tf.layers.flatten(pool2)

    # Première couche entièrement connectée
    fc1 = tf.layers.dense(flat, units=120, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # Deuxième couche entièrement connectée
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # Couche de sortie softmax
    output = tf.layers.dense(fc2, units=10, activation=tf.nn.softmax,
                             kernel_initializer=initializer)

    # Calcul de la perte
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                logits=output))
    # Calcul de la précision
    correct_prediction = tf.equal(tf.argmax(output, 1),
                                  tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32))

    # Optimiseur Adam
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return output, train_op, loss, accuracy
