#!/usr/bin/env python3
"""LeNet-5 modifié"""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):
    """modèle LeNet-5"""
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Première couche convolutionnelle
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(x)

    # Première couche de pooling
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Deuxième couche convolutionnelle
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)(pool1)

    # Deuxième couche de pooling
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Aplatissement des données pour les couches entièrement connectées
    flat = tf.layers.Flatten()(pool2)

    # Première couche entièrement connectée
    fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                          kernel_initializer=initializer)(flat)

    # Deuxième couche entièrement connectée
    fc2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                          kernel_initializer=initializer)(fc1)

    # Couche de sortie softmax
    logits = tf.layers.Dense(units=10, kernel_initializer=initializer)(fc2)
    output = tf.nn.softmax(logits)

    # Calcul de la perte
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)

    # Optimiseur Adam
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Calcul de la précision
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return output, train_op, loss, accuracy
