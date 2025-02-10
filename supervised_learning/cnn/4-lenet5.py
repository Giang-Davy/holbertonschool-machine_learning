#!/usr/bin/env python3
"""LeNet-5 modifié"""


import tensorflow.compat.v1 as tf


def lenet5(x, y):
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # First Convolutional Layer
    conv1 = tf.layers.conv2d(x, 6, (5, 5), padding='same', activation='relu', kernel_initializer=initializer)
    
    # First Max Pooling Layer
    pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='valid')
    
    # Second Convolutional Layer
    conv2 = tf.layers.conv2d(pool1, 16, (5, 5), padding='valid', activation='relu', kernel_initializer=initializer)
    
    # Second Max Pooling Layer
    pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='valid')
    
    # Flatten the output from the second pooling layer
    flat = tf.layers.flatten(pool2)
    
    # First Fully Connected Layer
    fc1 = tf.layers.dense(flat, 120, activation='relu', kernel_initializer=initializer)
    
    # Second Fully Connected Layer
    fc2 = tf.layers.dense(fc1, 84, activation='relu', kernel_initializer=initializer)
    
    # Output Layer
    output = tf.layers.dense(fc2, 10, activation='softmax', kernel_initializer=initializer)
    
    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    
    # Optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return output, optimizer, loss, accuracy
