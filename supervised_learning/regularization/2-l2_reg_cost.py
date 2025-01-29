&#!/usr/bin/env python3


import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculate the cost of a neural network with L2 regularization

    Arguments:
    cost -- tensor containing the cost of the network without L2 regularization
    model -- Keras model that includes layers with L2 regularization

    Returns:
    total_cost -- tensor containing the total cost for each layer of the network, accounting for L2 regularization
    """
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_weights])
    l2_cost = cost + l2_loss
    return l2_cost
