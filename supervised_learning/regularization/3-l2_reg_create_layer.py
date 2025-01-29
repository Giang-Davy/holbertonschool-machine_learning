#!/usr/bin/env python3
"""fonction"""


import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Adds L2 regularization to the cost function of the neural network.
    
    cost: Tensor
        The cost of the network without L2 regularization.
    model: tf.keras.Model
        The model that includes layers with L2 regularization.
    
    Returns:
    Tensor: The total cost, including the L2 regularization term.
    """
    l2_loss = 0
    # Iterate over all layers in the model
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            # Add the L2 regularization penalty for each layer
            l2_loss += tf.reduce_sum(layer.kernel_regularizer(layer.kernel))
    
    # Total cost is the original cost plus the L2 regularization penalty
    return cost + l2_loss
