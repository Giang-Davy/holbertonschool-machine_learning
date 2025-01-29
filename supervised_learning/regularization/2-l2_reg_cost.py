#!/usr/bin/env python3
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a neural network with L2 regularization for each layer.

    Args:
    - cost: Tensor, the original cost of the model without L2 regularization.
    - model: Keras model, the neural network model with L2 regularization.

    Returns:
    - Tensor, the total cost with L2 regularization for each layer.
    """
    reg_losses = []
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            reg_loss = tf.reduce_sum(layer.kernel_regularizer(layer.kernel))
            reg_losses.append(reg_loss)
    
    reg_losses = tf.convert_to_tensor(reg_losses, dtype=tf.float32)
    total_cost = cost + tf.reduce_sum(reg_losses)
    
    return tf.concat([reg_losses, [total_cost]], axis=0)&
