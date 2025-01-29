#!/usr/bin/env python3
"""fonction"""

import tensorflow as tf

def l2_reg_cost(cost, model):
    l2_loss = [tf.reduce_sum(layer.losses) for layer in model.layers]
    total_cost = tf.concat([[cost], l2_loss], axis=0)
    return total_cost[1:]  # Return the total cost for each layer excluding the first cost
