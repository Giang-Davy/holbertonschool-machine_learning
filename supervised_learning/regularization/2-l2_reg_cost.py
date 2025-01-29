#!/usr/bin/env python3

import tensorflow as tf

def l2_reg_cost(cost, model):
    l2_loss = sum(tf.reduce_sum(layer.losses) for layer in model.layers)
    total_cost = cost + l2_loss
    return total_cost
