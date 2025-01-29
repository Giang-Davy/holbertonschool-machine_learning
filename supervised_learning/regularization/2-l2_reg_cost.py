#!/usr/bin/env python3
"""fonction"""


import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calcul du coût avec régularisation L2"""
    l2_regularization = 0
    
    # Calculer la somme des carrés des poids pour chaque couche avec L2
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            l2_regularization += tf.reduce_sum(tf.square(layer.kernel))
    
    # Calculer le coût total avec la régularisation
    total_cost = cost + l2_regularization
    return total_cost
