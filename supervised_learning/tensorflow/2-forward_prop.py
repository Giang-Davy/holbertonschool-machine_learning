#!/usr/bin/env python3
"""fonction"""


import tensorflow.compat.v1 as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """fffffffff"""
    input_layer = x
    for i in range(len(layer_sizes)):
        input_layer = create_layer(input_layer, layer_sizes[i], activations[i])
    return input_layer
