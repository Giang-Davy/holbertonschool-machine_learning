#!/usr/bin/env python3
"""fonction"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """apprentissage decroissant"""
    decayed_learning_rate = alpha / (1 + decay_rate * (
        global_step // decay_step))
    return decayed_learning_rate
