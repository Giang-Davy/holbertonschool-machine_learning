#!/usr/bin/env python3
"""1-q_init.py"""


import numpy as np


def q_init(env):
    """cree une q table"""

    q = np.zeros((env.observation_space.n, env.action_space.n))

    return q
