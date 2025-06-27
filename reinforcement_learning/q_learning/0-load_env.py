#!/usr/bin/env python3
"""0-load_env.py"""


import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """charger l'environnement du lac gelé"""

    # update du code avec 'render_mode="ansi"'
    env = gym.make('FrozenLake', map_name=map_name,
                   is_slippery=is_slippery,
                   desc=desc, render_mode="ansi")

    return env
